"""Train and evaluate the Phase 2 instrument classifier in separate steps."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import yaml

from groovemirror_ml.data import load_annotations
from groovemirror_ml.data.phase2 import load_split_manifest
from groovemirror_ml.evaluation.instruments import (
    INSTRUMENTS,
    dataset_event_metrics,
    patch_metrics,
    prediction_document,
)
from groovemirror_ml.features import AudioConfig, detect_onsets, load_audio
from groovemirror_ml.features.patches import PatchConfig, annotation_targets, event_patches
from groovemirror_ml.instrument_models import fit_classical, fit_cnn, load_model, save_model

RUN_SCHEMA = 1
THRESHOLD_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
KINDS = ('classical', 'cnn')


@dataclass
class Recording:
    events: list[dict]
    targets: np.ndarray
    patches: np.ndarray
    detected_times: np.ndarray
    detected_patches: np.ndarray


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def config_version(audio: dict, patch: dict) -> str:
    serialized = json.dumps({'audio': audio, 'patch': patch}, sort_keys=True, separators=(',', ':'))
    return patch['version'] + '+' + hashlib.sha256(serialized.encode()).hexdigest()


def write_json(path: Path, document: dict) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def checked_input(root: Path, relative: str, digest: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or sha256(path) != digest:
        raise ValueError(f'Input path or checksum does not match the manifest: {relative}')
    return path


def read_recording(row: dict, root: Path, audio_config: AudioConfig, patch_config: PatchConfig,
                   *, detected: bool) -> Recording:
    audio_path = checked_input(root, row['audio_path'], row['audio_sha256'])
    xml = checked_input(root, row['raw_annotation_path'], row['raw_annotation_sha256'])
    audio = load_audio(audio_path, audio_config)
    events = load_annotations(xml, duration_seconds=audio.duration_seconds)
    times, targets = annotation_targets(events)
    patches = event_patches(audio, times, patch_config)
    if detected:
        detected_times = detect_onsets(audio, audio_config).times_seconds
        detected_patches = event_patches(audio, detected_times, patch_config)
    else:
        detected_times = np.empty(0, dtype=np.float64)
        detected_patches = np.empty((0, *patch_config.shape), dtype=np.float32)
    return Recording(events, targets, patches, detected_times, detected_patches)


def load_partition(manifest: dict, split: str, root: Path, audio: AudioConfig,
                   patch: PatchConfig) -> list[Recording]:
    if split not in ('train', 'val', 'test'):
        raise ValueError(f'Unknown partition: {split}')
    rows = [row for row in manifest['recordings'] if row['split'] == split]
    if not rows:
        raise ValueError(f'Empty {split} partition')
    # Filter before opening audio or annotations. Training never invokes this for test.
    return [read_recording(row, root, audio, patch, detected=split != 'train') for row in rows]


def stack(rows: list[Recording]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([row.patches for row in rows]), np.concatenate([row.targets for row in rows])


def choose_thresholds(y: np.ndarray, probabilities: np.ndarray) -> list[float]:
    """Choose per-class patch F1 on validation. Ties favor 0.5, then higher."""
    thresholds = []
    for column, _ in enumerate(INSTRUMENTS):
        actual = y[:, column] == 1
        if not actual.any() or actual.all():
            thresholds.append(0.5)
            continue
        def objective(limit: float) -> tuple[float, float, float]:
            guess = probabilities[:, column] >= limit
            score = 2 * np.count_nonzero(actual & guess) / (actual.sum() + guess.sum())
            return float(score), -abs(round(limit - 0.5, 6)), limit
        thresholds.append(max(THRESHOLD_GRID, key=objective))
    return thresholds


def event_prediction(model, times: np.ndarray, patches: np.ndarray, thresholds: list[float],
                     model_version: str, preprocessing_version: str) -> dict:
    probabilities = model.predict_proba(patches) if len(patches) else np.empty((0, len(INSTRUMENTS)))
    return prediction_document(times, probabilities, thresholds, model_version=model_version,
                               preprocessing_version=preprocessing_version)


def score_events(model, rows: list[Recording], thresholds: list[float], model_version: str,
                 preprocessing_version: str, tolerance: float) -> dict:
    pairs = ((row.events, event_prediction(model, row.detected_times, row.detected_patches,
                                          thresholds, model_version, preprocessing_version))
             for row in rows)
    return dataset_event_metrics(pairs, tolerance_seconds=tolerance)


def source_lineage() -> dict:
    package = Path(__file__).resolve().parent
    hashes = {path.relative_to(package).as_posix(): sha256(path) for path in sorted(package.rglob('*.py'))}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=package,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'],
                                         cwd=package, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return {'git_commit': None, 'git_dirty': None, 'source_sha256_by_path': hashes}
    return {'git_commit': commit, 'git_dirty': bool(status), 'source_sha256_by_path': hashes}


def environment() -> dict:
    libraries = ('numpy', 'scipy', 'librosa', 'soxr', 'soundfile', 'scikit-learn', 'torch', 'PyYAML')
    import torch

    thread_vars = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
    return {'python': platform.python_version(), 'platform': platform.platform(), 'device': 'cpu',
            'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
            'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'torch_mkldnn_enabled': torch.backends.mkldnn.enabled,
            'thread_environment': {key: os.environ.get(key) for key in thread_vars},
            'libraries': {lib: metadata.version(lib) for lib in libraries}}


def train(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError('Run output already exists; specify a new run directory')
    manifest = load_split_manifest(args.manifest)
    if not 1 <= args.epochs <= 100 or not 1 <= args.batch_size <= 512:
        raise ValueError('epochs must be 1–100 and batch size must be 1–512')
    if not np.isfinite(args.tolerance) or not 0 <= args.tolerance <= 1:
        raise ValueError('tolerance must be finite and in [0, 1] seconds')
    audio = AudioConfig(**yaml.safe_load(args.audio_config.read_text(encoding='utf-8')))
    patch = PatchConfig(sample_rate=audio.sample_rate)
    preprocessing = config_version(audio.to_dict(), patch.to_dict())
    root = args.data_root.resolve()
    x_train, y_train = stack(load_partition(manifest, 'train', root, audio, patch))
    val = load_partition(manifest, 'val', root, audio, patch)
    x_val, y_val = stack(val)
    args.output.mkdir(parents=True, exist_ok=False)
    runs = {}
    for kind in KINDS:
        if kind == 'classical':
            model, history = fit_classical(x_train, y_train, seed=args.seed)
        else:
            model, history = fit_cnn(x_train, y_train, x_val, y_val, seed=args.seed,
                                     epochs=args.epochs, batch_size=args.batch_size)
        artifact = args.output / f'{kind}.npz'
        save_model(model, artifact)
        digest = sha256(artifact)
        restored = load_model(artifact)
        probabilities = model.predict_proba(x_val)
        if not np.array_equal(probabilities, restored.predict_proba(x_val)):
            raise ValueError(f'{kind} artifact reload changed validation predictions')
        version = f'{kind}-sha256:{digest}'
        thresholds = choose_thresholds(y_val, probabilities)
        validation_events = score_events(restored, val, thresholds, version, preprocessing, args.tolerance)
        runs[kind] = {'artifact_sha256': digest, 'model_version': version, 'thresholds': thresholds,
                      'history': history, 'train_patches': patch_metrics(y_train, model.predict_proba(x_train), thresholds),
                      'validation_patches': patch_metrics(y_val, probabilities, thresholds),
                      'validation_events': validation_events, 'artifact_reload_equal': True}
    best = max(KINDS, key=lambda kind: runs[kind]['validation_events']['macro']['f1'])
    frozen = {'schema_version': RUN_SCHEMA, 'manifest_version': manifest['manifest_version'],
              'manifest_sha256': sha256(args.manifest), 'instruments': list(INSTRUMENTS),
              'audio_config': audio.to_dict(), 'patch_config': patch.to_dict(),
              'preprocessing_version': preprocessing, 'tolerance_seconds': args.tolerance,
              'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
              'code': source_lineage(), 'environment': environment(), 'selection_split': 'val',
              'selection_rule': 'validation detected-event macro F1; ties favor classical',
              'threshold_rule': 'per-class validation reference-patch F1 on fixed grid',
              'selected_model': best, 'models': runs}
    freeze_path = args.output / 'frozen.json'
    write_json(freeze_path, frozen)
    digest = sha256(freeze_path)
    with (args.output / 'frozen.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest + '\n')
    print(f'Frozen {best}: {freeze_path} (SHA-256 {digest})')


def load_candidate(directory: Path, *, reviewed_digest: str | None = None
                   ) -> tuple[dict, dict, object, AudioConfig, PatchConfig, str]:
    frozen_bytes = (directory / 'frozen.json').read_bytes()
    freeze_digest = hashlib.sha256(frozen_bytes).hexdigest()
    recorded_digest = (directory / 'frozen.sha256').read_text(encoding='ascii').strip()
    if recorded_digest != freeze_digest or (reviewed_digest is not None and reviewed_digest.lower() != freeze_digest):
        raise ValueError('Frozen metadata checksum does not match the pinned candidate')
    frozen = json.loads(frozen_bytes)
    kind = frozen['selected_model']
    if (frozen['schema_version'] != RUN_SCHEMA or frozen['selection_split'] != 'val'
            or kind not in KINDS or frozen['instruments'] != list(INSTRUMENTS)):
        raise ValueError('Unsupported frozen candidate metadata')
    audio = AudioConfig(**frozen['audio_config'])
    patch = PatchConfig.from_dict(frozen['patch_config'])
    if frozen['preprocessing_version'] != config_version(audio.to_dict(), patch.to_dict()):
        raise ValueError('Frozen preprocessing settings do not match the recorded version')
    row = frozen['models'][kind]
    path = directory / f'{kind}.npz'
    digest = sha256(path)
    if row['artifact_sha256'] != digest or row['model_version'] != f'{kind}-sha256:{digest}':
        raise ValueError('Frozen model artifact checksum mismatch')
    return frozen, row, load_model(path), audio, patch, freeze_digest


def evaluate(args: argparse.Namespace) -> None:
    frozen, model_row, model, audio, patch, frozen_digest = load_candidate(
        args.run, reviewed_digest=args.frozen_sha256)
    if sha256(args.manifest) != frozen['manifest_sha256']:
        raise ValueError('Split manifest differs from the frozen run')
    if args.output.exists():
        raise ValueError('Holdout report already exists; retain the original result')
    if not args.output.parent.is_dir():
        raise ValueError('Holdout output parent must be an existing directory')
    if source_lineage()['source_sha256_by_path'] != frozen['code']['source_sha256_by_path']:
        raise ValueError('Evaluation package sources differ from the frozen training run')
    manifest = load_split_manifest(args.manifest)
    rows = load_partition(manifest, 'test', args.data_root.resolve(), audio, patch)
    x, y = stack(rows)
    report = {'schema_version': RUN_SCHEMA, 'split': 'test', 'recordings': len(rows),
              'manifest_sha256': frozen['manifest_sha256'], 'frozen_sha256': frozen_digest,
              'model_version': model_row['model_version'], 'preprocessing_version': frozen['preprocessing_version'],
              'patch_metrics': patch_metrics(y, model.predict_proba(x), model_row['thresholds']),
              'event_metrics': score_events(model, rows, model_row['thresholds'], model_row['model_version'],
                                            frozen['preprocessing_version'], frozen['tolerance_seconds'])}
    write_json(args.output, report)
    print(f'Held-out evaluation: {args.output}')


def predict(args: argparse.Namespace) -> None:
    frozen, model_row, model, audio_config, patch_config, _ = load_candidate(args.run)
    audio = load_audio(args.wav, audio_config)
    times = detect_onsets(audio, audio_config).times_seconds
    document = event_prediction(model, times, event_patches(audio, times, patch_config),
                                model_row['thresholds'], model_row['model_version'], frozen['preprocessing_version'])
    write_json(args.output, document)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(required=True)
    train_args = modes.add_parser('train', help='Fit both models on train and select on validation')
    eval_args = modes.add_parser('evaluate', help='Evaluate a frozen candidate on test once')
    pred_args = modes.add_parser('predict', help='Produce multi-label predictions for a WAV')
    for command in (train_args, eval_args):
        command.add_argument('--manifest', type=Path, required=True)
        command.add_argument('--data-root', type=Path, default=Path.cwd())
    for command in (train_args, eval_args, pred_args):
        command.add_argument('--output', type=Path, required=True)
    for command in (eval_args, pred_args):
        command.add_argument('--run', type=Path, required=True)
    train_args.add_argument('--audio-config', type=Path, default=Path('configs/audio.yaml'))
    train_args.add_argument('--tolerance', type=float, default=0.05)
    train_args.add_argument('--seed', type=int, default=0)
    train_args.add_argument('--epochs', type=int, default=12)
    train_args.add_argument('--batch-size', type=int, default=64)
    pred_args.add_argument('wav', type=Path)
    train_args.set_defaults(command=train)
    eval_args.add_argument('--frozen-sha256', required=True, help='Digest pinned before inspecting test results')
    eval_args.set_defaults(command=evaluate)
    pred_args.set_defaults(command=predict)
    args = parser.parse_args()
    try:
        args.command(args)
    except (OSError, KeyError, TypeError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        print(f'Phase 2: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
