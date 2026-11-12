"""Reproducible Phase 1 WAV onset-baseline command (no classifier/services)."""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, TypeVar

import numpy as np
import yaml

from groovemirror_ml.data import load_annotations
from groovemirror_ml.evaluation import EvaluationConfig, evaluate_onsets
from groovemirror_ml.features import AudioConfig, detect_onsets, detector_parameters, load_audio


ConfigType = TypeVar("ConfigType", AudioConfig, EvaluationConfig)


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML typo protection: repeated keys must not silently overwrite values."""


def _unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("Configuration keys must be strings")
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _load_config(path: Path, kind: type[ConfigType]) -> tuple[ConfigType, dict[str, str]]:
    content = path.read_bytes()
    if len(content) > 1024 * 1024:
        raise ValueError(f"Configuration exceeds 1 MiB: {path}")
    try:
        values = yaml.load(content.decode("utf-8-sig"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must be a flat YAML mapping: {path}")
    unknown = set(values) - {field.name for field in fields(kind)}
    if unknown:
        raise ValueError(f"Unknown {kind.__name__} keys in {path}: {', '.join(sorted(unknown))}")
    return kind(**values), {"path": str(path.resolve()), "sha256": _digest(content)}


def _software_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("groovemirror", "numpy", "scipy", "librosa", "soundfile", "soxr", "matplotlib", "PyYAML"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _code_lineage() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    source_files = [{"path": str(path.relative_to(package)).replace(os.sep, "/"),
                     "sha256": _file_hash(path)} for path in sorted(package.rglob("*.py"))]
    code: dict[str, Any] = {
        "git_commit": None, "git_status": "unversioned", "git_dirty": None,
        "source_sha256": _canonical_hash(source_files), "source_files": source_files,
        "source_hash_scope": "all Python source files beneath groovemirror_ml package",
    }
    try:
        commit = subprocess.run(["git", "-C", str(package), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5, check=False)
        if commit.returncode == 0:
            status = subprocess.run(["git", "-C", str(package), "status", "--porcelain"],
                                    capture_output=True, text=True, timeout=5, check=False)
            code["git_commit"] = commit.stdout.strip()
            code["git_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
            code["git_status"] = ("modified" if code["git_dirty"] else "clean") if status.returncode == 0 else "status_unknown"
    except (OSError, subprocess.SubprocessError):
        code["git_status"] = "unversioned_or_git_unavailable"
    return code


def _plot(path: Path, audio: Any, detected: Any, references: list[float], recording_id: str, hop_length: int) -> None:
    # Object-oriented Agg canvas works headlessly without switching global backend.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(13, 7), constrained_layout=True)
    FigureCanvasAgg(figure)
    waveform, strength, timeline = figure.subplots(3, 1, sharex=True,
                                                 gridspec_kw={"height_ratios": [2, 1, 1]})
    stride = max(1, int(np.ceil(audio.samples.size / 100_000)))
    x = np.arange(0, audio.samples.size, stride) / audio.sample_rate
    waveform.plot(x, audio.samples[::stride], color="0.35", linewidth=0.5)
    waveform.set_ylabel("Amplitude")
    waveform.set_title(f"{recording_id}: Phase 1 onset baseline (no instrument classification)")
    for axis in (waveform, strength):
        axis.vlines(references, 0, 1, transform=axis.get_xaxis_transform(),
                    color="tab:green", alpha=0.35, linewidth=0.8)
        axis.vlines(detected.times_seconds, 0, 1, transform=axis.get_xaxis_transform(),
                    color="tab:orange", alpha=0.5, linewidth=0.8, linestyle="--")
    envelope_times = np.arange(detected.envelope.size) * hop_length / audio.sample_rate
    strength.plot(envelope_times, detected.envelope, color="tab:blue", linewidth=0.7)
    strength.set_ylabel("Spectral flux")
    timeline.eventplot([references, detected.times_seconds], lineoffsets=[1, 0],
                       linelengths=0.7, colors=["tab:green", "tab:orange"])
    timeline.set_yticks([0, 1], ["Detected", "Reference"])
    timeline.set_ylim(-0.6, 1.6)
    timeline.set_xlim(0, audio.duration_seconds)
    timeline.set_xlabel("Recording time (seconds); identical reference timestamps count once")
    figure.savefig(path, format="png", dpi=140,
                   metadata={"Software": "GrooveMirror Phase 1 onset baseline"})
    figure.clear()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a WAV and evaluate a librosa onset-only baseline.")
    parser.add_argument("audio", type=Path, help="Input WAV recording")
    parser.add_argument("--annotations", type=Path, required=True, help="Normalized JSON recording_id/events annotations")
    parser.add_argument("--config", type=Path, default=Path("configs/audio.yaml"), help="Flat AudioConfig YAML")
    parser.add_argument("--evaluation-config", type=Path, default=Path("configs/evaluation.yaml"), help="Flat EvaluationConfig YAML")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--recording-id", help="Optional ID; must agree with the normalized annotation record")
    return parser


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    """Analyze one recording and atomically publish individual output files.

    Validation and analysis occur before creating output artifacts. Temporary
    files are cleaned on failure; existing input files are never overwritten.
    A report represents one recording, not a held-out dataset benchmark.
    """
    start = time.perf_counter()
    audio_config, audio_config_file = _load_config(args.config, AudioConfig)
    evaluation_config, evaluation_config_file = _load_config(args.evaluation_config, EvaluationConfig)
    output = args.output_dir.expanduser().resolve()
    source_paths = {path.expanduser().resolve() for path in
                    (args.audio, args.annotations, args.config, args.evaluation_config)}
    if any(output / filename in source_paths for filename in ("report.json", "onsets.png")):
        raise ValueError("Output artifacts would overwrite an input; choose another --output-dir")
    audio = load_audio(args.audio, audio_config)
    annotation_content = args.annotations.read_bytes()
    if len(annotation_content) > 16 * 1024 * 1024:
        raise ValueError("Normalized annotation JSON exceeds 16 MiB")
    try:
        annotation_record = json.loads(annotation_content.decode("utf-8-sig"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Expected normalized annotation JSON: {exc}") from exc
    events = load_annotations(args.annotations, duration_seconds=audio.duration_seconds)
    if not isinstance(annotation_record, dict):
        raise ValueError("Annotations must contain a normalized JSON object")
    recording_id = annotation_record.get("recording_id")
    if not isinstance(recording_id, str) or not recording_id.strip():
        raise ValueError("Annotations require a nonempty recording_id")
    if args.recording_id is not None and args.recording_id != recording_id:
        raise ValueError("--recording-id differs from the annotation recording_id")
    references = sorted({float(event["time"]) for event in events})
    detected = detect_onsets(audio, audio_config)
    metrics = evaluate_onsets(references, detected.times_seconds, evaluation_config)
    configuration = {
        "audio": audio_config.to_dict(), "evaluation": evaluation_config.to_dict(),
        "detector": detector_parameters(audio_config),
        "config_files": {"audio": audio_config_file, "evaluation": evaluation_config_file},
    }
    configuration["effective_sha256"] = _canonical_hash({
        key: configuration[key] for key in ("audio", "evaluation", "detector")
    })
    source_metadata = {key: annotation_record[key] for key in
                       ("source", "provenance", "recording_group", "dataset_version", "annotation_version", "split")
                       if key in annotation_record}
    report: dict[str, Any] = {
        "schema_version": "1.0", "analysis_type": "phase1_onset_baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "recording_id": recording_id,
        "audio": audio.to_dict(), "configuration": configuration,
        "onsets": {"reference_times_seconds": references,
                   "predicted_times_seconds": detected.times_seconds.tolist(),
                   "strengths": detected.strengths.tolist(),
                   "annotation_event_count": len(events),
                   "unique_reference_onset_count": len(references)},
        "metrics": metrics.to_dict(),
        "evaluation_convention": {
            "scope": "single_recording_instrument_agnostic_onset_detection",
            "reference_policy": "identical annotation timestamps collapse to one onset across all instruments",
            "prediction_policy": "duplicates preserved and counted separately",
            "matching": "maximum_cardinality_then_minimum_total_absolute_error_ordered_sparse_dp",
            "tie_break": "earliest DP node in stable reference/prediction traversal",
            "tolerance_boundary": "inclusive; 1e-12 seconds floating-point slack except at zero tolerance",
            "signed_error": "prediction minus reference; positive=late, negative=early",
            "empty_denominator": 0.0, "empty_timing_statistic": None,
            "offset_correction_seconds": 0.0,
            "maximum_events_per_input": 100000, "maximum_candidate_pairs": 1000000,
        },
        "lineage": {
            "audio": {"path": str(audio.source_path), "sha256": _file_hash(audio.source_path)},
            "annotations": {"path": str(args.annotations.resolve()),
                            "sha256": _digest(annotation_content), "metadata": source_metadata},
            "code": _code_lineage(), "software": _software_versions(),
        },
        "artifacts": {"report": "report.json", "onset_plot": "onsets.png"},
        "limitations": [
            "One recording only; this is not a held-out benchmark or an acceptance-target claim.",
            "No instrument classifier, per-instrument metrics, rhythm ability score, or recording-latency calibration.",
            "Frame-grid timing, boundary padding and resampling can affect detections near the start/end.",
            "Near-simultaneous but unequal annotation times remain distinct and may lower onset-only recall.",
            "Stereo arithmetic-mean mixing may cancel anti-phase signals; no amplitude normalization is applied.",
            "Onset strengths are raw spectral flux, not calibrated confidence probabilities.",
            "Waveform is decimated for display only; detection uses the complete validated signal.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        for suffix in (".png", ".json"):
            descriptor, filename = tempfile.mkstemp(prefix=".baseline-", suffix=suffix, dir=output)
            os.close(descriptor)
            temporary_paths.append(Path(filename))
        plot_temp, report_temp = temporary_paths
        _plot(plot_temp, audio, detected, references, recording_id, audio_config.hop_length)
        report["analysis_duration_seconds"] = time.perf_counter() - start
        # allow_nan=False ensures no non-standard NaN/Infinity escapes into artifacts.
        report_temp.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(plot_temp, output / "onsets.png")
        os.replace(report_temp, output / "report.json")
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_baseline(args)
    except (ValueError, OSError, RuntimeError, TypeError) as exc:
        print(f"groovemirror-baseline: error: {exc}", file=sys.stderr)
        return 2
    metrics = report["metrics"]
    timing = metrics["mean_absolute_error_seconds"]
    timing_text = "n/a (no matched events)" if timing is None else f"{timing * 1000:.3f} ms"
    median = metrics["median_absolute_error_seconds"]
    median_text = "n/a (no matched events)" if median is None else f"{median * 1000:.3f} ms"
    print(f"recording_id={report['recording_id']} precision={metrics['precision']:.6f} "
          f"recall={metrics['recall']:.6f} f1={metrics['f1']:.6f} "
          f"matched={metrics['true_positives']} mean_absolute_error={timing_text} "
          f"median_absolute_error={median_text}")
    print(f"Artifacts: {args.output_dir.resolve() / 'report.json'}; {args.output_dir.resolve() / 'onsets.png'}")
    return 0
