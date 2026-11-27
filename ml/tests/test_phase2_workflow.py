"""Synthetic tests of split boundaries and the frozen-run interface."""
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest

from groovemirror_ml import phase2


class FakeModel:
    def predict_proba(self, inputs):
        return np.tile([[0.8, 0.7, 0.4]], (len(inputs), 1))


def args(tmp_path):
    root = tmp_path.resolve()
    manifest = root / 'manifest.json'
    manifest.write_text('{}', encoding='utf-8')
    audio = root / 'audio.yaml'
    audio.write_text('{}', encoding='utf-8')
    return Namespace(data_root=root, manifest=manifest, audio_config=audio,
                     output=root / 'run', epochs=1, seed=0, tolerance=0.05, batch_size=2)


def test_training_never_opens_test_audio_and_holds_frozen_config(tmp_path, monkeypatch):
    options = args(tmp_path)
    calls = []
    manifest = {'manifest_version': 'synthetic-v1', 'recordings': [
        {'split': split} for split in ('train', 'val', 'test')]}
    monkeypatch.setattr(phase2, 'load_split_manifest', lambda path: manifest)

    def fake_read(row, *_args, detected):
        calls.append(row['split'])
        if row['split'] == 'test':
            pytest.fail('Training attempted to open held-out files')
        assert detected == (row['split'] == 'val')
        patches = np.zeros((2, 40, 24), dtype=np.float32)
        y = np.array([[1, 1, 0], [0, 0, 1]])
        refs = [{'time': 0.3, 'instrument': label} for label in ('kick', 'snare')]
        return phase2.Recording(refs, y, patches, np.array([0.3, 0.6]), patches)

    monkeypatch.setattr(phase2, 'read_recording', fake_read)
    monkeypatch.setattr(phase2, 'fit_classical', lambda *_a, **_k: (FakeModel(), {}))
    monkeypatch.setattr(phase2, 'fit_cnn', lambda *_a, **_k: (FakeModel(), {}))
    monkeypatch.setattr(phase2, 'save_model', lambda model, path: path.write_bytes(b'fake model'))
    monkeypatch.setattr(phase2, 'load_model', lambda path: FakeModel())
    phase2.train(options)
    assert calls == ['train', 'val']
    frozen, row, *_ = phase2.load_candidate(options.output)
    assert frozen['selected_model'] == 'classical'  # deterministic tie
    assert frozen['selection_split'] == 'val'
    assert row['thresholds'] and row['artifact_reload_equal']
    holdout = Namespace(**vars(options))
    holdout.run = options.output
    holdout.frozen_sha256 = phase2.sha256(options.output / 'frozen.json')
    holdout.output = options.data_root / 'test.json'
    options.manifest.write_text('{"changed":true}', encoding='utf-8')
    with pytest.raises(ValueError, match='differs'):
        phase2.evaluate(holdout)
    frozen['patch_config']['hop_length'] = 128
    (options.output / 'frozen.json').write_text(json.dumps(frozen), encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        phase2.load_candidate(options.output)
    new_digest = phase2.sha256(options.output / 'frozen.json')
    (options.output / 'frozen.sha256').write_text(new_digest, encoding='ascii')
    with pytest.raises(ValueError, match='checksum'):
        phase2.load_candidate(options.output, reviewed_digest=holdout.frozen_sha256)
    with pytest.raises(ValueError, match='preprocessing'):
        phase2.load_candidate(options.output)


def test_resists_file_escape_and_checksum_change(tmp_path):
    root = (tmp_path / 'dataset').resolve()
    root.mkdir()
    local = root / 'data.txt'
    local.write_bytes(b'initial')
    digest = phase2.sha256(local)
    assert phase2.checked_input(root, 'data.txt', digest) == local
    local.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        phase2.checked_input(root, 'data.txt', digest)
    with pytest.raises(ValueError, match='path'):
        phase2.checked_input(root, '../outside.txt', digest)


def test_threshold_selection_uses_fixed_grid_and_handles_missing_classes():
    targets = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1]])
    scores = np.array([[0.4, 0.3, 0.1], [0.3, 0.2, 0.2], [0.55, 0.1, 0.3]])
    assert phase2.choose_thresholds(targets, scores) == [0.4, 0.5, 0.5]


def test_prediction_schema_retains_simultaneous_labels():
    document = phase2.event_prediction(FakeModel(), np.array([1.25]), np.zeros((1, 40, 24)),
                                       [0.5, 0.5, 0.5], 'classical-v1', 'logmel-v1')
    assert document['schema_version'] == 1
    assert document['events'][0]['time_seconds'] == 1.25
    assert document['events'][0]['labels'] == ['kick', 'snare']
    assert set(document['events'][0]['probabilities']) == set(phase2.INSTRUMENTS)


def test_reports_are_not_overwritten(tmp_path):
    path = Path(tmp_path) / 'report.json'
    phase2.write_json(path, {'first': 1})
    with pytest.raises(FileExistsError):
        phase2.write_json(path, {'replacement': 2})
    assert json.loads(path.read_text(encoding='utf-8')) == {'first': 1}
