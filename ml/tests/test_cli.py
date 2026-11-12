"""CLI smoke uses labelled synthetic fixture audio, not benchmark scores."""
from importlib import metadata
import json
import subprocess
import sys

import numpy as np
import pytest


def command(audio, annotations, audio_config, evaluation_config, output_dir):
    return [sys.executable, "-m", "groovemirror_ml", str(audio), "--annotations", str(annotations),
            "--config", str(audio_config), "--evaluation-config", str(evaluation_config),
            "--output-dir", str(output_dir)]


@pytest.fixture
def configs(tmp_path):
    audio = tmp_path / "synthetic-audio.yaml"
    audio.write_text("sample_rate: 22050\nhop_length: 256\nn_fft: 1024\n", encoding="utf-8")
    evaluation = tmp_path / "synthetic-evaluation.yaml"
    evaluation.write_text("onset_tolerance_seconds: 0.05\n", encoding="utf-8")
    return audio, evaluation


def test_cli_synthetic_end_to_end(synthetic_audio, annotations_file, configs, tmp_path):
    sr = 22050
    samples = np.zeros(sr * 2)
    rng = np.random.default_rng(7)
    for time in (0.25, 0.75, 1.25):
        start = int(time * sr)
        samples[start:start + 600] = rng.normal(0, 0.2, 600) * np.exp(-np.arange(600) / 100)
    audio = synthetic_audio(samples)
    annotations = annotations_file([
        {"time": 0.25, "instrument": "kick"},
        {"time": 0.25, "instrument": "hihat"},
        {"time": 0.75, "instrument": "snare"},
        {"time": 1.25, "instrument": "kick"},
    ])
    output = tmp_path / "synthetic-output"
    completed = subprocess.run(command(audio, annotations, *configs, output), capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert {"metrics", "onsets", "audio", "lineage"} <= report.keys()
    assert report["schema_version"] == "1.0"
    assert report["analysis_type"] == "phase1_onset_baseline"
    assert report["lineage"]["software"]["groovemirror"] == metadata.version("groovemirror")
    for source in ("audio", "annotations"):
        assert len(report["lineage"][source]["sha256"]) == 64
    assert report["configuration"]["audio"]["sample_rate"] == 22050
    assert report["configuration"]["evaluation"]["onset_tolerance_seconds"] == 0.05
    metrics = report["metrics"]
    assert {"precision", "recall", "f1", "true_positives", "false_positives", "false_negatives", "matches",
            "mean_absolute_error_seconds", "median_absolute_error_seconds", "onset_tolerance_seconds"} <= metrics.keys()
    assert metrics["onset_tolerance_seconds"] == 0.05
    for key in ("precision", "recall", "f1"):
        assert 0 <= metrics[key] <= 1
    onsets = report["onsets"]
    assert onsets["reference_times_seconds"] == [0.25, 0.75, 1.25]
    assert len(onsets["predicted_times_seconds"]) == len(onsets["strengths"])
    assert all(np.isfinite(time) for time in onsets["predicted_times_seconds"])
    assert (output / "onsets.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "precision" in completed.stdout.lower()
    assert "recall" in completed.stdout.lower()
    assert "f1" in completed.stdout.lower()


@pytest.mark.parametrize("invalid_yaml", [
    "sample_rate: 22050\nsample_rate: 16000\n",
    "sample_rate_typo: 22050\n", "max_duration_seconds: .nan\n", "[22050]\n",
    "sample_rate: true\n", "hop_length: 2.5\n",
])
def test_cli_invalid_config_is_actionable(invalid_yaml, annotations_file, configs, tmp_path, capsys):
    from groovemirror_ml.cli import main

    audio_config, evaluation_config = configs
    audio_config.write_text(invalid_yaml, encoding="utf-8")
    code = main([str(tmp_path / "missing.wav"), "--annotations", str(annotations_file([])),
                 "--config", str(audio_config), "--evaluation-config", str(evaluation_config),
                 "--output-dir", str(tmp_path / "invalid-config-output")])
    assert code == 2
    error = capsys.readouterr().err
    assert error.strip()
    assert "traceback" not in error.lower()
    assert not (tmp_path / "invalid-config-output" / "report.json").exists()


def test_cli_corrupt_audio_actionable_error(annotations_file, configs, tmp_path):
    corrupt = tmp_path / "synthetic-corrupt.wav"
    corrupt.write_bytes(b"not audio")
    output = tmp_path / "error-output"
    completed = subprocess.run(command(corrupt, annotations_file([]), *configs, output), capture_output=True, text=True, timeout=120)
    assert completed.returncode == 2
    assert completed.stderr.strip()
    assert "traceback" not in completed.stderr.lower()
    assert not (output / "report.json").exists()


def test_cli_invalid_annotation_range(synthetic_audio, annotations_file, configs, tmp_path):
    completed = subprocess.run(command(synthetic_audio(), annotations_file([{"time": 2, "instrument": "kick"}]),
                                       *configs, tmp_path / "invalid-output"), capture_output=True, text=True, timeout=120)
    assert completed.returncode == 2
    assert completed.stderr.strip()
    assert "traceback" not in completed.stderr.lower()
