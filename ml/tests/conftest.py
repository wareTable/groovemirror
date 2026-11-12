"""Synthetic fixtures only; these are not real-dataset benchmark recordings."""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def synthetic_audio(tmp_path):
    """Write generated samples to an isolated WAV; never commit fixture audio."""
    def write(samples=None, *, sample_rate=22050, subtype="FLOAT", name="synthetic.wav"):
        if samples is None:
            times = np.arange(sample_rate, dtype=np.float64) / sample_rate
            samples = 0.1 * np.sin(2 * np.pi * 440 * times)
        path = tmp_path / name
        sf.write(path, np.asarray(samples), sample_rate, subtype=subtype)
        return path
    return write


@pytest.fixture
def annotations_file(tmp_path):
    """Write explicitly synthetic normalized event annotations."""
    def write(events, *, recording_id="synthetic-test", name="synthetic.json"):
        path = tmp_path / name
        path.write_text(json.dumps({"recording_id": recording_id, "events": events}), encoding="utf-8")
        return path
    return write
