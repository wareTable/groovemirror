"""Synthetic tests for the shared Phase 2 feature/target contract."""
import json
from pathlib import Path

import librosa
import numpy as np
import pytest

from groovemirror_ml.features.audio import AudioData, load_audio
from groovemirror_ml.features.patches import (
    CLASS_LABELS,
    PatchConfig,
    annotation_targets,
    event_patches,
)


def _audio(samples, sample_rate=22050):
    samples = np.asarray(samples, dtype=np.float64)
    return AudioData(
        samples, sample_rate, sample_rate, 1, samples.size / sample_rate,
        Path("synthetic.wav"), 0.0,
    )


def test_config_roundtrip_and_default_shape():
    config = PatchConfig()
    encoded = json.loads(json.dumps(config.to_dict()))
    assert encoded["version"] == "logmel-v1"
    assert PatchConfig.from_dict(encoded) == config
    assert config.shape == (40, 24)
    with pytest.raises(ValueError, match="version"):
        PatchConfig.from_dict({"version": "future"})
    with pytest.raises(ValueError, match="version"):
        PatchConfig.from_dict({})
    del encoded["hop_length"]
    with pytest.raises(ValueError, match="fields"):
        PatchConfig.from_dict(encoded)


@pytest.mark.parametrize(
    "changes", [{"sample_rate": 0}, {"n_fft": 1023}, {"hop_length": 1025},
                {"pre_frames": -1}, {"post_frames": 0}, {"n_mels": True}],
)
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        PatchConfig(**changes)


def test_targets_merge_only_exact_timestamps_in_canonical_order():
    near = np.nextafter(0.125, np.inf)
    events = [
        {"time": near, "instrument": "hihat"},
        {"time": 0.125, "instrument": "snare"},
        {"time": 0.0, "instrument": "hihat"},
        {"time": 0.125, "instrument": "kick"},
        {"time": 0.125, "instrument": "kick"},
    ]
    assert CLASS_LABELS == ("kick", "snare", "hihat")
    times, targets = annotation_targets(iter(events))
    assert times.dtype == np.float64
    assert targets.dtype == np.float32
    np.testing.assert_array_equal(times, [0.0, 0.125, near])
    np.testing.assert_array_equal(targets, [[0, 0, 1], [1, 1, 0], [0, 0, 1]])


@pytest.mark.parametrize("event", [None, {}, {"time": np.nan, "instrument": "kick"},
                                  {"time": -1, "instrument": "kick"},
                                  {"time": True, "instrument": "kick"},
                                  {"time": 0, "instrument": "ride"}])
def test_targets_reject_invalid_events(event):
    with pytest.raises(ValueError):
        annotation_targets([event])


def test_empty_times_and_empty_waveform():
    times, targets = annotation_targets([])
    assert times.shape == (0,) and times.dtype == np.float64
    assert targets.shape == (0, 3) and targets.dtype == np.float32
    patches = event_patches(_audio([]), times)
    assert patches.shape == (0, 40, 24) and patches.dtype == np.float32
    np.testing.assert_array_equal(event_patches(_audio([]), [0]), -100.0)


def test_short_clip_pads_both_edges_without_wrap_or_shifting():
    audio = _audio([1.0, 0.0, -0.5])
    # Both frame grids include windows lying entirely outside the source.
    end, start = event_patches(audio, [audio.duration_seconds, 0])
    np.testing.assert_array_equal(end[:, :2], -100.0)
    np.testing.assert_array_equal(start[:, :3], -100.0)
    assert end[:, 4].max() > -100.0 and start[:, 4].max() > -100.0
    np.testing.assert_array_equal(end[:, 6:], -100.0)
    np.testing.assert_array_equal(start[:, 7:], -100.0)
    assert not np.array_equal(end, start)


def test_patch_matches_independent_zero_padded_librosa_reference():
    config = PatchConfig()
    audio = _audio(np.random.default_rng(7).normal(size=2051))
    centers = [audio.samples.size, 0, 637, 1]
    times = [center / config.sample_rate for center in centers[:-1]]
    times.append(0.5 / config.sample_rate)  # halfway between samples 0 and 1
    padded = np.pad(audio.samples, config.n_fft + config.hop_length * config.shape[1])
    pad = (padded.size - audio.samples.size) // 2
    mel = librosa.filters.mel(sr=config.sample_rate, n_fft=config.n_fft,
                              n_mels=config.n_mels, dtype=np.float64)
    patches = event_patches(audio, iter(times), config)
    for center, patch in zip(centers, patches, strict=True):
        start = pad + center - config.pre_frames * config.hop_length - config.n_fft // 2
        end = start + config.n_fft + (config.shape[1] - 1) * config.hop_length
        stft = librosa.stft(padded[start:end], n_fft=config.n_fft,
                            hop_length=config.hop_length, center=False)
        expected = librosa.power_to_db(mel @ np.abs(stft)**2, top_db=None)
        np.testing.assert_allclose(patch, expected, atol=5e-6, rtol=0)


def test_shared_audio_loader_stereo_resampling_and_independent_levels(synthetic_audio):
    rng = np.random.default_rng(10)
    mono = rng.uniform(-0.2, 0.2, 44100)
    stereo = np.column_stack([mono / 2, mono * 1.5])
    audio = load_audio(synthetic_audio(stereo, sample_rate=44100, subtype="FLOAT"))
    assert audio.original_channels == 2 and audio.original_sample_rate == 44100
    assert audio.sample_rate == PatchConfig().sample_rate
    patch = event_patches(audio, [0.3])
    half = event_patches(_audio(audio.samples / 2), [0.3])
    assert patch.shape == (1, 40, 24) and patch.dtype == np.float32
    np.testing.assert_allclose(patch - half, 20 * np.log10(2), atol=5e-6)


def test_thousands_of_events_keep_order_and_match_individual_rows():
    audio = _audio(np.random.default_rng(123).uniform(-0.2, 0.2, 22050))
    times = np.random.default_rng(456).uniform(0.0, 1.0, 2049)
    times[-1] = times[0]
    patches = event_patches(audio, times)
    selected = [2048, 42, 41, 1, 0]
    assert patches.shape == (2049, 40, 24)
    assert np.isfinite(patches).all()
    np.testing.assert_array_equal(patches[0], patches[-1])
    np.testing.assert_array_equal(patches[selected], event_patches(audio, times[selected]))


@pytest.mark.parametrize("times", [[-0.001], [1.001], [np.nan], [np.inf], [[0.0]], ["0"], [True]])
def test_invalid_candidate_times(times):
    with pytest.raises(ValueError, match="times_seconds"):
        event_patches(_audio(np.zeros(22050)), times)


def test_sample_rate_mismatch_does_not_resample_implicitly():
    with pytest.raises(ValueError, match="sample rate"):
        event_patches(_audio([1.0], sample_rate=44100), [0])
