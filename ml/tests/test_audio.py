"""Audio contracts use generated synthetic WAVs, not committed audio."""
import numpy as np
import pytest

from groovemirror_ml.features import AudioConfig, AudioValidationError, detect_onsets, load_audio


def test_valid_mono_audio(synthetic_audio):
    path = synthetic_audio()
    audio = load_audio(path)
    assert audio.sample_rate == 22050
    assert audio.original_sample_rate == 22050
    assert audio.original_channels == 1
    assert audio.samples.ndim == 1
    assert audio.duration_seconds == pytest.approx(1)
    assert np.all(np.isfinite(audio.samples))
    assert audio.rms > 0


def test_stereo_downmix_is_arithmetic_mean(synthetic_audio):
    values = np.column_stack([np.full(22050, 0.1), np.full(22050, 0.3)])
    audio = load_audio(synthetic_audio(values))
    assert audio.original_channels == 2
    np.testing.assert_allclose(audio.samples, 0.2, atol=1e-7)


def test_resampling_preserves_duration(synthetic_audio):
    audio = load_audio(synthetic_audio(sample_rate=44100), AudioConfig(sample_rate=16000))
    assert audio.original_sample_rate == 44100
    assert audio.sample_rate == 16000
    assert len(audio.samples) == 16000
    assert audio.duration_seconds == pytest.approx(1, abs=1 / 16000)
    assert np.all(np.isfinite(audio.samples))


def test_corrupt_audio(tmp_path):
    path = tmp_path / "synthetic-corrupt.wav"
    path.write_bytes(b"not a WAV recording")
    with pytest.raises(AudioValidationError):
        load_audio(path)


def test_missing_audio(tmp_path):
    with pytest.raises(AudioValidationError):
        load_audio(tmp_path / "missing.wav")


@pytest.mark.parametrize("samples", [np.zeros(22050), np.empty(0), np.column_stack([np.full(22050, 0.1), np.full(22050, -0.1)])])
def test_empty_silent_or_cancelled_audio(synthetic_audio, samples):
    with pytest.raises(AudioValidationError):
        load_audio(synthetic_audio(samples))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_audio_samples(synthetic_audio, bad):
    samples = np.full(22050, 0.1)
    samples[100] = bad
    with pytest.raises(AudioValidationError):
        load_audio(synthetic_audio(samples, subtype="FLOAT"))


def test_file_size_limit(synthetic_audio):
    with pytest.raises(AudioValidationError, match="max_file_size_mb"):
        load_audio(synthetic_audio(), AudioConfig(max_file_size_mb=0.001))


def test_duration_limit(synthetic_audio):
    with pytest.raises(AudioValidationError):
        load_audio(synthetic_audio(), AudioConfig(max_duration_seconds=0.5))


@pytest.mark.parametrize("field", ["max_duration_seconds", "max_file_size_mb", "silence_rms_threshold"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_audio_config_rejected(field, bad):
    with pytest.raises(ValueError):
        AudioConfig(**{field: bad})


def test_synthetic_percussive_onsets_are_finite_bounded_and_repeatable(synthetic_audio):
    sample_rate = 22050
    samples = np.zeros(sample_rate * 2)
    rng = np.random.default_rng(42)
    burst = rng.normal(0, 0.2, 1000) * np.exp(-np.arange(1000) / 150)
    for time in (0.25, 0.75, 1.25, 1.75):
        start = int(time * sample_rate)
        samples[start:start + len(burst)] += burst
    audio = load_audio(synthetic_audio(samples))
    result = detect_onsets(audio)
    repeat = detect_onsets(audio)
    assert len(result.times_seconds) > 0
    assert len(result.times_seconds) == len(result.strengths)
    assert np.all(np.isfinite(result.times_seconds))
    assert np.all(np.isfinite(result.strengths))
    assert np.all((result.times_seconds >= 0) & (result.times_seconds <= audio.duration_seconds))
    assert np.all(np.diff(result.times_seconds) >= 0)
    np.testing.assert_array_equal(result.times_seconds, repeat.times_seconds)
    np.testing.assert_array_equal(result.strengths, repeat.strengths)
