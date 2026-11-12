"""Shared Phase 1 WAV validation and spectral-flux onset preprocessing.

Channels are mixed with the arithmetic mean (no peak normalization), then
resampled with librosa's explicit soxr_hq, fix=True, scale=False settings.
All timestamps remain relative to the original recording; no latency/mean-error
correction is applied. Source duration is frames / original sample rate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


class AudioValidationError(ValueError):
    """An actionable validation/decode failure for the input recording."""


# Secondary decoded-allocation guard, independent of potentially deceptive headers.
MAX_DECODED_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    hop_length: int = 256
    n_fft: int = 1024
    max_duration_seconds: float = 300.0
    max_file_size_mb: float = 100.0
    silence_rms_threshold: float = 1e-7
    backtrack: bool = False
    res_type: str = "soxr_hq"

    def __post_init__(self) -> None:
        for name in ("sample_rate", "hop_length", "n_fft"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_fft < self.hop_length or self.n_fft < 2:
            raise ValueError("n_fft must be at least 2 and at least hop_length")
        if self.n_fft > 65536 or self.sample_rate > 384000:
            raise ValueError("n_fft must be <=65536 and sample_rate must be <=384000")
        for name in ("max_duration_seconds", "max_file_size_mb", "silence_rms_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_duration_seconds == 0 or self.max_file_size_mb == 0:
            raise ValueError("max_duration_seconds and max_file_size_mb must be positive")
        if not isinstance(self.backtrack, bool):
            raise ValueError("backtrack must be a boolean")
        if self.res_type != "soxr_hq":
            raise ValueError("Phase 1 res_type must be 'soxr_hq' for reproducible preprocessing")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    original_sample_rate: int
    original_channels: int
    duration_seconds: float
    source_path: Path
    rms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "sample_rate": self.sample_rate,
            "original_sample_rate": self.original_sample_rate,
            "original_channels": self.original_channels,
            "duration_seconds": self.duration_seconds,
            "processed_sample_count": int(self.samples.size),
            "processed_duration_seconds": self.samples.size / self.sample_rate,
            "rms": self.rms,
            "channel_mixing": "arithmetic_mean",
            "normalization": "none",
        }


@dataclass(frozen=True)
class OnsetResult:
    times_seconds: np.ndarray
    strengths: np.ndarray
    frames: np.ndarray
    envelope: np.ndarray


def _rms(samples: np.ndarray) -> float:
    peak = float(np.max(np.abs(samples)))
    if peak == 0:
        return 0.0
    # Scaling avoids overflow when validating floating-point WAV amplitudes.
    return peak * float(np.sqrt(np.mean(np.square(samples / peak))))


def load_audio(path: str | Path, config: AudioConfig = AudioConfig()) -> AudioData:
    """Decode a nonempty finite nonsilent WAV, checking limits before allocation.

    max_file_size_mb uses MiB (1024**2 bytes). Decoded float64 allocation is
    independently capped at 256 MiB. Silence is measured after channel mixing
    and again after resampling, so cancelling stereo channels are rejected.
    """
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".wav":
        raise AudioValidationError(f"Expected a .wav recording: {source}")
    try:
        if not source.is_file():
            raise AudioValidationError(f"Audio file does not exist or is not a regular file: {source}")
        size = source.stat().st_size
        if size == 0:
            raise AudioValidationError(f"Audio file is empty: {source}")
        if size > config.max_file_size_mb * 1024**2:
            raise AudioValidationError(f"Audio exceeds max_file_size_mb={config.max_file_size_mb} MiB: {source}")
        with sf.SoundFile(str(source), mode="r") as stream:
            if stream.format not in {"WAV", "WAVEX", "RF64"}:
                raise AudioValidationError(f"Expected decoded WAV format, found {stream.format}: {source}")
            rate, channels, frames = stream.samplerate, stream.channels, stream.frames
            if rate <= 0 or channels <= 0 or frames <= 0:
                raise AudioValidationError(f"Audio must have positive sample rate, channels and duration: {source}")
            duration = frames / rate
            if duration > config.max_duration_seconds:
                raise AudioValidationError(f"Audio duration {duration:.6f}s exceeds max_duration_seconds={config.max_duration_seconds}")
            if frames * channels * 8 > MAX_DECODED_BYTES:
                raise AudioValidationError("Decoded audio exceeds the 256 MiB processing limit")
            data = stream.read(frames=frames, dtype="float64", always_2d=True)
            if data.shape != (frames, channels):
                raise AudioValidationError(f"Truncated/corrupt WAV: expected {frames} frames, decoded {data.shape[0]}")
    except AudioValidationError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise AudioValidationError(f"Audio is unreadable or corrupt ({source}): {exc}") from exc
    if not np.all(np.isfinite(data)):
        raise AudioValidationError(f"Audio contains nonfinite samples (NaN or infinity): {source}")
    # Divide first to avoid overflowing a channel sum for a floating-point WAV.
    mono = np.sum(data / channels, axis=1)
    if not np.all(np.isfinite(mono)):
        raise AudioValidationError("Mono mixing produced nonfinite samples")
    if _rms(mono) <= config.silence_rms_threshold:
        raise AudioValidationError(f"Audio is silent/effectively silent after mono mixing (RMS <= {config.silence_rms_threshold})")
    target_samples = math.ceil(frames * config.sample_rate / rate)
    if target_samples * 8 > MAX_DECODED_BYTES:
        raise AudioValidationError("Resampled audio exceeds the 256 MiB processing limit")
    if rate != config.sample_rate:
        try:
            mono = librosa.resample(mono, orig_sr=rate, target_sr=config.sample_rate,
                                    res_type=config.res_type, fix=True, scale=False)
        except (ValueError, RuntimeError) as exc:
            raise AudioValidationError(f"Audio resampling failed: {exc}") from exc
    if mono.size == 0 or not np.all(np.isfinite(mono)):
        raise AudioValidationError("Resampling produced empty or nonfinite audio")
    rms = _rms(mono)
    if rms <= config.silence_rms_threshold:
        raise AudioValidationError("Audio is silent/effectively silent after resampling")
    mono = np.ascontiguousarray(mono, dtype=np.float64)
    mono.setflags(write=False)
    return AudioData(mono, config.sample_rate, rate, channels, duration, source, rms)


def detector_parameters(config: AudioConfig) -> dict[str, Any]:
    """Fixed, explicit DSP parameters, also stored in every report."""
    return {
        "method": "librosa_positive_spectral_flux_log_mel",
        "n_fft": config.n_fft, "hop_length": config.hop_length,
        "window": "hann", "center": True, "pad_mode": "constant",
        "n_mels": 128, "fmin": 0.0, "fmax": config.sample_rate / 2,
        "mel_norm": "slaney", "htk": False, "power": 2.0,
        "db_ref": 1.0, "db_amin": 1e-10, "top_db": 80.0,
        "lag": 1, "max_size": 1, "detrend": False, "aggregate": "mean",
        "peak_normalize": True, "delta": 0.07,
        "pre_max": int(0.03 * config.sample_rate // config.hop_length),
        "post_max": 1,
        "pre_avg": int(0.10 * config.sample_rate // config.hop_length),
        "post_avg": int(0.10 * config.sample_rate // config.hop_length) + 1,
        "wait": int(0.03 * config.sample_rate // config.hop_length),
        "backtrack": config.backtrack,
        "strength_definition": "raw spectral-flux envelope at returned frame; not a probability",
    }


def detect_onsets(audio: AudioData, config: AudioConfig = AudioConfig()) -> OnsetResult:
    """Return frame-grid onsets; near-boundary detections are not extrapolated.

    A centered log-mel spectrogram feeds positive spectral flux, then explicit
    librosa peak-picking settings. The flux's centering compensation is applied
    by librosa; frames_to_time receives no additional n_fft offset. Backtracking
    uses local minima of the same envelope when enabled. No instrument labels
    or confidence probabilities are inferred.
    """
    if audio.sample_rate != config.sample_rate:
        raise ValueError("AudioData sample_rate differs from AudioConfig; preprocess with the same config")
    y = np.asarray(audio.samples)
    if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
        raise AudioValidationError("Onset detection expects finite nonempty mono samples")
    if _rms(y) <= config.silence_rms_threshold:
        raise AudioValidationError("Onset detection expects nonsilent audio")
    # Bound spectrogram allocation too (128 mel bins plus STFT intermediates).
    frame_count = 1 + y.size // config.hop_length
    if frame_count * (config.n_fft // 2 + 1) * 16 > MAX_DECODED_BYTES:
        raise AudioValidationError("STFT exceeds the 256 MiB intermediate processing limit; increase hop_length")
    params = detector_parameters(config)
    mel = librosa.feature.melspectrogram(
        y=y, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length,
        win_length=config.n_fft, window="hann", center=True, pad_mode="constant",
        power=2.0, n_mels=128, fmin=0.0, fmax=config.sample_rate / 2,
        htk=False, norm="slaney",
    )
    if not np.all(np.isfinite(mel)):
        raise AudioValidationError("Spectral computation overflowed; audio amplitude is outside the numeric processing range")
    log_mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=80.0)
    envelope = librosa.onset.onset_strength(
        S=log_mel, sr=config.sample_rate, hop_length=config.hop_length,
        n_fft=config.n_fft, lag=1, max_size=1, detrend=False, center=True,
        aggregate=np.mean,
    )
    frames = librosa.onset.onset_detect(
        onset_envelope=envelope, sr=config.sample_rate, hop_length=config.hop_length,
        backtrack=config.backtrack, energy=envelope, units="frames", normalize=True,
        sparse=True, **{name: params[name] for name in
                        ("pre_max", "post_max", "pre_avg", "post_avg", "wait", "delta")},
    )
    frames = np.asarray(frames, dtype=np.int64)
    times = librosa.frames_to_time(frames, sr=config.sample_rate, hop_length=config.hop_length)
    keep = (times >= 0) & (times < audio.duration_seconds)
    frames, times = frames[keep], times[keep]
    return OnsetResult(times, np.asarray(envelope[frames], dtype=float), frames, envelope)
