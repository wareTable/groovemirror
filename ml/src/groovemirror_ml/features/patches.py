"""Versioned, unfitted log-mel preprocessing for candidate drum events."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from numbers import Real

import librosa
import numpy as np
from scipy.fft import rfft
from scipy.signal.windows import hann

from .audio import AudioData

CLASS_LABELS = ("kick", "snare", "hihat")
PATCH_VERSION = "logmel-v1"
_CLASS_INDEX = {label: index for index, label in enumerate(CLASS_LABELS)}
_MIN_POWER = 1e-10
_FLOOR_DB = -100.0
_MAX_BATCH_EVENTS = 64
_MAX_BATCH_FRAME_SAMPLES = 1 << 20


@dataclass(frozen=True)
class PatchConfig:
    """Serialized DSP parameters, independent of training data.

    Frame centers run from -pre_frames to post_frames-1, inclusive, relative
    to the event sample. The post side includes the frame centered on the
    event. Default shape is 40 mel bands by 24 frames (~279 ms at hop stride).
    """

    version: str = PATCH_VERSION
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 40
    pre_frames: int = 4
    post_frames: int = 20

    def __post_init__(self) -> None:
        if self.version != PATCH_VERSION:
            raise ValueError(f"Unsupported patch config version: {self.version!r}")
        for name in ("sample_rate", "n_fft", "hop_length", "n_mels", "pre_frames", "post_frames"):
            value = getattr(self, name)
            minimum = 0 if name == "pre_frames" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not 2 <= self.n_fft <= 65536 or self.n_fft % 2:
            raise ValueError("n_fft must be even and between 2 and 65536")
        if self.hop_length > self.n_fft or self.n_mels > self.n_fft // 2 + 1:
            raise ValueError("hop_length or n_mels exceeds the corresponding FFT dimension")
        if self.n_mels * (self.n_fft // 2 + 1) > 1 << 20:
            raise ValueError("The mel filter bank exceeds the coefficient limit")
        if self.sample_rate > 384000:
            raise ValueError("sample_rate must be <= 384000")
        if self.shape[1] * self.n_fft > _MAX_BATCH_FRAME_SAMPLES:
            raise ValueError("A single patch exceeds the frame sample limit")

    @property
    def shape(self) -> tuple[int, int]:
        """One patch's (mel bands, frames) dimensions."""
        return self.n_mels, self.pre_frames + self.post_frames

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, int | str]) -> PatchConfig:
        if values.get("version") != PATCH_VERSION:
            raise ValueError(f"Unsupported patch config version: {values.get('version')!r}")
        if values.keys() != cls.__dataclass_fields__.keys():
            raise ValueError("Serialized patch config must contain exactly the versioned fields")
        return cls(**values)


@lru_cache(maxsize=8)
def _filters(sample_rate: int, n_fft: int, n_mels: int) -> tuple[np.ndarray, np.ndarray]:
    window = hann(n_fft, sym=False)
    mel = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=0.0,
        fmax=sample_rate / 2, htk=False, norm="slaney", dtype=np.float64,
    )
    window.setflags(write=False)
    mel.setflags(write=False)
    return window, mel


def event_patches(
    audio: AudioData,
    times_seconds: Iterable[float],
    config: PatchConfig = PatchConfig(),
) -> np.ndarray:
    """Return float32 (events, mel bands, frames) in input order.

    Timestamps in [0, source duration] are rounded to the nearest processed
    sample (half up), not snapped to the recording-wide hop grid. Centered
    Hann FFT frames use actual mono samples wherever available and zero-pad
    outside *either* recording edge without shifting the event. All-zero
    frames are -100 dB; other values use 10*log10 of mel power with reference
    1.0 and the same absolute floor. No per-event or fitted normalization.
    """
    if audio.sample_rate != config.sample_rate:
        raise ValueError("AudioData sample rate differs from PatchConfig; use matching AudioConfig")
    if audio.samples.ndim != 1 or audio.samples.dtype.kind not in "fiu":
        raise ValueError("AudioData samples must be real mono audio")
    times = np.asarray(times_seconds if isinstance(times_seconds, np.ndarray) else list(times_seconds))
    if times.ndim != 1 or times.dtype.kind not in "fiu":
        raise ValueError("times_seconds must be a one-dimensional sequence of real numbers")
    times = times.astype(np.float64, copy=False)
    if not np.all(np.isfinite(times) & (times >= 0) & (times <= audio.duration_seconds)):
        raise ValueError("times_seconds must be finite and within the recording duration")

    output = np.empty((times.size, *config.shape), dtype=np.float32)
    if not times.size:
        return output
    if not audio.samples.size:
        output.fill(_FLOOR_DB)
        return output

    n_frames = config.shape[1]
    batch_size = min(_MAX_BATCH_EVENTS, _MAX_BATCH_FRAME_SAMPLES // (n_frames * config.n_fft))
    span = config.n_fft + (n_frames - 1) * config.hop_length
    offsets = np.arange(span) - config.pre_frames * config.hop_length - config.n_fft // 2
    window, mel = _filters(config.sample_rate, config.n_fft, config.n_mels)
    for start in range(0, times.size, batch_size):
        stop = min(start + batch_size, times.size)
        centers = np.floor(times[start:stop] * config.sample_rate + 0.5).astype(np.int64)
        indices = centers[:, None] + offsets
        valid = (indices >= 0) & (indices < audio.samples.size)
        clips = np.where(valid, audio.samples[indices.clip(0, audio.samples.size - 1)], 0.0)
        frames = np.lib.stride_tricks.sliding_window_view(clips, config.n_fft, axis=-1)
        spectrum = rfft(frames[:, ::config.hop_length, :] * window, axis=-1, workers=1)
        power = spectrum.real**2 + spectrum.imag**2
        # Fixed reduction order makes each event independent of batch boundaries.
        mel_power = np.einsum("mf,btf->bmt", mel, power, optimize=False)
        np.maximum(mel_power, _MIN_POWER, out=mel_power)
        np.log10(mel_power, out=mel_power)
        if not np.isfinite(mel_power).all():
            raise ValueError("AudioData produced nonfinite log-mel power")
        output[start:stop] = 10.0 * mel_power
    return output


def annotation_targets(
    events: Iterable[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Group exact timestamps from loader events into sorted multi-hot rows.

    Repeated labels are idempotent. Close but distinct timestamps remain
    distinct even when their audio patches round to the same sample.
    """
    grouped: dict[float, set[int]] = {}
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"Event {index} must be a mapping")
        time, label = event.get("time"), event.get("instrument")
        if isinstance(time, (bool, np.bool_)) or not isinstance(time, Real):
            raise ValueError(f"Event {index} time must be a finite nonnegative number")
        try:
            seconds = float(time)
        except OverflowError as exc:
            raise ValueError(f"Event {index} time must be a finite nonnegative number") from exc
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Event {index} time must be a finite nonnegative number")
        if not isinstance(label, str) or label not in _CLASS_INDEX:
            raise ValueError(f"Event {index} instrument must be one of {CLASS_LABELS}")
        grouped.setdefault(seconds, set()).add(_CLASS_INDEX[label])
    times = np.asarray(sorted(grouped), dtype=np.float64)
    targets = np.zeros((times.size, len(CLASS_LABELS)), dtype=np.float32)
    for row, time in enumerate(times):
        for column in grouped[time]:
            targets[row, column] = 1.0
    return times, targets
