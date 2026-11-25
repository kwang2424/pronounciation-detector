"""Clip hygiene shared by everything that plays synthesised speech to a learner.

Three separate complaints from one training session traced back to clip edges: a
word heard as starting with a plosive it does not have ("haute", whose h is
silent), a short word heard as truncated ("rue" sounding like "re"), and nasal
vowels heard as cut off at the end ("sans", "son") — the tail being exactly where
French carries nasality. Padding keeps the tail; short fades stop an abrupt edge
reading as a burst.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Silence added around each clip. Neural TTS trims hard on isolated words.
PAD_HEAD_MS = 60
PAD_TAIL_MS = 220
#: Fade at each end; long enough to kill the transient, too short to touch speech.
FADE_MS = 12


class AudioQualityError(ValueError):
    """Raised when a recording fails silence, duration, or SNR hygiene gates."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class AudioCheck:
    wav: object
    duration_s: float
    voiced_s: float
    snr_db: float | None
    clipped_frac: float
    warnings: tuple[str, ...]
    reject_reason: str | None

    @property
    def ok(self) -> bool:
        return self.reject_reason is None


def reject_message(reason: str, snr_db: float | None = None) -> str:
    if reason == "empty":
        return "The recording is empty. Record a sentence and try again."
    if reason == "silent":
        return "No speech detected (the recording is too quiet or silent). Check your microphone."
    if reason == "too-short":
        return "The recording is too short to score. Hold the sentence a bit longer."
    if reason == "low-snr":
        snr_txt = f" ({snr_db:.1f} dB SNR)" if snr_db is not None else ""
        return f"Too much background noise{snr_txt} to score reliably. Move somewhere quieter."
    return f"Recording rejected ({reason})."


def preprocess_recording(
    wav,
    sr: int = 16000,
    *,
    pad_ms: float = 300.0,
    trim_over_ms: float = 1200.0,
    min_speech_ms: float = 120.0,
    min_snr_db: float = 6.0,
    min_rms: float = 0.005,
) -> AudioCheck:
    """Validate mic audio quality (silence, clipping, SNR) and VAD-trim leading/trailing pauses."""
    import numpy as np

    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1).astype(np.float32)
    if x.size == 0:
        return AudioCheck(np.zeros(0, dtype=np.float32), 0.0, 0.0, None, 0.0, (), "empty")

    clipped_frac = float(np.mean(np.abs(x) >= 0.99))
    warnings: list[str] = []
    if clipped_frac >= 0.01:
        warnings.append(
            f"Audio is clipping ({clipped_frac:.1%} of samples near full scale) — "
            "step back from the microphone or lower input gain."
        )

    win = max(1, int(sr * 0.02))
    hop = max(1, int(sr * 0.01))
    if x.size < win:
        return AudioCheck(x, x.size / sr, 0.0, None, clipped_frac, tuple(warnings), "too-short")

    starts = range(0, x.size - win + 1, hop)
    frames = np.stack([x[i:i + win] for i in starts])
    rms = np.sqrt((frames ** 2).mean(axis=1))
    peak_rms = float(rms.max())
    if peak_rms < min_rms:
        return AudioCheck(x, x.size / sr, 0.0, None, clipped_frac, tuple(warnings), "silent")

    speech_rms = float(np.percentile(rms, 90))
    quiet = rms[rms < peak_rms * 0.25]
    noise_rms = 0.0
    snr_db: float | None = None
    if len(quiet) >= 3:
        noise_rms = float(np.median(quiet))
        if noise_rms > 1e-6:
            snr_db = float(20.0 * np.log10(max(speech_rms, 1e-12) / noise_rms))
            if snr_db < min_snr_db:
                return AudioCheck(x, x.size / sr, 0.0, snr_db, clipped_frac, tuple(warnings), "low-snr")
    else:
        win_fn = np.hanning(win)
        spec = np.abs(np.fft.rfft(frames * win_fn, axis=1)) ** 2 + 1e-12
        flatness = np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)
        med_flatness = float(np.median(flatness))
        p10 = float(np.percentile(rms, 10))
        est_snr = float(20.0 * np.log10(max(speech_rms, 1e-12) / max(p10, 1e-12)))
        if med_flatness > 0.45 and est_snr < min_snr_db:
            return AudioCheck(x, x.size / sr, 0.0, est_snr, clipped_frac, tuple(warnings), "low-snr")

    noise_gate = (noise_rms * 2.5) if (len(quiet) >= 3 and noise_rms > 1e-6) else 0.0
    vad_thresh = max(min_rms, noise_gate, peak_rms * 0.05)
    active = np.flatnonzero(rms >= vad_thresh)
    voiced_s = len(active) * hop / sr
    if len(active) == 0 or voiced_s < min_speech_ms / 1000.0:
        return AudioCheck(x, x.size / sr, voiced_s, snr_db, clipped_frac, tuple(warnings), "too-short")

    speech_start = int(active[0] * hop)
    speech_end = min(x.size, int(active[-1] * hop) + win)
    trigger = int(sr * trim_over_ms / 1000.0)
    pad = int(sr * pad_ms / 1000.0)
    trim_start = max(0, speech_start - pad) if speech_start > trigger else 0
    trim_end = min(x.size, speech_end + pad) if (x.size - speech_end) > trigger else x.size
    trimmed = x[trim_start:trim_end]
    return AudioCheck(trimmed, trimmed.size / sr, voiced_s, snr_db, clipped_frac, tuple(warnings), None)


def polish(data, sr: int):
    """Pad and fade a clip; returns int16 samples."""
    import numpy as np

    x = np.asarray(data)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float64)

    fade = max(1, int(sr * FADE_MS / 1000))
    if x.size > 2 * fade:
        x[:fade] *= np.linspace(0.0, 1.0, fade)
        x[-fade:] *= np.linspace(1.0, 0.0, fade)
    head = np.zeros(int(sr * PAD_HEAD_MS / 1000))
    tail = np.zeros(int(sr * PAD_TAIL_MS / 1000))
    return np.concatenate([head, x, tail]).astype(np.int16)
