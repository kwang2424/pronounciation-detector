"""Intonation for shadowing: your pitch line against a model's.

The scorer checks individual sounds and is blind to melody and rhythm, which
listeners lean on heavily. Shadowing (speaking along with a model) is the usual
way to practise them, and a visual pitch display is a long-standing aid for it
(Hardison 2004 found it helped English-speaking learners of French).

This shows the two contours on one time axis and reports a few robust facts:

- **pitch range**: how far your voice moves, against the model. A flat, narrow
  range is the most common learner pattern and the easiest to hear once seen.
- **shape match**: correlation of the two contours after aligning them in time.
- **ending**: whether each version rises, falls or stays level at the end, which
  is where questions and statements differ.
- **timing**: how long you took against the model.

It is a visual aid, not a score: a sentence has more than one natural
intonation, and the model is one voice. Pitch is in semitones from each
speaker's own median, so a low and a high voice compare fairly.

Pitch tracking is YIN (de Cheveigné & Kawahara 2002) in plain numpy, so it needs
no model download.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

SR = 16000
HOP_S = 0.01
FMIN, FMAX = 70.0, 450.0
YIN_THRESHOLD = 0.15
VOICED_MAX = 0.4
MAX_EXCURSION_ST = 12.0
#: Frames quieter than this (dB below the loudest) are never called voiced.
VOICING_DB = 35.0
MIN_VOICED_S = 0.05
#: Ending: how far back the syllables before the last one are looked for, and the
#: movement (semitones) that counts as a rise or fall.
END_S = 0.8
END_ST = 1.5
#: Longer utterances are aligned linearly; DTW memory grows with the square.
MAX_DTW_FRAMES = 3000


def load(path) -> np.ndarray:
    """Read any audio file as mono float32 at 16 kHz."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(data.mean(axis=1), sr, SR)


def resample(x: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    """Windowed-sinc low-pass then interpolation: plenty for pitch and energy."""
    x = np.asarray(x, dtype=np.float32)
    if sr == target or len(x) == 0:
        return x
    if target < sr:
        cutoff = 0.45 * target / sr
        n = np.arange(-32, 33)
        h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(len(n))
        x = np.convolve(x, h / h.sum(), mode="same").astype(np.float32)
    t = np.arange(int(len(x) * target / sr)) * (sr / target)
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    if len(x) < size:
        x = np.pad(x, (0, size - len(x)))
    n = 1 + (len(x) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def n_frames(x: np.ndarray, sr: int = SR) -> int:
    """Frames per signal: one per 10 ms hop, the same for every per-frame measure."""
    return 1 + len(x) // int(sr * HOP_S)


def energy_db(x: np.ndarray, sr: int = SR, win_s: float = 0.025) -> np.ndarray:
    """Frame energy in dB, one value per 10 ms hop, aligned with `pitch`."""
    hop, win = int(sr * HOP_S), int(sr * win_s)
    f = frames(np.pad(x, (win // 2, win)), win, hop)[:n_frames(x, sr)]
    return 10 * np.log10(np.mean(f.astype(np.float64) ** 2, axis=1) + 1e-10)


def pitch(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """F0 in Hz every 10 ms, NaN where unvoiced."""
    hop = int(sr * HOP_S)
    tmin, tmax = int(sr / FMAX), int(sr / FMIN)
    w = 2 * tmax                    # integration window: two periods of the lowest pitch
    count = n_frames(x, sr)
    f = frames(np.pad(x, (w // 2, w + tmax)), w + tmax, hop)[:count].astype(np.float64)
    f -= f.mean(axis=1, keepdims=True)

    # Difference function d(tau) = E(0) + E(tau) - 2 r(tau), via FFT.
    nfft = 1 << int(np.ceil(np.log2(2 * (w + tmax))))
    spec_full = np.fft.rfft(f, nfft)
    spec_head = np.fft.rfft(f[:, :w], nfft)
    r = np.fft.irfft(np.conj(spec_head) * spec_full, nfft)[:, :tmax + 1]
    sq = np.cumsum(np.pad(f ** 2, ((0, 0), (1, 0))), axis=1)
    taus = np.arange(tmax + 1)
    e0 = sq[:, w][:, None]
    et = sq[:, taus + w] - sq[:, taus]
    d = np.maximum(e0 + et - 2 * r, 0.0)

    # Cumulative mean normalised difference.
    cum = np.cumsum(d[:, 1:], axis=1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cum, 1e-12)

    f0 = np.full(count, np.nan)
    window = cmnd[:, tmin:tmax]
    # The first dip under the threshold is the period (later dips are multiples of
    # it). Voices that are less periodic than the textbook — creaky, breathy, or
    # espeak's low voices, median 0.29 — rarely dip under 0.15, so the threshold
    # rises to just above each frame's best dip, up to VOICED_MAX; noise sits ~0.85.
    best = window.min(axis=1)
    below = window < np.maximum(YIN_THRESHOLD, best + 0.1)[:, None]
    for i in np.flatnonzero(best < VOICED_MAX):
        k = int(np.argmax(below[i]))
        while k + 1 < window.shape[1] and window[i, k + 1] < window[i, k]:
            k += 1                                   # walk down to the local minimum
        tau = k + tmin
        if 0 < k < window.shape[1] - 1:               # parabolic refinement
            a, b, c = window[i, k - 1], window[i, k], window[i, k + 1]
            denom = a - 2 * b + c
            if denom > 0:
                tau += 0.5 * (a - c) / denom
        f0[i] = sr / tau

    loud = energy_db(x, sr)
    if len(loud):
        f0[loud < loud.max() - VOICING_DB] = np.nan
    # More than an octave from the speaker's typical pitch is a tracking error
    # (onsets and fricatives produce stray 400 Hz readings), not intonation.
    voiced = f0[~np.isnan(f0)]
    if len(voiced):
        f0[np.abs(12 * np.log2(f0 / np.median(voiced))) > MAX_EXCURSION_ST] = np.nan
    return _clean(f0)


def _clean(f0: np.ndarray) -> np.ndarray:
    """Drop voiced blips too short to be a syllable and median-smooth the rest."""
    out = f0.copy()
    min_len = int(MIN_VOICED_S / HOP_S)
    voiced = ~np.isnan(out)
    i = 0
    while i < len(out):
        if voiced[i]:
            j = i
            while j < len(out) and voiced[j]:
                j += 1
            if j - i < min_len:
                out[i:j] = np.nan
            else:
                seg = out[i:j]
                padded = np.pad(seg, 2, mode="edge")
                out[i:j] = np.median(frames(padded, 5, 1), axis=1)[:j - i]
            i = j
        else:
            i += 1
    return out


def semitones(f0: np.ndarray) -> np.ndarray:
    """Pitch relative to the speaker's own median, so voices compare fairly."""
    voiced = f0[~np.isnan(f0)]
    if len(voiced) == 0:
        return f0.copy()
    return 12 * np.log2(f0 / np.median(voiced))


def speech_span(x: np.ndarray, sr: int = SR, floor_db: float = 30.0) -> tuple[int, int]:
    """First and last 10 ms frame within `floor_db` of the loudest frame."""
    e = energy_db(x, sr)
    idx = np.flatnonzero(e > e.max() - floor_db) if len(e) else []
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, len(e))


def features(x: np.ndarray, sr: int = SR, bands: int = 20) -> np.ndarray:
    """Normalised log band energies every 10 ms, for aligning two takes."""
    hop, win = int(sr * HOP_S), int(sr * 0.025)
    f = frames(np.pad(x, (win // 2, win)), win, hop)[:n_frames(x, sr)] * np.hanning(win)
    mag = np.abs(np.fft.rfft(f, 512, axis=1))
    edges = np.geomspace(4, mag.shape[1] - 1, bands + 1).astype(int)
    feat = np.stack([mag[:, a:b + 1].mean(axis=1) for a, b in pairwise(edges)], 1)
    feat = np.log(feat + 1e-6)
    return (feat - feat.mean(0)) / (feat.std(0) + 1e-6)


def dtw_path(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    """Monotonic alignment of two feature sequences (row recurrence vectorised)."""
    n, m = len(a), len(b)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    D = np.full((n, m), np.inf)
    D[0] = np.cumsum(cost[0])
    for i in range(1, n):
        c = cost[i]
        diag = np.concatenate(([np.inf], D[i - 1, :-1]))
        best = c + np.minimum(diag, D[i - 1])
        # D[i,j] = min_k<=j (best[k] + sum c[k+1..j]) = S[j] + min_k<=j (best[k] - S[k])
        S = np.cumsum(c)
        D[i] = S + np.minimum.accumulate(best - S)
    i, j, path = n - 1, m - 1, [(n - 1, m - 1)]
    while i > 0 or j > 0:
        options = []
        if i > 0 and j > 0:
            options.append((D[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            options.append((D[i - 1, j], i - 1, j))
        if j > 0:
            options.append((D[i, j - 1], i, j - 1))
        _, i, j = min(options)
        path.append((i, j))
    return path[::-1]


@dataclass
class Contour:
    times: np.ndarray        # seconds, on the model's time axis
    model: np.ndarray        # semitones re the model's median, NaN unvoiced
    yours: np.ndarray        # semitones re your median, warped onto the model's time
    model_range: float
    your_range: float
    correlation: float | None
    model_ending: str
    your_ending: str
    duration_ratio: float
    aligned: str             # "dtw" or "linear"


def _range(st: np.ndarray) -> float:
    v = st[~np.isnan(st)]
    return float(np.percentile(v, 90) - np.percentile(v, 10)) if len(v) >= 10 else 0.0


def ending(st: np.ndarray) -> str:
    """'rise', 'fall' or 'level' at the end of the utterance.

    The movement is measured two ways and the larger one counts: within the last
    voiced stretch (a rise on the final syllable), and from the syllables just
    before it to the very end (a step up or down onto the final syllable, which is
    how espeak, and many speakers, mark it).
    """
    idx = np.flatnonzero(~np.isnan(st))
    if len(idx) < 6:
        return "unclear"
    last = idx[-1]
    first = last
    while first - 1 >= 0 and not np.isnan(st[first - 1]):
        first -= 1
    end = float(np.median(st[max(first, last - 2):last + 1]))
    within = end - float(np.median(st[first:first + 3]))
    before = idx[(idx < first) & (idx >= last - int(END_S / HOP_S))]
    step = end - float(np.median(st[before])) if len(before) >= 3 else 0.0
    # The final syllable's own movement is what is heard last, so it wins; a step
    # onto it counts only when the syllable does not move the other way (a step
    # up into a final word that then falls is a statement, not a question).
    if abs(within) >= END_ST:
        move = within
    elif within * step < 0 and abs(within) >= 0.5:
        move = 0.0
    else:
        move = step
    if move >= END_ST:
        return "rise"
    if move <= -END_ST:
        return "fall"
    return "level"


def compare(model: np.ndarray, yours: np.ndarray, sr: int = SR) -> Contour:
    """Both takes trimmed to their speech, your pitch mapped onto the model's time."""
    m0, m1 = speech_span(model, sr)
    y0, y1 = speech_span(yours, sr)
    hop = int(sr * HOP_S)
    model, yours = model[m0 * hop:m1 * hop], yours[y0 * hop:y1 * hop]
    if len(model) < sr * 0.2 or len(yours) < sr * 0.2:
        raise ValueError("A recording is too short or silent to compare.")

    ms, ys = semitones(pitch(model, sr)), semitones(pitch(yours, sr))
    if len(ms) <= MAX_DTW_FRAMES and len(ys) <= MAX_DTW_FRAMES:
        path = dtw_path(features(model, sr), features(yours, sr))
        how = "dtw"
    else:
        path = [(i, min(len(ys) - 1, round(i * len(ys) / len(ms)))) for i in range(len(ms))]
        how = "linear"
    warped = np.full(len(ms), np.nan)
    for i in range(len(ms)):
        js = [j for k, j in path if k == i] if how == "dtw" else [path[i][1]]
        vals = ys[[j for j in js if j < len(ys)]]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            warped[i] = float(np.median(vals))

    both = ~np.isnan(ms) & ~np.isnan(warped)
    corr = None
    if both.sum() >= 20 and np.std(ms[both]) > 0 and np.std(warped[both]) > 0:
        corr = float(np.corrcoef(ms[both], warped[both])[0, 1])
    return Contour(times=np.arange(len(ms)) * HOP_S, model=ms, yours=warped,
                   model_range=_range(ms), your_range=_range(ys), correlation=corr,
                   model_ending=ending(ms), your_ending=ending(ys),
                   duration_ratio=len(yours) / len(model), aligned=how)


def describe(c: Contour) -> str:
    lines = []
    if c.model_range > 0:
        share = c.your_range / c.model_range
        msg = (f"**Pitch range:** yours {c.your_range:.1f} semitones, the model's "
               f"{c.model_range:.1f}")
        if share < 0.7:
            msg += " — noticeably flatter. Exaggerate the rises and falls a little."
        elif share > 1.4:
            msg += " — wider than the model. Fine if it sounds natural to you."
        else:
            msg += " — similar."
        lines.append(msg)
    if c.correlation is not None:
        word = ("close" if c.correlation >= 0.7 else "partly similar" if c.correlation >= 0.4
                else "different")
        lines.append(f"**Shape:** {word} (correlation {c.correlation:.2f}).")
    # Only a clear model ending is worth comparing against: a "level" model ending
    # is often a weak synthetic question rise rather than a real target.
    if c.model_ending in ("rise", "fall") and c.your_ending != "unclear":
        if c.model_ending == c.your_ending:
            lines.append(f"**Ending:** both {c.model_ending} — matches.")
        else:
            yours = "stays level" if c.your_ending == "level" else c.your_ending + "s"
            lines.append(f"**Ending:** the model's pitch {c.model_ending}s at the end, "
                         f"yours {yours}. Questions and statements differ here.")
    pace = c.duration_ratio
    lines.append(f"**Timing:** you took {pace:.1f}× the model's time"
                 + (" — slow down less, or try the Slow model." if pace > 1.5 else "."))
    return "\n\n".join(lines)


def plot(c: Contour, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3), dpi=110)
    ax.plot(c.times, c.model, color="#2563eb", lw=2.5, label="Model")
    ax.plot(c.times, c.yours, color="#f97316", lw=2.5, label="You", alpha=0.9)
    ax.axhline(0, color="#9ca3af", lw=0.8, ls=":")
    ax.set_xlabel("time (s, on the model's timing)")
    ax.set_ylabel("semitones from\nyour usual pitch")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig
