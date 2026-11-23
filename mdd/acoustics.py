"""Tell apart the three things a "cut off" syllable can actually be.

Stød, a glottal stop and a plain short vowel all sound "clipped" to an untrained
ear, and a spectral-distance number cannot separate them either. They differ in
a way instruments read easily:

* **stød** — creaky voice. Energy dips but phonation *continues*; the glottal
  pulses go irregular, so periodicity collapses while amplitude stays well above
  zero.
* **glottal stop** — closure. Energy goes to (near) zero: a silent gap.
* **no stød** — no interior dip worth the name.

So the test is two numbers measured at the quietest interior moment: how much
energy is left, and how periodic it still is. This is what settles whether a
voice renders stød, which listening cannot do reliably and separation ratios
cannot do at all.

Deliberately simple DSP (framed RMS + normalised autocorrelation) so it has no
dependencies beyond numpy and can be read and checked by hand.
"""
from __future__ import annotations

from dataclasses import dataclass

#: RMS below this fraction of peak marks the dip as "quiet" — but quiet alone does
#: NOT mean closure: creak is quiet in RMS precisely because its pulses are sparse.
QUIET_FRAC = 0.55
#: The discriminator. Creak keeps discrete glottal pulses, so instantaneous peak
#: amplitude stays well above zero even while RMS collapses; a closure has neither.
#: Calibrating on RMS alone misclassified synthetic creak as a closure, which is
#: exactly the error that would wrongly clear a voice for stod training.
CLOSURE_PEAK_FRAC = 0.06
#: Peak-to-RMS within the dip. Sparse pulses (creak) give a high crest factor;
#: modal voice and silence both give a low one.
CREAK_CREST = 3.0
#: Normalised autocorrelation below this counts as aperiodic (creak or noise).
APERIODIC = 0.45
#: Frames quieter than this fraction of peak are trimmed as leading/trailing silence.
EDGE_FRAC = 0.10
#: A voice that creaks on more than this share of *all* words has creaky phonation
#: as a voice quality, so creak on any one word carries no information about stød.
#: This is the same idea as measuring separation against a jitter floor: a signal
#: only means something relative to that source's own baseline.
BASELINE_CREAK_RATE = 0.40


@dataclass
class Anatomy:
    """What the quietest interior moment of a word looks like."""

    duration: float
    #: RMS at the quietest interior frame, as a fraction of the word's peak RMS.
    dip_frac: float
    #: Instantaneous peak amplitude there, as a fraction of the word's peak. This
    #: is what separates creak (pulses remain) from closure (nothing remains).
    dip_peak_frac: float
    #: Peak-to-RMS ratio in the dip frame: high means sparse pulses.
    dip_crest: float
    #: Normalised autocorrelation peak there: ~1 periodic, ~0 aperiodic.
    dip_periodicity: float
    #: Median periodicity across the word, for comparison.
    median_periodicity: float
    #: Where the dip sits, as a fraction through the voiced span.
    dip_position: float
    kind: str
    #: Length of the voiced span only. Neural TTS pads clips with near-silence, so
    #: total clip duration says more about the padding than about the word.
    voiced_duration: float = 0.0

    def describe(self) -> str:
        return {
            "creak": "creaky dip — phonation continues but goes irregular (stød-like)",
            "closure": "silent gap — phonation stops (glottal stop, not stød)",
            "smooth": "no interior dip (no stød rendered)",
            "unclear": "dip present but neither clearly creaky nor a closure",
            "too-short": "too short to analyse",
        }[self.kind]


def _frames(x, sr: int, win_ms: float = 25.0, hop_ms: float = 5.0):
    import numpy as np

    win, hop = int(sr * win_ms / 1000), int(sr * hop_ms / 1000)
    if len(x) < win * 3:
        return np.empty((0, win)), hop
    idx = range(0, len(x) - win, hop)
    return np.stack([x[i:i + win] for i in idx]), hop


def _periodicity(frame, sr: int, fmin: float = 60.0, fmax: float = 400.0) -> float:
    """Peak of the normalised autocorrelation within a plausible F0 lag range."""
    import numpy as np

    f = frame - frame.mean()
    energy = float(f @ f)
    if energy <= 0:
        return 0.0
    ac = np.correlate(f, f, mode="full")[len(f) - 1:]
    lo, hi = int(sr / fmax), min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    return float(np.clip(ac[lo:hi].max() / energy, 0.0, 1.0))


def analyse(samples, sr: int) -> Anatomy:
    """Measure the quietest interior moment of one word."""
    import numpy as np

    x = np.asarray(samples, dtype=float)
    if x.ndim > 1:
        x = x[:, 0]
    peak = np.abs(x).max()
    if peak > 0:
        x = x / peak
    frames, _ = _frames(x, sr)
    if len(frames) < 7:
        return Anatomy(len(x) / sr, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, "too-short")

    rms = np.sqrt((frames ** 2).mean(axis=1))
    loud = rms.max()
    if loud <= 0:
        return Anatomy(len(x) / sr, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, "too-short")

    # Trim leading/trailing silence, then ignore the first and last voiced frames
    # so a word's own onset and offset are not mistaken for an interior dip.
    voiced = np.flatnonzero(rms >= loud * EDGE_FRAC)
    if len(voiced) < 7:
        return Anatomy(len(x) / sr, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, "too-short")
    lo, hi = voiced[0], voiced[-1]
    margin = max(2, (hi - lo) // 8)
    interior = slice(lo + margin, hi - margin + 1)
    seg = rms[interior]
    if len(seg) < 3:
        return Anatomy(len(x) / sr, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, "too-short")

    j = int(np.argmin(seg))
    dip_idx = interior.start + j
    dip_frac = float(seg[j] / loud)
    dip_frame = frames[dip_idx]
    dip_peak = float(np.abs(dip_frame).max())          # x is peak-normalised to 1.0
    dip_rms = float(np.sqrt((dip_frame ** 2).mean()))
    crest = dip_peak / dip_rms if dip_rms > 0 else 0.0
    dip_per = _periodicity(dip_frame, sr)
    med_per = float(np.median([_periodicity(f, sr) for f in frames[lo:hi + 1]]))
    pos = (dip_idx - lo) / max(hi - lo, 1)
    hop = int(sr * 5.0 / 1000)
    voiced_dur = (hi - lo) * hop / sr

    if dip_peak < CLOSURE_PEAK_FRAC:
        kind = "closure"                    # nothing left at all: phonation stopped
    elif dip_frac > QUIET_FRAC:
        kind = "smooth"                     # never really dipped
    elif crest >= CREAK_CREST or (dip_per < APERIODIC and dip_per < med_per * 0.8):
        kind = "creak"                      # pulses remain but irregular/sparse
    else:
        kind = "unclear"
    return Anatomy(len(x) / sr, dip_frac, dip_peak, crest, dip_per, med_per, float(pos),
                   kind, voiced_dur)


def compare(a: Anatomy, b: Anatomy) -> str:
    """One line on what differs between a pair — the question ears cannot settle."""
    if a.kind == b.kind == "smooth":
        return "neither word has an interior dip: no stød rendered in either"
    if b.kind == "creak" and a.kind in ("smooth", "unclear"):
        return "second word has a creaky dip the first lacks — consistent with stød"
    if b.kind == "closure" and a.kind in ("smooth", "unclear"):
        return "second word has a SILENT GAP the first lacks — glottal stop, not stød"
    if a.kind == b.kind:
        ratio = b.duration / a.duration if a.duration else 1.0
        if abs(ratio - 1) > 0.12:
            return (f"same dip type in both; the second is {ratio:.0%} the duration "
                    f"of the first — a length difference, not stød")
        return f"both look the same ({a.kind}): no stød distinction rendered"
    return f"first is {a.kind}, second is {b.kind}"


@dataclass
class VoiceBaseline:
    """How creaky a voice is overall — the context every per-word verdict needs.

    A voice with creaky phonation throughout will show a creaky dip on stød and
    non-stød words alike, and picking out the pairs where it happened to land on
    the stød member is cherry-picking noise. Measured on real Danish neural
    voices, one creaked on 9 of 12 words (including three with no stød) while
    another creaked on 1 (the non-stød member of its pair). Neither rendered
    stød; without this baseline the first looked like it did.
    """

    voice: str
    n_words: int
    creak_rate: float

    @property
    def diagnostic(self) -> bool:
        return 0.0 < self.creak_rate <= BASELINE_CREAK_RATE

    def verdict(self) -> str:
        if self.creak_rate == 0:
            return (f"{self.voice}: no creak on any of {self.n_words} words — this voice "
                    f"renders no stød, so pair differences are something else")
        if self.creak_rate > BASELINE_CREAK_RATE:
            return (f"{self.voice}: creaks on {self.creak_rate:.0%} of {self.n_words} words — "
                    f"creaky voice quality, NOT a stød distinction; per-pair creak here is "
                    f"not evidence")
        return (f"{self.voice}: creaks on {self.creak_rate:.0%} of {self.n_words} words — "
                f"sparse enough to be phonemic; check it lands on the stød member")


def baseline(voice: str, anatomies) -> VoiceBaseline:
    items = list(anatomies)
    if not items:
        return VoiceBaseline(voice, 0, 0.0)
    creaky = sum(1 for a in items if a.kind == "creak")
    return VoiceBaseline(voice, len(items), creaky / len(items))


@dataclass(frozen=True)
class VowelMeasurement:
    duration_ms: float
    duration_ratio: float | None
    f1_hz: float | None
    f2_hz: float | None
    bounded: bool


def estimate_formants(samples, sr: int = 16000) -> tuple[float | None, float | None]:
    """Estimate (F1, F2) in Hz from a short vocalic slice using pure-NumPy LPC."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size < int(sr * 0.015) or np.abs(x).max() < 1e-6:
        return None, None

    if sr > 10000:
        target_sr = 8000
        n_out = max(1, int(round(x.size * target_sr / sr)))
        spec = np.fft.rfft(x)
        new_spec = np.zeros(n_out // 2 + 1, dtype=complex)
        k = min(spec.size, new_spec.size)
        new_spec[:k] = spec[:k]
        x = np.fft.irfft(new_spec, n=n_out) * (n_out / x.size)
        sr = target_sr

    if x.size < 16:
        return None, None
    x = x[1:] - 0.97 * x[:-1]
    x = x * np.hamming(x.size)

    order = min(8, x.size - 2)
    if order < 4:
        return None, None
    ac = np.correlate(x, x, mode="full")[x.size - 1:x.size + order]
    if ac.size <= order or ac[0] <= 1e-12:
        return None, None

    idx = np.arange(order)
    R = ac[np.abs(idx[:, None] - idx[None, :])].copy()
    R[np.diag_indices(order)] += 1e-6 * ac[0]
    try:
        coeffs = np.linalg.solve(R, -ac[1:order + 1])
    except np.linalg.LinAlgError:
        return None, None

    roots = np.roots(np.concatenate(([1.0], coeffs)))
    roots = roots[(roots.imag > 0) & (np.abs(roots) > 0.0) & (np.abs(roots) < 1.0)]
    if roots.size == 0:
        return None, None

    freqs = np.arctan2(roots.imag, roots.real) * sr / (2.0 * np.pi)
    bws = -sr / (2.0 * np.pi) * np.log(np.abs(roots))
    mask = (freqs >= 180.0) & (freqs <= 3200.0) & (bws <= 400.0)
    formants = sorted(float(f) for f in freqs[mask])
    if not formants:
        return None, None
    f1 = round(formants[0], 1)
    f2 = round(formants[1], 1) if len(formants) >= 2 else None
    return f1, f2


def measure_vowel(wav, sr: int, segs, idx: int) -> VowelMeasurement | None:
    """Measure vocalic duration and central F1/F2 bounded around CTC spike `segs[idx]`."""
    import numpy as np

    x = np.asarray(wav, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size == 0 or not segs or idx < 0 or idx >= len(segs):
        return None
    total_s = x.size / sr
    if total_s < segs[idx].t_start:
        return None

    center = 0.5 * (segs[idx].t_start + segs[idx].t_end)
    left_s = 0.5 * (segs[idx - 1].t_start + segs[idx - 1].t_end) if idx > 0 else 0.0
    right_s = 0.5 * (segs[idx + 1].t_start + segs[idx + 1].t_end) if idx + 1 < len(segs) else total_s
    if right_s <= left_s:
        return None

    win = max(1, int(sr * 0.020))
    hop = max(1, int(sr * 0.005))
    if x.size < win:
        return None
    n_frames = (x.size - win) // hop + 1
    frame_centers = (np.arange(n_frames) * hop + win / 2.0) / sr
    in_bounds = np.flatnonzero((frame_centers >= left_s) & (frame_centers <= right_s))
    if in_bounds.size == 0:
        return None
    left_idx, right_idx = int(in_bounds[0]), int(in_bounds[-1])

    win_fn = np.hanning(win)
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    low_mask = (freqs >= 80.0) & (freqs <= 2500.0)
    win_indices = np.arange(left_idx, right_idx + 1)
    frames = np.stack([x[k * hop:k * hop + win] for k in win_indices])
    rms_win = np.sqrt((frames ** 2).mean(axis=1))
    local_peak = float(rms_win.max())
    if local_peak < 1e-4:
        return None

    power = np.abs(np.fft.rfft(frames * win_fn, axis=1)) ** 2
    tot_power = power.sum(axis=1)
    low_ratio = np.divide(
        power[:, low_mask].sum(axis=1),
        tot_power,
        out=np.zeros_like(tot_power),
        where=tot_power > 1e-12,
    )
    vocalic = (rms_win >= 0.25 * local_peak) & (low_ratio >= 0.60)

    local_centers = frame_centers[win_indices]
    near = np.flatnonzero(np.abs(local_centers - center) <= 0.025)
    anchor_rel = int(np.argmin(np.abs(local_centers - center)))
    if not vocalic[anchor_rel]:
        near_voc = [int(i) for i in near if vocalic[i]]
        if not near_voc:
            return None
        anchor_rel = max(near_voc, key=lambda i: float(rms_win[i]))

    lo_rel = anchor_rel
    while lo_rel > 0 and vocalic[lo_rel - 1]:
        lo_rel -= 1
    hi_rel = anchor_rel
    while hi_rel + 1 < len(vocalic) and vocalic[hi_rel + 1]:
        hi_rel += 1

    lo = int(win_indices[lo_rel])
    hi = int(win_indices[hi_rel])
    dur_ms = (hi - lo + 1) * 5.0
    bounded = (lo > left_idx) or (hi < right_idx)

    duration_ratio: float | None = None
    if len(segs) >= 2:
        centers_ms = [0.5 * (s.t_start + s.t_end) * 1000.0 for s in segs]
        intervals = [b - a for a, b in zip(centers_ms, centers_ms[1:]) if b - a > 15.0]
        if intervals:
            median_phone_ms = float(np.median(intervals))
            duration_ratio = dur_ms / max(median_phone_ms, 50.0)

    v_start = lo * hop
    v_end = min(x.size, hi * hop + win)
    mid_center = (v_start + v_end) // 2
    half_25 = int(sr * 0.025)
    s0 = max(v_start, mid_center - half_25)
    s1 = min(v_end, mid_center + half_25)
    f1, f2 = estimate_formants(x[s0:s1], sr)

    return VowelMeasurement(
        duration_ms=round(dur_ms, 1),
        duration_ratio=round(duration_ratio, 2) if duration_ratio is not None else None,
        f1_hz=f1,
        f2_hz=f2,
        bounded=bounded,
    )
