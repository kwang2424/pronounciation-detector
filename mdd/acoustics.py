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
        return Anatomy(len(x) / sr, 1.0, 1.0, 1.0, 0.0, "too-short")

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

    if dip_peak < CLOSURE_PEAK_FRAC:
        kind = "closure"                    # nothing left at all: phonation stopped
    elif dip_frac > QUIET_FRAC:
        kind = "smooth"                     # never really dipped
    elif crest >= CREAK_CREST or (dip_per < APERIODIC and dip_per < med_per * 0.8):
        kind = "creak"                      # pulses remain but irregular/sparse
    else:
        kind = "unclear"
    return Anatomy(len(x) / sr, dip_frac, dip_peak, crest, dip_per, med_per, float(pos), kind)


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
