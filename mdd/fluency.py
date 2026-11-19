"""Fluency measures for the 4/3/2 activity.

4/3/2 (Maurice 1983; Nation 1989): tell the same short story three times, in four,
three and then two minutes. Repetition frees attention, and the shrinking time
pushes you to say it faster with fewer pauses; Nation found speech rate rose and
hesitations fell across the rounds. This module measures each round so you can
see whether that happened.

The measures are the standard "utterance fluency" ones (De Jong et al. 2013):

- **speech rate**: syllables per second, pauses included;
- **articulation rate**: syllables per second of actual speaking;
- **pauses**: silences of 250 ms or more inside the talk, per minute and their mean
  length (250 ms is the usual cut-off; De Jong & Bosker 2013);
- **mean length of run**: syllables between pauses;
- **longest run**: the longest stretch without a pause.

No speech model is needed. Pauses come from loudness; syllables are counted as
loudness peaks in voiced stretches, after De Jong & Wempe (2009). That count is an
estimate, typically within 10–20% of a hand count, but its error is roughly the
same in every round, so changes between rounds are more trustworthy than any
single number.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from . import prosody
from .progress import set_aside, write_json_atomic

PAUSE_S = 0.25
CLICK_S = 0.06
#: Speech threshold: this far from the noise floor towards the loud end, in dB.
THRESHOLD_FRAC = 0.35
MIN_SPREAD_DB = 12.0
#: A syllable peak must stand this far above the valley before it. 3 dB and 30 ms
#: smoothing counted espeak sentences to within ~10% (sum 0.98 of the true count);
#: De Jong & Wempe used 2 dB on natural speech, so expect some difference.
DIP_DB = 3.0
SMOOTH_S = 0.03

ROUNDS = {1: 240, 2: 180, 3: 120}       # round -> seconds


@dataclass
class Measures:
    duration: float           # first to last speech, seconds
    phonation: float          # seconds of speech
    pauses: int
    pause_time: float
    syllables: int
    longest_run: float

    @property
    def speech_rate(self) -> float:
        return self.syllables / self.duration if self.duration else 0.0

    @property
    def articulation_rate(self) -> float:
        return self.syllables / self.phonation if self.phonation else 0.0

    @property
    def pauses_per_min(self) -> float:
        return 60 * self.pauses / self.duration if self.duration else 0.0

    @property
    def mean_pause(self) -> float:
        return self.pause_time / self.pauses if self.pauses else 0.0

    @property
    def mean_run(self) -> float:
        return self.syllables / (self.pauses + 1)

    def row(self) -> dict:
        return {"duration": self.duration, "speech_rate": self.speech_rate,
                "articulation_rate": self.articulation_rate,
                "pauses_per_min": self.pauses_per_min, "mean_pause": self.mean_pause,
                "mean_run": self.mean_run, "longest_run": self.longest_run}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def speech_mask(x: np.ndarray, sr: int = prosody.SR) -> tuple[np.ndarray, float]:
    """Per-frame speech/silence, with clicks removed and short gaps kept as speech."""
    e = prosody.energy_db(x, sr)
    floor, loud = np.percentile(e, 5), np.percentile(e, 95)
    threshold = floor + max(MIN_SPREAD_DB, THRESHOLD_FRAC * (loud - floor))
    mask = e > threshold
    for a, b in _runs(mask):                       # isolated clicks are not speech
        if (b - a) * prosody.HOP_S < CLICK_S:
            mask[a:b] = False
    for a, b in _runs(~mask):                      # gaps under 250 ms are not pauses
        if 0 < a and b < len(mask) and (b - a) * prosody.HOP_S < PAUSE_S:
            mask[a:b] = True
    return mask, float(threshold)


def syllable_peaks(x: np.ndarray, sr: int = prosody.SR, mask=None,
                   threshold: float | None = None) -> np.ndarray:
    """Frame indices of syllable nuclei: voiced loudness peaks with a 2 dB dip."""
    if mask is None:
        mask, threshold = speech_mask(x, sr)
    e = prosody.energy_db(x, sr)
    k = max(1, int(SMOOTH_S / prosody.HOP_S))
    smooth = np.convolve(e, np.ones(k) / k, mode="same")
    voiced = ~np.isnan(prosody.pitch(x, sr))
    near_voiced = np.convolve(voiced, np.ones(5), mode="same") > 0

    peaks, valley = [], np.inf
    for i in range(1, len(smooth) - 1):
        valley = min(valley, smooth[i])
        if not (smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]):
            continue
        if not (mask[i] and near_voiced[i] and smooth[i] > threshold):
            continue
        if smooth[i] - valley >= DIP_DB:
            peaks.append(i)
            valley = smooth[i]
        elif peaks and smooth[i] > smooth[peaks[-1]]:
            # Higher, with no real dip since the last peak: the same syllable.
            peaks[-1] = i
            valley = smooth[i]
    return np.array(peaks, dtype=int)


def measure(x: np.ndarray, sr: int = prosody.SR) -> Measures:
    mask, threshold = speech_mask(x, sr)
    runs = _runs(mask)
    if not runs:
        raise ValueError("No speech found in that recording.")
    start, end = runs[0][0], runs[-1][1]
    hop = prosody.HOP_S
    pauses = [(b - a) * hop for (_, a), (b, _) in pairwise(runs)]
    return Measures(duration=(end - start) * hop,
                    phonation=sum(b - a for a, b in runs) * hop,
                    pauses=len(pauses), pause_time=float(sum(pauses)),
                    syllables=len(syllable_peaks(x, sr, mask, threshold)),
                    longest_run=max(b - a for a, b in runs) * hop)


def measure_file(path) -> Measures:
    return measure(prosody.load(path))


# ------------------------------------------------------------------ history
def default_path() -> Path:
    env = os.environ.get("MDD_FLUENCY")
    return Path(env) if env else Path.home() / ".mdd" / "fluency.json"


class FluencyLog:
    """Each 4/3/2 session's rounds, keyed by session id (saved after every round)."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else default_path()
        self.load_error = None
        self.kept_as: Path | None = None
        self.data = {"version": 1, "sessions": []}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("version") == 1:
                    self.data = data
                else:
                    self.load_error = f"{self.path} is not a fluency log"
            except (OSError, ValueError) as exc:
                self.load_error = f"could not read {self.path}: {exc}"

    def record(self, session_id: str, lang: str, topic: str, rnd: int, m: Measures,
               now: float | None = None) -> None:
        sessions = self.data["sessions"]
        entry = next((s for s in sessions if s["id"] == session_id), None)
        if entry is None:
            entry = {"id": session_id, "at": time.time() if now is None else now,
                     "lang": lang, "topic": topic, "rounds": {}}
            sessions.append(entry)
        entry["topic"] = topic or entry["topic"]
        entry["rounds"][str(rnd)] = asdict(m)

    def sessions(self, lang: str) -> list[dict]:
        return [s for s in self.data["sessions"] if s["lang"] == lang]

    def save(self) -> Path:
        if self.load_error:                 # never overwrite what could not be read
            self.kept_as = set_aside(self.path)
            self.load_error = None
        return write_json_atomic(self.path, self.data)


def from_dict(d: dict) -> Measures:
    return Measures(**d)
