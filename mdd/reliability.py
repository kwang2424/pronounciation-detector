"""How much to trust a given flag, from the evaluation results.

A flag is not evidence in itself. The native control measures how often the
recogniser flags a phone on speech that is *known correct*, and the synthetic
positives measure how often it catches an error that is *known present*. Both
vary enormously by phone: on German native speech /d/ is reported as [t] in 2.8%
of its occurrences, while final devoicing errors are caught 0% of the time no
matter how wrong the speaker is.

Without that context a learner reads every row of the report as equally real,
and will chase a phantom /d/ while an invisible final-devoicing error goes
unmentioned. This module attaches the context, reading the committed results
rather than hard-coding numbers that would drift from them.

Missing results (a language not yet evaluated) yield None — unknown, stated as
unknown, never guessed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results"

#: A phone falsely flagged on this share of its native occurrences is one whose
#: flags should be read with suspicion.
NOISY_FP = 0.05
#: An error type caught this rarely is effectively invisible: no flag says nothing.
BLIND_RECALL = 0.30


@dataclass(frozen=True)
class Reliability:
    #: Share of this phone's native occurrences that were falsely flagged.
    false_positive_rate: float | None = None
    #: What the recogniser usually mis-hears it as on correct speech.
    usual_confusion: str | None = None
    #: Whether the substitution actually observed is that usual confusion.
    matches_usual: bool = False

    @property
    def level(self) -> str:
        if self.false_positive_rate is None:
            return "unknown"
        # The rate alone understates the signature errors. German /d/ is falsely
        # flagged on only 2.8% of its occurrences, but 10 of those 11 flags were
        # specifically d->[t] — so seeing exactly d->[t] is far weaker evidence
        # than the phone's overall rate suggests. A different substitution for the
        # same phone is correspondingly stronger.
        if self.matches_usual and self.false_positive_rate >= 0.02:
            return "noisy"
        if self.false_positive_rate >= NOISY_FP:
            return "noisy"
        if self.false_positive_rate >= 0.02:
            return "fair"
        return "solid"

    def note(self) -> str:
        if self.false_positive_rate is None:
            return "not yet evaluated for this language"
        pct = f"{self.false_positive_rate:.1%}"
        if self.level == "noisy":
            if self.matches_usual:
                return (f"treat with suspicion — this exact substitution is the "
                        f"recogniser's habitual error on native speech ({pct} of "
                        f"occurrences)")
            extra = f", usually as [{self.usual_confusion}]" if self.usual_confusion else ""
            return f"treat with suspicion — falsely flagged on {pct} of native speech{extra}"
        if self.level == "fair":
            return f"fairly reliable — {pct} false-positive rate on native speech"
        return f"reliable — {pct} false-positive rate on native speech"


def _results_path(lang: str, name: str) -> Path:
    return RESULTS / (name if lang == "de" else f"{lang}/{name}")


@lru_cache(maxsize=8)
def _native(lang: str) -> dict[str, dict]:
    path = _results_path(lang, "native_control.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for row in data.get("per_phone", []):
        canonical = row.get("canonical")
        if canonical and canonical != "(insertion)":
            heard = row.get("heard_as") or []
            out[canonical] = {"rate": row.get("rate"),
                              "heard": heard[0][0] if heard else None}
    return out


@lru_cache(maxsize=8)
def blind_spots(lang: str) -> dict[str, float]:
    """Error types the recogniser essentially never catches, name -> recall.

    Their absence from a report means nothing, which is worth saying out loud:
    German final devoicing and vowel-length errors are at 0% recall, so a clean
    report is not evidence that they were pronounced correctly.
    """
    path = _results_path(lang, "synthetic_errors.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for row in data.get("rows", []):
        recall = (row.get("recall") or {}).get("-2.0")
        if recall is not None and recall <= BLIND_RECALL:
            out[row.get("name", "?")] = recall
    return out


def reliability(canonical: str | None, lang: str = "de",
                realized: str | None = None) -> Reliability:
    if canonical is None:
        return Reliability()
    row = _native(lang).get(canonical)
    if row is None:
        # Absent from the table means it was never falsely flagged in the control.
        return Reliability(0.0, None) if _native(lang) else Reliability()
    heard = row.get("heard")
    return Reliability(row.get("rate"), heard,
                       matches_usual=realized is not None and realized == heard)
