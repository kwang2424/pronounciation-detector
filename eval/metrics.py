"""Standard Mispronunciation Detection and Diagnosis (MDD) evaluation metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MDDCounts:
    """Confusion-matrix counts for phoneme-level MDD detection and diagnosis.

    ta: true acceptance (correct phone, not flagged)
    fr: false rejection (correct phone, wrongly flagged)
    tr: true rejection (mispronounced phone, correctly flagged)
    fa: false acceptance (mispronounced phone, missed)
    cd: correct diagnosis (correctly flagged AND realized matches annotation)
    de: diagnosis error (correctly flagged, wrong realized phone)
    """

    ta: int = 0
    fr: int = 0
    tr: int = 0
    fa: int = 0
    cd: int = 0
    de: int = 0

    def __add__(self, other: MDDCounts) -> MDDCounts:
        return MDDCounts(
            ta=self.ta + other.ta,
            fr=self.fr + other.fr,
            tr=self.tr + other.tr,
            fa=self.fa + other.fa,
            cd=self.cd + other.cd,
            de=self.de + other.de,
        )


def compute_metrics(c: MDDCounts) -> dict:
    """Compute Precision, Recall, F1, MCC, FAR, FRR, conditional diag acc, and DER."""
    precision = c.tr / (c.tr + c.fr) if (c.tr + c.fr) > 0 else None
    recall = c.tr / (c.tr + c.fa) if (c.tr + c.fa) > 0 else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    denom = (c.tr + c.fr) * (c.tr + c.fa) * (c.ta + c.fr) * (c.ta + c.fa)
    mcc = ((c.tr * c.ta) - (c.fr * c.fa)) / math.sqrt(denom) if denom > 0 else None
    far = c.fa / (c.fa + c.tr) if (c.fa + c.tr) > 0 else None
    frr = c.fr / (c.ta + c.fr) if (c.ta + c.fr) > 0 else None
    cond_diag_acc = c.cd / c.tr if c.tr > 0 else None
    der = c.de / (c.cd + c.de) if (c.cd + c.de) > 0 else None
    return {
        "ta": c.ta,
        "fr": c.fr,
        "tr": c.tr,
        "fa": c.fa,
        "cd": c.cd,
        "de": c.de,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "far": far,
        "frr": frr,
        "cond_diag_acc": cond_diag_acc,
        "der": der,
    }


def recommend_threshold(metrics_by_tau: dict[str, dict], *, max_frr: float = 0.05) -> float | None:
    """Choose the best threshold keeping FRR below `max_frr` (maximizing F1 when available)."""
    ok: list[tuple[float, dict]] = []
    for k, m in metrics_by_tau.items():
        frr = m.get("frr")
        if frr is None or frr >= max_frr:
            continue
        ok.append((float(k), m))
    if not ok:
        return None
    with_f1 = [(tau, m) for tau, m in ok if m.get("f1") is not None]
    if with_f1:
        return max(with_f1, key=lambda kv: (kv[1]["f1"], -kv[1]["frr"], kv[0]))[0]
    return max(tau for tau, _ in ok)
