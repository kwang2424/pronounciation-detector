"""Feature-aware Needleman-Wunsch alignment of canonical vs realised phones."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import panphon.distance

_dist = panphon.distance.Distance()
GAP = 0.9            # cost of insertion / deletion
LENGTH_ONLY = 0.35   # same quality, different length (Stadt/Staat)


@lru_cache(maxsize=None)
def sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if a.rstrip("ː") == b.rstrip("ː"):
        return LENGTH_ONLY
    try:
        d = _dist.weighted_feature_edit_distance(a, b)
        # panphon weighted distances for a single segment pair typically fall in ~[0, 8]
        return min(1.0, d / 8.0) if d > 0 else 0.15
    except Exception:
        return 1.0


@dataclass
class AlignOp:
    canonical: str | None
    realized: str | None
    op: Literal["match", "sub", "del", "ins"]
    cost: float


def align(canon: list[str], real: list[str]) -> list[AlignOp]:
    n, m = len(canon), len(real)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * GAP
    for j in range(1, m + 1):
        D[0][j] = j * GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(
                D[i - 1][j - 1] + sub_cost(canon[i - 1], real[j - 1]),
                D[i - 1][j] + GAP,
                D[i][j - 1] + GAP,
            )
    ops: list[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(D[i][j] - (D[i - 1][j - 1] + sub_cost(canon[i - 1], real[j - 1]))) < 1e-9:
            c = sub_cost(canon[i - 1], real[j - 1])
            ops.append(AlignOp(canon[i - 1], real[j - 1], "match" if c == 0 else "sub", c))
            i, j = i - 1, j - 1
        elif i > 0 and abs(D[i][j] - (D[i - 1][j] + GAP)) < 1e-9:
            ops.append(AlignOp(canon[i - 1], None, "del", GAP))
            i -= 1
        else:
            ops.append(AlignOp(None, real[j - 1], "ins", GAP))
            j -= 1
    ops.reverse()
    return ops
