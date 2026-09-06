"""Compare two sets of eval results, e.g. an archived baseline against the current run.

  python -m eval.compare eval/results/v1 eval/results
"""
import json
import sys
from pathlib import Path

from .common import fmt_pct


def load(d: Path, name: str) -> dict | None:
    p = d / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def delta(a, b) -> str:
    if a is None or b is None:
        return "–"
    return f"{100 * (b - a):+.1f} pt"


def main(before: Path, after: Path) -> None:
    L = [f"# {before} → {after}", ""]
    n1, n2 = load(before, "native_control"), load(after, "native_control")
    if n1 and n2:
        L += ["## Native control: phone-level false-positive rate (natural voices)", "",
              "| τ | before | after | change |", "|---|---|---|---|"]
        for t in n2["thresholds"]:
            a, b = n1["natural"]["fpr"].get(str(t)), n2["natural"]["fpr"].get(str(t))
            L.append(f"| {t:g} | {fmt_pct(a)} | {fmt_pct(b)} | {delta(a, b)} |")
        L += ["", f"Recommended τ: {n1['recommended_tau']} → {n2['recommended_tau']}", "",
              "| voice | disagreement before | after |", "|---|---|---|"]
        for v in n2["voices"]:
            L.append(f"| {v} | {fmt_pct(n1['voices'].get(v, {}).get('disagreement_rate'))} | "
                     f"{fmt_pct(n2['voices'][v]['disagreement_rate'])} |")
        L.append("")
    s1, s2 = load(before, "synthetic_errors"), load(after, "synthetic_errors")
    if s1 and s2:
        r1 = {r["name"]: r for r in s1["rows"]}
        L += ["## Synthetic errors: recall at τ = -2 and diagnosis accuracy", "",
              "| error | n | recall before | after | diagnosis before | after |", "|---|---|---|---|---|---|"]
        for r in s2["rows"]:
            o = r1.get(r["name"])
            if r["error"]:
                continue
            L.append(f"| {r['name']} | {r['n']} | {fmt_pct(o['recall']['-2.0']) if o else '–'} | "
                     f"{fmt_pct(r['recall']['-2.0'])} | {fmt_pct(o['diag_acc']) if o else '–'} | {fmt_pct(r['diag_acc'])} |")
        o1, o2 = s1["overall"], s2["overall"]
        L.append(f"| **all** | {o2['n']} | {fmt_pct(o1['recall']['-2.0'])} | {fmt_pct(o2['recall']['-2.0'])} | "
                 f"{fmt_pct(o1['diag_acc'])} | {fmt_pct(o2['diag_acc'])} |")
    print("\n".join(L))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
