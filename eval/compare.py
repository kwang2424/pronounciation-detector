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
        r2 = {r["name"]: r for r in s2["rows"]}
        names = [r["name"] for r in s2["rows"]] + [r["name"] for r in s1["rows"] if r["name"] not in r2]
        L += ["## Synthetic errors: recall at τ = -2 and diagnosis accuracy", "",
              "| error | n | recall before | after | diagnosis before | after |", "|---|---|---|---|---|---|"]
        for name in names:
            o = r1.get(name)
            r = r2.get(name)
            if r is None:
                L.append(f"| {name} (removed) | {o['n']} | {fmt_pct(o['recall']['-2.0'])} | – | "
                         f"{fmt_pct(o['diag_acc'])} | – |")
                continue
            if r.get("error") and o is None:
                continue
            rec_after = "–" if r.get("error") else fmt_pct(r["recall"]["-2.0"])
            diag_after = "–" if r.get("error") else fmt_pct(r["diag_acc"])
            L.append(f"| {name} | {r['n']} | {fmt_pct(o['recall']['-2.0']) if o else '–'} | "
                     f"{rec_after} | {fmt_pct(o['diag_acc']) if o else '–'} | {diag_after} |")
        o1, o2 = s1["overall"], s2["overall"]
        L.append(f"| **all** | {o2['n']} | {fmt_pct(o1['recall']['-2.0'])} | {fmt_pct(o2['recall']['-2.0'])} | "
                 f"{fmt_pct(o1['diag_acc'])} | {fmt_pct(o2['diag_acc'])} |")
        if o1.get("cond_diag_acc") and o2.get("cond_diag_acc"):
            L += ["", "### Threshold τ = -1 summary (recall / cond_diag / DER)", "",
                  f"- Recall@-1: {fmt_pct(o1['recall'].get('-1.0'))} → {fmt_pct(o2['recall'].get('-1.0'))}",
                  f"- Cond diagnosis@-1: {fmt_pct(o1['cond_diag_acc'].get('-1.0'))} → {fmt_pct(o2['cond_diag_acc'].get('-1.0'))}",
                  f"- DER@-1: {fmt_pct((o1.get('der') or {}).get('-1.0'))} → {fmt_pct((o2.get('der') or {}).get('-1.0'))}"]
    print("\n".join(L))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
