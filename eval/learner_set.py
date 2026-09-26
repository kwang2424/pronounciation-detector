"""Tier 4: the scorer on your own voice, from the Test set tab's recordings.

Every word was recorded as your best attempt and as a deliberate error, so each
contrast gets:

- caught         — deliberate errors flagged on the target sound. Certain label.
- false alarms   — best attempts flagged on the target sound. Errs high: a best
                   attempt that was genuinely off is still counted here.
- diagnosed      — deliberate errors heard as the error you made (ü heard as u).
- hears a diff   — the recogniser heard something different in your two versions,
                   flagged or not. Low here means it cannot tell them apart; high
                   with a low catch rate means it hears it but is too lenient.

Faked errors are clearer than natural ones: read the catch rate as "catches clear
errors", an upper bound for subtle ones.

  python -m eval.learner_set de
  python -m eval.learner_set fr --root D:/mdd-testset

The report stays next to your recordings (it describes your voice); paste it
wherever it is useful.
"""
if __name__ == "__main__":
    from mdd._utf8 import ensure_utf8_mode
    ensure_utf8_mode()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

from mdd.pipeline import GOP_THRESHOLD, _length_only  # noqa: E402
from mdd.testset import contrast, contrasts, default_root, recorded  # noqa: E402

from .common import analyse_cached, cache_stats, is_flagged  # noqa: E402

SWEEP = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a rate: with 20 takes, 15/20 is really 'somewhere 53–89%'."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _flag(p: dict, tau: float, length: bool) -> bool:
    if is_flagged(p, tau):
        return True
    # The app never flags length alone, so a length-only difference is stored as
    # a "match" whose realised vowel differs; scoring it here needs no re-run.
    return (length and p["op"] == "match" and p["realized"] not in (None, p["canonical"])
            and _length_only(p["canonical"], p["realized"])
            and (p["gop"] is None or p["gop"] < tau))


def target_hits(rep: dict, c) -> list[dict]:
    """The target phones of one take, plus any sound inserted right next to one.

    An error like French 'bonn' can come out as the nasal vowel *plus* an inserted
    n, which is a correct detection even though the vowel itself matched.
    """
    phones = rep["phones"]
    canon = [i for i, p in enumerate(phones) if p["op"] != "ins"]
    if c.position == "first":
        canon = canon[:1]
    elif c.position == "last":
        canon = canon[-1:]
    idx = [i for i in canon if phones[i]["canonical"] in c.targets]
    near = {j for i in idx for j in (i - 1, i + 1)
            if 0 <= j < len(phones) and phones[j]["op"] == "ins"}
    return [phones[i] for i in sorted(set(idx) | near)]


def heard(hits: list[dict]) -> str:
    return " ".join(p["realized"] or "∅" for p in hits) or "?"


def score(take, rep: dict, tau: float = GOP_THRESHOLD) -> dict:
    c = contrast(take.lang, take.contrast)
    hits = target_hits(rep, c)
    return {
        "word": take.word, "kind": take.kind, "take": take.take,
        "flagged": any(_flag(p, tau, False) for p in hits),
        "flagged_with_length": any(_flag(p, tau, True) for p in hits),
        "any_flag": any(_flag(p, tau, False) for p in rep["phones"]),
        "diagnosed": any(p["realized"] in c.heard_as for p in hits if p["op"] != "ins"),
        "heard": heard(hits),
    }


def summarise(rows: list[dict], length: bool = False) -> dict:
    key = "flagged_with_length" if length else "flagged"
    best = [r for r in rows if r["kind"] == "best"]
    error = [r for r in rows if r["kind"] == "error"]
    pairs = {(r["word"], r["take"]): {} for r in rows}
    for r in rows:
        pairs[(r["word"], r["take"])][r["kind"]] = r["heard"]
    complete = [p for p in pairs.values() if len(p) == 2]
    return {
        "n_best": len(best), "n_error": len(error),
        "caught": sum(r[key] for r in error),
        "false_alarms": sum(r[key] for r in best),
        "diagnosed": sum(r["diagnosed"] for r in error),
        "pairs": len(complete),
        "differ": sum(p["best"] != p["error"] for p in complete),
        "heard_error": Counter(r["heard"] for r in error).most_common(3),
        "heard_best": Counter(r["heard"] for r in best).most_common(3),
    }


def verdict(s: dict) -> str:
    if not s["n_error"] or not s["n_best"]:
        return "not enough takes yet"
    catch = s["caught"] / s["n_error"]
    fa = s["false_alarms"] / s["n_best"]
    differ = s["differ"] / s["pairs"] if s["pairs"] else None
    if differ is not None and differ < 0.5:
        return ("hears no reliable difference between your two versions: either it "
                "is blind to this, or your versions sound alike")
    if catch >= 0.7 and fa <= 0.2:
        return "works for your voice"
    if catch < 0.5 and differ is not None and differ >= 0.7:
        return "hears the difference but rarely flags it: too lenient here"
    if fa > 0.3:
        return ("often flags your best attempt: false alarms, or your best attempt "
                "is genuinely off — a native listener can tell which")
    return "mixed: usable as a hint, not a verdict"


def _pct(k: int, n: int) -> str:
    if not n:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, {100 * lo:.0f}–{100 * hi:.0f})"


def evaluate(root: Path, lang: str, rec=None) -> tuple[dict, str]:
    """Analyse every recorded take (cached) and build the report."""
    reports = []
    for take, wav in recorded(root, lang):
        cache = root / lang / ".cache" / take.contrast / wav.name
        if rec is None:
            rep = analyse_cached(take.word, wav, cache, lang)
        else:     # tests pass a stand-in recogniser
            from mdd.pipeline import analyse
            rep = analyse(take.word, str(wav), rec, lang=lang)
        reports.append((take, rep))

    data = {"lang": lang, "threshold": GOP_THRESHOLD, "contrasts": {}, "sweep": {}}
    lines = [f"# Your voice, {lang}: {len(reports)} takes",
             "",
             f"At the app's threshold ({GOP_THRESHOLD}). Rates show a 95% range; "
             "deliberate errors are clearer than natural ones, so *caught* is an upper bound.",
             "",
             "| Contrast | Caught | False alarms | Diagnosed | Hears a diff | Verdict |",
             "|---|---|---|---|---|---|"]
    for c in contrasts(lang):
        rows = [score(t, r) for t, r in reports if t.contrast == c.id]
        if not rows:
            continue
        s = summarise(rows)
        data["contrasts"][c.id] = {**s, "rows": rows, "verdict": verdict(s)}
        lines.append(f"| {c.label} | {_pct(s['caught'], s['n_error'])} | "
                     f"{_pct(s['false_alarms'], s['n_best'])} | "
                     f"{_pct(s['diagnosed'], s['n_error'])} | "
                     f"{_pct(s['differ'], s['pairs'])} | {verdict(s)} |")
        if c.length:
            sl = summarise(rows, length=True)
            data["contrasts"][c.id]["with_length_flags"] = sl
            lines.append(f"| ↳ with length flags on | {_pct(sl['caught'], sl['n_error'])} | "
                         f"{_pct(sl['false_alarms'], sl['n_best'])} | | | "
                         f"off in the app (37% false alarms on native speech) |")

    if data["contrasts"]:
        lines += ["", "## What it heard on the target sound", "",
                  "| Contrast | Best attempt | Deliberate error |", "|---|---|---|"]
        for cid, s in data["contrasts"].items():
            fmt = lambda pairs: ", ".join(f"`{h}` ×{n}" for h, n in pairs)  # noqa: E731
            lines.append(f"| {contrast(lang, cid).label} | {fmt(s['heard_best'])} | "
                         f"{fmt(s['heard_error'])} |")

        lines += ["", "## Threshold", "",
                  "| Threshold | Caught | False alarms |", "|---|---|---|"]
        for tau in SWEEP:
            rows = [score(t, r, tau) for t, r in reports]
            s = summarise(rows)
            data["sweep"][str(tau)] = {"caught": s["caught"], "n_error": s["n_error"],
                                       "false_alarms": s["false_alarms"], "n_best": s["n_best"]}
            mark = " ← app" if tau == GOP_THRESHOLD else ""
            lines.append(f"| {tau}{mark} | {_pct(s['caught'], s['n_error'])} | "
                         f"{_pct(s['false_alarms'], s['n_best'])} |")
    return data, "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(prog="eval.learner_set", description=__doc__.split("\n")[0])
    ap.add_argument("lang", nargs="?", default="de")
    ap.add_argument("--root", default=None, help=f"test-set folder (default: {default_root()})")
    args = ap.parse_args()
    root = Path(args.root) if args.root else default_root()
    if not recorded(root, args.lang):
        raise SystemExit(f"No recordings under {root / args.lang} — record some in the "
                         f"app's Test set tab first.")
    data, md = evaluate(root, args.lang)
    out = root / args.lang
    (out / "report.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"(analysed {cache_stats['miss']} new takes, {cache_stats['hit']} from cache; "
          f"report saved to {out / 'report.md'})")


if __name__ == "__main__":
    main()
