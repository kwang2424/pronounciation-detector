"""Tier 1 (design doc section 7.1): native negative control.

Synthesise every sentence with several native German voices, run the pipeline, and count
every flag as a false positive. Reports phone-level FPR per threshold and per voice, plus
which canonical phones attract the most false flags.

  python -m eval.native_control [--n 100] [--voices edge:de-DE-KatjaNeural,espeak,...]
"""
if __name__ == "__main__":
    from mdd._utf8 import ensure_utf8_mode
    ensure_utf8_mode()

import argparse  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

from .common import CACHE, DEFAULT_LANG, THRESHOLDS, analyse_cached, cache_stats, fmt_pct, is_flagged, key, load_sentences, results_name, write_results  # noqa: E402
from .tts import default_voices, synth_missing  # noqa: E402

FPR_TARGET = 0.05


def collect(voices: list[str], sentences: list[str],
            lang: str = DEFAULT_LANG) -> dict[str, list[tuple[str, dict]]]:
    reports = {}
    for voice in voices:
        vdir = CACHE / "native" / f"{lang}_{voice.replace(':', '_')}" if lang != DEFAULT_LANG \
            else CACHE / "native" / voice.replace(":", "_")
        items = [(t, vdir / f"{key(t)}.wav") for t in sentences]
        print(f"[{voice}] synthesising {sum(1 for _, p in items if not p.exists())} of {len(items)} clips", flush=True)
        synth_missing(voice, items)
        reps = []
        for i, (t, wav) in enumerate(items, 1):
            reps.append((t, analyse_cached(t, wav, wav.with_suffix(".json"), lang)))
            if i % 25 == 0 or i == len(items):
                print(f"[{voice}] analysed {i}/{len(items)}", flush=True)
        reports[voice] = reps
    return reports


def rates(reps: list[tuple[str, dict]]) -> dict:
    phones = [p for _, r in reps for p in r["phones"]]
    n_canon = sum(1 for p in phones if p["op"] != "ins")
    res = {
        "n_sentences": len(reps),
        "n_phones": n_canon,
        "disagreement_rate": sum(1 for p in phones if p["op"] != "match") / max(n_canon, 1),
        "fpr": {}, "sentence_flag_rate": {},
    }
    for tau in THRESHOLDS:
        res["fpr"][str(tau)] = sum(1 for p in phones if is_flagged(p, tau)) / max(n_canon, 1)
        res["sentence_flag_rate"][str(tau)] = (
            sum(1 for _, r in reps if any(is_flagged(p, tau) for p in r["phones"])) / max(len(reps), 1))
    return res


def per_phone(reps: list[tuple[str, dict]], tau: float, top: int = 20) -> list[dict]:
    flags, total, heard = Counter(), Counter(), defaultdict(Counter)
    for _, r in reps:
        for p in r["phones"]:
            k = p["canonical"] or "(insertion)"
            if p["op"] != "ins":
                total[k] += 1
            if is_flagged(p, tau):
                flags[k] += 1
                heard[k][p["realized"] or "∅"] += 1
    return [{"canonical": k, "false_flags": n, "occurrences": total[k],
             "rate": n / total[k] if total[k] else None,
             "heard_as": heard[k].most_common(3)} for k, n in flags.most_common(top)]


def recommend(res: dict) -> float | None:
    ok = [t for t in THRESHOLDS if res["fpr"][str(t)] < FPR_TARGET]
    return max(ok) if ok else None


def recommend_robust(per_voice: dict[str, dict]) -> tuple[float | None, dict]:
    """Threshold from the MEDIAN voice, not the pooled total.

    Pooling assumes the voices are comparable. In the French run they were not:
    disagreement ranged from 8.5% to 47.1%, and the worst voice alone dragged the
    pooled rate past the target at every threshold, recommending -8 — a setting
    at which almost nothing is ever flagged. The median voice is unmoved by one
    bad talker, and the spread is reported so a bad one is visible rather than
    silently averaged in.
    """
    natural = {v: r for v, r in per_voice.items() if not v.startswith("espeak")}
    if not natural:
        return None, {}
    medians = {}
    for tau in THRESHOLDS:
        rates = sorted(r["fpr"][str(tau)] for r in natural.values())
        mid = len(rates) // 2
        medians[str(tau)] = (rates[mid] if len(rates) % 2
                             else (rates[mid - 1] + rates[mid]) / 2)
    ok = [t for t in THRESHOLDS if medians[str(t)] < FPR_TARGET]
    spread = {v: r["disagreement_rate"] for v, r in natural.items()}
    return (max(ok) if ok else None), {"median_fpr": medians, "disagreement": spread}


def markdown(summary: dict) -> str:
    voices = list(summary["voices"])
    L = ["# Native negative control", "",
         f"{summary['natural']['n_sentences']} native clips across {len(voices)} voices "
         f"({summary['natural']['n_phones']} canonical phones from natural voices). Every flag is a false positive.",
         "", "Threshold τ is the GOP cutoff: a phone is flagged only if the free decode disagrees "
         "**and** its GOP is below τ. Less negative τ = stricter = more flags.", "",
         "## Phone-level false-positive rate", "",
         "| τ | natural voices | all voices | " + " | ".join(v.split(":")[-1] for v in voices) + " |",
         "|---|---|---|" + "---|" * len(voices)]
    for t in THRESHOLDS:
        L.append(f"| {t:g} | {fmt_pct(summary['natural']['fpr'][str(t)])} | {fmt_pct(summary['all']['fpr'][str(t)])} | "
                 + " | ".join(fmt_pct(summary["voices"][v]["fpr"][str(t)]) for v in voices) + " |")
    L += ["", "## Sentences with at least one flag", "",
          "| τ | natural voices | " + " | ".join(v.split(":")[-1] for v in voices) + " |",
          "|---|---|" + "---|" * len(voices)]
    for t in THRESHOLDS:
        L.append(f"| {t:g} | {fmt_pct(summary['natural']['sentence_flag_rate'][str(t)])} | "
                 + " | ".join(fmt_pct(summary["voices"][v]["sentence_flag_rate"][str(t)]) for v in voices) + " |")
    L += ["", "## Alignment disagreement before the GOP gate", "",
          "Share of canonical phones where the free decode disagrees at all (what the FPR would be with no GOP gate).", "",
          "| voice | disagreement |", "|---|---|"]
    for v in voices:
        L.append(f"| {v} | {fmt_pct(summary['voices'][v]['disagreement_rate'])} |")
    rec = summary["recommended_tau"]
    med = summary.get("recommended_tau_median_voice")
    detail = summary.get("median_voice") or {}
    L += ["", "## Recommended threshold", "",
          (f"Largest τ with natural-voice FPR below {FPR_TARGET:.0%}: **τ = {rec:g}** "
           f"(FPR {fmt_pct(summary['natural']['fpr'][str(rec)])})." if rec is not None
           else f"No threshold in the sweep keeps pooled natural-voice FPR below "
                f"{FPR_TARGET:.0%}."),
          ""]
    if detail:
        spread = detail["disagreement"]
        lo, hi = min(spread.values()), max(spread.values())
        L += [(f"By the **median voice** (robust to one bad talker): "
               f"**τ = {med:g}** (median FPR {fmt_pct(detail['median_fpr'][str(med)])})."
               if med is not None else
               "No threshold keeps even the median voice below the target."),
              "",
              f"Per-voice disagreement ranges {fmt_pct(lo)}–{fmt_pct(hi)}. "
              + ("A wide spread means the voices are not measuring the same thing — "
                 "check for a dialect mismatch or an unusual speaking style before "
                 "trusting the pooled number."
                 if hi > 2 * lo else "The voices agree closely, so pooling is safe."),
              ""]
    by_voice = summary.get("per_phone_by_voice") or {}
    if len(by_voice) > 1:
        rates = {v: summary["voices"][v]["disagreement_rate"] for v in by_voice}
        worst = max(rates, key=rates.get)
        best = min(rates, key=rates.get)
        if rates[worst] > 2 * rates[best]:
            L += ["", "## Per-voice breakdown (voices disagree by more than 2x)", "",
                  f"`{worst}` disagrees {rates[worst] / rates[best]:.1f}x as often as "
                  f"`{best}`. If the same phones dominate both, it is a degree "
                  f"difference — speaking rate or recording. If different phones "
                  f"dominate, it is an accent, and that voice is measuring something "
                  f"other than the pipeline.", ""]
            for v in sorted(by_voice, key=lambda k: -rates[k]):
                top = ", ".join(f"{r['canonical']} {fmt_pct(r['rate'])}"
                                for r in by_voice[v][:6])
                L += [f"- **{v}** ({fmt_pct(rates[v])} disagreement): {top}"]
            L += [""]
    L += [f"## Most falsely flagged phones (natural voices, τ = {summary['per_phone_tau']:g})", "",
          "| canonical | false flags | occurrences | rate | heard as |", "|---|---|---|---|---|"]
    for row in summary["per_phone"]:
        heard = ", ".join(f"{h} ×{n}" for h, n in row["heard_as"])
        L.append(f"| {row['canonical']} | {row['false_flags']} | {row['occurrences']} | {fmt_pct(row['rate'])} | {heard} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="use only the first N sentences")
    ap.add_argument("--lang", default=DEFAULT_LANG, help="language code (de, fr, ...)")
    ap.add_argument("--voices", default=None,
                    help="comma-separated voice specs; defaults to this language's neural voices")
    ap.add_argument("--per-phone-tau", type=float, default=-2.0)
    a = ap.parse_args()
    voices = [v for v in (a.voices or ",".join(default_voices(a.lang))).split(",") if v]
    reports = collect(voices, load_sentences(a.n, a.lang), a.lang)
    # espeak is a formant synthesiser and is kept out of the headline FPR, which
    # is meant to describe natural voices. startswith, not ==: the spec carries a
    # voice name for non-German languages ("espeak:fr").
    natural = [tr for v, reps in reports.items() if not v.startswith("espeak") for tr in reps]
    everything = [tr for reps in reports.values() for tr in reps]
    summary = {
        "thresholds": THRESHOLDS,
        "voices": {v: rates(reps) for v, reps in reports.items()},
        "natural": rates(natural or everything),
        "all": rates(everything),
        "per_phone_tau": a.per_phone_tau,
        "per_phone": per_phone(natural or everything, a.per_phone_tau),
        # Per voice as well as pooled: when one talker's disagreement is several
        # times another's, the pooled table describes that talker, not the
        # language, and the only way to tell an accent apart from a recording
        # problem is to see which phones each voice actually loses.
        "per_phone_by_voice": {v: per_phone(reps, a.per_phone_tau, top=8)
                               for v, reps in reports.items()
                               if not v.startswith("espeak")},
    }
    summary["lang"] = a.lang
    summary["recommended_tau"] = recommend(summary["natural"])
    robust, detail = recommend_robust(summary["voices"])
    summary["recommended_tau_median_voice"] = robust
    summary["median_voice"] = detail
    path = write_results(results_name("native_control", a.lang), summary, markdown(summary))
    hit, miss = cache_stats["hit"], cache_stats["miss"]
    if miss == 0:
        print(f"\nAll {hit} clips came from cache — the numbers are unchanged by design. "
              f"A code change that alters scoring moves the cache key; if you expected "
              f"one to, check you are on the branch that has it.")
    else:
        print(f"\nRe-analysed {miss} of {hit + miss} clips "
              f"({hit} from cache) — these numbers are new.")
    print(f"wrote {path}")
    print(markdown(summary))


if __name__ == "__main__":
    main()
