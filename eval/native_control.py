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

from .common import CACHE, DEFAULT_LANG, THRESHOLDS, analyse_cached, fmt_pct, is_flagged, key, load_sentences, results_name, write_results  # noqa: E402
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
    L += ["", "## Recommended threshold", "",
          (f"Largest τ with natural-voice FPR below {FPR_TARGET:.0%}: **τ = {rec:g}** "
           f"(FPR {fmt_pct(summary['natural']['fpr'][str(rec)])})." if rec is not None
           else f"No threshold in the sweep keeps natural-voice FPR below {FPR_TARGET:.0%}."),
          "", f"## Most falsely flagged phones (natural voices, τ = {summary['per_phone_tau']:g})", "",
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
    }
    summary["lang"] = a.lang
    summary["recommended_tau"] = recommend(summary["natural"])
    path = write_results(results_name("native_control", a.lang), summary, markdown(summary))
    print(f"\nwrote {path}")
    print(markdown(summary))


if __name__ == "__main__":
    main()
