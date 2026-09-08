"""Tier 2 (design doc section 7.2): synthetic positives.

For each error type in eval.errors.CATALOG, pick sentences containing the canonical phone in
a valid context, inject the error into espeak's phoneme string, synthesise, and check that
the pipeline flags that phone (recall) and names the substitute correctly (diagnosis).
The same sentences synthesised unmodified, also from phonemes, serve as controls: a flag at
the same position in the control is a false alarm the injection didn't cause.

  python -m eval.synthetic_errors [--n 100] [--per-type 8]
"""
if __name__ == "__main__":
    from mdd._utf8 import ensure_utf8_mode
    ensure_utf8_mode()

import argparse  # noqa: E402
from collections import Counter  # noqa: E402

from mdd.normalize import tokenize  # noqa: E402

from .common import CACHE, THRESHOLDS, analyse_cached, canonical_tokens_by_word, fmt_pct, is_flagged, key, load_sentences, write_results  # noqa: E402
from .errors import CATALOG, MNEMONIC, ErrorType  # noqa: E402
from mdd.validate import SEPARATION_THRESHOLD  # noqa: E402

from .tts import Espeak, PhonemeGrid  # noqa: E402

REPORT_TAUS = [-6.0, -4.0, -2.0, -1.0, 0.0]


def build_grids(esp: Espeak, sentences: list[str]):
    """espeak grid per sentence, verified token-for-token against the pipeline's own G2P."""
    grids, skipped = {}, []
    for text in sentences:
        try:
            g = esp.phoneme_grid(text)
        except ValueError as e:
            skipped.append((text, str(e)))
            continue
        pipeline_words = canonical_tokens_by_word(text)
        esp_words = []
        for word in g.words:
            toks, posmap = [], []
            for pos, ipa in word:
                sub = tokenize(ipa)
                toks += sub
                posmap += [pos] * len(sub)
            esp_words.append((toks, posmap))
        if [t for _, t in pipeline_words] != [t for t, _ in esp_words]:
            skipped.append((text, "sentence-level espeak phonemes differ from per-word G2P"))
            continue
        for toks, posmap in esp_words:
            for tok, pos in zip(toks, posmap):
                if posmap.count(pos) == 1:
                    MNEMONIC.setdefault(tok, PhonemeGrid.split_stress(g.mnemonic_at(pos))[1])
        grids[text] = (g, pipeline_words, esp_words)
    return grids, skipped


def candidates(err: ErrorType, grids, per_type: int):
    out = []
    for text, (g, pw, ew) in grids.items():
        for wi, (toks, posmap) in enumerate(ew):
            hit = next((ti for ti, _ in enumerate(toks) if err.ok(toks, ti) and posmap.count(posmap[ti]) == 1), None)
            if hit is not None:
                out.append((text, wi, hit))
                break   # at most one injection per sentence per error type
        if len(out) >= per_type:
            break
    return out


def phone_at(rep: dict, j: int):
    """The report entry for canonical index j, and the entry right after it (for insertions)."""
    ci = -1
    for k, p in enumerate(rep["phones"]):
        if p["op"] != "ins":
            ci += 1
            if ci == j:
                nxt = rep["phones"][k + 1] if k + 1 < len(rep["phones"]) else None
                return p, nxt
    raise IndexError(j)


def evaluate(err: ErrorType, rep: dict, ctl: dict, j: int) -> dict:
    p, nxt = phone_at(rep, j)
    c, _ = phone_at(ctl, j)
    if err.kind == "sub":
        detected = {t: is_flagged(p, t) for t in THRESHOLDS}
        diag_ok = p["op"] == "sub" and p["realized"] in (err.realized, *err.also_ok)
        heard = p["realized"] if p["op"] != "match" else "(match)"
    elif err.kind == "del":
        detected = {t: is_flagged(p, t) for t in THRESHOLDS}
        diag_ok = p["op"] == "del"
        heard = "∅" if p["op"] == "del" else (p["realized"] if p["op"] != "match" else "(match)")
    else:   # insertion after the canonical phone
        extra = err.realized.split()[1:]
        ins_ok = nxt is not None and nxt["op"] == "ins"
        detected = {t: is_flagged(p, t) or (ins_ok and is_flagged(nxt, t)) for t in THRESHOLDS}
        diag_ok = ins_ok and nxt["realized"] in (extra[0], *err.also_ok)
        heard = f"+{nxt['realized']}" if ins_ok else (p["realized"] if p["op"] != "match" else "(match)")
    return {"detected": detected, "diag_ok": diag_ok, "heard": heard, "gop": p["gop"],
            "control_flagged": {t: is_flagged(c, t) for t in THRESHOLDS}, "control_gop": c["gop"]}


def run(esp: Espeak, grids, per_type: int) -> dict:
    results = {}
    for err in CATALOG:
        cands = candidates(err, grids, per_type)
        entry = {"kind": err.kind, "canonical": err.canonical, "realized": err.realized, "note": err.note, "cases": []}
        try:
            new_mn = None if err.realized is None else "".join(MNEMONIC[r] for r in err.realized.split())
        except KeyError as e:
            entry["error"] = f"no espeak mnemonic known for {e}"
            results[err.name] = entry
            continue
        if not cands:
            entry["error"] = "no sentence contains this phone in the required context"
        for text, wi, ti in cands:
            g, pw, ew = grids[text]
            pos = ew[wi][1][ti]
            j = sum(len(t) for _, t in pw[:wi]) + ti
            inj = CACHE / "synthetic" / f"{key(err.name, text, str(wi), str(ti))}.wav"
            ctl = CACHE / "synthetic" / f"ctl_{key(text)}.wav"
            if not inj.exists():
                esp.synth_phonemes(g.with_replacement(pos, new_mn), inj)
            if not ctl.exists():
                esp.synth_phonemes(g.render(), ctl)
            rep = analyse_cached(text, inj, inj.with_suffix(".json"))
            ctl_rep = analyse_cached(text, ctl, ctl.with_suffix(".json"))
            case = evaluate(err, rep, ctl_rep, j)
            case.update({"text": text, "word": pw[wi][0],
                         "rendered": _rendered(ctl, inj)})
            entry["cases"].append(case)
        results[err.name] = entry
        n = len(entry["cases"])
        print(f"{err.name:18} n={n:2}  recall@-2={fmt_pct(_recall(entry, -2.0)) if n else 'n/a':>6}  "
              f"diag={fmt_pct(_diag(entry)) if n else 'n/a':>6}  {entry.get('error', '')}", flush=True)
    return results


def _rendered(control_wav, injected_wav) -> float | None:
    """How far the injected audio differs from the control, against synthesis jitter.

    An injection espeak does not actually render produces identical audio, and the
    pipeline then scores 0% recall on an error that is not in the signal — a
    limit of the test material reported as a limit of the recogniser. Measured:
    final_k->ɡ separates at 0.93x, final_t->d at 1.01x, final_p->b at 0.99x, all
    at the jitter floor, while long_a->short reaches 7.91x and is a real blind
    spot. Three of the four "undetectable" rows were untestable, not undetectable.
    """
    import soundfile as sf

    from mdd.validate import _distance, _spectrum

    try:
        a, _ = sf.read(str(control_wav))
        b, _ = sf.read(str(injected_wav))
    except Exception:
        return None
    sa = _spectrum([int(v * 32768) for v in a])
    sb = _spectrum([int(v * 32768) for v in b])
    if sa is None or sb is None:
        return None
    # Same-file distance is zero for espeak's cached output, so compare against a
    # small absolute floor on the unit-normalised scale, as mdd.validate does.
    from mdd.validate import MIN_FLOOR
    return float(_distance(sa, sb) / MIN_FLOOR)


def _rendered_ratio(entry) -> float | None:
    vals = [c["rendered"] for c in entry["cases"] if c.get("rendered") is not None]
    return sum(vals) / len(vals) if vals else None


def _recall(entry, tau):
    cs = entry["cases"]
    return sum(c["detected"][tau] for c in cs) / len(cs) if cs else None


def _diag(entry):
    cs = entry["cases"]
    return sum(c["diag_ok"] for c in cs) / len(cs) if cs else None


def _control_fp(entry, tau):
    cs = entry["cases"]
    return sum(c["control_flagged"][tau] for c in cs) / len(cs) if cs else None


def summarise(results: dict) -> dict:
    rows = []
    for name, e in results.items():
        cs = e["cases"]
        rows.append({
            "name": name, "kind": e["kind"], "canonical": e["canonical"], "realized": e["realized"], "n": len(cs),
            "recall": {str(t): _recall(e, t) for t in THRESHOLDS},
            "diag_acc": _diag(e),
            "control_fp": {str(t): _control_fp(e, t) for t in THRESHOLDS},
            "heard_as": Counter(c["heard"] for c in cs).most_common(3),
            # Whether espeak rendered the injection at all. A row with a low
            # value has an untestable error, not an undetectable one.
            "rendered": _rendered_ratio(e),
            "error": e.get("error"), "note": e["note"],
        })
    all_cases = [c for e in results.values() for c in e["cases"]]
    overall = {
        "n": len(all_cases),
        "recall": {str(t): sum(c["detected"][t] for c in all_cases) / max(len(all_cases), 1) for t in THRESHOLDS},
        "diag_acc": sum(c["diag_ok"] for c in all_cases) / max(len(all_cases), 1),
        "control_fp": {str(t): sum(c["control_flagged"][t] for c in all_cases) / max(len(all_cases), 1) for t in THRESHOLDS},
    }
    return {"thresholds": THRESHOLDS, "rows": rows, "overall": overall}


def markdown(summary: dict, skipped) -> str:
    L = ["# Synthetic positives", "",
         f"{summary['overall']['n']} injected errors across {sum(1 for r in summary['rows'] if r['n'])} error types, "
         "synthesised with espeak-ng from modified phoneme strings. Recall = the injected phone was flagged at τ. "
         "Diagnosis = the alignment named the substitute exactly (independent of τ). Control FP = the same "
         "position was flagged in the unmodified synthesis, i.e. a false alarm espeak's audio causes on its own.", "",
         "| error | n | " + " | ".join(f"recall@{t:g}" for t in REPORT_TAUS) + " | diagnosis | control FP@-2 | heard as |",
         "|---|---|" + "---|" * len(REPORT_TAUS) + "---|---|---|"]
    for r in summary["rows"]:
        if r["error"]:
            L.append(f"| {r['name']} | 0 | " + " | ".join("–" for _ in REPORT_TAUS) + f" | – | – | _{r['error']}_ |")
            continue
        heard = ", ".join(f"{h} ×{n}" for h, n in r["heard_as"])
        rendered = r.get("rendered")
        untestable = rendered is not None and rendered < SEPARATION_THRESHOLD
        cells = ("n/a " * len(REPORT_TAUS)).split() if untestable else [
            fmt_pct(r["recall"][str(t)]) for t in REPORT_TAUS]
        note = " ⚠️ not rendered" if untestable else ""
        L.append(f"| {r['name']}{note} | {r['n']} | " + " | ".join(cells)
                 + f" | {fmt_pct(r['diag_acc'])} | {fmt_pct(r['control_fp']['-2.0'])} | {heard} |")
    o = summary["overall"]
    L.append(f"| **all** | {o['n']} | " + " | ".join(fmt_pct(o["recall"][str(t)]) for t in REPORT_TAUS)
             + f" | {fmt_pct(o['diag_acc'])} | {fmt_pct(o['control_fp']['-2.0'])} | |")
    if skipped:
        L += ["", f"Skipped {len(skipped)} sentences whose sentence-level espeak phonemes differ from per-word G2P:", ""]
        L += [f"- {t}: {why}" for t, why in skipped]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="use only the first N sentences")
    ap.add_argument("--per-type", type=int, default=8)
    a = ap.parse_args()
    esp = Espeak()
    grids, skipped = build_grids(esp, load_sentences(a.n))
    print(f"{len(grids)} sentences usable, {len(skipped)} skipped", flush=True)
    results = run(esp, grids, a.per_type)
    summary = summarise(results)
    summary["skipped"] = skipped
    path = write_results("synthetic_errors", summary, markdown(summary, skipped))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
