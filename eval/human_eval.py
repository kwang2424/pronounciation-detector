"""Tier 3 (design doc section 7.3): real human German L1/L2 benchmark & manifest CLI.

Supports three subcommands:
  python -m eval.human_eval init --out manifest.jsonl
  python -m eval.human_eval import-dir recordings/ --out manifest.jsonl
  python -m eval.human_eval score manifest.jsonl [--out eval/results/human_eval]
"""
from __future__ import annotations

if __name__ == "__main__":
    from mdd._utf8 import ensure_utf8_mode

    ensure_utf8_mode()

import argparse  # noqa: E402
import json  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

from mdd.pipeline import GOP_THRESHOLD, analyse  # noqa: E402
from mdd.recorded import AUDIO_SUFFIXES  # noqa: E402

from .common import CACHE, DEFAULT_LANG, RESULTS, THRESHOLDS, analyse_cached, fmt_pct, is_flagged, key, load_sentences  # noqa: E402
from .metrics import MDDCounts, compute_metrics, recommend_threshold  # noqa: E402

_PRIORITY_MARKERS = (
    "Staat", "Stadt", "Hütte", "König", "Pferd", "Tür", "Löffel", "Küche", "Buch",
    "Zeit", "Straße", "Hund", "Tag", "halb", "Fahrrad", "Brücke", "Österreich",
    "Schlüssel", "Kühlschrank", "zwölf", "Frühling", "Mädchen", "Fluss", "Witz", "Fuß",
)


def load_manifest(path: Path | str) -> list[dict]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        return list(data) if isinstance(data, list) else list(data.get("entries", []))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_manifest(entries: list[dict], path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        lines = [json.dumps(e, ensure_ascii=False) for e in entries]
        p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return p


def _match_canonical_errors(
    canon_positions: list[tuple[int, dict]],
    canon_errors: list[dict],
) -> tuple[dict[int, dict], list[dict]]:
    matched: dict[int, dict] = {}
    unmatched: list[dict] = []
    for err in canon_errors:
        target_canon = err.get("canonical")
        target_widx = err.get("word_idx")
        target_word = (err.get("word") or "").casefold()
        occ = int(err.get("occurrence", 0))

        pool = []
        for k, p in canon_positions:
            if p.get("canonical") != target_canon:
                continue
            if target_widx is not None and p.get("word_idx") is not None:
                if p.get("word_idx") != target_widx:
                    continue
            elif target_word and (p.get("word") or "").casefold() != target_word:
                continue
            pool.append(k)

        avail = [k for k in pool if k not in matched]
        chosen = None
        if 0 <= occ < len(pool) and pool[occ] not in matched:
            chosen = pool[occ]
        elif avail:
            chosen = avail[0]

        if chosen is not None:
            matched[chosen] = err
        else:
            unmatched.append(err)
    return matched, unmatched


def _match_insertion_errors(
    ins_positions: list[tuple[int, dict]],
    ins_errors: list[dict],
) -> tuple[dict[int, dict], list[dict]]:
    matched: dict[int, dict] = {}
    unmatched: list[dict] = []
    for err in ins_errors:
        target_widx = err.get("word_idx")
        target_word = (err.get("word") or "").casefold()
        target_real = err.get("realized")

        pool = []
        for k, p in ins_positions:
            if k in matched:
                continue
            if target_widx is not None and p.get("word_idx") is not None:
                if p.get("word_idx") != target_widx:
                    continue
            elif target_word and (p.get("word") or "").casefold() != target_word:
                continue
            pool.append((k, p))

        exact = [k for k, p in pool if target_real is None or p.get("realized") == target_real]
        if exact:
            matched[exact[0]] = err
        elif pool:
            matched[pool[0][0]] = err
        else:
            unmatched.append(err)
    return matched, unmatched


def score_entry(entry: dict, rep: dict, thresholds=THRESHOLDS) -> dict:
    """Compute exact phoneme-level MDDCounts for one utterance across thresholds."""
    phones = rep.get("phones", [])
    canon_positions = [(k, p) for k, p in enumerate(phones) if p.get("op") != "ins"]
    ins_positions = [(k, p) for k, p in enumerate(phones) if p.get("op") == "ins"]

    errors = entry.get("errors") or []
    canon_errors = [e for e in errors if e.get("op") != "ins" and e.get("canonical") is not None]
    ins_errors = [e for e in errors if e.get("op") == "ins" or e.get("canonical") is None]

    matched_canon, unmatched_canon = _match_canonical_errors(canon_positions, canon_errors)
    matched_ins, unmatched_ins = _match_insertion_errors(ins_positions, ins_errors)
    unmatched_total = len(unmatched_canon) + len(unmatched_ins)

    counts_by_tau: dict[str, MDDCounts] = {}
    for tau in thresholds:
        ta = fr = tr = fa = cd = de = 0
        for k, p in canon_positions:
            flagged = is_flagged(p, tau)
            if k in matched_canon:
                err = matched_canon[k]
                exp_real = err.get("realized")
                is_del = err.get("op") == "del" or exp_real is None
                if flagged:
                    tr += 1
                    diag_ok = (p["op"] == "del") if is_del else (p["op"] == "sub" and p.get("realized") == exp_real)
                    if diag_ok:
                        cd += 1
                    else:
                        de += 1
                else:
                    fa += 1
            else:
                if flagged:
                    fr += 1
                else:
                    ta += 1

        for k, p in ins_positions:
            flagged = is_flagged(p, tau)
            if k in matched_ins:
                err = matched_ins[k]
                exp_real = err.get("realized")
                if flagged:
                    tr += 1
                    if exp_real is None or p.get("realized") == exp_real:
                        cd += 1
                    else:
                        de += 1
                else:
                    fa += 1
            elif flagged:
                fr += 1

        fa += unmatched_total
        counts_by_tau[str(tau)] = MDDCounts(ta=ta, fr=fr, tr=tr, fa=fa, cd=cd, de=de)

    # Per-phone FRR and per-error details at default GOP_THRESHOLD
    default_tau = GOP_THRESHOLD
    phone_occ: Counter[str] = Counter()
    phone_fr: Counter[str] = Counter()
    error_outcomes: list[tuple[str, bool, bool]] = []
    for k, p in canon_positions:
        canon = p.get("canonical") or "?"
        if k in matched_canon:
            err = matched_canon[k]
            exp_real = err.get("realized")
            is_del = err.get("op") == "del" or exp_real is None
            label = f"{canon}→{'∅' if is_del else exp_real}"
            flagged = is_flagged(p, default_tau)
            diag_ok = flagged and ((p["op"] == "del") if is_del else (p["op"] == "sub" and p.get("realized") == exp_real))
            error_outcomes.append((label, flagged, diag_ok))
        else:
            phone_occ[canon] += 1
            if is_flagged(p, default_tau):
                phone_fr[canon] += 1
    for k, p in ins_positions:
        if k in matched_ins:
            err = matched_ins[k]
            label = f"∅→{err.get('realized') or '?'}"
            flagged = is_flagged(p, default_tau)
            diag_ok = flagged and (err.get("realized") is None or p.get("realized") == err.get("realized"))
            error_outcomes.append((label, flagged, diag_ok))
        elif is_flagged(p, default_tau):
            phone_fr["(insertion)"] += 1
    for err in unmatched_canon + unmatched_ins:
        label = f"{err.get('canonical') or '∅'}→{err.get('realized') or '∅'}"
        error_outcomes.append((label, False, False))

    return {
        "counts_by_tau": counts_by_tau,
        "phone_occ": phone_occ,
        "phone_fr": phone_fr,
        "error_outcomes": error_outcomes,
    }


def evaluate_manifest(
    entries: list[dict],
    *,
    lang: str = DEFAULT_LANG,
    manifest_dir: Path | None = None,
    dry_run: bool = False,
    thresholds=THRESHOLDS,
) -> dict:
    """Evaluate a human L1/L2 manifest end-to-end and return summary metrics."""
    overall_counts = {str(t): MDDCounts() for t in thresholds}
    by_type_counts: dict[str, dict[str, MDDCounts]] = defaultdict(
        lambda: {str(t): MDDCounts() for t in thresholds}
    )
    utt_by_type: Counter[str] = Counter()
    total_occ: Counter[str] = Counter()
    total_fr: Counter[str] = Counter()
    err_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # n, tr, cd

    for idx, entry in enumerate(entries):
        text = entry["text"]
        stype = entry.get("speaker_type") or "l2"
        utt_by_type[stype] += 1

        if entry.get("realized_ipa") is not None:
            rep = analyse(text, realized_ipa=entry["realized_ipa"], lang=lang)
        elif dry_run:
            raise ValueError(f"Entry {entry.get('id', idx)!r} has no realized_ipa for --dry-run")
        else:
            audio = entry.get("audio")
            if not audio:
                raise ValueError(f"Entry {entry.get('id', idx)!r} is missing 'audio'")
            wav_path = Path(audio)
            if not wav_path.is_absolute() and manifest_dir is not None:
                wav_path = manifest_dir / wav_path
            cache_json = CACHE / "human" / f"{key(lang, str(wav_path), text)}.json"
            rep = analyse_cached(text, wav_path, cache_json, lang=lang)

        scored = score_entry(entry, rep, thresholds=thresholds)
        for tk, c in scored["counts_by_tau"].items():
            overall_counts[tk] = overall_counts[tk] + c
            by_type_counts[stype][tk] = by_type_counts[stype][tk] + c
        total_occ.update(scored["phone_occ"])
        total_fr.update(scored["phone_fr"])
        for label, det, diag in scored["error_outcomes"]:
            err_stats[label][0] += 1
            err_stats[label][1] += int(det)
            err_stats[label][2] += int(diag)

    metrics_by_tau = {tk: compute_metrics(c) for tk, c in overall_counts.items()}
    by_speaker_type = {
        stype: {
            "n_utterances": utt_by_type[stype],
            "metrics_by_tau": {tk: compute_metrics(c) for tk, c in tcounts.items()},
        }
        for stype, tcounts in sorted(by_type_counts.items())
    }
    per_phone_frr = [
        {
            "canonical": k,
            "false_rejects": total_fr[k],
            "occurrences": total_occ[k],
            "frr": total_fr[k] / total_occ[k] if total_occ[k] else None,
        }
        for k in sorted(set(total_occ) | set(total_fr), key=lambda p: (-total_fr[p], -total_occ[p], p))
    ]
    per_error = [
        {
            "error": label,
            "n": n,
            "tr": tr,
            "recall": tr / n if n else None,
            "cd": cd,
            "cond_diag_acc": cd / tr if tr else None,
            "der": 1.0 - (cd / tr) if tr else None,
        }
        for label, (n, tr, cd) in sorted(err_stats.items(), key=lambda kv: (-kv[1][0], kv[0]))
    ]
    return {
        "lang": lang,
        "n_utterances": len(entries),
        "thresholds": list(thresholds),
        "default_tau": GOP_THRESHOLD,
        "recommended_tau": recommend_threshold(metrics_by_tau),
        "metrics_by_tau": metrics_by_tau,
        "by_speaker_type": by_speaker_type,
        "per_phone_frr": per_phone_frr,
        "per_error": per_error,
    }


def markdown(summary: dict) -> str:
    L = [
        "# Human German Benchmark (Tier 3)",
        "",
        f"Utterances: {summary['n_utterances']} | Language: `{summary['lang']}`",
        "",
        "## Overall MDD metrics by threshold",
        "",
        "| τ | precision | recall | F1 | MCC | FAR | FRR | cond_diag | DER |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tau in summary["thresholds"]:
        m = summary["metrics_by_tau"].get(str(tau))
        if not m:
            continue
        mcc = "n/a" if m["mcc"] is None else f"{m['mcc']:.3f}"
        L.append(
            f"| {tau:g} | {fmt_pct(m['precision'])} | {fmt_pct(m['recall'])} | {fmt_pct(m['f1'])} | "
            f"{mcc} | {fmt_pct(m['far'])} | {fmt_pct(m['frr'])} | "
            f"{fmt_pct(m['cond_diag_acc'])} | {fmt_pct(m['der'])} |"
        )
    rec = summary.get("recommended_tau")
    L += ["", f"Recommended τ (FRR < 5%): **{rec:g}**" if rec is not None else "Recommended τ (FRR < 5%): **none**"]

    by_st = summary.get("by_speaker_type") or {}
    if by_st:
        dtau = str(float(summary.get("default_tau", GOP_THRESHOLD)))
        L += ["", f"## Split by speaker_type (at τ = {float(dtau):g})", "",
              "| speaker_type | utterances | precision | recall | F1 | FRR | DER |",
              "|---|---|---|---|---|---|---|"]
        for stype, data in by_st.items():
            m = data["metrics_by_tau"].get(dtau) or {}
            L.append(
                f"| {stype} | {data['n_utterances']} | {fmt_pct(m.get('precision'))} | "
                f"{fmt_pct(m.get('recall'))} | {fmt_pct(m.get('f1'))} | "
                f"{fmt_pct(m.get('frr'))} | {fmt_pct(m.get('der'))} |"
            )

    fr_rows = [r for r in summary.get("per_phone_frr") or [] if r["false_rejects"] > 0][:20]
    if fr_rows:
        L += ["", f"## Most false-rejected phones (at τ = {summary.get('default_tau', GOP_THRESHOLD):g})", "",
              "| canonical | false_rejects | occurrences | FRR |", "|---|---|---|---|"]
        for r in fr_rows:
            L.append(f"| {r['canonical']} | {r['false_rejects']} | {r['occurrences']} | {fmt_pct(r['frr'])} |")

    err_rows = summary.get("per_error") or []
    if err_rows:
        L += ["", f"## Annotated errors (at τ = {summary.get('default_tau', GOP_THRESHOLD):g})", "",
              "| error | n | detected | recall | cond_diag | DER |", "|---|---|---|---|---|---|"]
        for r in err_rows:
            L.append(
                f"| {r['error']} | {r['n']} | {r['tr']} | {fmt_pct(r['recall'])} | "
                f"{fmt_pct(r['cond_diag_acc'])} | {fmt_pct(r['der'])} |"
            )
    return "\n".join(L) + "\n"


def init_manifest(
    out_path: Path | str,
    *,
    lang: str = DEFAULT_LANG,
    n: int = 25,
    speaker: str = "learner1",
    speaker_type: str = "l2",
) -> list[dict]:
    sentences = load_sentences(lang=lang)
    prioritized: list[str] = []
    rest: list[str] = []
    for s in sentences:
        if any(tok in s for tok in _PRIORITY_MARKERS):
            prioritized.append(s)
        else:
            rest.append(s)
    chosen = (prioritized + rest)[:n]
    entries = [
        {
            "id": f"{lang}-{i:03d}",
            "audio": f"clips/{lang}-{i:03d}.wav",
            "text": sent,
            "speaker": speaker,
            "speaker_type": speaker_type,
            "errors": [],
        }
        for i, sent in enumerate(chosen, 1)
    ]
    write_manifest(entries, out_path)
    return entries


def import_dir(
    audio_dir: Path | str,
    out_path: Path | str,
    *,
    speaker: str = "speaker1",
    speaker_type: str = "l1",
) -> list[dict]:
    adir = Path(audio_dir)
    out = Path(out_path)
    entries = []
    for path in sorted(adir.rglob("*")):
        if path.suffix.lower() not in AUDIO_SUFFIXES or not path.is_file():
            continue
        txt_path = path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = " ".join(txt_path.read_text(encoding="utf-8").split())
        if not text:
            continue
        try:
            audio_ref = path.resolve().relative_to(out.parent.resolve()).as_posix()
        except ValueError:
            audio_ref = str(path)
        entries.append({
            "id": path.stem,
            "audio": audio_ref,
            "text": text,
            "speaker": speaker,
            "speaker_type": speaker_type,
            "errors": [],
        })
    write_manifest(entries, out)
    return entries


def main():
    ap = argparse.ArgumentParser(prog="eval.human_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create a starter recording manifest")
    p_init.add_argument("--out", type=Path, required=True)
    p_init.add_argument("--lang", default=DEFAULT_LANG)
    p_init.add_argument("--n", type=int, default=25)
    p_init.add_argument("--speaker", default="learner1")
    p_init.add_argument("--speaker-type", default="l2")

    p_imp = sub.add_parser("import-dir", help="Build manifest from paired audio + .txt files")
    p_imp.add_argument("audio_dir", type=Path)
    p_imp.add_argument("--out", type=Path, required=True)
    p_imp.add_argument("--speaker", default="speaker1")
    p_imp.add_argument("--speaker-type", default="l1")

    p_score = sub.add_parser("score", help="Score a human manifest and write JSON + MD reports")
    p_score.add_argument("manifest", type=Path)
    p_score.add_argument("--lang", default=DEFAULT_LANG)
    p_score.add_argument("--out", type=Path, default=RESULTS / "human_eval")
    p_score.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    if args.cmd == "init":
        entries = init_manifest(args.out, lang=args.lang, n=args.n, speaker=args.speaker, speaker_type=args.speaker_type)
        print(f"Wrote {len(entries)} starter items to {args.out}")
    elif args.cmd == "import-dir":
        entries = import_dir(args.audio_dir, args.out, speaker=args.speaker, speaker_type=args.speaker_type)
        print(f"Imported {len(entries)} paired recordings into {args.out}")
    else:
        entries = load_manifest(args.manifest)
        summary = evaluate_manifest(entries, lang=args.lang, manifest_dir=args.manifest.parent, dry_run=args.dry_run)
        out_base = args.out
        out_base.parent.mkdir(parents=True, exist_ok=True)
        out_base.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path = out_base.with_suffix(".md")
        md_path.write_text(markdown(summary), encoding="utf-8")
        print(f"Wrote {md_path}")
