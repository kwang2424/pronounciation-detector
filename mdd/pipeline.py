"""End-to-end: audio + text -> per-phone detection/diagnosis report."""
if __name__ == "__main__":
    from ._utf8 import ensure_utf8_mode
    ensure_utf8_mode()

import json  # noqa: E402
from dataclasses import asdict, dataclass

from .align import align
from .diagnose import tip_for
from .g2p import text_to_ipa_words
from .languages import LanguageProfile, get
from .normalize import tokenize

GOP_THRESHOLD = -2.0   # calibrate on native Common Voice (§7 of the design doc)


@dataclass
class PhoneResult:
    word: str | None
    canonical: str | None
    realized: str | None
    op: str
    gop: float | None
    flagged: bool
    tip: str | None


def analyse(text: str, wav_path: str | None = None, recognizer=None, realized_ipa: str | None = None,
            threshold: float = GOP_THRESHOLD,
            lang: LanguageProfile | str | None = None) -> dict:
    """If `realized_ipa` is given, skip audio (useful for tests / synthetic eval)."""
    profile = lang if isinstance(lang, LanguageProfile) else get(lang)
    words = text_to_ipa_words(text, profile)
    canon_tokens, word_of = [], []
    for w, ipa in words:
        toks = tokenize(ipa, profile)
        canon_tokens += toks
        word_of += [w] * len(toks)

    gops = None
    if realized_ipa is None:
        wav = recognizer.load_audio(wav_path)
        logp = recognizer.log_probs(wav)
        realized_ipa = recognizer.greedy_decode(logp)
        gops = [s.gop for s in recognizer.gop(logp, canon_tokens)]
    real_tokens = tokenize(realized_ipa, profile)

    results, ci = [], 0
    for op in align(canon_tokens, real_tokens):
        gop = None
        if op.op != "ins":
            gop = gops[ci] if gops else None
            word = word_of[ci]
            ci += 1
        else:
            word = word_of[min(ci, len(word_of) - 1)] if word_of else None
        disagree = op.op != "match"
        flagged = disagree and (gop is None or gop < threshold)
        results.append(PhoneResult(word, op.canonical, op.realized, op.op, gop, flagged,
                                   tip_for(op.canonical, op.realized, profile) if flagged else None))

    word_scores = {}
    for r in results:
        if r.word and r.gop is not None:
            word_scores[r.word] = min(word_scores.get(r.word, 0.0), r.gop)
    n_canon = sum(1 for r in results if r.op != "ins")
    return {
        "text": text,
        "lang": profile.code,
        "canonical": " ".join(canon_tokens),
        "realized": " ".join(real_tokens),
        "phones": [asdict(r) for r in results],
        "word_scores": word_scores,
        "overall": 1 - sum(r.flagged for r in results) / max(n_canon, 1),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="mdd")
    ap.add_argument("text")
    ap.add_argument("wav", nargs="?")
    ap.add_argument("--ipa", help="skip audio; supply realised IPA directly")
    ap.add_argument("--threshold", type=float, default=GOP_THRESHOLD)
    ap.add_argument("--lang", default=None, help="language code (de, da, ko)")
    a = ap.parse_args()
    rec = None
    if a.ipa is None:
        from .recognizer import PhoneRecognizer
        rec = PhoneRecognizer()
    rep = analyse(a.text, a.wav, rec, a.ipa, a.threshold, a.lang)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
