"""End-to-end: audio + text -> per-phone detection/diagnosis report."""
if __name__ == "__main__":
    from ._utf8 import ensure_utf8_mode
    ensure_utf8_mode()

import json  # noqa: E402
from dataclasses import asdict, dataclass

from .align import align
from .diagnose import tip_for
from .g2p import text_to_ipa_words
from .normalize import MULTI, tokenize

GOP_THRESHOLD = -1.0   # native TTS voices give 2.6% phone FPR here (4.3% with the gate off at 0);
                       # kept slightly conservative for noisier microphone audio. See eval/results/.
VERSION = 3            # bump when alignment/flagging rules change (invalidates eval caches)
FLAG_LENGTH_ONLY = False   # aː vs a etc.: 37% false-positive rate on native speech, so off by default
INS_MIN_PROB = 0.6     # an inserted phone counts only if the recogniser was this sure of it (spurious
                       # insertions on native speech have median confidence 0.38, real phones 0.95)

# espeak's G2P writes coda r as consonantal ʁ (Bier -> biːʁ), but native speakers vocalise it
# to [ɐ] or merge it into the preceding vowel. Accept those realisations as a match; only
# English [ɹ] or a wrong consonant remains an error. Onset r (rot, Brot) stays strict.
_VOWEL_START = set("aeiouyæøœɐɑɒɔəɘɛɜɪʊʏ")
_CODA_R_OK = {None, "ɐ", "ɜ", "ə", "a"}


def _is_vowel(tok: str) -> bool:
    return tok[:1] in _VOWEL_START


def _coda_flags(toks: list[str]) -> list[bool]:
    return [i == len(toks) - 1 or not _is_vowel(toks[i + 1]) for i in range(len(toks))]


@dataclass
class PhoneResult:
    word: str | None
    canonical: str | None
    realized: str | None
    op: str
    gop: float | None
    flagged: bool
    tip: str | None
    conf: float | None = None   # recogniser confidence in the realised phone (insertions only)


def _token_confidences(spikes: list[tuple[str, float]], real_tokens: list[str]) -> list[float] | None:
    """Carry each spike's confidence over to the normalised token list. Returns None if the
    per-spike tokenisation cannot be reconciled with tokenize() of the joined string."""
    toks, conf = [], []
    for tok, c in spikes:
        for sub in tokenize(tok):
            toks.append(sub)
            conf.append(c)
    i = 0
    while i < len(toks) - 1:          # tokenize() merges e.g. a+ɪ -> aɪ across spikes
        if toks[i] + toks[i + 1] in MULTI:
            toks[i:i + 2] = [toks[i] + toks[i + 1]]
            conf[i:i + 2] = [min(conf[i], conf[i + 1])]
        else:
            i += 1
    return conf if toks == real_tokens else None


def _length_only(a: str | None, b: str | None) -> bool:
    return bool(a and b) and a != b and a.rstrip("ː") == b.rstrip("ː")


def analyse(text: str, wav_path: str | None = None, recognizer=None, realized_ipa: str | None = None,
            threshold: float = GOP_THRESHOLD, flag_length: bool = FLAG_LENGTH_ONLY) -> dict:
    """If `realized_ipa` is given, skip audio (useful for tests / synthetic eval)."""
    words = text_to_ipa_words(text)
    canon_tokens, word_of, coda_of = [], [], []
    for w, ipa in words:
        toks = tokenize(ipa)
        canon_tokens += toks
        word_of += [w] * len(toks)
        coda_of += _coda_flags(toks)

    gops, confs = None, None
    if realized_ipa is None:
        wav = recognizer.load_audio(wav_path)
        logp = recognizer.log_probs(wav)
        spikes = recognizer.greedy_spikes(logp)
        realized_ipa = "".join(tok for tok, _ in spikes)
        gops = [s.gop for s in recognizer.gop(logp, canon_tokens)]
    real_tokens = tokenize(realized_ipa)
    if gops is not None:
        confs = _token_confidences(spikes, real_tokens)

    results, ci, ri = [], 0, 0
    for op in align(canon_tokens, real_tokens):
        gop, conf, kind = None, None, op.op
        if op.op != "del":
            conf = confs[ri] if confs else None
            ri += 1
        if op.op != "ins":
            gop = gops[ci] if gops else None
            word = word_of[ci]
            if op.canonical == "ʁ" and coda_of[ci] and op.realized in _CODA_R_OK:
                kind = "match"
            if not flag_length and _length_only(op.canonical, op.realized):
                kind = "match"
            ci += 1
        else:
            word = word_of[min(ci, len(word_of) - 1)] if word_of else None
        if kind == "ins":
            flagged = conf is None or conf >= INS_MIN_PROB
        else:
            flagged = kind != "match" and (gop is None or gop < threshold)
        results.append(PhoneResult(word, op.canonical, op.realized, kind, gop, flagged,
                                   tip_for(op.canonical, op.realized) if flagged else None,
                                   conf if kind == "ins" else None))

    word_scores = {}
    for r in results:
        if r.word and r.gop is not None:
            word_scores[r.word] = min(word_scores.get(r.word, 0.0), r.gop)
    n_canon = sum(1 for r in results if r.op != "ins")
    return {
        "text": text,
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
    a = ap.parse_args()
    rec = None
    if a.ipa is None:
        from .recognizer import PhoneRecognizer
        rec = PhoneRecognizer()
    rep = analyse(a.text, a.wav, rec, a.ipa, a.threshold)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
