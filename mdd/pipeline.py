"""End-to-end: audio + text -> per-phone detection/diagnosis report."""
if __name__ == "__main__":
    from ._utf8 import ensure_utf8_mode
    ensure_utf8_mode()

import json  # noqa: E402
from dataclasses import asdict, dataclass

from .acoustics import measure_vowel
from .align import align
from .audio import AudioQualityError, preprocess_recording, reject_message
from .diagnose import tip_for
from .g2p import text_to_ipa_words
from .languages import LanguageProfile, get
from .normalize import tokenize, tokenize_with_conf

GOP_THRESHOLD = -1.0   # native TTS voices give 2.6% phone FPR here (4.3% with the gate off at 0);
                       # kept slightly conservative for noisier microphone audio. See eval/results/.
VERSION = 6            # bump when alignment/flagging rules change (invalidates eval caches)
FLAG_LENGTH_ONLY = False   # aː vs a etc.: 37% false-positive rate on native speech, so off by default
INS_MIN_PROB = 0.6     # an inserted phone counts only if the recogniser was this sure of it (spurious
                       # insertions on native speech have median confidence 0.38, real phones 0.95)
LONG_VOWEL_MIN_MS = 85.0
LONG_VOWEL_MIN_RATIO = 1.20
SHORT_VOWEL_MAX_MS = 155.0
SHORT_VOWEL_MAX_RATIO = 2.55
_SHORT_LENGTH_TARGETS = frozenset({"a", "y"})
_LONG_LENGTH_TARGETS = frozenset({"aː", "yː"})

# Coda r: espeak's G2P writes it as a consonant (German Bier -> biːʁ), but speakers vocalise it
# to [ɐ] or merge it into the preceding vowel. Which realisations count as native is per-language
# (`profile.coda_r_ok`); onset r stays strict everywhere.
_VOWEL_START = set("aeiouyæøœɐɑɒɔəɘɛɜɪʊʏɯʌ")


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
    word_idx: int | None = None
    acoustics: dict | None = None


def _token_confidences(spikes: list[tuple[str, float]], real_tokens: list[str],
                       profile: LanguageProfile) -> list[float] | None:
    """Carry each spike's confidence over to the normalised token list. Returns None if the
    per-spike tokenisation cannot be reconciled with tokenize() of the joined string."""
    toks, conf = tokenize_with_conf(spikes, profile)
    return conf if toks == real_tokens else None


def _length_only(a: str | None, b: str | None) -> bool:
    return bool(a and b) and a != b and a.rstrip("ː") == b.rstrip("ː")


def analyse(text: str, wav_path: str | None = None, recognizer=None, realized_ipa: str | None = None,
            threshold: float = GOP_THRESHOLD, flag_length: bool = FLAG_LENGTH_ONLY,
            lang: LanguageProfile | str | None = None,
            check_audio: bool = True, check_vowel_duration: bool = True) -> dict:
    """If `realized_ipa` is given, skip audio (useful for tests / synthetic eval)."""
    profile = lang if isinstance(lang, LanguageProfile) else get(lang)
    words = text_to_ipa_words(text, profile)
    canon_tokens, word_of, word_idx_of, coda_of, is_word_final = [], [], [], [], []
    for wi, (w, ipa) in enumerate(words):
        toks = tokenize(ipa, profile)
        canon_tokens += toks
        word_of += [w] * len(toks)
        word_idx_of += [wi] * len(toks)
        coda_of += _coda_flags(toks)
        is_word_final += [j == len(toks) - 1 for j in range(len(toks))]

    from_audio = realized_ipa is None
    wav = None
    segs = []
    gops, confs, audio_meta = None, None, None
    if from_audio:
        wav = recognizer.load_audio(wav_path)
        if check_audio:
            audio_check = preprocess_recording(wav, sr=16000)
            if not audio_check.ok:
                raise AudioQualityError(
                    audio_check.reject_reason,
                    reject_message(audio_check.reject_reason, audio_check.snr_db),
                )
            wav = audio_check.wav
            audio_meta = {
                "duration_s": round(audio_check.duration_s, 3),
                "voiced_s": round(audio_check.voiced_s, 3),
                "snr_db": round(audio_check.snr_db, 1) if audio_check.snr_db is not None else None,
                "clipped_frac": round(audio_check.clipped_frac, 4),
                "warnings": list(audio_check.warnings),
            }
        logp = recognizer.log_probs(wav)
        spikes = recognizer.greedy_spikes(logp)
        realized_ipa = "".join(tok for tok, _ in spikes)
        segs = recognizer.gop(logp, canon_tokens)
        gops = [s.gop for s in segs] if len(segs) == len(canon_tokens) else None
    real_tokens = tokenize(realized_ipa, profile)
    if gops is not None:
        confs = _token_confidences(spikes, real_tokens, profile)

    results, ci, ri = [], 0, 0
    for op in align(canon_tokens, real_tokens):
        gop, conf, kind = None, None, op.op
        cur_ci = None
        if op.op != "del":
            conf = confs[ri] if confs else None
            ri += 1
        if op.op != "ins":
            cur_ci = ci
            gop = gops[ci] if gops else None
            word = word_of[ci]
            word_idx = word_idx_of[ci]
            if (op.canonical == profile.r_canonical and coda_of[ci]
                    and op.realized in profile.coda_r_ok):
                kind = "match"
            if op.realized is not None and op.realized in profile.allow.get(op.canonical, ()):
                kind = "match"
            if (profile.allow_final_ig and op.canonical == "ç" and op.realized == "k"
                    and is_word_final[ci] and ci > 0
                    and canon_tokens[ci - 1] == "ɪ"
                    and word_idx_of[ci - 1] == word_idx_of[ci]
                    and word.casefold().endswith("ig")):
                kind = "match"
            if not flag_length and _length_only(op.canonical, op.realized):
                kind = "match"
            ci += 1
        else:
            if word_of:
                prev_ci = max(ci - 1, 0)
                word = word_of[prev_ci]
                word_idx = word_idx_of[prev_ci]
            else:
                word, word_idx = None, None
        if kind == "ins":
            flagged = conf is None or conf >= INS_MIN_PROB
        else:
            flagged = kind != "match" and (gop is None or gop < threshold)

        realized_out = op.realized
        tip = tip_for(op.canonical, realized_out, profile) if flagged else None
        phone_acoustics = None
        if (from_audio and wav is not None and gops is not None and cur_ci is not None
                and check_vowel_duration and _is_vowel(op.canonical)):
            vm = measure_vowel(wav, 16000, segs, cur_ci)
            if vm is not None:
                phone_acoustics = {
                    "duration_ms": vm.duration_ms,
                    "duration_ratio": vm.duration_ratio,
                    "f1_hz": vm.f1_hz,
                    "f2_hz": vm.f2_hz,
                }
                if (profile.code == "de" and vm.bounded and not flagged
                        and (kind == "match" or _length_only(op.canonical, op.realized))):
                    has_coda_r = (
                        cur_ci + 1 < len(canon_tokens)
                        and canon_tokens[cur_ci + 1] == profile.r_canonical
                        and coda_of[cur_ci + 1]
                    )
                    if (op.canonical in _LONG_LENGTH_TARGETS and not has_coda_r
                            and vm.duration_ms < LONG_VOWEL_MIN_MS
                            and (vm.duration_ratio is None or vm.duration_ratio < LONG_VOWEL_MIN_RATIO)):
                        kind = "sub"
                        realized_out = op.canonical.rstrip("ː")
                        flagged = True
                        phone_acoustics["duration_flagged"] = True
                        if op.canonical == "yː":
                            tip = (f"Vowel [yː] was too short ({vm.duration_ms:.0f} ms) — "
                                   f"hold it longer while keeping your tongue forward as for /i/ with rounded lips.")
                        else:
                            tip = (f"Vowel [{op.canonical}] was too short ({vm.duration_ms:.0f} ms) — "
                                   f"hold it about twice as long.")
                    elif (op.canonical in _SHORT_LENGTH_TARGETS and cur_ci < len(canon_tokens) - 1
                          and vm.duration_ms > SHORT_VOWEL_MAX_MS
                          and vm.duration_ratio is not None and vm.duration_ratio > SHORT_VOWEL_MAX_RATIO):
                        kind = "sub"
                        realized_out = f"{op.canonical}ː"
                        flagged = True
                        phone_acoustics["duration_flagged"] = True
                        tip = (f"Vowel [{op.canonical}] was held too long ({vm.duration_ms:.0f} ms) — "
                               f"keep it crisp and short.")

        results.append(PhoneResult(word, op.canonical, realized_out, kind, gop, flagged,
                                   tip,
                                   conf if kind == "ins" else None,
                                   word_idx,
                                   phone_acoustics))

    word_scores = {}
    word_min_gop: dict[int, float] = {}
    for r in results:
        if r.word and r.gop is not None:
            word_scores[r.word] = min(word_scores.get(r.word, 0.0), r.gop)
        if r.word_idx is not None and r.gop is not None:
            word_min_gop[r.word_idx] = min(word_min_gop.get(r.word_idx, 0.0), r.gop)
    word_scores_by_index = [
        {"index": wi, "word": w, "min_gop": word_min_gop[wi]}
        for wi, (w, _) in enumerate(words)
        if wi in word_min_gop
    ]
    n_canon = sum(1 for r in results if r.op != "ins")
    report = {
        "text": text,
        "lang": profile.code,
        "canonical": " ".join(canon_tokens),
        "realized": " ".join(real_tokens),
        "words": [w for w, _ in words],
        "phones": [asdict(r) for r in results],
        "word_scores": word_scores,
        "word_scores_by_index": word_scores_by_index,
        "overall": max(0.0, 1 - sum(r.flagged for r in results) / max(n_canon, 1)),
    }
    if audio_meta is not None:
        report["audio"] = audio_meta
    return report


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="mdd")
    ap.add_argument("text")
    ap.add_argument("wav", nargs="?")
    ap.add_argument("--ipa", help="skip audio; supply realised IPA directly")
    ap.add_argument("--threshold", type=float, default=GOP_THRESHOLD)
    ap.add_argument("--flag-length", action="store_true", help="also flag vowel-length-only errors")
    ap.add_argument("--lang", default=None, help="language code (de, da, ko)")
    a = ap.parse_args()
    rec = None
    if a.ipa is None:
        from .recognizer import PhoneRecognizer
        rec = PhoneRecognizer()
    rep = analyse(a.text, a.wav, rec, a.ipa, a.threshold, a.flag_length, a.lang)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
