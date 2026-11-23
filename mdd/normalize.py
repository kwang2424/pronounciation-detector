"""Map espeak-flavoured IPA (from G2P or the CTC recognizer) into one token set.

Tokens are strings. Long vowels keep the length mark attached ("aː" is one
token). Affricates and diphthongs are single tokens. Stress and tie bars are
removed. Which marks survive is language-specific: German strips the glottal
stop (predictable before initial vowels), Danish keeps it because that is how
espeak spells stød, which is phonemic there.
"""
import re
import unicodedata

from .languages import LanguageProfile, get

#: espeak marks a mid-utterance language switch inline, e.g. "dos" in French
#: comes back as "(en)dɒs(fr)". Parentheses never occur in IPA, so stripping the
#: markers is safe, and leaving them in would tokenise into per-letter garbage.
_LANG_SWITCH = re.compile(r"\([a-z]{2}(?:-[a-z]{2,})?\)")

SYLLABIC = "̩"


def tokenize_with_conf(
    spikes: list[tuple[str, float]],
    profile: LanguageProfile | str | None = None,
) -> tuple[list[str], list[float]]:
    """Tokenise `(text, confidence)` spikes while propagating per-token confidence."""
    if not isinstance(profile, LanguageProfile):
        profile = get(profile)
    strip = profile.strip_set()
    multi = sorted((unicodedata.normalize("NFD", a) for a in profile.multi), key=len, reverse=True)

    raw_chars: list[str] = []
    raw_confs: list[float] = []
    for tok, c in spikes:
        for ch in tok:
            raw_chars.append(ch)
            raw_confs.append(c)

    joined = "".join(raw_chars)
    drop = [False] * len(raw_chars)
    for m in _LANG_SWITCH.finditer(joined):
        for idx in range(m.start(), m.end()):
            drop[idx] = True

    nfd_chars: list[str] = []
    nfd_confs: list[float] = []
    for ch, c, is_dropped in zip(raw_chars, raw_confs, drop):
        if is_dropped:
            continue
        for dec in unicodedata.normalize("NFD", ch):
            if dec not in strip:
                nfd_chars.append(dec)
                nfd_confs.append(c)

    s = "".join(nfd_chars)
    toks: list[str] = []
    tok_confs: list[float] = []
    i = 0
    while i < len(s):
        start = i
        hit = next((a for a in multi if s.startswith(a, i)), None)
        if hit:
            tok, i = hit, i + len(hit)
        else:
            tok, i = s[i], i + 1
        # absorb length mark / syllabic mark / other combining marks
        while i < len(s) and (s[i] == "ː" or s[i] == SYLLABIC or unicodedata.combining(s[i])):
            tok += s[i]
            i += 1
        toks.append(tok)
        tok_confs.append(min(nfd_confs[start:i]))

    out: list[str] = []
    out_confs: list[float] = []
    for t, c in zip(toks, tok_confs):
        t = unicodedata.normalize("NFC", t)
        if t in profile.r_variants:
            t = profile.r_canonical
        t = profile.equiv.get(t, t)
        # syllabic n/m/l <-> ən/əm/əl : represent as schwa + consonant
        if t.endswith(SYLLABIC):
            out.append("ə")
            out_confs.append(c)
            t = t[:-1]
        if t:
            out.append(t)
            out_confs.append(c)
    return out, out_confs


def tokenize(ipa: str, profile: LanguageProfile | str | None = None) -> list[str]:
    return tokenize_with_conf([(ipa, 1.0)], profile)[0]
