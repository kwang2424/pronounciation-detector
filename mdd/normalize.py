"""Map espeak-flavoured IPA (from G2P or the CTC recognizer) into one token set.

Tokens are strings. Long vowels keep the length mark attached ("aː" is one
token). Affricates and diphthongs are single tokens. Stress and tie bars are
removed. Which marks survive is language-specific: German strips the glottal
stop (predictable before initial vowels), Danish keeps it because that is how
espeak spells stød, which is phonemic there.
"""
import unicodedata

from .languages import LanguageProfile, get

SYLLABIC = "̩"


def tokenize(ipa: str, profile: LanguageProfile | str | None = None) -> list[str]:
    if not isinstance(profile, LanguageProfile):
        profile = get(profile)
    strip = profile.strip_set()
    # Longest-first so "ɑw" wins over "ɑ", and "tɕh" over "tɕ".
    multi = sorted(profile.multi, key=len, reverse=True)

    s = unicodedata.normalize("NFD", ipa)
    s = "".join(ch for ch in s if ch not in strip)
    toks: list[str] = []
    i = 0
    while i < len(s):
        hit = next((a for a in multi if s.startswith(unicodedata.normalize("NFD", a), i)), None)
        if hit:
            n = len(unicodedata.normalize("NFD", hit))
            tok, i = s[i:i + n], i + n
        else:
            tok, i = s[i], i + 1
        # absorb length mark / syllabic mark / other combining marks
        while i < len(s) and (s[i] == "ː" or s[i] == SYLLABIC or unicodedata.combining(s[i])):
            tok += s[i]
            i += 1
        toks.append(tok)

    out: list[str] = []
    for t in toks:
        t = unicodedata.normalize("NFC", t)
        if t in profile.r_variants:
            t = profile.r_canonical
        t = profile.equiv.get(t, t)
        # syllabic n/m/l <-> ən/əm/əl : represent as schwa + consonant
        if t.endswith(SYLLABIC):
            out.append("ə")
            t = t[:-1]
        if t:
            out.append(t)
    return out
