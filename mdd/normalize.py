"""Map espeak-flavoured IPA (from G2P or the CTC recognizer) into one token set.

Tokens are strings. Long vowels keep the length mark attached ("aː" is one
token). Affricates are single tokens. Stress, glottal stops and tie bars
are removed. Regional r variants collapse to "ʁ".
"""
import re
import unicodedata

AFFRICATES = ("pf", "ts", "tʃ", "dʒ")
DIPHTHONGS = ("aɪ", "aʊ", "ɔʏ", "ɔɪ", "ɔy", "ɔø")
MULTI = DIPHTHONGS + AFFRICATES
R_VARIANTS = {"r", "ʀ", "ɾ", "ʁ"}          # all accepted as canonical German r
STRIP = {"ˈ", "ˌ", "ʔ", "͡", "‿", ".", " ", "|", "̯", "̃"}
SYLLABIC = "̩"

# recogniser sometimes emits these for the same phone
EQUIV = {
    "ɐ̯": "ɐ", "ɛ̃": "ɛ", "ɑ": "a", "ɑː": "aː", "ɹ": "ɹ",  # keep English r distinct!
    "ɔø": "ɔʏ", "ɔy": "ɔʏ", "ɔɪ": "ɔʏ",   # espeak writes eu/äu as ɔø
    "g": "ɡ",                              # ASCII g -> IPA ɡ (U+0261), which espeak and panphon use
    # inventory mismatches found by the native-control eval: the recogniser never emits ʏ or ɛː
    "ʏ": "y",                              # short ü: model says y for espeak's ʏ (100% false flags otherwise)
    "ɛː": "eː",                            # long ä: merged with eː by the model (and by most speakers)
}


def tokenize(ipa: str) -> list[str]:
    s = unicodedata.normalize("NFD", ipa)
    s = "".join(ch for ch in s if ch not in STRIP)
    toks: list[str] = []
    i = 0
    while i < len(s):
        # affricate?
        hit = next((a for a in MULTI if s.startswith(a, i)), None)
        if hit:
            tok, i = hit, i + len(hit)
        else:
            tok, i = s[i], i + 1
        # absorb length mark / syllabic mark / combining marks
        while i < len(s) and (s[i] == "ː" or s[i] == SYLLABIC or unicodedata.combining(s[i])):
            tok += s[i]
            i += 1
        toks.append(tok)
    # normalisation passes
    out = []
    for t in toks:
        t = unicodedata.normalize("NFC", t)
        if t in R_VARIANTS:
            t = "ʁ"
        t = EQUIV.get(t, t)
        # syllabic n/m/l <-> ən/əm/əl : represent as schwa + consonant
        if t.endswith(SYLLABIC):
            out.append("ə")
            t = t[:-1]
        out.append(t)
    return out
