"""Catalogue of learner errors to inject (design doc section 6), expressed on the pipeline's
canonical token inventory. `realized` is one IPA token, a space-separated sequence
(insertion after substitution), or None (deletion). `context(word_tokens, idx)` restricts
positions."""
from dataclasses import dataclass
from typing import Callable

Ctx = Callable[[list[str], int], bool]


_VOWEL_START = set("aeiouyæøœɐɑɒɔəɘɛɜɪʊʏɯʌ")


def word_final(w, i):
    return i == len(w) - 1


def before_t_or_p(w, i):
    return i + 1 < len(w) and w[i + 1] in ("t", "p")


def before_vowel(w, i):
    return i + 1 < len(w) and w[i + 1][:1] in _VOWEL_START


def not_final_ig(w, i):
    return not (len(w) > 3 and i == len(w) - 1 and w[i - 1] == "ɪ")


@dataclass(frozen=True)
class ErrorType:
    name: str
    canonical: str
    realized: str | None
    context: Ctx | None = None
    note: str = ""
    also_ok: tuple[str, ...] = ()   # other realisations counted as a correct diagnosis

    def ok(self, word: list[str], i: int) -> bool:
        return word[i] == self.canonical and (self.context is None or self.context(word, i))

    @property
    def kind(self) -> str:
        if self.realized is None:
            return "del"
        return "ins" if " " in self.realized else "sub"


CATALOG: list[ErrorType] = [
    ErrorType("ü_long→uː",       "yː", "uː"),
    ErrorType("ü_short→ʊ",       "y",  "ʊ"),
    ErrorType("ö_long→oː",       "øː", "oː"),
    ErrorType("ö_short→ɔ",       "œ",  "ɔ"),
    ErrorType("ich→k",           "ç",  "k", not_final_ig),
    ErrorType("ich→sch",         "ç",  "ʃ"),
    ErrorType("ach→k",           "x",  "k"),
    ErrorType("ach→h",           "x",  "h"),
    ErrorType("z→voiced_z",      "ts", "z"),
    ErrorType("w→english_w",     "v",  "w", note="[w] is borrowed from espeak's base inventory"),
    ErrorType("v→voiced",        "f",  "v"),
    ErrorType("st_initial→s",    "ʃ",  "s",  before_t_or_p),
    ErrorType("final_t→d",       "t",  "d",  word_final),
    ErrorType("final_k→ɡ",       "k",  "ɡ",  word_final),
    ErrorType("final_p→b",       "p",  "b",  word_final),
    ErrorType("long_a→short",    "aː", "a"),
    ErrorType("schwa→eː",        "ə",  "eː", word_final),
    ErrorType("schwa_dropped",   "ə",  None, word_final),
    ErrorType("ei→iː",           "aɪ", "iː"),
    ErrorType("eu→uː",           "ɔʏ", "uː"),
    ErrorType("ng→ng+ɡ",         "ŋ",  "ŋ ɡ", also_ok=("k",), note="word-final ɡ is heard devoiced"),
    ErrorType("ü_long→iː",       "yː", "iː"),
    ErrorType("ü_short→ɪ",       "y",  "ɪ"),
    ErrorType("ö_long→eː",       "øː", "eː"),
    ErrorType("ö_short→ɛ",       "œ",  "ɛ"),
    ErrorType("i_long→ɪ",        "iː", "ɪ"),
    ErrorType("u_long→ʊ",        "uː", "ʊ"),
    ErrorType("o_long→ɔ",        "oː", "ɔ"),
    ErrorType("e_long→ɛ",        "eː", "ɛ"),
    ErrorType("short_a→long",    "a",  "aː"),
    ErrorType("ich→ach",         "ç",  "x"),
    ErrorType("ach→ich",         "x",  "ç"),
    ErrorType("r_onset→english_r", "ʁ", "ɹ", before_vowel, note="[ɹ] via espeak base approximant r/"),
    ErrorType("final_s→z",       "s",  "z",  word_final),
]

# IPA token -> espeak German mnemonic. Seeded from probing espeak; extended at runtime
# from whatever the sentence list yields.
MNEMONIC: dict[str, str] = {
    "ç": "C", "x": "x", "k": "k", "ɡ": "g", "ʃ": "S", "s": "s", "z": "z", "ts": "ts", "tʃ": "tS", "pf": "pF",
    "œ": "W", "øː": "Y:", "yː": "y:", "y": "y", "uː": "u:", "ʊ": "U", "oː": "o:", "ɔ": "O",
    "ʁ": "r", "ɹ": "r/", "v": "v", "f": "f", "w": "w", "h": "h", "t": "t", "d": "d", "p": "p", "b": "b",
    "ə": "@", "aɪ": "aI", "aʊ": "aU", "ɔʏ": "OY", "ŋ": "N", "n": "n", "m": "m", "l": "l", "j": "j",
    "aː": "A:", "a": "a", "eː": "e:", "ɛ": "E", "iː": "i:", "ɪ": "I", "ɜ": "3",
}
