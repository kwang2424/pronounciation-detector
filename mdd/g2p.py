"""Reference text -> canonical IPA via espeak-ng (Standard German)."""
from phonemizer import phonemize
from phonemizer.separator import Separator

_SEP = Separator(phone="", word="|", syllable="")


def text_to_ipa_words(text: str, lang: str = "de") -> list[tuple[str, str]]:
    """Return [(orthographic_word, ipa_string), ...] keeping word alignment."""
    words = [w for w in text.replace("\n", " ").split(" ") if w]
    ipa = phonemize(
        words,
        language=lang,
        backend="espeak",
        separator=_SEP,
        strip=True,
        with_stress=False,
        preserve_punctuation=False,
        njobs=1,
    )
    out = []
    for w, p in zip(words, ipa):
        p = p.replace("|", "").strip()
        clean = w.strip(".,;:!?\"'()")
        if p:
            out.append((clean, p))
    return out
