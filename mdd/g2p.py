"""Reference text -> canonical IPA via espeak-ng, for any profiled language."""
from phonemizer import phonemize
from phonemizer.separator import Separator

from .languages import DEFAULT

_SEP = Separator(phone="", word="|", syllable="")


def text_to_ipa_words(text: str, lang=None) -> list[tuple[str, str]]:
    """Return [(orthographic_word, ipa_string), ...] keeping word alignment.

    `lang` accepts an espeak code or a LanguageProfile.
    """
    lang = getattr(lang, "code", None) or lang or DEFAULT
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
