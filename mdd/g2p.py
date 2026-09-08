"""Reference text -> canonical IPA via espeak-ng, for any profiled language.

Phonemised a sentence at a time, not word by word, because some languages apply
sandhi across word boundaries and phonemising each word alone silently loses it.
French liaison is the case that forced this: per-word, `les amis` comes back as
`le ami`, while a speaker says `lezami`. The recogniser then hears a /z/ the
canonical does not contain, and the pipeline reports an inserted phone — liaison
was the single largest false-positive category in the French native control
(z x30, t x19). Sentence-level phonemisation gives `lez|ami` and the flag
disappears. It also avoids espeak reading a lone letter as its name: `Il y a`
per-word gives `iɡʁɛk` ("i grec") where a sentence gives `i`.

Word alignment still has to survive, since every report is per word. espeak
occasionally splits or merges words (contractions especially), so when the
sentence-level word count does not match the orthographic one, this falls back to
per-word phonemisation for that text and records it in `last_fallbacks`.

German is unaffected: across the 102 evaluation sentences both routes agree
token for token, with no word-count mismatches.
"""
import unicodedata

from phonemizer import phonemize
from phonemizer.separator import Separator

from .languages import DEFAULT

_SEP = Separator(phone="", word="|", syllable="")

#: Texts whose sentence-level phonemisation could not be aligned to their words,
#: and so fell back to per-word. Diagnostic only; reset on each call.
last_fallbacks: list[str] = []


def _is_punctuation(token: str) -> bool:
    return all(unicodedata.category(ch).startswith("P") or ch.isspace() for ch in token)


def _phonemize(texts: list[str], lang: str) -> list[str]:
    return phonemize(
        texts,
        language=lang,
        backend="espeak",
        separator=_SEP,
        strip=True,
        with_stress=False,
        preserve_punctuation=False,
        njobs=1,
    )


def text_to_ipa_words(text: str, lang=None) -> list[tuple[str, str]]:
    """Return [(orthographic_word, ipa_string), ...] keeping word alignment.

    `lang` accepts an espeak code or a LanguageProfile.
    """
    lang = getattr(lang, "phonemizer_language", None) or lang or DEFAULT
    # French typography spaces punctuation off ("soir ?"), so a bare split yields
    # tokens espeak produces nothing for — a spurious word-count mismatch that
    # would send the whole sentence down the per-word path and lose its liaison.
    words = [w for w in text.replace("\n", " ").split(" ")
             if w and not _is_punctuation(w)]
    if not words:
        return []

    ipa = [w for w in _phonemize([text], lang)[0].split("|") if w.strip()]
    if len(ipa) != len(words):
        # espeak split or merged something; per-word keeps the alignment honest
        # at the cost of any cross-word sandhi in this one text.
        last_fallbacks.append(text)
        ipa = [p.replace("|", "").strip() for p in _phonemize(words, lang)]

    out = []
    for w, p in zip(words, ipa):
        p = p.replace("|", "").strip()
        clean = w.strip(".,;:!?\"'()")
        if p:
            out.append((clean, p))
    return out
