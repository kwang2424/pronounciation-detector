"""Reference text -> canonical IPA via espeak-ng, for any profiled language."""
import os
import shutil

# No system espeak-ng and no explicit library path: fall back to the DLL/.so
# bundled in the `espeakng-loader` pip package (no admin install needed).
if "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ and shutil.which("espeak-ng") is None:
    try:
        import espeakng_loader
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeakng_loader.get_library_path()
        os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
    except ImportError:
        pass

from phonemizer import phonemize  # noqa: E402
from phonemizer.separator import Separator  # noqa: E402

from .languages import DEFAULT  # noqa: E402

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
