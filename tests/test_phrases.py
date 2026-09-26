"""The practice phrases must contain the sounds they claim to train."""
import pytest

from mdd import g2p
from mdd.languages import get
from mdd.phrases import PHRASES, find, for_sound, phrases
from mdd.pipeline import analyse

ALL = [(lang, p) for lang, ps in PHRASES.items() for p in ps]


def _expected(text, lang):
    g2p.last_fallbacks.clear()
    rep = analyse(text, realized_ipa="a", lang=lang)
    return {p["canonical"] for p in rep["phones"] if p["canonical"]}


@pytest.mark.parametrize("lang,phrase", ALL, ids=[p.text for _, p in ALL])
def test_phrase_contains_its_target_sounds(lang, phrase):
    missing = set(phrase.targets) - _expected(phrase.text, lang)
    assert not missing, f"espeak does not produce {missing} in {phrase.text!r}"


@pytest.mark.parametrize("lang,phrase", ALL, ids=[p.text for _, p in ALL])
def test_phrase_needs_no_word_by_word_fallback(lang, phrase):
    """A fallback means espeak mangled the sentence (e.g. a language switch)."""
    _expected(phrase.text, lang)
    assert not g2p.last_fallbacks


@pytest.mark.parametrize("lang,phrase", ALL, ids=[p.text for _, p in ALL])
def test_linked_contrasts_exist(lang, phrase):
    ids = {c.id for c in get(lang).contrasts}
    assert set(phrase.contrasts) <= ids


def test_known_espeak_artifacts_stay_out():
    """espeak reads French 'rose' as [ʁɔz]; a reference voice would teach that."""
    texts = " ".join(p.text for p in phrases("fr"))
    assert "rose" not in texts and "jeunes" not in texts


def test_lookup_and_suggestions():
    assert find("de", "Die Tür ist grün.").targets == ("yː",)
    assert find("de", "not a phrase") is None
    assert phrases("da") == ()
    # By-ear phrases are never suggested for a sound the score is tracking.
    assert all(not p.by_ear for p in for_sound("de", "a"))
    assert find("de", "Der Staat und die Stadt.").by_ear
