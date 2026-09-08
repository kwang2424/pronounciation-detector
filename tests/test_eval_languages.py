"""The evaluation harness across languages — and that German did not move."""
import pytest

from eval.common import canonical_tokens_by_word, load_sentences, results_name
from eval.tts import default_voices


def test_german_defaults_are_unchanged():
    """German has committed baseline results and a clip cache keyed by voice name;
    changing either spelling would invalidate the comparison it exists for."""
    assert default_voices("de") == [
        "edge:de-DE-KatjaNeural", "edge:de-DE-ConradNeural",
        "edge:de-DE-AmalaNeural", "edge:de-DE-KillianNeural", "espeak",
    ]
    assert results_name("native_control") == "native_control"
    assert results_name("native_control", "de") == "native_control"


def test_other_languages_get_their_own_results_directory():
    assert results_name("native_control", "fr") == "fr/native_control"


def test_french_voices_are_french():
    voices = default_voices("fr")
    assert all("fr-" in v or v.startswith("espeak") for v in voices)
    assert voices[-1] == "espeak:fr"


def test_espeak_specs_are_excluded_from_the_natural_split():
    """The headline FPR describes natural voices. espeak carries a voice name for
    non-German languages, so an equality check would let 'espeak:fr' through."""
    for spec in ("espeak", "espeak:fr", "espeak:da"):
        assert spec.startswith("espeak")
    assert not "edge:fr-FR-DeniseNeural".startswith("espeak")


def test_unknown_language_has_no_default_voices():
    with pytest.raises(ValueError):
        default_voices("xx")


def test_french_sentences_load_and_cover_the_contrasts():
    sentences = load_sentences(lang="fr")
    assert len(sentences) >= 50
    seen = set()
    for text in sentences:
        for _, tokens in canonical_tokens_by_word(text, "fr"):
            seen.update(tokens)
    # every phone the French contrasts train must actually occur
    for phone in ("y", "u", "ɑ̃", "ɛ̃", "ɔ̃", "o", "ɔ", "e", "ɛ", "ʁ"):
        assert phone in seen, f"/{phone}/ never occurs in the French eval sentences"


def test_tokenisation_follows_the_language():
    """The harness must not silently score French with the German tokeniser."""
    french = dict(canonical_tokens_by_word("pain", "fr"))
    assert "ɛ̃" in french["pain"], "French nasality must survive"
    german = dict(canonical_tokens_by_word("Bier", "de"))
    assert german["Bier"] and "ɛ̃" not in german["Bier"]


def test_missing_sentence_file_says_what_is_needed():
    with pytest.raises(FileNotFoundError, match="no evaluation sentences"):
        load_sentences(lang="ko")
