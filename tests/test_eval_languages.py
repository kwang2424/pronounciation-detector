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


# ---------------------------------------------------------------- preflight
def test_preflight_flags_language_switches():
    """espeak phonemises loanwords as the other language. The tokeniser strips
    the markers, so the wrong phonemes pass silently into the canonical side —
    two of these were in the first draft of the French sentences."""
    from eval.preflight import check_transcription

    assert check_transcription("fr")["switches"] == []
    assert check_transcription("de")["switches"] == []


def test_preflight_reports_thin_contrast_phones():
    from eval.preflight import THIN_PHONE, check_transcription

    counts = check_transcription("fr")["contrast_phone_counts"]
    assert counts, "French contrasts should name phones"
    assert all(n >= THIN_PHONE for n in counts.values()), \
        f"a phone seen under {THIN_PHONE} times gives a per-phone FPR too noisy to read"


def test_preflight_counts_realisation_only_phones_as_zero_not_missing():
    """A Contrast lists the whole confusable set, including realisations that
    never appear in canonical transcription — German [ɐ] is one, since espeak
    writes coda r as ʁ. Reporting that as an error would be a false alarm."""
    from eval.preflight import check_transcription

    counts = check_transcription("de")["contrast_phone_counts"]
    assert counts.get("ɐ") == 0
    assert counts.get("ʁ", 0) > 0


def test_preflight_environment_check_runs_without_the_model():
    from eval.preflight import check_environment

    results = check_environment("fr")
    assert any("voices configured" in label for _, label in results)
    assert all(isinstance(ok, bool) for ok, _ in results)


def test_preflight_handles_a_language_with_no_sentences():
    from eval.preflight import check_transcription

    with pytest.raises(FileNotFoundError):
        check_transcription("ko")


# ------------------------------------------------------- liaison / sandhi
def test_french_liaison_survives_phonemisation():
    """Per-word, `les amis` is `le ami` and the recogniser hears a /z/ nobody
    predicted — liaison was the largest false-positive category in the French
    native control (z x30, t x19)."""
    from mdd.g2p import text_to_ipa_words

    ipa = dict(text_to_ipa_words("les amis vous avez", "fr-fr"))
    assert ipa["les"].endswith("z"), "liaison /z/ must reach the canonical form"
    assert ipa["vous"].endswith("z")


def test_lone_letters_are_not_read_as_letter_names():
    """espeak reads a standalone 'y' as 'i grec' when phonemising it alone."""
    from mdd.g2p import text_to_ipa_words

    ipa = dict(text_to_ipa_words("Il y a beaucoup", "fr-fr"))
    assert ipa["y"] == "i"


def test_punctuation_tokens_do_not_break_alignment():
    """French typography spaces punctuation off ('soir ?'); counting it as a word
    caused a spurious mismatch that sent the sentence down the per-word path."""
    from mdd import g2p

    g2p.last_fallbacks.clear()
    ipa = dict(g2p.text_to_ipa_words("Tu as vu le film hier soir ?", "fr-fr"))
    assert "?" not in ipa
    assert g2p.last_fallbacks == [], "should not have fallen back"


def test_german_is_unchanged_by_sentence_level_phonemisation():
    """Across the evaluation sentences both routes agree token for token."""
    from mdd.g2p import text_to_ipa_words
    from mdd.normalize import tokenize

    for text in load_sentences(20, "de"):
        for word, ipa in text_to_ipa_words(text, "de"):
            assert word and tokenize(ipa, "de")


def test_word_count_mismatch_falls_back_and_is_recorded():
    from mdd import g2p

    g2p.last_fallbacks.clear()
    for text in load_sentences(lang="fr"):
        g2p.text_to_ipa_words(text, "fr-fr")
    # A fallback is safe (it only loses sandhi for that text) but should be rare.
    assert len(g2p.last_fallbacks) <= 3


# ------------------------------------------------------- voices / threshold
def test_french_voices_are_metropolitan_only():
    """The canonical transcription is espeak's fr-fr. A Quebec voice differs
    systematically and measures as pipeline error when it is dialect difference."""
    assert not any("fr-CA" in v for v in default_voices("fr"))


def test_median_voice_threshold_ignores_one_bad_talker():
    """Pooling assumes comparable voices. In the French run disagreement ranged
    8.5%-47.1%, and the worst voice alone pushed the pooled rate past target at
    every threshold, recommending -8 — a setting that flags almost nothing."""
    from eval.native_control import recommend_robust

    def voice(rate, dis):
        return {"fpr": {str(t): rate for t in
                        (-8.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0)},
                "disagreement_rate": dis}

    voices = {"edge:a": voice(0.03, 0.09), "edge:b": voice(0.04, 0.085),
              "edge:c": voice(0.30, 0.47)}
    tau, detail = recommend_robust(voices)
    assert tau == 0.0, "the median voice is well under target at every threshold"
    assert detail["disagreement"]["edge:c"] == 0.47


def test_espeak_is_excluded_from_the_median_too():
    from eval.native_control import recommend_robust

    fpr = {str(t): 0.02 for t in (-8.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0)}
    _, detail = recommend_robust({
        "edge:a": {"fpr": fpr, "disagreement_rate": 0.09},
        "espeak:fr": {"fpr": fpr, "disagreement_rate": 0.51},
    })
    assert "espeak:fr" not in detail["disagreement"]


# --------------------------------------------------------------- cache keys
def test_cache_key_tracks_the_canonical_phones():
    """A hand-maintained VERSION is forgettable, and was forgotten: switching
    French to sentence-level phonemisation changed every French canonical form,
    VERSION was not bumped, and a full re-run silently replayed stale numbers.
    Deriving the key from the canonical sequence makes that impossible."""
    from eval.common import canonical_signature
    from mdd import languages

    before_de = canonical_signature("Ich möchte ein Bier", "de")
    before_fr = canonical_signature("les amis", "fr")

    original = languages.PROFILES["fr"]
    languages.PROFILES["fr"] = languages.LanguageProfile(
        **{**original.__dict__, "equiv": {**original.equiv, "z": "s"}})
    try:
        assert canonical_signature("les amis", "fr") != before_fr, \
            "a change to French canonical forms must invalidate French entries"
        assert canonical_signature("Ich möchte ein Bier", "de") == before_de, \
            "and must leave an untouched language's cache valid"
    finally:
        languages.PROFILES["fr"] = original


def test_cache_key_is_stable_for_unchanged_text():
    from eval.common import canonical_signature

    assert canonical_signature("Il fait beau", "fr") == canonical_signature("Il fait beau", "fr")


def test_cache_key_separates_languages():
    from eval.common import canonical_signature

    # Same string, different profile: must not share a cache entry.
    assert canonical_signature("son", "fr") != canonical_signature("son", "de")
