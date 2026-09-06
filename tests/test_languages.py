"""Profile loading, and the language-specific normalisation each one needs."""
import pytest

from mdd.diagnose import tip_for
from mdd.languages import PROFILES, get
from mdd.normalize import tokenize
from mdd.pipeline import analyse


def test_every_profile_loads_and_has_contrasts():
    for code, profile in PROFILES.items():
        assert profile.code == code
        assert profile.contrasts, f"{code} has no contrasts"
        assert profile.tips, f"{code} has no tips"
        for contrast in profile.contrasts:
            assert profile.contrast(contrast.id) is contrast
            for group in contrast.pairs:
                assert len(group) >= 2, f"{code}/{contrast.id} has a one-word 'minimal set'"


def test_danish_keeps_stod_but_german_strips_glottal_stop():
    # espeak spells Danish stod with a glottal stop; it is phonemic there.
    assert "ʔ" in tokenize("hʔun", "da")
    # German's glottal stop is automatic before initial vowels and carries nothing.
    assert "ʔ" not in tokenize("ʔapfəl", "de")


def test_danish_greek_epsilon_is_folded_to_ipa():
    # espeak's Danish voice emits U+03B5 GREEK SMALL LETTER EPSILON, which
    # panphon cannot score. It must normalise to U+025B LATIN SMALL LETTER OPEN E.
    assert tokenize("ʋεn", "da") == tokenize("ʋɛn", "da")
    assert "ε" not in tokenize("ʋεn", "da")


def test_danish_r_offglide_is_one_token():
    assert tokenize("mʔoɐ̯", "da") == ["m", "ʔ", "o", "ɐ̯"]


def test_espeak_question_mark_artifact_is_dropped():
    # espeak's Danish voice emits a literal '?' for some aspirated stops.
    assert tokenize("t?ɑk", "da") == ["t", "a", "k"]


def test_multi_character_tokens_prefer_the_longest_match():
    assert tokenize("tɕhɐ", "ko")[0] == "tɕh"


def test_danish_soft_d_substitution_is_diagnosed():
    rep = analyse("mad", realized_ipa="mad", lang="da")
    flagged = [(p["canonical"], p["realized"]) for p in rep["phones"] if p["flagged"]]
    assert ("ð", "d") in flagged
    assert "Blødt d" in tip_for("ð", "d", "da")


def test_dropped_stod_gets_a_targeted_tip():
    rep = analyse("hund", realized_ipa="hun", lang="da")
    tips = [p["tip"] for p in rep["phones"] if p["flagged"]]
    assert any("Stød" in t for t in tips)


def test_language_is_reported_and_defaults_to_german():
    assert analyse("Ich", realized_ipa="ɪç")["lang"] == "de"
    assert analyse("mad", realized_ipa="mað", lang="da")["lang"] == "da"


def test_unknown_language_is_rejected():
    with pytest.raises(KeyError):
        get("xx")
