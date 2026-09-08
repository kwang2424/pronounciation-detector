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


# ------------------------------------------------------------------- French
def test_french_nasal_vowels_survive_tokenisation():
    """The combining tilde used to live in BASE_STRIP, inherited from German where
    ɛ̃ is only a recogniser artifact. That silently erased every French nasal
    vowel — paix and pain both came out /pɛ/, collapsing the contrast."""
    assert tokenize("pɛ̃", "fr") == ["p", "ɛ̃"]
    assert tokenize("pɛ", "fr") == ["p", "ɛ"]
    assert tokenize("pɛ̃", "fr") != tokenize("pɛ", "fr")
    assert tokenize("sɑ̃", "fr") == ["s", "ɑ̃"]


def test_german_still_folds_the_nasal_artifact():
    """German has no nasal vowels; the recogniser's ɛ̃ is noise and must fold."""
    assert tokenize("ɛ̃", "de") == ["ɛ"]


def test_french_uses_different_names_for_g2p_and_synthesis():
    """espeak's phonemiser rejects "fr" and its synthesiser rejects "fr-fr"."""
    french = get("fr")
    assert french.phonemizer_language == "fr-fr"
    assert french.synth_voice == "fr"
    german = get("de")
    assert german.phonemizer_language == german.synth_voice == "de"


def test_language_switch_markers_are_stripped():
    """espeak flags a mid-utterance switch inline: French 'dos' comes back as
    '(en)dɒs(fr)', which would otherwise tokenise into per-letter garbage."""
    assert tokenize("(en)dɒs(fr)", "fr") == ["d", "ɔ", "s"]
    # oʊ is not a German diphthong, so it splits — the point here is only that
    # no bracket or language code survives into the token stream.
    assert tokenize("(en)hɛloʊ(de)", "de") == ["h", "ɛ", "l", "o", "ʊ"]


def test_french_nasal_vowels_are_single_tokens():
    for nasal in ("ɑ̃", "ɛ̃", "ɔ̃"):
        assert tokenize(nasal, "fr") == [nasal]


def test_french_diagnosis_names_the_error():
    rep = analyse("tu", realized_ipa="tu", lang="fr")
    flagged = [(p["canonical"], p["realized"], p["tip"]) for p in rep["phones"] if p["flagged"]]
    assert flagged and flagged[0][:2] == ("y", "u")
    assert "round your lips" in flagged[0][2]


def test_french_nasal_omission_is_diagnosed():
    rep = analyse("pain", realized_ipa="pɛn", lang="fr")
    tips = [p["tip"] for p in rep["phones"] if p["flagged"]]
    assert any("nasal vowel" in t for t in tips)
