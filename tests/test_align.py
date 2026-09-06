from mdd.pipeline import analyse


def flagged(rep):
    return [(p["word"], p["canonical"], p["realized"]) for p in rep["phones"] if p["flagged"]]


def test_perfect_utterance_has_no_flags():
    rep = analyse("Ich möchte ein Bier", realized_ipa="ɪç mœçtə aɪn biːɾ")
    assert flagged(rep) == []


def test_umlaut_and_ich_laut_errors_are_substitutions():
    # learner says "ich mochte" with /k/ for ch and /o/ for ö
    rep = analyse("Ich möchte", realized_ipa="ɪk mɔktə")
    f = flagged(rep)
    assert ("Ich", "ç", "k") in f
    assert ("möchte", "œ", "ɔ") in f
    assert ("möchte", "ç", "k") in f
    assert all(p["op"] == "sub" for p in rep["phones"] if p["flagged"])


def test_english_r_is_flagged_but_regional_r_is_not():
    assert flagged(analyse("rot", realized_ipa="ʀoːt")) == []
    assert ("rot", "ʁ", "ɹ") in flagged(analyse("rot", realized_ipa="ɹoːt"))


def test_vowel_length_is_off_by_default_but_available():
    assert flagged(analyse("Staat", realized_ipa="ʃtat")) == []
    assert flagged(analyse("Staat", realized_ipa="ʃtat", flag_length=True)) == [("Staat", "aː", "a")]


def test_short_ue_and_long_ae_use_the_recogniser_inventory():
    assert flagged(analyse("Hütte", realized_ipa="hytə")) == []          # model writes ʏ as y
    assert flagged(analyse("Käse", realized_ipa="keːzə")) == []          # model writes ɛː as eː
    assert ("Hütte", "y", "ʊ") in flagged(analyse("Hütte", realized_ipa="hʊtə"))


def test_eu_diphthong_is_one_token_and_gets_its_tip():
    f = analyse("neu", realized_ipa="nuː")
    bad = [p for p in f["phones"] if p["flagged"]]
    assert [(p["canonical"], p["realized"]) for p in bad] == [("ɔʏ", "uː")]
    assert "oy" in bad[0]["tip"]


def test_final_devoicing_tip_matches_ipa_g():
    f = analyse("Tag", realized_ipa="taːg")
    bad = [p for p in f["phones"] if p["flagged"]]
    assert [(p["canonical"], p["realized"]) for p in bad] == [("k", "ɡ")]
    assert "devoicing" in bad[0]["tip"]


def test_coda_r_vocalised_or_dropped_is_native():
    assert flagged(analyse("Bier", realized_ipa="biːɐ")) == []
    assert flagged(analyse("Bier", realized_ipa="biː")) == []
    assert flagged(analyse("Garten", realized_ipa="ɡaatən")) == []


def test_onset_r_and_english_r_still_flagged():
    assert ("rot", "ʁ", None) in flagged(analyse("rot", realized_ipa="oːt"))
    assert ("Bier", "ʁ", "ɹ") in flagged(analyse("Bier", realized_ipa="biːɹ"))
