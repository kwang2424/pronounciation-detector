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


def test_vowel_length():
    f = flagged(analyse("Staat", realized_ipa="ʃtat"))
    assert f == [("Staat", "aː", "a")]
