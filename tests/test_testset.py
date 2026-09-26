"""The personal test set: recording plan, storage, and the evaluator."""
import numpy as np
import pytest
import soundfile as sf

from eval.learner_set import evaluate, summarise, verdict, wilson
from mdd import g2p, testset
from mdd.pipeline import analyse
from mdd.testset import CONTRASTS, Take

ALL = [(lang, c, w) for lang, cs in CONTRASTS.items() for c in cs for w in c.words]


@pytest.mark.parametrize("lang,c,word", ALL, ids=[f"{lang}-{w}" for lang, _, w in ALL])
def test_every_word_has_its_target_sound_where_the_contrast_says(lang, c, word):
    g2p.last_fallbacks.clear()
    canon = [p["canonical"] for p in analyse(word, realized_ipa="a", lang=lang)["phones"]
             if p["canonical"]]
    assert not g2p.last_fallbacks
    under_test = {"first": canon[:1], "last": canon[-1:]}.get(c.position, canon)
    assert set(under_test) & set(c.targets), f"{word}: {canon}"


def test_plan_covers_every_word_both_ways_twice():
    assert len(testset.plan("de")) == 8 * 5 * 2 * testset.TAKES
    assert len(testset.plan("fr")) == 4 * 5 * 2 * testset.TAKES
    first = testset.plan("de")[:2]
    assert [t.kind for t in first] == ["best", "error"] and first[0].word == first[1].word


def _tone(path, seconds=0.6, amp=0.3, sr=16000):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (amp * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    return path


def test_save_stores_the_label_in_the_file_name(tmp_path):
    take = Take("de", "u-umlaut", "Tür", "error", 1)
    dest = testset.save(tmp_path, take, _tone(tmp_path / "in.wav"))
    assert dest.name == "Tür__error__1.wav" and dest.parent.name == "u-umlaut"
    assert testset.recorded(tmp_path, "de") == [(take, dest)]
    assert take not in testset.pending(tmp_path, "de")


def test_save_refuses_silent_or_truncated_takes(tmp_path):
    take = Take("de", "u-umlaut", "Tür", "best", 1)
    with pytest.raises(ValueError, match="silent"):
        testset.save(tmp_path, take, _tone(tmp_path / "quiet.wav", amp=0.0))
    with pytest.raises(ValueError, match="too short"):
        testset.save(tmp_path, take, _tone(tmp_path / "short.wav", seconds=0.1))
    assert testset.recorded(tmp_path, "de") == []


def test_wilson_interval_is_honest_about_small_samples():
    lo, hi = wilson(15, 20)
    assert 0.5 < lo < 0.55 and 0.88 < hi < 0.92
    assert wilson(0, 0) == (0.0, 1.0)


class Segment:
    """What analyse() reads from mdd.recognizer.Segment (which needs torch)."""

    def __init__(self, gop):
        self.gop = gop


class FakeRecognizer:
    """Hears whatever `heard` maps each file to; every mismatch is confidently bad."""

    def __init__(self, heard):
        self.heard = heard

    def load_audio(self, path):
        return path

    def log_probs(self, wav):
        return wav

    def greedy_spikes(self, path):
        from pathlib import Path
        return [(tok, 0.99) for tok in self.heard[Path(path).name].split()]

    def gop(self, logp, canonical):
        return [Segment(-5.0) for _ in canonical]


def _record(root, lang, contrast, word, heard_best, heard_error, takes=(1, 2)):
    heard = {}
    for n in takes:
        for kind, ipa in (("best", heard_best), ("error", heard_error)):
            t = Take(lang, contrast, word, kind, n)
            testset.save(root, t, _tone(root / "tmp.wav"))
            heard[t.filename] = ipa
    return heard


def test_evaluator_catches_a_clear_error_and_names_it(tmp_path):
    heard = _record(tmp_path, "de", "u-umlaut", "Tür", "t yː ʁ", "t u ʁ")
    data, md = evaluate(tmp_path, "de", FakeRecognizer(heard))
    s = data["contrasts"]["u-umlaut"]
    assert (s["caught"], s["n_error"], s["false_alarms"], s["diagnosed"]) == (2, 2, 0, 2)
    assert s["verdict"] == "works for your voice"
    assert "| ü | 2/2" in md and "`u` ×2" in md


def test_length_is_reported_both_as_the_app_scores_it_and_with_length_flags(tmp_path):
    """The app never flags length alone; the report must show it hears it anyway."""
    heard = _record(tmp_path, "de", "vowel-length", "Staat", "ʃ t aː t", "ʃ t a t")
    data, md = evaluate(tmp_path, "de", FakeRecognizer(heard))
    s = data["contrasts"]["vowel-length"]
    assert s["caught"] == 0 and s["differ"] == 2
    assert s["with_length_flags"]["caught"] == 2
    assert "hears the difference but rarely flags" in s["verdict"]
    assert "with length flags on" in md


def test_only_the_final_consonant_counts_for_devoicing(tmp_path):
    """'Tag' starts with t too; a flag there says nothing about devoicing."""
    heard = _record(tmp_path, "de", "final-devoicing", "Tag", "d aː k", "t aː ɡ")
    s = evaluate(tmp_path, "de", FakeRecognizer(heard))[0]["contrasts"]["final-devoicing"]
    assert s["false_alarms"] == 0 and s["caught"] == 2 and s["diagnosed"] == 2


def test_an_inserted_n_after_a_nasal_vowel_counts_as_catching_it(tmp_path):
    heard = _record(tmp_path, "fr", "nasal", "bon", "b ɔ̃", "b ɔ̃ n")
    s = evaluate(tmp_path, "fr", FakeRecognizer(heard))[0]["contrasts"]["nasal"]
    assert s["caught"] == 2 and s["false_alarms"] == 0


def test_verdict_when_the_two_versions_sound_the_same():
    rows = [{"word": "Tür", "take": 1, "kind": k, "flagged": False, "flagged_with_length": False,
             "diagnosed": False, "heard": "yː"} for k in ("best", "error")]
    assert "no reliable difference" in verdict(summarise(rows))


# -------------------------------------------------------------------- the app
def test_the_tab_walks_through_the_plan_and_can_undo(tmp_path):
    import app

    take, card, progress, _ = app.ts_show("German", [])
    assert take == testset.plan("de")[0] and "as well as you can" in card
    assert "0 of 160 recorded" in progress

    take2, card, progress, _ = app.ts_save("German", take, str(_tone(tmp_path / "a.wav")), [])
    assert take2.kind == "error" and "English 'oo'" in card and "1 of 160" in progress

    skipped, take3, *_ = app.ts_skip("German", take2, [])
    assert take3 != take2 and skipped == [take2]

    redo, card, progress, _ = app.ts_undo("German", skipped)
    assert redo == take and "(redo)" in card and "0 of 160" in progress


def test_saving_silence_is_refused_in_the_tab(tmp_path):
    import app

    take, *_ = app.ts_show("German", [])
    with pytest.raises(Exception, match="silent"):
        app.ts_save("German", take, str(_tone(tmp_path / "q.wav", amp=0.0)), [])
    with pytest.raises(Exception, match="Record the take first"):
        app.ts_save("German", take, None, [])
