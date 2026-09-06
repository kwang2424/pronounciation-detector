"""Perception training: trial construction, staircase, and honest gating."""
import pytest

from mdd import synth
from mdd.hvpt import MAX_CHOICES, MIN_CHOICES, PerceptionUnavailable, Session
from mdd.languages import get
from mdd.validate import check_contrast

needs_audio = pytest.mark.skipif(not synth.available(), reason="espeak-ng unavailable")


def _answer_all(session, n, correctly=True):
    for _ in range(n):
        trial = session.next_trial()
        pick = trial.target if correctly else next(c for c in trial.choices if c != trial.target)
        session.record(trial, pick)


def test_danish_stod_is_excluded_because_espeak_cannot_render_it():
    # espeak gives hun and hund the same transcription, so a stod trial would be
    # unanswerable and would score the learner as failing a real contrast.
    session = Session("da", seed=1)
    assert "stod" not in session.contrast_ids
    assert "stod" in session.skipped
    assert "soft-d" in session.contrast_ids


def test_stod_pairs_are_transcription_identical():
    profile = get("da")
    report = check_contrast(profile.contrast("stod"), profile, audio=False)
    assert not report.usable
    assert all(not c.transcription_distinct for c in report.checks)


def test_korean_perception_is_refused_with_a_reason():
    with pytest.raises(PerceptionUnavailable) as exc:
        Session("ko")
    assert "Korean" in str(exc.value)


def test_german_contrasts_all_survive_validation():
    session = Session("de", seed=1)
    assert session.skipped == {}
    assert set(session.contrast_ids) == {c.id for c in get("de").contrasts}


def test_trial_target_is_among_choices_and_scored_by_index_or_word():
    session = Session("de", seed=2)
    trial = session.next_trial()
    assert trial.target in trial.choices
    assert trial.is_correct(trial.target)
    assert trial.is_correct(trial.answer_index)
    assert not trial.is_correct(next(c for c in trial.choices if c != trial.target))


def test_staircase_widens_after_two_correct_and_narrows_after_one_wrong():
    session = Session("da", seed=3)
    assert session.difficulty == MIN_CHOICES
    _answer_all(session, 8, correctly=True)
    assert session.difficulty == MAX_CHOICES
    _answer_all(session, 1, correctly=False)
    assert session.difficulty == MAX_CHOICES - 1


def test_difficulty_is_clamped_at_both_ends():
    session = Session("da", seed=4)
    _answer_all(session, 30, correctly=True)
    assert session.difficulty == MAX_CHOICES
    _answer_all(session, 30, correctly=False)
    assert session.difficulty == MIN_CHOICES


def test_talker_never_repeats_on_consecutive_trials():
    session = Session("de", seed=5)
    talkers = [session.next_trial().talker for _ in range(25)]
    assert all(a != b for a, b in zip(talkers, talkers[1:]))


def test_practice_is_steered_toward_the_weakest_contrast():
    session = Session("da", seed=6)
    ids = session.contrast_ids
    for cid in ids:                                  # every contrast seen once, all wrong
        trial = session.next_trial(cid)
        session.record(trial, next(c for c in trial.choices if c != trial.target))
    strong, weak = ids[0], ids[1:]
    for _ in range(10):
        trial = session.next_trial(strong)
        session.record(trial, trial.target)
    picked = {session.next_trial().contrast_id for _ in range(10)}
    assert strong not in picked
    assert picked <= set(weak)


def test_report_records_accuracy_and_the_worst_confusion():
    session = Session("da", seed=7)
    cid = "soft-d"
    for _ in range(6):
        trial = session.next_trial(cid)
        session.record(trial, next(c for c in trial.choices if c != trial.target))
    report = session.report()
    assert report["lang"] == "da"
    assert report["accuracy"] == 0.0
    assert report["contrasts"][cid]["seen"] == 6
    assert report["contrasts"][cid]["worst_confusion"]["count"] >= 1
    assert "stod" in report["skipped"]


@needs_audio
def test_trial_renders_audio():
    session = Session("da", seed=8)
    rate, samples = session.next_trial().audio()
    assert rate > 0 and len(samples) > 1000


@needs_audio
def test_different_talkers_produce_different_renderings():
    a = synth.synthesize("hund", "da", synth.TALKERS[0])[1]
    b = synth.synthesize("hund", "da", synth.TALKERS[3])[1]
    assert len(a) != len(b) or a != b
