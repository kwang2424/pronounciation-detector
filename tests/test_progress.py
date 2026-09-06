"""Cross-session persistence: totals must stay honest across saves and restarts."""
import json

import pytest

from mdd.hvpt import Session
from mdd.progress import MAX_SESSIONS, SCHEMA_VERSION, Progress


@pytest.fixture
def store(tmp_path):
    return Progress(tmp_path / "progress.json")


def _report(seen, correct, cid="soft-d", worst=None):
    entry = {"label": cid, "seen": seen, "correct": correct,
             "accuracy": correct / seen if seen else 0.0, "tip": ""}
    if worst:
        entry["worst_confusion"] = worst
    return {"contrasts": {cid: entry}}


def test_round_trip(store):
    store.record_session("da", _report(10, 6), "s1", now=1000.0)
    store.save()
    back = Progress.load(store.path)
    h = back.history("da")["soft-d"]
    assert (h.seen, h.correct, h.last_practiced) == (10, 6, 1000.0)
    assert h.accuracy == 0.6


def test_saving_the_same_session_repeatedly_does_not_inflate_totals(store):
    """The app saves after every answer; that must upsert, not accumulate."""
    for seen in range(1, 7):                      # 1, 2, ... 6 trials answered so far
        store.record_session("da", _report(seen, 0), "same-session", now=1000.0)
    h = store.history("da")["soft-d"]
    assert h.seen == 6, "running totals were added instead of replaced"
    assert len(store.sessions("da")) == 1


def test_distinct_sessions_accumulate(store):
    store.record_session("da", _report(6, 0), "s1", now=1000.0)
    store.record_session("da", _report(4, 4), "s2", now=2000.0)
    h = store.history("da")["soft-d"]
    assert (h.seen, h.correct) == (10, 4)
    assert [r.trials for r in store.sessions("da")] == [6, 4]
    assert h.last_practiced == 2000.0


def test_upsert_keeps_the_original_start_time(store):
    store.record_session("da", _report(1, 0), "s1", now=1000.0)
    store.record_session("da", _report(5, 2), "s1", now=9999.0)
    assert store.sessions("da")[0].at == 1000.0


def test_confusions_aggregate_across_sessions(store):
    worst = {"target": "mad", "picked": "mat", "count": 3}
    store.record_session("da", _report(6, 0, worst=worst), "s1", now=1.0)
    store.record_session("da", _report(6, 0, worst=worst), "s2", now=2.0)
    assert store.history("da")["soft-d"].confusions[("mad", "mat")] == 6


def test_languages_are_kept_separate(store):
    store.record_session("da", _report(4, 1), "s1", now=1.0)
    store.record_session("de", _report(9, 9, cid="ich-ach"), "s2", now=2.0)
    assert set(store.history("da")) == {"soft-d"}
    assert set(store.history("de")) == {"ich-ach"}
    assert store.is_empty("ko")


def test_trimming_preserves_lifetime_totals(store):
    for i in range(MAX_SESSIONS + 20):
        store.record_session("da", _report(2, 1), f"s{i}", now=float(i))
    assert len(store.sessions("da")) == MAX_SESSIONS
    h = store.history("da")["soft-d"]
    assert h.seen == (MAX_SESSIONS + 20) * 2, "archived sessions dropped from lifetime totals"
    assert h.correct == MAX_SESSIONS + 20


def test_empty_report_writes_nothing(store):
    store.record_session("da", {"contrasts": {}}, "s1", now=1.0)
    assert store.is_empty("da")


def test_unreadable_file_yields_an_empty_store_and_leaves_it_alone(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text("{not json", encoding="utf-8")
    store = Progress.load(path)
    assert store.load_error and "could not read" in store.load_error
    assert store.is_empty("da")
    assert path.read_text(encoding="utf-8") == "{not json", "corrupt file must not be clobbered"


def test_unknown_schema_version_is_not_rewritten(tmp_path):
    path = tmp_path / "progress.json"
    original = json.dumps({"version": SCHEMA_VERSION + 99, "languages": {"da": {}}})
    path.write_text(original, encoding="utf-8")
    store = Progress.load(path)
    assert store.load_error and "schema version" in store.load_error
    assert path.read_text(encoding="utf-8") == original


def test_missing_file_is_simply_empty(tmp_path):
    store = Progress.load(tmp_path / "nope.json")
    assert store.load_error is None
    assert store.is_empty()


def test_save_leaves_no_temp_files_behind(store):
    store.record_session("da", _report(1, 1), "s1", now=1.0)
    store.save()
    assert [p.name for p in store.path.parent.iterdir()] == ["progress.json"]


# ---------------------------------------------------------------- integration
def test_session_seeds_from_history_and_reports_lifetime(store):
    first = Session("da", seed=1, progress=store)
    for _ in range(6):
        t = first.next_trial("soft-d")
        first.record(t, next(c for c in t.choices if c != t.target))
        first.save()

    second = Session("da", seed=2, progress=Progress.load(store.path))
    assert second.prior["soft-d"].seen == 6
    assert second.combined("soft-d").seen == 6
    t = second.next_trial("soft-d")
    second.record(t, t.target)
    assert second.combined("soft-d").seen == 7
    assert second.report()["contrasts"]["soft-d"]["seen"] == 1, "this sitting only"
    assert second.report()["lifetime"]["soft-d"]["seen"] == 7, "running total"


def test_practice_is_steered_by_lifetime_not_just_this_sitting(store):
    first = Session("da", seed=1, progress=store)
    for cid in first.contrast_ids:
        for _ in range(5):
            t = first.next_trial(cid)
            first.record(t, t.target if cid != "soft-d" else
                         next(c for c in t.choices if c != t.target))
    first.save()

    second = Session("da", seed=7, progress=Progress.load(store.path))
    assert {second.next_trial().contrast_id for _ in range(8)} == {"soft-d"}


def test_save_is_a_no_op_without_a_store_or_without_trials(store):
    assert Session("da", seed=1).save() is None
    assert Session("da", seed=1, progress=store).save() is None
