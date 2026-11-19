"""A history file that cannot be read must never be overwritten.

Every store starts empty when its file is corrupt or from a newer version, and
the first save used to replace it — while the app told the learner the old file
was "left untouched". Saving now renames it aside and keeps going.
"""
import json

from mdd import fluency
from mdd.progress import SCHEMA_VERSION, Progress, set_aside
from mdd.review import Review


def _corrupt(path, text="{not json"):
    path.write_text(text, encoding="utf-8")
    return text


def _kept(path):
    (kept,) = path.parent.glob(f"{path.stem}.unreadable-*{path.suffix}")
    return kept


def test_progress_keeps_a_corrupt_file_when_saving(tmp_path):
    path = tmp_path / "progress.json"
    original = _corrupt(path)
    store = Progress.load(path)
    store.record_session("de", {"contrasts": {"x": {"seen": 2, "correct": 1}}}, "s1", now=1.0)
    store.save()
    assert _kept(path).read_text(encoding="utf-8") == original
    assert store.kept_as == _kept(path)
    assert Progress.load(path).history("de")["x"].seen == 2


def test_progress_keeps_a_newer_schema_when_saving(tmp_path):
    path = tmp_path / "progress.json"
    original = _corrupt(path, json.dumps({"version": SCHEMA_VERSION + 1, "languages": {}}))
    store = Progress.load(path)
    store.record_session("de", {"contrasts": {"x": {"seen": 1, "correct": 1}}}, "s1", now=1.0)
    store.save()
    assert _kept(path).read_text(encoding="utf-8") == original


def test_review_keeps_a_corrupt_file_when_saving(tmp_path):
    path = tmp_path / "review.json"
    original = _corrupt(path)
    r = Review.load(path)
    r.record("de", "Tür", [{"word": "Tür", "canonical": "yː", "realized": "u",
                            "flagged": True}], now=1.0)
    r.save()
    assert _kept(path).read_text(encoding="utf-8") == original
    assert "tür" in Review.load(path).items("de")


def test_fluency_keeps_a_corrupt_log_and_carries_on(tmp_path):
    path = tmp_path / "fluency.json"
    original = _corrupt(path)
    log = fluency.FluencyLog(path)
    assert log.load_error
    log.record("s1", "de", "", 1, fluency.Measures(60, 50, 5, 10, 150, 8))
    log.save()
    assert _kept(path).read_text(encoding="utf-8") == original
    assert fluency.FluencyLog(path).sessions("de")


def test_a_second_unreadable_file_does_not_replace_the_first_backup(tmp_path):
    path = tmp_path / "review.json"
    _corrupt(path, "first")
    first = set_aside(path)
    _corrupt(path, "second")
    second = set_aside(path)
    assert first != second
    assert {first.read_text(), second.read_text()} == {"first", "second"}


def test_a_readable_file_is_simply_updated(tmp_path):
    path = tmp_path / "review.json"
    Review(path).save()
    Review.load(path).save()
    assert not list(tmp_path.glob("*.unreadable-*"))


def test_set_aside_without_a_file_is_a_no_op(tmp_path):
    assert set_aside(tmp_path / "nope.json") is None


def test_the_app_says_where_an_unreadable_review_file_went(monkeypatch):
    import os
    from pathlib import Path

    import app

    path = Path(os.environ["MDD_REVIEW"])
    _corrupt(path)
    assert "unreadable" in app.review_status("German")

    def fake_analyse(text, audio=None, recognizer=None, **kw):
        return {"phones": [{"word": "Tür", "canonical": "yː", "realized": "u", "flagged": True,
                            "gop": None, "tip": "tip", "op": "sub"}],
                "overall": 0.5, "canonical": "", "realized": ""}

    monkeypatch.setattr(app, "analyse", fake_analyse)
    monkeypatch.setattr(app, "get_recognizer", lambda: None)
    summary, *_ = app.run("German", "Tür", "take.wav", "", -1.0)
    assert "kept as" in summary and _kept(path).read_text() == "{not json"
