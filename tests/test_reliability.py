"""Flags are not equally trustworthy, and the report has to say so."""
import json

import pytest

from mdd.reliability import Reliability, blind_spots, reliability


def test_the_habitual_substitution_is_marked_noisy():
    """German /d/ is falsely flagged on only 2.8% of its native occurrences, but
    10 of those 11 flags were specifically d->[t]. Seeing exactly d->[t] is much
    weaker evidence than the phone's overall rate suggests."""
    assert reliability("d", "de", "t").level == "noisy"
    assert "habitual" in reliability("d", "de", "t").note()


def test_a_different_substitution_of_the_same_phone_is_stronger():
    assert reliability("d", "de", "b").level != "noisy"


def test_a_phone_the_recogniser_never_mishears_is_solid():
    assert reliability("ɪ", "de", "e").level == "solid"
    assert reliability("ɡ", "de", "k").level == "solid"


def test_unevaluated_languages_say_unknown_rather_than_guessing():
    assert reliability("y", "fr", "u").level == "unknown"
    assert "not yet evaluated" in reliability("y", "fr", "u").note()


def test_blind_spots_are_reported_from_the_synthetic_eval():
    """Their absence from a report means nothing: German final devoicing and
    vowel-length errors are caught 0% of the time however wrong the speaker is,
    so a clean report is not evidence they were right."""
    spots = blind_spots("de")
    assert "final_t→d" in spots and spots["final_t→d"] == 0.0
    assert "final_k→ɡ" in spots
    assert "long_a→short" in spots


def test_no_blind_spots_claimed_for_an_unevaluated_language():
    assert blind_spots("fr") == {}


def test_missing_results_never_fabricate_a_number():
    assert Reliability().level == "unknown"
    assert Reliability().false_positive_rate is None


def test_corrupt_results_are_survivable(tmp_path, monkeypatch):
    import mdd.reliability as rel

    bad = tmp_path / "results"
    (bad / "xx").mkdir(parents=True)
    (bad / "xx" / "native_control.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(rel, "RESULTS", bad)
    rel._native.cache_clear()
    rel.blind_spots.cache_clear()
    try:
        assert rel.reliability("d", "xx", "t").level == "unknown"
        assert rel.blind_spots("xx") == {}
    finally:
        rel._native.cache_clear()
        rel.blind_spots.cache_clear()


def test_the_report_surfaces_both_caveats():
    import app

    summary, _, rows, _ = app.run("German", "du bist gut genug", None,
                                  "t u p e s t ɡ u k e n u", -1.0)
    assert "weigh those rows less" in summary
    assert "not evidence these were right" in summary
    levels = {r[3] for r in rows}
    assert "noisy" in levels and "solid" in levels


def test_results_files_still_have_the_shape_this_reads():
    """A guard against the eval output drifting away from the reader."""
    from mdd.reliability import RESULTS

    native = json.loads((RESULTS / "native_control.json").read_text(encoding="utf-8"))
    assert native["per_phone"] and "canonical" in native["per_phone"][0]
    assert "rate" in native["per_phone"][0] and "heard_as" in native["per_phone"][0]

    synth = json.loads((RESULTS / "synthetic_errors.json").read_text(encoding="utf-8"))
    assert synth["rows"] and "recall" in synth["rows"][0]
    assert "-2.0" in synth["rows"][0]["recall"]
