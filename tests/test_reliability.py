"""Flags are not equally trustworthy, and the report has to say so."""
import json


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


def test_zero_recall_is_not_claimed_as_blindness_without_evidence(tmp_path, monkeypatch):
    """Low recall only means the recogniser is blind if the error was actually in
    the audio. Rows without a `rendered` measurement are withheld rather than
    guessed at, and unrendered injections (final_t/k/p/s devoicing, rendered <
    1.35x espeak jitter floor) are excluded from committed blind_spots("de")."""
    import mdd.reliability as rel
    from mdd.reliability import RESULTS

    # 1) Legacy rows predating `rendered` are still withheld
    legacy = tmp_path / "results" / "xx"
    legacy.mkdir(parents=True)
    (legacy / "synthetic_errors.json").write_text(json.dumps({"rows": [
        {"name": "final_k→ɡ", "recall": {"-2.0": 0.0, "-1.0": 0.0}},
    ]}), encoding="utf-8")
    monkeypatch.setattr(rel, "RESULTS", tmp_path / "results")
    rel.blind_spots.cache_clear()
    try:
        assert rel.blind_spots("xx") == {}
    finally:
        monkeypatch.undo()
        rel.blind_spots.cache_clear()

    # 2) Live committed German results filter unrendered rows and keep only audible blind spots
    synth = json.loads((RESULTS / "synthetic_errors.json").read_text(encoding="utf-8"))
    by_name = {r["name"]: r for r in synth["rows"]}
    spots = blind_spots("de")
    assert spots, "calibrated v6 results should report verified audible blind spots"
    for unrendered in ("final_t→d", "final_k→ɡ", "final_p→b", "final_s→z"):
        assert by_name[unrendered]["rendered"] < 1.35
        assert unrendered not in spots
    for name, recall in spots.items():
        row = by_name[name]
        assert row["rendered"] >= 1.35
        assert recall <= 0.30
        assert row["recall"]["-1.0"] == recall


def test_a_verified_blind_spot_is_reported(tmp_path, monkeypatch):
    """Once audibility is measured, a genuine blind spot must still surface —
    long_a→short renders at 7.91x and really is undetectable."""
    import mdd.reliability as rel

    results = tmp_path / "results" / "xx"
    results.mkdir(parents=True)
    (results / "synthetic_errors.json").write_text(json.dumps({"rows": [
        {"name": "long_a→short", "recall": {"-2.0": 0.0}, "rendered": 7.91},
        {"name": "final_k→ɡ", "recall": {"-2.0": 0.0}, "rendered": 0.93},
        {"name": "ü_long→uː", "recall": {"-2.0": 0.875}, "rendered": 2.71},
    ]}), encoding="utf-8")
    monkeypatch.setattr(rel, "RESULTS", tmp_path / "results")
    rel.blind_spots.cache_clear()
    try:
        spots = rel.blind_spots("xx")
        assert spots == {"long_a→short": 0.0}, "audible-and-missed only"
    finally:
        rel.blind_spots.cache_clear()


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
    assert native["per_phone_complete"] is True
    assert "-1.0" in native["per_phone_by_tau"]

    synth = json.loads((RESULTS / "synthetic_errors.json").read_text(encoding="utf-8"))
    assert synth["rows"] and "recall" in synth["rows"][0]
    assert "-2.0" in synth["rows"][0]["recall"]
    assert "rendered" in synth["rows"][0]
    assert "cond_diag_acc" in synth["rows"][0]
    assert "der" in synth["rows"][0]
    assert "-1.0" in synth["overall"]["metrics"]
