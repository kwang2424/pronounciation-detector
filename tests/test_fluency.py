"""Pauses, syllables and the 4/3/2 history."""
import json

import numpy as np
import pytest

from mdd import fluency as F
from mdd import prosody as P
from mdd import synth


def say(text, lang="de", wpm=150, variant=""):
    synth.synthesize("warm up", lang)
    sr, s = synth.synthesize(text, lang, synth.Talker(variant, wpm))
    return P.resample(np.array(s, np.float32) / 32768, sr)


def silence(seconds):
    return np.zeros(int(P.SR * seconds), np.float32)


A, B = "Ich möchte fünf Brötchen, bitte.", "Das ist sehr gut."


def test_pauses_are_silences_of_250_ms_or_more():
    x = np.concatenate([silence(0.6), say(A), silence(0.8), say(B), silence(0.15),
                        say(B), silence(1.2), say(A), silence(0.6)])
    m = F.measure(x)
    assert m.pauses == 2, "the 0.15 s gap is not a pause; leading/trailing silence is not either"
    assert abs(m.pause_time - 2.0) < 0.3
    assert m.duration < len(x) / P.SR - 1.0


def test_background_noise_does_not_change_the_pauses():
    x = np.concatenate([say(A), silence(0.8), say(B)])
    noisy = x + 0.003 * np.random.default_rng(1).standard_normal(len(x)).astype(np.float32)
    assert F.measure(noisy).pauses == F.measure(x).pauses == 1


def test_syllable_estimate_is_close_to_a_hand_count():
    cases = [("de", A, 8), ("de", "Zehn Katzen sitzen zu Hause.", 8),
             ("de", "Am Wochenende haben wir einen langen Spaziergang am See gemacht.", 17),
             ("fr", "Hier soir, nous avons mangé dans un petit restaurant italien.", 17),
             ("fr", "Le train rouge arrive à Paris.", 8)]
    est = sum(F.measure(say(t, lang)).syllables for lang, t, _ in cases)
    true = sum(n for *_, n in cases)
    assert 0.8 < est / true < 1.15


def test_faster_speech_measures_faster():
    slow, fast = F.measure(say(A + " " + B, wpm=120)), F.measure(say(A + " " + B, wpm=190))
    assert fast.speech_rate > 1.2 * slow.speech_rate


def test_silence_is_refused():
    with pytest.raises(ValueError, match="No speech"):
        F.measure(silence(2.0))


def test_log_keeps_rounds_per_session(tmp_path):
    path = tmp_path / "fluency.json"
    m = F.Measures(60.0, 45.0, 10, 15.0, 200, 8.0)
    log = F.FluencyLog(path)
    log.record("s1", "de", "Urlaub", 1, m, now=1.0)
    log.record("s1", "de", "", 2, m)
    log.record("s2", "fr", "Vacances", 1, m)
    log.save()
    again = F.FluencyLog(path)
    (s1,) = again.sessions("de")
    assert s1["topic"] == "Urlaub" and set(s1["rounds"]) == {"1", "2"}
    assert F.from_dict(s1["rounds"]["1"]).mean_run == pytest.approx(200 / 11)


def test_measures_derived_values():
    m = F.Measures(duration=120.0, phonation=90.0, pauses=12, pause_time=24.0,
                   syllables=360, longest_run=9.0)
    assert m.speech_rate == 3.0 and m.articulation_rate == 4.0
    assert m.pauses_per_min == 6.0 and m.mean_pause == 2.0
    assert json.dumps(m.row())
