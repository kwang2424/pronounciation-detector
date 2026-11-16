"""Pitch tracking and the shadowing comparison."""
from itertools import pairwise

import numpy as np
import pytest

from mdd import prosody as P
from mdd import synth

SR = P.SR


def harmonic(f0, seconds=1.0):
    f0s = np.full(int(SR * seconds), float(f0)) if np.isscalar(f0) else f0
    ph = 2 * np.pi * np.cumsum(f0s) / SR
    return (0.3 * np.sin(ph) + 0.15 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)).astype(np.float32)


def say(text, lang="de", variant="", wpm=150, pitch=50, rng=50):
    synth.synthesize("warm up", lang)       # espeak's first call renders slightly differently
    sr, s = synth.synthesize(text, lang, synth.Talker(variant, wpm, pitch, rng))
    return P.resample(np.array(s, np.float32) / 32768, sr)


@pytest.mark.parametrize("hz", [80, 150, 380])
def test_pitch_of_a_steady_voice(hz):
    f0 = P.pitch(harmonic(hz))
    assert np.mean(~np.isnan(f0)) > 0.9
    assert abs(np.nanmedian(f0) - hz) / hz < 0.01


def test_noise_and_silence_are_unvoiced():
    noise = (0.1 * np.random.default_rng(0).standard_normal(SR)).astype(np.float32)
    assert np.isnan(P.pitch(noise)).all()
    mixed = np.concatenate([np.zeros(SR // 2, np.float32), harmonic(200, 0.5)])
    assert np.isnan(P.pitch(mixed)[:45]).all()


def test_a_low_espeak_voice_is_tracked():
    """Regression: a 15 ms window held barely one period of an 80 Hz voice, and
    espeak's low voices never cleared the textbook 0.15 YIN threshold — most
    vowels came out unvoiced."""
    f0 = P.pitch(say("Meine Freunde sind weit weg.", variant="m1", wpm=130, pitch=40))
    assert np.mean(~np.isnan(f0)) > 0.35


def test_a_monotone_take_is_called_flat():
    text = "Ich möchte heute Abend ins Kino gehen."
    c = P.compare(say(text), say(text, rng=0))
    assert c.your_range < 0.3 * c.model_range
    assert "flatter" in P.describe(c)


def test_a_higher_voice_with_the_same_melody_matches():
    """Semitones from each speaker's own median make voices comparable."""
    text = "Ich möchte heute Abend ins Kino gehen."
    c = P.compare(say(text), say(text, pitch=85))
    assert c.correlation > 0.8


def test_question_and_statement_endings():
    assert P.ending(P.semitones(P.pitch(say("Ist das gut?")))) == "rise"
    assert P.ending(P.semitones(P.pitch(say("Das ist sehr gut.")))) == "fall"
    c = P.compare(say("Das ist sehr gut."), say("Ist das gut?"))
    assert "Questions and statements differ" in P.describe(c)


def test_a_level_model_ending_is_not_held_against_you():
    c = P.Contour(np.arange(3), np.zeros(3), np.zeros(3), 3.0, 3.0, None,
                  "level", "rise", 1.0, "dtw")
    assert "Ending" not in P.describe(c)


def test_timing_and_alignment():
    text = "Ich möchte fünf Brötchen, bitte."
    c = P.compare(say(text), say(text, wpm=110))
    assert c.aligned == "dtw" and 1.15 < c.duration_ratio < 1.6 and c.correlation > 0.8


def test_long_takes_fall_back_to_linear_alignment(monkeypatch):
    monkeypatch.setattr(P, "MAX_DTW_FRAMES", 50)
    text = "Ich möchte fünf Brötchen, bitte."
    assert P.compare(say(text), say(text)).aligned == "linear"


def test_too_short_is_refused():
    with pytest.raises(ValueError, match="too short"):
        P.compare(harmonic(150, 0.1), harmonic(150))


def test_dtw_path_is_monotonic_and_complete():
    a = P.features(harmonic(np.linspace(120, 240, SR)))
    b = P.features(harmonic(np.linspace(120, 240, int(SR * 1.5))))
    path = P.dtw_path(a, b)
    assert path[0] == (0, 0) and path[-1] == (len(a) - 1, len(b) - 1)
    assert all(i2 >= i1 and j2 >= j1 for (i1, j1), (i2, j2) in pairwise(path))


def test_plot_renders():
    text = "Das ist sehr gut."
    fig = P.plot(P.compare(say(text), say(text)), "test")
    assert len(fig.axes[0].lines) >= 2
