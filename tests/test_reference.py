"""Hearing the target: reference audio for the expected sentence or word."""
import numpy as np
import pytest
import soundfile as sf

from mdd import reference
from mdd.audio import PAD_HEAD_MS


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(reference, "_neural_failed_at", None)


class FakeNeural:
    """Stands in for the network voice; records what it was asked for."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, text, voice, rate):
        self.calls.append((text, voice, rate))
        if self.fail:
            raise OSError("network unreachable")
        sr = 24000
        t = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
        return (np.cos(2 * np.pi * 180 * t) * 15000).astype(np.int16), sr


def test_uses_the_neural_voice_when_it_is_reachable(tmp_path):
    fake = FakeNeural()
    ref = reference.speak("Die Tür ist grün", "de", root=tmp_path, neural=fake)
    assert ref.source == "neural" and ref.voice == "de-DE-KatjaNeural"
    assert ref.path.exists()


def test_french_uses_the_voice_the_evaluation_favoured(tmp_path):
    fake = FakeNeural()
    reference.speak("Je voudrais une bière", "fr", root=tmp_path, neural=fake)
    assert fake.calls[0][1] == "fr-FR-HenriNeural"


def test_repeat_plays_come_from_the_cache(tmp_path):
    fake = FakeNeural()
    first = reference.speak("Guten Tag", "de", root=tmp_path, neural=fake)
    second = reference.speak("Guten Tag", "de", root=tmp_path, neural=fake)
    assert first.path == second.path
    assert len(fake.calls) == 1


def test_text_is_spoken_as_a_complete_utterance(tmp_path):
    """A bare word makes a voice clip its last sound — where French nasality lives."""
    fake = FakeNeural()
    reference.speak("sans", "fr", root=tmp_path, neural=fake)
    reference.speak("Wie geht's?", "de", root=tmp_path, neural=fake)
    assert fake.calls[0][0] == "sans."
    assert fake.calls[1][0] == "Wie geht's?"


def test_clips_are_padded_and_faded(tmp_path):
    ref = reference.speak("Tag", "de", root=tmp_path, neural=FakeNeural())
    data, sr = sf.read(str(ref.path), dtype="int16")
    head = int(sr * PAD_HEAD_MS / 1000)
    assert not np.any(data[:head]), "head padding must be silence"
    assert abs(int(data[head])) < 1000, "the fake starts at full scale; the fade must remove that step"


def test_slow_asks_for_a_slower_rate_and_caches_separately(tmp_path):
    fake = FakeNeural()
    normal = reference.speak("Brötchen", "de", root=tmp_path, neural=fake)
    slow = reference.speak("Brötchen", "de", slow=True, root=tmp_path, neural=fake)
    assert fake.calls[1][2] == reference.SLOW_RATE
    assert normal.path != slow.path and slow.slow


def test_falls_back_to_espeak_and_says_so(tmp_path):
    ref = reference.speak("Die Tür ist grün", "de", root=tmp_path, neural=FakeNeural(fail=True))
    assert ref.source == "espeak" and ref.path.exists()
    assert "espeak" in reference.describe(ref) and "internet" in reference.describe(ref)


def test_after_a_failure_it_stops_paying_the_timeout(tmp_path):
    failing = FakeNeural(fail=True)
    reference.speak("eins", "de", root=tmp_path, neural=failing)
    reference.speak("zwei", "de", root=tmp_path, neural=failing)
    assert len(failing.calls) == 1, "second call must skip straight to espeak"


def test_it_retries_the_neural_voice_after_the_cooldown(tmp_path, monkeypatch):
    """One network hiccup must not condemn the whole session to the robotic voice."""
    reference.speak("eins", "de", root=tmp_path, neural=FakeNeural(fail=True))
    monkeypatch.setattr(reference, "RETRY_AFTER_S", -1.0)
    ok = FakeNeural()
    assert reference.speak("zwei", "de", root=tmp_path, neural=ok).source == "neural"


def test_espeak_slow_is_actually_slower(tmp_path):
    normal = reference.speak("Die Tür ist grün", "de", root=tmp_path, neural=None)
    slow = reference.speak("Die Tür ist grün", "de", slow=True, root=tmp_path, neural=None)
    assert sf.info(str(slow.path)).duration > sf.info(str(normal.path)).duration * 1.15


# -------------------------------------------------------------------- the app
def test_hear_sentence_needs_text():
    import app

    with pytest.raises(Exception, match="Enter a German sentence"):
        app.hear_sentence("German", None, False)


@pytest.mark.parametrize("shape", ["pandas", "list", "dict"])
def test_clicking_a_row_plays_that_word(shape, monkeypatch, tmp_path):
    """Gradio hands the table back in more than one shape; all must work."""
    import pandas as pd

    import app

    monkeypatch.setenv("MDD_REFERENCE", str(tmp_path))
    rows = [["möchte", "œ", "ɔ", "noisy", "", "tip"], ["Bier", "ʁ", "ɹ", "solid", "", "tip"]]
    table = {"pandas": pd.DataFrame(rows), "list": rows,
             "dict": {"headers": list("abcdef"), "data": rows}}[shape]

    class Evt:
        index = [1, 3]                                   # any column in row 1

    path, note = app.hear_word("German", False, table, Evt())
    assert path and note.startswith("**Bier**")


def test_a_click_outside_the_rows_is_ignored():
    import app

    class Evt:
        index = [9, 0]

    assert app.hear_word("German", False, [["x", "", "", "", "", ""]], Evt()) == (None, "")
