"""Hearing the target: reference audio for the expected sentence or word."""
from pathlib import Path

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


def test_the_app_serves_clips_from_where_gradio_allows(monkeypatch, tmp_path):
    """The cache lives in ~/.mdd, which Gradio refuses to serve (InvalidPathError
    on Windows); the app must hand it a copy in the temp directory instead."""
    import app

    cache = tmp_path / "home" / ".mdd" / "reference"
    monkeypatch.setenv("MDD_REFERENCE", str(cache))
    served, _ = app.hear_sentence("German", "Guten Tag", False)
    assert Path(served).parent == app._TMP
    assert not Path(served).is_relative_to(cache)
    assert Path(served).read_bytes() == next(cache.rglob("*.wav")).read_bytes()


def _rate_after_new_clip(head: str) -> float:
    """Set 0.5x on an <audio>, load a different clip into it, read the rate back."""
    import base64
    import io

    sync_api = pytest.importorskip("playwright.sync_api")
    buf = io.BytesIO()
    sf.write(buf, np.zeros(8000, dtype=np.int16), 8000, format="WAV")
    clip = "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path="/opt/pw-browsers/chromium")
        except Exception:
            try:
                browser = pw.chromium.launch()
            except Exception as exc:
                pytest.skip(f"no browser: {exc}")
        page = browser.new_page()
        page.set_content(f"<html><head>{head}</head><body><audio></audio></body></html>")
        rate = page.evaluate("""async (clip) => {
            const a = document.querySelector('audio');
            const loaded = () => new Promise(r => a.addEventListener('loadedmetadata', r, {once: true}));
            a.src = clip; await loaded();
            a.playbackRate = 0.5;                       // what the player's speed button does
            a.src = clip + '#next'; await loaded();     // switching to another word
            return a.playbackRate; }""", clip)
        browser.close()
    return rate


def test_the_chosen_speed_survives_switching_words():
    """Gradio's speed button said 0.5x but the next word played at 1x."""
    import app

    assert _rate_after_new_clip("") == 1.0, "the browser behaviour this works around"
    assert _rate_after_new_clip(app.KEEP_PLAYBACK_RATE) == 0.5
    assert app.LAUNCH_OPTIONS["head"] is app.KEEP_PLAYBACK_RATE
