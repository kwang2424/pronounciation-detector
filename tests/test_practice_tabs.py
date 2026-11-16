"""The Shadow, Fluency and Level check tabs, driven through their handlers."""
import numpy as np
import pytest
import soundfile as sf

import app
from mdd import fluency, synth, vocab


def wav(tmp_path, name, text, lang="de", wpm=150, pad=0.0):
    synth.synthesize("warm up", lang)
    sr, s = synth.synthesize(text, lang, synth.Talker("", wpm))
    data = np.array(s, np.int16)
    if pad:
        data = np.concatenate([data, np.zeros(int(sr * pad), np.int16), data])
    path = tmp_path / name
    sf.write(str(path), data, sr)
    return str(path)


def test_shadow_needs_a_model_and_a_take(tmp_path):
    with pytest.raises(Exception, match="Get a model first"):
        app.sh_compare(None, wav(tmp_path, "a.wav", "Das ist gut."))
    with pytest.raises(Exception, match="Record yourself"):
        app.sh_compare(wav(tmp_path, "a.wav", "Das ist gut."), None)


def test_shadow_compares_against_the_voice(tmp_path, monkeypatch):
    monkeypatch.setenv("MDD_REFERENCE", str(tmp_path / "ref"))
    from mdd import reference

    # espeak only: a test must not depend on reaching the neural voice.
    monkeypatch.setattr(app, "speak", lambda text, lang, slow=False:
                        reference.speak(text, lang, slow, neural=None))
    model, note = app.sh_model("German", "Ich möchte fünf Brötchen, bitte.", False)
    assert model and "Correct pronunciation" in note
    take = wav(tmp_path, "take.wav", "Ich möchte fünf Brötchen, bitte.", wpm=120)
    fig, text = app.sh_compare(model, take)
    assert fig.axes and "Pitch range" in text and "not a score" in text


def test_a_fluency_session_moves_through_the_rounds(tmp_path):
    sid, rounds, *_ = app.fl_new("German")
    take = wav(tmp_path, "r1.wav", "Ich möchte fünf Brötchen, bitte.", pad=0.8)
    rounds, rows, note, nxt, _, cleared = app.fl_analyse(
        "German", sid, rounds, app.ROUND_LABELS[0], "Urlaub", take)
    assert rows[0][0] == "1" and nxt["value"] == app.ROUND_LABELS[1] and cleared is None
    assert "fill the time" in note, "a 5-second take for a 4-minute round"
    rounds, rows, note, *_ = app.fl_analyse("German", sid, rounds, app.ROUND_LABELS[1], "Urlaub",
                                            wav(tmp_path, "r2.wav", "Ich möchte fünf Brötchen, bitte.",
                                                wpm=190))
    assert [r[0] for r in rows] == ["1", "2"] and "Round 1 → 2" in note
    (saved,) = fluency.FluencyLog().sessions("de")
    assert saved["id"] == sid and set(saved["rounds"]) == {"1", "2"}


def test_fluency_history_compares_round_1_with_round_3(tmp_path):
    log = fluency.FluencyLog()
    for n, syl in ((1, 120), (3, 180)):
        log.record("old", "de", "Urlaub", n, fluency.Measures(60, 50, 5, 10, syl, 8), now=0)
    log.save()
    hist = app._fluency_history("German")
    assert "| Urlaub | 2.00 | 3.00 |" in hist


def test_level_check_import_analyse_and_mark(tmp_path):
    export = tmp_path / "deck.txt"
    export.write_text("#separator:tab\ndie Tür\tdoor\ngehen\tto go\nWetter\tweather\n",
                      encoding="utf-8")
    note = app.lv_import("German", str(export), 1, "A word (e.g. die Tür)")
    assert "Imported 3 notes" in note
    text = "Das Wetter ist schön, aber die Tür ist zu. Wir gehen spazieren."
    summary, rows, _, download = app.lv_analyse("German", text, None, 250)
    unknown = [r[0] for r in rows]
    assert "Wetter" not in unknown and "Tür" not in unknown and "spazieren" in unknown
    assert download
    with open(download, encoding="utf-8") as fh:
        assert fh.read().startswith(rows[0][0])
    before = float(summary.split("%")[0].split()[-1])
    summary, rows, *_ = app.lv_mark("German", ["spazieren"], text, None, 250)
    assert "spazieren" not in [r[0] for r in rows]
    assert float(summary.split("%")[0].split()[-1]) > before
    assert "spazieren" in vocab.Known("de").words


def test_level_check_reads_subtitle_files(tmp_path):
    srt = tmp_path / "ep.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nIch bin hier.\n", encoding="utf-8")
    summary, *_ = app.lv_analyse("German", "", str(srt), 500)
    assert summary.startswith("### 100%")


def test_level_check_errors():
    with pytest.raises(Exception, match="Paste a transcript"):
        app.lv_analyse("German", "  ", None, 0)
    with pytest.raises(Exception, match="Choose your Anki export"):
        app.lv_import("German", None, 1, "A sentence")
