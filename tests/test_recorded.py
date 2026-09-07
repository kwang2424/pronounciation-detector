"""Recorded native talkers as a stimulus source — the only route for stød."""
import unicodedata

import numpy as np
import pytest
import soundfile as sf

from mdd.hvpt import Session
from mdd.languages import get
from mdd.recorded import MIN_TALKERS, RecordedTalkers, coverage, wordlist

SR = 22050


def _write(path, seed=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 0.45, int(SR * 0.45), endpoint=False)
    x = (0.4 * np.sin(2 * np.pi * (140 + seed * 7) * t) * 32767).astype(np.int16)
    sf.write(str(path), x, SR)


@pytest.fixture
def library(tmp_path):
    """A full Danish set: every contrast word from four talkers."""
    root = tmp_path / "rec"
    profile = get("da")
    for i, word in enumerate(wordlist(profile)):
        for j, talker in enumerate(["anna", "bo", "cecilie", "david"]):
            _write(root / "da" / talker / f"{word}.wav", seed=i * 4 + j)
    return root


def test_nested_and_flat_layouts_both_index(tmp_path):
    _write(tmp_path / "da" / "anna" / "hund.wav")
    _write(tmp_path / "da" / "hund__bo.wav")
    store = RecordedTalkers(tmp_path, "da")
    assert store.talkers_for("hund") == ["anna", "bo"]


def test_decomposed_filenames_still_match():
    """macOS stores filenames NFD, so 'bønder' off a Mac is not the same string
    as the NFC 'bønder' in the contrast tables unless it is normalised."""
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp())
    _write(root / "da" / "anna" / (unicodedata.normalize("NFD", "bønder") + ".wav"))
    store = RecordedTalkers(root, "da")
    assert store.talkers_for("bønder") == ["anna"]
    assert store.talkers_for(unicodedata.normalize("NFD", "bønder")) == ["anna"]


def test_a_bare_word_with_no_talker_is_ignored(tmp_path):
    """A clip with no talker attached cannot contribute to talker variability."""
    _write(tmp_path / "da" / "hund.wav")
    assert len(RecordedTalkers(tmp_path, "da")) == 0


def test_coverage_flags_missing_and_thin_words(tmp_path):
    profile = get("da")
    _write(tmp_path / "da" / "anna" / "mad.wav")     # one word, one talker
    reports = {r.contrast_id: r for r in coverage(RecordedTalkers(tmp_path, "da"), profile)}
    soft_d = reports["soft-d"]
    assert not soft_d.usable
    assert "no recordings for" in soft_d.summary()
    assert "mat" in soft_d.summary()


def test_partial_coverage_does_not_count_as_usable(tmp_path):
    """Every word in a set needs talkers; otherwise trials silently narrow to
    whatever exists, which is item memorisation rather than training."""
    profile = get("da")
    for word in ("mad", "mat", "bad"):               # 'bat' deliberately missing
        for talker in ("anna", "bo", "cecilie"):
            _write(tmp_path / "da" / talker / f"{word}.wav")
    report = next(r for r in coverage(RecordedTalkers(tmp_path, "da"), profile)
                  if r.contrast_id == "soft-d")
    assert not report.usable


def test_too_few_talkers_is_not_ready(tmp_path):
    profile = get("da")
    for word in wordlist(profile):
        for talker in list("ab")[:MIN_TALKERS - 1]:
            _write(tmp_path / "da" / talker / f"{word}.wav")
    assert all(not r.usable for r in coverage(RecordedTalkers(tmp_path, "da"), profile))


def test_recordings_make_stod_trainable(library):
    """The whole point: no synthesiser renders stød, recordings do."""
    synth_session = Session("da", seed=1)
    assert "stod" in synth_session.skipped

    recorded = Session("da", seed=1, recordings=RecordedTalkers(library, "da"))
    assert "stod" in recorded.contrast_ids
    assert recorded.skipped == {}


def test_trials_play_the_recordings(library):
    session = Session("da", seed=2, recordings=RecordedTalkers(library, "da"))
    trial = session.next_trial("stod")
    rate, samples = trial.audio()
    assert rate == SR
    assert len(samples) > 1000
    assert trial.talker in RecordedTalkers(library, "da").talkers


def test_talkers_rotate_over_real_speakers(library):
    session = Session("da", seed=3, recordings=RecordedTalkers(library, "da"))
    talkers = [session.next_trial().talker for _ in range(12)]
    assert set(talkers) <= set(RecordedTalkers(library, "da").talkers)
    assert all(a != b for a, b in zip(talkers, talkers[1:]))


def test_recorded_pairs_are_still_gated_on_their_own_audio(tmp_path):
    """Recordings are not trusted blindly: if two words were recorded as the same
    audio, the gate must still reject them."""
    profile = get("da")
    for word in wordlist(profile):
        for talker in ("anna", "bo", "cecilie"):
            _write(tmp_path / "da" / talker / f"{word}.wav", seed=0)   # all identical
    session_words = RecordedTalkers(tmp_path, "da")
    with pytest.raises(Exception):
        Session("da", seed=1, recordings=session_words)


def test_wordlist_covers_every_contrast_word():
    profile = get("da")
    words = set(wordlist(profile))
    for contrast in profile.contrasts:
        assert set(contrast.words()) <= words
