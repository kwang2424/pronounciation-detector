"""Spaced review of flagged words."""
import json

import pytest

from mdd.review import DAY, INTERVALS, MIN_SOUND_ATTEMPTS, Review, gap, word_key

T0 = 1_700_000_000.0
SENTENCE = "Ich möchte fünf Brötchen."


def ph(word, canonical, realized=None, flagged=False):
    return {"word": word, "canonical": canonical,
            "realized": realized if realized is not None else canonical, "flagged": flagged,
            "gop": None, "tip": "tip" if flagged else None, "op": "sub" if flagged else "match"}


def attempt(*, möchte_ok=True, ich_ok=True):
    return [ph("Ich", "ɪ"), ph("Ich", "ç", "ç" if ich_ok else "k", flagged=not ich_ok),
            ph("möchte", "m"), ph("möchte", "œ", "œ" if möchte_ok else "ɔ", flagged=not möchte_ok),
            ph("möchte", "ç"), ph("fünf", "f"), ph("fünf", "y"),
            ph("Brötchen.", "b"), ph("Brötchen.", "øː")]


@pytest.fixture
def review(tmp_path):
    return Review(tmp_path / "review.json")


def test_a_flagged_word_joins_the_queue_due_in_ten_minutes(review):
    out = review.record("de", SENTENCE, attempt(möchte_ok=False), now=T0)
    assert out.added == ["möchte"]
    item = review.items("de")["möchte"]
    assert item.sentence == SENTENCE and item.errors == {"œ→ɔ": 1}
    assert review.due("de", now=T0) == []
    assert [i.word for i in review.due("de", now=T0 + INTERVALS[0])] == ["möchte"]


def test_clean_reviews_space_out_until_learned(review):
    review.record("de", SENTENCE, attempt(möchte_ok=False), now=T0)
    now, gaps = T0, []
    for _ in INTERVALS[1:]:
        now = review.items("de")["möchte"].due
        out = review.record("de", SENTENCE, attempt(), now=now)
        gaps.append(out.advanced[0][1])
    assert gaps == list(INTERVALS[1:]), "each clean review lengthens the gap"
    out = review.record("de", SENTENCE, attempt(), now=review.items("de")["möchte"].due)
    assert out.learned == ["möchte"]
    assert review.counts("de", now=now + 100 * DAY) == {"due": 0, "tracked": 0, "learned": 1}


def test_a_new_flag_sends_the_word_back_to_the_start(review):
    review.record("de", SENTENCE, attempt(möchte_ok=False), now=T0)
    review.record("de", SENTENCE, attempt(), now=T0 + INTERVALS[0])        # -> 1 day
    out = review.record("de", SENTENCE, attempt(möchte_ok=False), now=T0 + 2 * DAY)
    assert out.failed == ["möchte"]
    item = review.items("de")["möchte"]
    assert item.box == 0 and item.due == T0 + 2 * DAY + INTERVALS[0]
    assert item.errors == {"œ→ɔ": 2}


def test_saying_it_early_does_not_advance_it(review):
    """The gap is the point; a repeat a minute later only shows short-term memory."""
    review.record("de", SENTENCE, attempt(möchte_ok=False), now=T0)
    out = review.record("de", SENTENCE, attempt(), now=T0 + 60)
    assert out.early == ["möchte"] and out.advanced == []
    assert review.items("de")["möchte"].box == 0


def test_noisy_flags_neither_add_nor_fail_a_word(review):
    """German d→t also happens on native speech, so it is weak evidence."""
    noisy = [ph("du", "d", "t", flagged=True), ph("du", "uː")]
    out = review.record("de", "du bist", noisy, now=T0)
    assert out.added == [] and "du" not in review.items("de")

    review.record("de", "du bist", [ph("du", "d"), ph("du", "uː", "u", flagged=True)], now=T0)
    out = review.record("de", "du bist", noisy, now=T0 + DAY)
    assert out.noisy == ["du"] and out.failed == []
    assert review.items("de")["du"].box == 0 and review.items("de")["du"].flagged == 1


def test_an_inserted_sound_is_weak_evidence(review):
    out = review.record("de", "gut", [ph("gut", "ɡ"), ph("gut", None, "ə", flagged=True)], now=T0)
    assert out.added == []


def test_a_new_substitution_on_a_mishearable_phone_still_counts(review):
    """ö is misheard natively as [ɛ] only; ö said as [ɔ] is a real error."""
    habitual = attempt()
    habitual[3] = ph("möchte", "œ", "ɛ", flagged=True)
    assert review.record("de", SENTENCE, habitual, now=T0).added == []
    assert review.record("de", SENTENCE, attempt(möchte_ok=False), now=T0).added == ["möchte"]


def test_a_repeated_word_counts_once_per_attempt(review):
    review.record("de", "Die Katze, die Maus.", [ph("Die", "iː", "i", flagged=True)], now=T0)
    out = review.record("de", "Die Katze, die Maus.",
                        [ph("Die", "iː"), ph("die", "iː")], now=T0 + INTERVALS[0])
    assert out.advanced == [("Die", INTERVALS[1])]
    assert review.items("de")["die"].box == 1


def test_next_review_picks_the_sentence_with_most_due_words(review):
    review.record("de", "Der Hund.", [ph("Hund", "ʊ", "u", flagged=True)], now=T0)
    review.record("de", SENTENCE, attempt(möchte_ok=False, ich_ok=False), now=T0 + 1)
    sentence, words = review.next_review("de", now=T0 + DAY)
    assert sentence == SENTENCE and sorted(words) == ["Ich", "möchte"]
    assert review.next_review("de", now=T0) is None


def test_trouble_sounds_are_rates_over_a_window(review):
    for i in range(MIN_SOUND_ATTEMPTS):
        review.record("de", SENTENCE, attempt(möchte_ok=i == 0), now=T0 + i)
    rows = dict((p, (bad, seen)) for p, bad, seen in review.trouble_sounds("de", now=T0 + 10))
    assert rows["œ"] == (2, 3)
    assert "y" not in rows, "never flagged, so not a trouble sound"
    assert review.trouble_sounds("de", now=T0 + 60 * DAY) == [], "outside the window"


def test_one_sentence_is_not_enough_to_call_a_sound_a_trouble_sound(review):
    """'Ich möchte' has three ich-Laute; that is one attempt, not three."""
    review.record("de", SENTENCE, attempt(ich_ok=False), now=T0)
    assert review.trouble_sounds("de", now=T0) == []


def test_history_survives_a_restart_and_is_per_language(tmp_path):
    path = tmp_path / "review.json"
    r = Review(path)
    r.record("de", SENTENCE, attempt(möchte_ok=False), now=T0)
    r.save()
    again = Review.load(path)
    assert "möchte" in again.items("de") and again.items("fr") == {}
    assert again.attempts("de") == 1


def test_an_unreadable_file_is_reported_not_fatal(tmp_path):
    path = tmp_path / "review.json"
    path.write_text("{not json", encoding="utf-8")
    r = Review.load(path)
    assert r.load_error and r.items("de") == {}
    path.write_text(json.dumps({"version": 99, "languages": {}}), encoding="utf-8")
    assert Review.load(path).load_error


def test_helpers():
    assert word_key("Brötchen.") == "brötchen" and word_key("„Tür“") == "tür"
    assert gap(600) == "10 minutes" and gap(DAY) == "1 day" and gap(16 * DAY) == "16 days"
    assert gap(3 * 3600) == "3 hours"


# -------------------------------------------------------------------- the app
def test_the_app_records_real_attempts_but_not_dry_runs(monkeypatch):
    import app

    app.run("German", SENTENCE, None, "ɪk mɔktə fʏnf bʁøːtçən", -1.0)
    assert Review.load().attempts("de") == 0, "typed IPA is not evidence about you"

    class FakeRecognizer:
        pass

    def fake_analyse(text, audio=None, recognizer=None, **kw):
        return {"phones": attempt(möchte_ok=False), "overall": 0.9,
                "canonical": "", "realized": ""}

    monkeypatch.setattr(app, "analyse", fake_analyse)
    monkeypatch.setattr(app, "get_recognizer", FakeRecognizer)
    summary, *_, status = app.run("German", SENTENCE, "take.wav", "", -1.0)
    assert "added to review: *möchte*" in summary
    assert "1 in rotation" in status
    assert Review.load().attempts("de") == 1


def test_picking_a_phrase_fills_the_sentence_and_says_what_to_listen_for():
    import app

    text, note = app.pick_phrase("German", "Der Staat und die Stadt.", "old")
    assert text == "Der Staat und die Stadt."
    assert "long a, then short a" in note and "by ear" in note
    assert "Long vs short vowels" in note, "links to the perception contrast"
    _, note = app.pick_phrase("French", "Un bon vin blanc.", "")
    assert "unreliable for nasal vowels" in note


def test_next_review_button(monkeypatch):
    import app

    with pytest.raises(Exception, match="Nothing is due"):
        app.next_review("German")
    r = Review.load()
    r.record("de", SENTENCE, attempt(möchte_ok=False), now=T0)
    r.save()
    text, note = app.next_review("German")
    assert text == SENTENCE and "*möchte* (was `œ→ɔ`)" in note


def test_status_suggests_a_phrase_for_the_worst_sound():
    import app

    r = Review.load()
    for i in range(3):
        r.record("de", SENTENCE, attempt(möchte_ok=False), now=__import__("time").time() - i)
    r.save()
    status = app.review_status("German")
    assert "`œ` 3 of 3" in status
    # Suggests an ö phrase other than the one just said three times.
    assert "try *Schöne Grüße aus Köln.*" in status
