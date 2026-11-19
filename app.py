"""Gradio UI: production feedback (record and be scored) plus perception training."""
from mdd._utf8 import ensure_utf8_mode

ensure_utf8_mode()

import datetime as _dt  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import uuid  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import gradio as gr  # noqa: E402

from mdd import fluency, prosody, testset, vocab  # noqa: E402
from mdd.g2p import clean_word  # noqa: E402
from mdd.hvpt import PerceptionUnavailable, Session  # noqa: E402
from mdd.languages import PROFILES  # noqa: E402
from mdd.phrases import CAVEATS, find, for_sound, phrases  # noqa: E402
from mdd.pipeline import GOP_THRESHOLD, analyse  # noqa: E402
from mdd.progress import Progress  # noqa: E402
from mdd.recorded import RecordedTalkers, default_root  # noqa: E402
from mdd.reference import describe, speak  # noqa: E402
from mdd.reliability import blind_spots, reliability  # noqa: E402
from mdd.review import Review, gap, word_key  # noqa: E402

_recognizer = None
_TMP = Path(tempfile.gettempdir()) / "mdd-stimuli"
_TMP.mkdir(exist_ok=True)

LANGS = {p.name: code for code, p in PROFILES.items()}
PERCEPTION_LANGS = {p.name: code for code, p in PROFILES.items() if p.hvpt_ready}


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        from mdd.recognizer import PhoneRecognizer
        _recognizer = PhoneRecognizer()
    return _recognizer


# --------------------------------------------------------------------------
# Production tab
# --------------------------------------------------------------------------
def run(lang_name: str, text: str, audio_path: str | None, ipa: str, threshold: float):
    code = LANGS[lang_name]
    # Gradio hands back None for a textbox the user cleared, not "".
    text = (text or "").strip()
    ipa = (ipa or "").strip()
    if not text:
        raise gr.Error(f"Enter a {lang_name} sentence first.")
    if ipa:
        rep = analyse(text, realized_ipa=ipa, threshold=threshold, lang=code)
    elif audio_path:
        rep = analyse(text, audio_path, get_recognizer(), threshold=threshold, lang=code)
    else:
        raise gr.Error("Record or upload audio, or paste realised IPA for a dry run.")

    flagged = [p for p in rep["phones"] if p["flagged"]]
    bad_words = {p["word"] for p in flagged}
    highlighted = []
    for w in _words(text):
        if highlighted:
            highlighted.append((" ", None))     # HighlightedText joins tokens as-is
        highlighted.append((w, "check" if w in bad_words else None))

    # A flag is not evidence on its own: on native German the recogniser reports
    # /d/ as [t] in 2.8% of its occurrences, so that row deserves less weight
    # than one for a phone it never mis-hears.
    rows = [[p["word"], p["canonical"] or "—", p["realized"] or "—",
             reliability(p["canonical"], code, p["realized"]).level,
             "" if p["gop"] is None else f"{p['gop']:.2f}", p["tip"]] for p in flagged]
    score = rep["overall"]
    summary = (f"**Score: {score:.0%}** · {len(flagged)} issue{'s' if len(flagged) != 1 else ''} "
               f"across {len(bad_words)} word{'s' if len(bad_words) != 1 else ''}"
               if flagged else f"**Score: {score:.0%}** · no issues detected")
    noisy = [f"`{p['canonical']}`→`{p['realized']}`" for p in flagged
             if reliability(p["canonical"], code, p["realized"]).level == "noisy"]
    if noisy:
        summary += ("\n\n⚠️ " + ", ".join(dict.fromkeys(noisy)) +
                    " occur on native speech too — weigh those rows less.")
    blind = blind_spots(code)
    if blind:
        worst = ", ".join(f"`{k}`" for k, v in sorted(blind.items(), key=lambda kv: kv[1])[:4])
        summary += (f"\n\n*Not detectable at all ({worst}): a clean report is not "
                    f"evidence these were right.*")
    if ipa:
        summary += "\n\n*Dry run — not saved to your review history.*"
    else:
        # Only real recordings count: typed IPA says nothing about how you speak.
        review = Review.load()
        outcome = review.record(code, text, rep["phones"])
        try:
            review.save()
        except OSError as exc:
            summary += f"\n\n⚠️ Could not save review history: {exc}"
        if review.kept_as:
            summary += (f"\n\n⚠️ The old review file could not be read; it was kept as "
                        f"`{review.kept_as}` and a new one started.")
        if outcome.summary():
            summary += "\n\n" + outcome.summary()
    ipa_view = f"**Expected:** `{rep['canonical']}`\n\n**Heard:** `{rep['realized']}`"
    return summary, highlighted, rows, ipa_view, review_status(lang_name)


def _words(text: str) -> list[str]:
    return [w for w in (clean_word(t) for t in text.split()) if w]


def _servable(path: Path) -> str:
    """A copy of a cached clip that Gradio will agree to serve.

    Gradio only serves files from the working directory or the system temp
    directory unless launch() is given allowed_paths, and the reference cache
    lives in ~/.mdd. Copying into temp, as the perception tab already does,
    works however the app is started.
    """
    dest = _TMP / f"reference-{path.parent.name}-{path.name}"
    if not dest.exists() or dest.stat().st_mtime < path.stat().st_mtime:
        shutil.copyfile(path, dest)
    return str(dest)


def hear_sentence(lang_name: str, text: str | None, slow: bool):
    """Play the sentence as it should sound, to compare against your recording."""
    text = (text or "").strip()
    if not text:
        raise gr.Error(f"Enter a {lang_name} sentence first.")
    ref = speak(text, LANGS[lang_name], bool(slow))
    return _servable(ref.path), describe(ref)


def _cell(table, row: int, col: int):
    """Read one cell from whatever shape Gradio hands back for a Dataframe."""
    try:
        if hasattr(table, "iloc"):
            return table.iloc[row, col]
        if isinstance(table, dict):
            return table["data"][row][col]
        return table[row][col]
    except (IndexError, KeyError, TypeError):
        return None


def hear_word(lang_name: str, slow: bool, table, evt: gr.SelectData):
    """Clicking a flagged row plays that word said correctly — the 'expected'
    column as sound rather than as an IPA symbol."""
    index = evt.index
    row = index[0] if isinstance(index, (list, tuple)) else index
    word = _cell(table, row, 0)
    if not word or word == "—":
        return None, ""
    ref = speak(str(word), LANGS[lang_name], bool(slow))
    return _servable(ref.path), f"**{word}** · " + describe(ref)


def on_lang_change(lang_name: str):
    profile = PROFILES[LANGS[lang_name]]
    return (gr.update(value=profile.example, label=f"{profile.name} sentence"),
            phrase_choices(lang_name), review_status(lang_name), "")


# ---------------------------------------------------------------- practice
def phrase_choices(lang_name: str):
    options = [(p.label, p.text) for p in phrases(LANGS[lang_name])]
    return gr.update(choices=options, value=None, interactive=bool(options),
                     label="Practice phrase" if options
                     else f"Practice phrase (none for {lang_name} yet)")


def pick_phrase(lang_name: str, choice: str | None, current: str | None):
    """Put the chosen phrase in the sentence box and say what to listen for."""
    code = LANGS[lang_name]
    phrase = find(code, choice or "")
    if phrase is None:
        return gr.update(value=current), ""
    note = [f"**Listen for:** {phrase.focus}."]
    if phrase.by_ear:
        note.append(f"👂 Practise this one by ear: {phrase.by_ear}.")
    if phrase.contrasts:
        labels = {c.id: c.label for c in PROFILES[code].contrasts}
        trained = ", ".join(labels.get(c, c) for c in phrase.contrasts)
        note.append(f"Pairs with perception training: *{trained}*.")
    if code in CAVEATS:
        note.append(CAVEATS[code])
    return phrase.text, " ".join(note)


def next_review(lang_name: str):
    """Load the sentence holding the most words that are due for review."""
    code = LANGS[lang_name]
    review = Review.load()
    nxt = review.next_review(code)
    if nxt is None:
        raise gr.Error("Nothing is due for review right now.")
    sentence, words = nxt
    items = review.items(code)
    focus = []
    for w in words:
        worst = items[word_key(w)].worst if word_key(w) in items else None
        focus.append(f"*{w}*" + (f" (was `{worst}`)" if worst else ""))
    return sentence, ("**Review:** focus on " + ", ".join(focus) +
                      ". Listen with 🔊 first, then record.")


def review_status(lang_name: str) -> str:
    code = LANGS[lang_name]
    review = Review.load()
    lines = []
    if review.load_error:
        lines.append(f"⚠️ Review history unreadable ({review.load_error}). Your next "
                     "recording starts a new one; the old file is renamed and kept.")
    c = review.counts(code)
    if not c["tracked"] and not c["learned"]:
        lines.append("**Review queue:** empty. Words the scorer flags in your recordings "
                     "land here and come back at spaced intervals.")
    else:
        head = f"**Review queue:** {c['due']} due now · {c['tracked']} in rotation"
        if c["learned"]:
            head += f" · {c['learned']} learned"
        if not c["due"]:
            wait = review.upcoming(code)
            if wait:
                head += f" · next due in {gap(wait)}"
        lines.append(head)
    trouble = review.trouble_sounds(code)[:3]
    if trouble:
        said = review.times_said(code)
        parts = []
        for phone, bad, seen in trouble:
            part = f"`{phone}` {bad} of {seen}"
            # The phrase for this sound you have said least, for variety.
            options = for_sound(code, phone)
            if options:
                part += f" → try *{min(options, key=lambda p: said[p.text]).text}*"
            parts.append(part)
        lines.append("**Trouble sounds (30 days):** " + " · ".join(parts))
    return "\n\n".join(lines)


# --------------------------------------------------------------------------
# Perception tab
# --------------------------------------------------------------------------
def _stimuli_for(code: str) -> RecordedTalkers | None:
    """Prefer recorded or neural clips over formant synthesis when any exist.

    espeak is intelligible enough to validate a contrast mechanically but thin to
    train on, which is the first thing a real user noticed. Anything in the
    stimuli directory — neural TTS from `eval.make_stimuli`, or real recordings —
    is used instead, and still has to pass the same gate.
    """
    try:
        store = RecordedTalkers(default_root(), code)
    except OSError:
        return None
    return store if len(store.talkers) >= 2 else None


def start_session(lang_name: str):
    code = PERCEPTION_LANGS[lang_name]
    store = Progress.load()
    stimuli = _stimuli_for(code)
    try:
        session = Session(code, progress=store, recordings=stimuli)
    except PerceptionUnavailable as exc:
        if stimuli is None:
            raise gr.Error(str(exc))
        # The recorded set failed the gate; fall back rather than blocking practice.
        gr.Warning(f"Recorded stimuli unusable ({exc}); falling back to synthesis.")
        stimuli = None
        try:
            session = Session(code, progress=store)
        except PerceptionUnavailable as exc2:
            raise gr.Error(str(exc2)) from exc2
    note = ""
    if session.skipped:
        skipped = ", ".join(f"`{k}`" for k in session.skipped)
        note = (f"\n\n*Not trained: {skipped} — the synthesiser cannot render "
                f"{'them' if len(session.skipped) > 1 else 'it'} distinctly, so a trial "
                f"would be unanswerable.*")
    if stimuli is not None:
        note += (f"\n\n*Stimuli: {len(stimuli.talkers)} recorded/neural talkers "
                 f"from {default_root()}.*")
    else:
        note += ("\n\n*Stimuli: espeak formant synthesis — robotic. Run "
                 "`python -m eval.make_stimuli " + code + "` for neural voices.*")
    if store.load_error:
        note += (f"\n\n*Starting from an empty history: {store.load_error}. "
                 f"Past practice is not lost — that file is renamed to "
                 f"`{store.path.stem}.unreadable-<date>.json` and kept, not overwritten.*")
    elif session.prior:
        done = sum(st.seen for st in session.prior.values())
        note += f"\n\n*Picking up {done} earlier trials from {store.path}.*"
    return (session, *_serve(session, f"Session started · {len(session.contrast_ids)} "
                                      f"contrasts{note}"))


def _serve(session: Session, message: str):
    """Build a new trial and the UI updates that present it."""
    trial = session.next_trial()
    contrast = session.profile.contrast(trial.contrast_id)
    path = trial.render(_TMP / f"trial-{len(session.history)}-{trial.talker_id}.wav")
    heading = f"### {contrast.label}\n{contrast.why}"
    return (trial, str(path), heading,
            gr.update(choices=list(trial.choices), value=None, visible=True),
            message, _stats(session), _trend(session))


def submit(session: Session, trial, choice: str | None):
    if session is None or trial is None:
        raise gr.Error("Start a session first.")
    if not choice:
        raise gr.Error("Pick what you heard.")
    correct = session.record(trial, choice)
    # Save after every answer: a training session that is closed rather than
    # formally ended is the normal case, and losing it would defeat the point.
    try:
        session.save()
    except OSError as exc:
        gr.Warning(f"Could not save progress: {exc}")
    contrast = session.profile.contrast(trial.contrast_id)
    if correct:
        message = f"✅ **{trial.target}** — correct."
    else:
        message = (f"❌ You picked **{choice}**; it was **{trial.target}**.\n\n"
                   f"*{contrast.tip}*")
    return (*_serve(session, message),)


def _stats(session: Session) -> str:
    rep = session.report()
    lines = []
    if session.history:
        lines += [f"**This session:** {session.accuracy():.0%} over {len(session.history)} "
                  f"trials · {session.difficulty} choices per trial", ""]
    else:
        lines += ["_No trials this session yet._", ""]

    lifetime = rep["lifetime"]
    if lifetime:
        lines += ["| Contrast | All trials | Accuracy | This session |", "|---|---|---|---|"]
        for cid, life in sorted(lifetime.items(), key=lambda kv: kv[1]["accuracy"]):
            now = rep["contrasts"].get(cid)
            today = f"{now['accuracy']:.0%} ({now['seen']})" if now else "—"
            lines.append(f"| {life['label']} | {life['seen']} | {life['accuracy']:.0%} | {today} |")
        lines += ["", "_Sorted weakest first — that is the order practice is steered in._"]
    return "\n".join(lines)


def _trend(session: Session) -> str:
    """Per-session accuracy over time: the signal a single sitting cannot show."""
    if session.progress is None:
        return ""
    records = session.progress.sessions(session.profile.code)
    if len(records) < 2:
        return ("### Trend\n_Two or more saved sessions are needed before a trend means "
                "anything. HVPT gains show up across a block of sessions, not within one._")
    rows = ["### Trend", "", "| Session | Trials | Accuracy |", "|---|---|---|"]
    for i, r in enumerate(records[-10:], start=max(1, len(records) - 9)):
        stamp = _dt.datetime.fromtimestamp(r.at, _dt.UTC).astimezone().strftime("%Y-%m-%d %H:%M")
        rows.append(f"| {i} · {stamp} | {r.trials} | {r.accuracy:.0%} |")
    return "\n".join(rows)


# --------------------------------------------------------------------------
# Test set tab
# --------------------------------------------------------------------------
TESTSET_LANGS = {PROFILES[code].name: code for code in testset.CONTRASTS}


def _ts_card(code: str, skipped: list) -> tuple:
    """The next take to record, its instruction card and the progress line."""
    root = testset.default_root()
    skip = set(skipped or [])
    todo = testset.pending(root, code, skip)
    done = testset.recorded(root, code)
    plan = testset.plan(code)
    per = []
    for c in testset.contrasts(code):
        n = sum(1 for t, _ in done if t.contrast == c.id)
        per.append(f"{c.label} {n}/{sum(1 for t in plan if t.contrast == c.id)}")
    progress = f"**{len(done)} of {len(plan)} recorded** · " + " · ".join(per)
    if done:
        progress += (f"\n\nScore what you have so far with `python -m eval.learner_set "
                     f"{code}` and paste the output.")
    if not todo:
        left = " (some skipped — reload the tab to return to them)" if skip else ""
        return None, f"### All done{left} 🎉", progress
    t = todo[0]
    c = testset.contrast(code, t.contrast)
    version = "your best attempt" if t.kind == "best" else "the deliberate error"
    hint = ("Listen with 🔊 first if you like, then record." if t.kind == "best" else
            "Exaggerate a little so the error is really there — that is the label.")
    card = (f"### {c.label} · {version} · take {t.take} of {testset.TAKES}\n\n"
            f"{testset.instruction(t)}\n\n*{hint}*")
    return t, card, progress


def ts_show(lang_name: str, skipped: list):
    take, card, progress = _ts_card(TESTSET_LANGS[lang_name], skipped)
    return take, card, progress, None


def ts_save(lang_name: str, take, audio_path: str | None, skipped: list):
    if take is None:
        raise gr.Error("Nothing left to record.")
    if not audio_path:
        raise gr.Error("Record the take first.")
    try:
        testset.save(testset.default_root(), take, audio_path)
    except ValueError as exc:
        raise gr.Error(str(exc))
    return ts_show(lang_name, skipped)


def ts_skip(lang_name: str, take, skipped: list):
    skipped = list(skipped or []) + ([take] if take is not None else [])
    return (skipped, *ts_show(lang_name, skipped))


def ts_undo(lang_name: str, skipped: list):
    """Delete the most recent take and show its card again."""
    code = TESTSET_LANGS[lang_name]
    done = testset.recorded(testset.default_root(), code)
    if not done:
        raise gr.Error("No takes recorded yet.")
    take, path = max(done, key=lambda tp: tp[1].stat().st_mtime)
    path.unlink()
    _, card, progress = _ts_card(code, skipped)
    c = testset.contrast(code, take.contrast)
    version = "your best attempt" if take.kind == "best" else "the deliberate error"
    card = (f"### {c.label} · {version} · take {take.take} of {testset.TAKES} (redo)\n\n"
            f"{testset.instruction(take)}")
    return take, card, progress, None


def ts_hear(lang_name: str, take):
    if take is None:
        raise gr.Error("Nothing left to record.")
    ref = speak(take.word, TESTSET_LANGS[lang_name])
    return _servable(ref.path)


# --------------------------------------------------------------------------
# Shadowing tab
# --------------------------------------------------------------------------
def sh_model(lang_name: str, text: str | None, slow: bool):
    """The voice's rendering of the sentence, as the model to shadow."""
    text = (text or "").strip()
    if not text:
        raise gr.Error(f"Enter a {lang_name} sentence, or upload a clip as the model.")
    ref = speak(text, LANGS[lang_name], bool(slow))
    note = describe(ref)
    if ref.source == "espeak":
        note += (" Its intonation is rule-based and its questions barely rise, so "
                 "compare the melody against it loosely.")
    return _servable(ref.path), note


def sh_compare(model_path: str | None, take_path: str | None):
    if not model_path:
        raise gr.Error("Get a model first: 🔊 the sentence, or upload a clip.")
    if not take_path:
        raise gr.Error("Record yourself first.")
    try:
        c = prosody.compare(prosody.load(model_path), prosody.load(take_path))
    except ValueError as exc:
        raise gr.Error(str(exc))
    note = prosody.describe(c)
    if c.aligned == "linear":
        note += ("\n\n*Long clip: timing was stretched evenly rather than matched "
                 "syllable by syllable.*")
    note += ("\n\n*A visual aid, not a score: a sentence can be said naturally in more "
             "than one way.*")
    return prosody.plot(c), note


def sh_pick(lang_name: str, choice: str | None, current: str | None):
    phrase = find(LANGS[lang_name], choice or "")
    return phrase.text if phrase else current


# --------------------------------------------------------------------------
# Fluency (4/3/2) tab
# --------------------------------------------------------------------------
TOPICS = (
    "Your last holiday or trip", "Your morning routine", "A film or series you liked, and why",
    "The place you grew up", "Your plans for next weekend", "Something that annoyed you recently",
    "Your favourite meal and how to make it", "A person who influenced you",
    "Why you are learning this language", "A day that went wrong",
)
FLUENCY_COLS = ["Round", "Length", "Syllables/s", "Syllables/s speaking", "Pauses/min",
                "Avg pause", "Syllables per run", "Longest run"]

#: Round lengths for the page's scripts, from the one definition in mdd.fluency.
#: A round label reads "Round 2 · 3 min", so its second word is the round number.
_ROUND_SECS_JS = (f"const secs = ({json.dumps(fluency.ROUNDS)})"
                  "[String(rnd).split(' ')[1]] || 240;")

#: Counts down in the page itself, so nothing runs on the server; a short beep
#: marks the end (the click that starts it allows the browser to play sound).
TIMER_JS = """(rnd) => {
  """ + _ROUND_SECS_JS + """
  const el = document.querySelector('#fluency-timer');
  if (!el) return;
  clearInterval(window.__mddTimer);
  const end = Date.now() + secs * 1000;
  const tick = () => {
    const left = Math.max(0, Math.round((end - Date.now()) / 1000));
    el.textContent = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, '0')}`;
    if (left === 0) {
      clearInterval(window.__mddTimer);
      el.textContent = "Time!";
      try {
        const ctx = new AudioContext(), o = ctx.createOscillator();
        o.connect(ctx.destination); o.frequency.value = 880; o.start(); o.stop(ctx.currentTime + 0.4);
      } catch (e) {}
    }
  };
  tick();
  window.__mddTimer = setInterval(tick, 250);
}"""


TIMER_RESET_JS = """(rnd) => {
  """ + _ROUND_SECS_JS + """
  clearInterval(window.__mddTimer);
  const el = document.querySelector('#fluency-timer');
  if (el) el.textContent = `${secs / 60}:00`;
}"""


def _round_label(n: int) -> str:
    return f"Round {n} · {fluency.ROUNDS[n] // 60} min"


ROUND_LABELS = [_round_label(n) for n in fluency.ROUNDS]


def _fmt_s(x: float) -> str:
    return f"{int(x // 60)}:{round(x % 60):02d}" if x >= 60 else f"{x:.1f}s"


def _fluency_rows(rounds: dict) -> list[list]:
    rows = []
    for n in sorted(rounds, key=int):
        m = fluency.from_dict(rounds[n])
        rows.append([n, _fmt_s(m.duration), f"{m.speech_rate:.2f}", f"{m.articulation_rate:.2f}",
                     f"{m.pauses_per_min:.1f}", f"{m.mean_pause:.2f}s", f"{m.mean_run:.1f}",
                     _fmt_s(m.longest_run)])
    return rows


def _fluency_change(rounds: dict) -> str:
    done = sorted(rounds, key=int)
    if len(done) < 2:
        return ""
    a, b = fluency.from_dict(rounds[done[0]]), fluency.from_dict(rounds[done[-1]])

    def pct(x, y):
        return f"{(y - x) / x:+.0%}" if x else "n/a"
    return (f"**Round {done[0]} → {done[-1]}:** speech rate {pct(a.speech_rate, b.speech_rate)}, "
            f"pauses per minute {pct(a.pauses_per_min, b.pauses_per_min)}, syllables per run "
            f"{pct(a.mean_run, b.mean_run)}. Faster with fewer pauses is the 4/3/2 effect; "
            "the syllable count is an estimate, so compare rounds rather than read single numbers.")


def _fluency_history(lang_name: str) -> str:
    log = fluency.FluencyLog()
    if log.load_error:
        return (f"⚠️ Fluency history unreadable ({log.load_error}). The next round "
                "starts a new one; the old file is renamed and kept.")
    rows = []
    for s in log.sessions(LANGS[lang_name])[-6:]:
        r = s["rounds"]
        if "1" in r and "3" in r:
            a, b = fluency.from_dict(r["1"]), fluency.from_dict(r["3"])
            day = _dt.datetime.fromtimestamp(s["at"], _dt.UTC).astimezone().strftime("%d %b")
            rows.append(f"| {day} | {s['topic'] or '—'} | {a.speech_rate:.2f} | {b.speech_rate:.2f} |")
    if not rows:
        return ""
    return ("**Past sessions** (syllables per second)\n\n| Date | Topic | Round 1 | Round 3 |\n"
            "|---|---|---|---|\n" + "\n".join(rows) +
            "\n\nOver weeks, round 1 creeping up towards round 3 means fluency is sticking.")


def fl_new(lang_name: str):
    return (str(uuid.uuid4()), {}, gr.update(value=ROUND_LABELS[0]), [], "",
            _fluency_history(lang_name), None)


def fl_topic():
    return random.choice(TOPICS)


def fl_analyse(lang_name: str, session_id: str, rounds: dict, rnd: str, topic: str | None,
               audio_path: str | None):
    if not audio_path:
        raise gr.Error("Record the round first.")
    n = int(rnd.split()[1])
    try:
        m = fluency.measure_file(audio_path)
    except ValueError as exc:
        raise gr.Error(str(exc))
    rounds = dict(rounds or {})
    rounds[str(n)] = asdict(m)
    note = _fluency_change(rounds)
    target = fluency.ROUNDS[n]
    if m.duration < 0.5 * target:
        note = (f"*That round was {_fmt_s(m.duration)} of {target // 60} minutes — the "
                "activity works best when you fill the time.*\n\n" + note)
    log = fluency.FluencyLog()
    log.record(session_id, LANGS[lang_name], (topic or "").strip(), n, m)
    try:
        log.save()
    except OSError as exc:
        note += f"\n\n⚠️ Not saved: {exc}"
    if log.kept_as:
        note += f"\n\n⚠️ The old fluency log could not be read; it was kept as `{log.kept_as}`."
    following = ROUND_LABELS[n] if n < len(ROUND_LABELS) else rnd
    return (rounds, _fluency_rows(rounds), note, gr.update(value=following),
            _fluency_history(lang_name), None)


# --------------------------------------------------------------------------
# Level check tab
# --------------------------------------------------------------------------
def _read_upload(path: str | None) -> str:
    if not path:
        return ""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise gr.Error("Could not read that file as text.")


def _how_common(z: float) -> str:
    return ("very common" if z >= 5 else "common" if z >= 4 else
            "less common" if z >= 3 else "rare")


def lv_import(lang_name: str, path: str | None, column: int, mode: str):
    if not path:
        raise gr.Error("Choose your Anki export (.txt) first.")
    code = LANGS[lang_name]
    rows = vocab.read_anki_export(_read_upload(path))
    if not rows:
        raise gr.Error("No notes found. In Anki: File → Export → 'Notes in Plain Text'.")
    col = int(column) - 1
    words = vocab.words_from_anki(rows, col, "word" if mode.startswith("A word") else "sentence",
                                  code)
    if not words:
        raise gr.Error(f"Field {int(column)} is empty in every note — try another field number.")
    known = vocab.Known(code)
    added = known.add(words)
    known.save()
    sample = ", ".join(r[col] for r in rows[:3] if col < len(r))
    return (f"Imported {len(rows)} notes from field {int(column)} (e.g. *{sample}*): "
            f"{added} new entries, {len(known.words)} known in total.")


def lv_analyse(lang_name: str, text: str | None, path: str | None, assume_top: int):
    code = LANGS[lang_name]
    body = (text or "").strip() or _read_upload(path)
    if not body.strip():
        raise gr.Error("Paste a transcript or upload a .txt/.srt/.vtt file.")
    known = vocab.Known(code)
    rep = vocab.analyse(body, code, known.words, int(assume_top))
    if not rep.total:
        raise gr.Error("No words found in that text.")
    summary = (f"### {rep.coverage:.0%} of the words are known\n\n"
               f"{rep.known} of {rep.total} running words · **{rep.verdict()}**\n\n"
               f"Guide: about 98% for comfortable reading, 95% workable, 90–95% for listening "
               f"with support. Based on {len(known.words)} known entries"
               + (f" plus the {int(assume_top)} most common words." if assume_top else "."))
    if rep.names:
        summary += f"\n\nNot counted (probably names): {', '.join(rep.names[:12])}."
    if code == "de":
        summary += ("\n\n*German names can't be told from nouns automatically — tick them "
                    "below as known; that is how coverage research counts them.*")
    rows = [[u.form, "" if u.lemma.casefold() == u.form.casefold() else u.lemma,
             _how_common(u.zipf), u.count, u.sentence] for u in rep.unknown]
    choices = [u.form for u in rep.unknown[:40]]
    out = _TMP / f"unknown-{code}.tsv"
    out.write_text(vocab.export_tsv(rep.unknown), encoding="utf-8")
    return (summary, rows, gr.update(choices=choices, value=[]),
            str(out) if rep.unknown else None)


def lv_mark(lang_name: str, picked: list | None, text, path, assume_top):
    if not picked:
        raise gr.Error("Tick the words you already know first.")
    known = vocab.Known(LANGS[lang_name])
    known.add(picked)
    known.save()
    return lv_analyse(lang_name, text, path, assume_top)


# --------------------------------------------------------------------------
#: Gradio's audio player keeps one <audio> element per component and changes the
#: speed through `playbackRate`. Loading a new clip resets `playbackRate` to
#: `defaultPlaybackRate` (the browser's media load algorithm), which is still 1,
#: while the speed button keeps showing the old choice — so switching words at
#: 0.5x played at 1x. Setting the default alongside keeps the choice across clips.
KEEP_PLAYBACK_RATE = """<script>
(() => {
  const rate = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, "playbackRate");
  Object.defineProperty(HTMLMediaElement.prototype, "playbackRate", {
    configurable: true,
    get() { return rate.get.call(this); },
    set(v) { rate.set.call(this, v); if (v > 0) this.defaultPlaybackRate = v; },
  });
})();
</script>"""

LAUNCH_OPTIONS = {"head": KEEP_PLAYBACK_RATE}

with gr.Blocks(title="Pronunciation trainer") as demo:
    gr.Markdown("# Pronunciation trainer\nPerception training and production feedback, "
                "driven by per-language contrast tables.")

    with gr.Tab("Say it (production)"):
        with gr.Row():
            with gr.Column():
                lang = gr.Dropdown(list(LANGS), value="German", label="Language")
                with gr.Row():
                    phrase = gr.Dropdown(choices=[(p.label, p.text) for p in phrases("de")],
                                         value=None, label="Practice phrase", scale=4)
                    review_btn = gr.Button("▶ Next review", scale=1)
                practice_note = gr.Markdown()
                text = gr.Textbox(label="German sentence", value=PROFILES["de"].example, lines=2)
                with gr.Row():
                    hear = gr.Button("🔊 Hear it said correctly")
                    slow = gr.Checkbox(label="Slow", value=False)
                reference = gr.Audio(label="Correct pronunciation", interactive=False,
                                     autoplay=True, type="filepath", elem_id="sentence-reference")
                reference_note = gr.Markdown()
                audio = gr.Audio(label="Your recording", sources=["microphone", "upload"],
                                 type="filepath")
                with gr.Accordion("Advanced", open=False):
                    threshold = gr.Slider(-6.0, 0.0, value=GOP_THRESHOLD, step=0.25,
                                          label="Strictness (GOP threshold)",
                                          info="Closer to 0 flags more; more negative flags less.")
                    ipa = gr.Textbox(label="Dry run: realised IPA (skips audio and model)",
                                     placeholder="ɪk mɔktə aɪn biːɹ")
                btn = gr.Button("Analyse", variant="primary")
                status = gr.Markdown()
            with gr.Column():
                summary = gr.Markdown()
                words = gr.HighlightedText(label="Words", color_map={"check": "#f59e0b"},
                                           show_legend=False)
                table = gr.Dataframe(
                    headers=["Word", "Expected", "Heard", "Confidence", "GOP", "Tip"],
                    datatype=["str"] * 6, wrap=True,
                    label="Flagged sounds — click a row to hear that word said correctly",
                    column_widths=["13%", "10%", "10%", "12%", "7%", "48%"])
                word_reference = gr.Audio(label="Selected word, said correctly",
                                          interactive=False, autoplay=True, type="filepath",
                                          elem_id="word-reference")
                word_note = gr.Markdown()
                ipa_view = gr.Markdown()
        lang.change(on_lang_change, lang, [text, phrase, status, practice_note])
        phrase.input(pick_phrase, [lang, phrase, text], [text, practice_note])
        review_btn.click(next_review, lang, [text, practice_note]).then(
            review_status, lang, status)
        demo.load(review_status, lang, status)
        hear.click(hear_sentence, [lang, text, slow], [reference, reference_note])
        table.select(hear_word, [lang, slow, table], [word_reference, word_note])
        btn.click(run, [lang, text, audio, ipa, threshold],
                  [summary, words, table, ipa_view, status])

    with gr.Tab("Hear it (perception)"):
        gr.Markdown(
            "Identify the word you hear. The voice changes every trial — hearing a contrast "
            "from many talkers is what builds a category that survives new speakers "
            "(Logan, Lively & Pisoni 1991). Perception gains transfer to production "
            "(Bradlow et al. 1997), so this is worth doing before you record anything."
        )
        session_state = gr.State()
        trial_state = gr.State()
        with gr.Row():
            with gr.Column():
                plang = gr.Dropdown(list(PERCEPTION_LANGS), value="Danish", label="Language")
                start = gr.Button("Start session", variant="primary")
                prompt = gr.Markdown()
                stimulus = gr.Audio(label="What did you hear?", interactive=False,
                                    autoplay=True, type="filepath")
                choices = gr.Radio([], label="Your answer", visible=False)
                answer = gr.Button("Submit answer", variant="primary")
            with gr.Column():
                feedback = gr.Markdown()
                stats = gr.Markdown()
                trend = gr.Markdown()
        outs = [trial_state, stimulus, prompt, choices, feedback, stats, trend]
        start.click(start_session, plang, [session_state, *outs])
        answer.click(submit, [session_state, trial_state, choices], outs)

    with gr.Tab("Test set"):
        gr.Markdown(
            "Record each word twice: **your best attempt**, then a **deliberate error**. "
            "You may not be able to judge your own pronunciation, but you always know "
            "which version you *meant*, and that is the label. It measures how well the "
            "scorer works on your voice: how often it catches a clear error, and how often "
            "it flags your best attempt. About 20–30 minutes for a full set; stop any time. "
            "Recordings stay on this computer in `~/.mdd/testset`.")
        ts_skipped = gr.State([])
        ts_take = gr.State()
        with gr.Row():
            with gr.Column():
                ts_lang = gr.Dropdown(list(TESTSET_LANGS), value="German", label="Language")
                ts_card = gr.Markdown()
                with gr.Row():
                    ts_hear_btn = gr.Button("🔊 Hear the word")
                    ts_ref = gr.Audio(label="Correct pronunciation", interactive=False,
                                      autoplay=True, type="filepath")
                ts_audio = gr.Audio(label="Your take", sources=["microphone", "upload"],
                                    type="filepath")
                with gr.Row():
                    ts_save_btn = gr.Button("Save take ✓", variant="primary")
                    ts_skip_btn = gr.Button("Skip")
                    ts_undo_btn = gr.Button("Undo last")
            with gr.Column():
                ts_progress = gr.Markdown()
        ts_outs = [ts_take, ts_card, ts_progress, ts_audio]
        demo.load(ts_show, [ts_lang, ts_skipped], ts_outs)
        ts_lang.change(lambda lang: ([], *ts_show(lang, [])), ts_lang, [ts_skipped, *ts_outs])
        ts_save_btn.click(ts_save, [ts_lang, ts_take, ts_audio, ts_skipped], ts_outs)
        ts_skip_btn.click(ts_skip, [ts_lang, ts_take, ts_skipped], [ts_skipped, *ts_outs])
        ts_undo_btn.click(ts_undo, [ts_lang, ts_skipped], ts_outs)
        ts_hear_btn.click(ts_hear, [ts_lang, ts_take], ts_ref)

    with gr.Tab("Shadow"):
        gr.Markdown(
            "Listen to the model a few times, then record yourself saying it with the same "
            "melody and rhythm — ideally speaking along with it **through headphones**. The "
            "chart puts your pitch line over the model's. Any clip can be the model: upload "
            "a sentence from a podcast or video to shadow a real speaker.")
        with gr.Row():
            with gr.Column():
                sh_lang = gr.Dropdown(list(LANGS), value="German", label="Language")
                sh_phrase = gr.Dropdown(choices=[(p.label, p.text) for p in phrases("de")],
                                        value=None, label="Practice phrase (optional)")
                sh_text = gr.Textbox(label="Sentence", value=PROFILES["de"].example, lines=2)
                with gr.Row():
                    sh_voice = gr.Button("🔊 Use the voice as the model")
                    sh_slow = gr.Checkbox(label="Slow", value=False)
                sh_model_audio = gr.Audio(label="Model (or upload your own clip)",
                                          sources=["upload"], type="filepath",
                                          elem_id="shadow-model")
                sh_model_note = gr.Markdown()
                sh_take = gr.Audio(label="You", sources=["microphone", "upload"],
                                   type="filepath")
                sh_btn = gr.Button("Compare", variant="primary")
            with gr.Column():
                sh_plot = gr.Plot(label="Pitch")
                sh_note = gr.Markdown()
        sh_lang.change(lambda lang: (phrase_choices(lang),
                                     gr.update(value=PROFILES[LANGS[lang]].example)),
                       sh_lang, [sh_phrase, sh_text])
        sh_phrase.input(sh_pick, [sh_lang, sh_phrase, sh_text], sh_text)
        sh_voice.click(sh_model, [sh_lang, sh_text, sh_slow], [sh_model_audio, sh_model_note])
        sh_btn.click(sh_compare, [sh_model_audio, sh_take], [sh_plot, sh_note])

    with gr.Tab("Fluency 4/3/2"):
        gr.Markdown(
            "Tell the **same story three times**: in 4 minutes, then 3, then 2. Repeating it "
            "frees your attention, and the shrinking time pushes you to speak faster with "
            "fewer pauses (Nation 1989). Record each round, then analyse it — the table "
            "shows whether you got more fluent from round to round.")
        fl_id = gr.State()
        fl_rounds = gr.State({})
        with gr.Row():
            with gr.Column():
                fl_lang = gr.Dropdown(list(LANGS), value="German", label="Language")
                with gr.Row():
                    fl_topic_box = gr.Textbox(label="Topic", scale=4,
                                              placeholder="Anything you can talk about")
                    fl_suggest = gr.Button("🎲 Suggest", scale=1)
                fl_round = gr.Radio(ROUND_LABELS, value=ROUND_LABELS[0], label="Round")
                with gr.Row():
                    fl_timer_btn = gr.Button("⏱ Start timer")
                    gr.HTML('<div id="fluency-timer" style="font-size:2rem;font-weight:600;'
                            'font-variant-numeric:tabular-nums;">4:00</div>')
                fl_audio = gr.Audio(label="This round", sources=["microphone", "upload"],
                                    type="filepath")
                with gr.Row():
                    fl_go = gr.Button("Analyse round", variant="primary")
                    fl_reset = gr.Button("New session")
            with gr.Column():
                fl_table = gr.Dataframe(headers=FLUENCY_COLS, datatype=["str"] * 8,
                                        label="Your rounds", interactive=False)
                fl_note = gr.Markdown()
                fl_hist = gr.Markdown()
        fl_new_outs = [fl_id, fl_rounds, fl_round, fl_table, fl_note, fl_hist, fl_audio]
        demo.load(fl_new, fl_lang, fl_new_outs)
        fl_reset.click(fl_new, fl_lang, fl_new_outs)
        fl_lang.change(fl_new, fl_lang, fl_new_outs)
        fl_suggest.click(fl_topic, None, fl_topic_box)
        fl_timer_btn.click(None, fl_round, None, js=TIMER_JS)
        fl_round.change(None, fl_round, None, js=TIMER_RESET_JS)
        fl_go.click(fl_analyse, [fl_lang, fl_id, fl_rounds, fl_round, fl_topic_box, fl_audio],
                    [fl_rounds, fl_table, fl_note, fl_round, fl_hist, fl_audio])

    with gr.Tab("Level check"):
        gr.Markdown(
            "Is a podcast, video or article at the right level? Paste its transcript (Easy "
            "German and InnerFrench publish theirs) to see what share of the words you know. "
            "About **95–98%** is the sweet spot for learning from it.")
        with gr.Row():
            with gr.Column():
                lv_lang = gr.Dropdown(list(LANGS), value="German", label="Language")
                lv_text = gr.Textbox(label="Transcript", lines=8,
                                     placeholder="Paste text here, or upload a file below")
                lv_file = gr.File(label="…or a .txt / .srt / .vtt file", type="filepath",
                                  file_types=[".txt", ".srt", ".vtt"])
                lv_top = gr.Slider(0, 5000, value=1000, step=250,
                                   label="Also count the N most common words as known",
                                   info="Covers words like der/und/le/et that are rarely on cards.")
                lv_go = gr.Button("Check level", variant="primary")
                with gr.Accordion("Your known words: import your Anki deck", open=False):
                    gr.Markdown("In Anki: **File → Export → Notes in Plain Text (.txt)**, "
                                "untick 'include HTML' if offered. Then pick which field holds "
                                "the target-language side.")
                    lv_anki = gr.File(label="Anki export (.txt)", type="filepath",
                                      file_types=[".txt"])
                    with gr.Row():
                        lv_col = gr.Number(value=1, precision=0, minimum=1,
                                           label="Field number (1 = front)")
                        lv_mode = gr.Radio(["A word (e.g. die Tür)", "A sentence"],
                                           value="A word (e.g. die Tür)",
                                           label="That field holds")
                    lv_import_btn = gr.Button("Import")
                    lv_import_note = gr.Markdown()
            with gr.Column():
                lv_summary = gr.Markdown()
                lv_known = gr.CheckboxGroup([], label="Already know some of these? Tick them")
                lv_mark_btn = gr.Button("Mark ticked words as known")
                lv_table = gr.Dataframe(headers=["Word", "Dictionary form (guess)",
                                                 "How common", "×", "Sentence"],
                                        datatype=["str", "str", "str", "number", "str"],
                                        label="Unknown words — most useful first",
                                        interactive=False, wrap=True,
                                        column_widths=["15%", "15%", "13%", "5%", "52%"])
                lv_download = gr.File(label="Unknown words for Anki (File → Import)")
        lv_outs = [lv_summary, lv_table, lv_known, lv_download]
        lv_go.click(lv_analyse, [lv_lang, lv_text, lv_file, lv_top], lv_outs)
        lv_mark_btn.click(lv_mark, [lv_lang, lv_known, lv_text, lv_file, lv_top], lv_outs)
        lv_import_btn.click(lv_import, [lv_lang, lv_anki, lv_col, lv_mode], lv_import_note)

    with gr.Tab("Coverage"):
        gr.Markdown("### What each language's G2P can and cannot be trusted with\n"
                    "Run `python -m mdd.validate` to re-check these against espeak-ng.")
        for code, profile in PROFILES.items():
            ready = "trains perception" if profile.hvpt_ready else "**perception disabled**"
            body = [f"## {profile.name} (`{code}`) — {ready}"]
            if profile.g2p_caveat:
                body.append(f"> {profile.g2p_caveat}")
            if profile.hvpt_caveat:
                body.append(f"> {profile.hvpt_caveat}")
            body.append("")
            for c in profile.contrasts:
                body.append(f"- **{c.label}** — {c.why}")
            gr.Markdown("\n".join(body))

if __name__ == "__main__":
    demo.launch(**LAUNCH_OPTIONS)
