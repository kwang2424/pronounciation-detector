"""Gradio UI: production feedback (record and be scored) plus perception training."""
from mdd._utf8 import ensure_utf8_mode

ensure_utf8_mode()

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import gradio as gr  # noqa: E402

from mdd.hvpt import PerceptionUnavailable, Session  # noqa: E402
from mdd.languages import PROFILES  # noqa: E402
from mdd.pipeline import GOP_THRESHOLD, analyse  # noqa: E402

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
    text = text.strip()
    if not text:
        raise gr.Error(f"Enter a {lang_name} sentence first.")
    if ipa.strip():
        rep = analyse(text, realized_ipa=ipa.strip(), threshold=threshold, lang=code)
    elif audio_path:
        rep = analyse(text, audio_path, get_recognizer(), threshold=threshold, lang=code)
    else:
        raise gr.Error("Record or upload audio, or paste realised IPA for a dry run.")

    flagged = [p for p in rep["phones"] if p["flagged"]]
    bad_words = {p["word"] for p in flagged}
    highlighted = [(w, "check" if w in bad_words else None) for w in _words(text)]

    rows = [[p["word"], p["canonical"] or "—", p["realized"] or "—", p["op"],
             "" if p["gop"] is None else f"{p['gop']:.2f}", p["tip"]] for p in flagged]
    score = rep["overall"]
    summary = (f"**Score: {score:.0%}** · {len(flagged)} issue{'s' if len(flagged) != 1 else ''} "
               f"across {len(bad_words)} word{'s' if len(bad_words) != 1 else ''}"
               if flagged else f"**Score: {score:.0%}** · no issues detected")
    ipa_view = f"**Expected:** `{rep['canonical']}`\n\n**Heard:** `{rep['realized']}`"
    return summary, highlighted, rows, ipa_view


def _words(text: str) -> list[str]:
    return [w for w in (t.strip(".,;:!?\"'()") for t in text.split()) if w]


def on_lang_change(lang_name: str):
    profile = PROFILES[LANGS[lang_name]]
    return gr.update(value=profile.example, label=f"{profile.name} sentence")


# --------------------------------------------------------------------------
# Perception tab
# --------------------------------------------------------------------------
def start_session(lang_name: str):
    code = PERCEPTION_LANGS[lang_name]
    try:
        session = Session(code)
    except PerceptionUnavailable as exc:
        raise gr.Error(str(exc))
    note = ""
    if session.skipped:
        skipped = ", ".join(f"`{k}`" for k in session.skipped)
        note = (f"\n\n*Not trained: {skipped} — the synthesiser cannot render "
                f"{'them' if len(session.skipped) > 1 else 'it'} distinctly, so a trial "
                f"would be unanswerable.*")
    return (session, *_serve(session, f"Session started · {len(session.contrast_ids)} "
                                      f"contrasts{note}"))


def _serve(session: Session, message: str):
    """Build a new trial and the UI updates that present it."""
    trial = session.next_trial()
    contrast = session.profile.contrast(trial.contrast_id)
    path = trial.render(_TMP / f"trial-{len(session.history)}-{trial.talker.variant}.wav")
    heading = f"### {contrast.label}\n{contrast.why}"
    return (trial, str(path), heading,
            gr.update(choices=list(trial.choices), value=None, visible=True),
            message, _stats(session))


def submit(session: Session, trial, choice: str | None):
    if session is None or trial is None:
        raise gr.Error("Start a session first.")
    if not choice:
        raise gr.Error("Pick what you heard.")
    correct = session.record(trial, choice)
    contrast = session.profile.contrast(trial.contrast_id)
    if correct:
        message = f"✅ **{trial.target}** — correct."
    else:
        message = (f"❌ You picked **{choice}**; it was **{trial.target}**.\n\n"
                   f"*{contrast.tip}*")
    return (*_serve(session, message),)


def _stats(session: Session) -> str:
    if not session.history:
        return "_No trials yet._"
    lines = [f"**{session.accuracy():.0%}** over {len(session.history)} trials · "
             f"{session.difficulty} choices per trial", "", "| Contrast | Trials | Correct |",
             "|---|---|---|"]
    for entry in session.report()["contrasts"].values():
        lines.append(f"| {entry['label']} | {entry['seen']} | {entry['accuracy']:.0%} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
with gr.Blocks(title="Pronunciation trainer") as demo:
    gr.Markdown("# Pronunciation trainer\nPerception training and production feedback, "
                "driven by per-language contrast tables.")

    with gr.Tab("Say it (production)"):
        with gr.Row():
            with gr.Column():
                lang = gr.Dropdown(list(LANGS), value="German", label="Language")
                text = gr.Textbox(label="German sentence", value=PROFILES["de"].example, lines=2)
                audio = gr.Audio(label="Your recording", sources=["microphone", "upload"],
                                 type="filepath")
                with gr.Accordion("Advanced", open=False):
                    threshold = gr.Slider(-6.0, 0.0, value=GOP_THRESHOLD, step=0.25,
                                          label="Strictness (GOP threshold)",
                                          info="Closer to 0 flags more; more negative flags less.")
                    ipa = gr.Textbox(label="Dry run: realised IPA (skips audio and model)",
                                     placeholder="ɪk mɔktə aɪn biːɹ")
                btn = gr.Button("Analyse", variant="primary")
            with gr.Column():
                summary = gr.Markdown()
                words = gr.HighlightedText(label="Words", color_map={"check": "#f59e0b"},
                                           show_legend=False)
                table = gr.Dataframe(headers=["Word", "Expected", "Heard", "Op", "GOP", "Tip"],
                                     datatype=["str"] * 6, label="Flagged sounds", wrap=True,
                                     column_widths=["14%", "11%", "11%", "8%", "8%", "48%"])
                ipa_view = gr.Markdown()
        lang.change(on_lang_change, lang, text)
        btn.click(run, [lang, text, audio, ipa, threshold], [summary, words, table, ipa_view])

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
        outs = [trial_state, stimulus, prompt, choices, feedback, stats]
        start.click(start_session, plang, [session_state, *outs])
        answer.click(submit, [session_state, trial_state, choices], outs)

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
    demo.launch()
