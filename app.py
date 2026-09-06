"""Gradio UI: type a German sentence, record it, see flagged phones and tips."""
from mdd._utf8 import ensure_utf8_mode

ensure_utf8_mode()

import gradio as gr  # noqa: E402

from mdd.pipeline import GOP_THRESHOLD, analyse  # noqa: E402

_recognizer = None


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        from mdd.recognizer import PhoneRecognizer
        _recognizer = PhoneRecognizer()
    return _recognizer


def run(text: str, audio_path: str | None, ipa: str, threshold: float):
    text = text.strip()
    if not text:
        raise gr.Error("Enter a German sentence first.")
    if ipa.strip():
        rep = analyse(text, realized_ipa=ipa.strip(), threshold=threshold)
    elif audio_path:
        rep = analyse(text, audio_path, get_recognizer(), threshold=threshold)
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


with gr.Blocks(title="German MDD") as demo:
    gr.Markdown("# German pronunciation check\nType a sentence, record yourself saying it, and get per-sound feedback.")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="German sentence", value="Ich möchte ein Bier", lines=2)
            audio = gr.Audio(label="Your recording", sources=["microphone", "upload"], type="filepath")
            with gr.Accordion("Advanced", open=False):
                threshold = gr.Slider(-6.0, 0.0, value=GOP_THRESHOLD, step=0.25, label="Strictness (GOP threshold)",
                                      info="Closer to 0 flags more; more negative flags less.")
                ipa = gr.Textbox(label="Dry run: realised IPA (skips audio and model)",
                                 placeholder="ɪk mɔktə aɪn biːɹ")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            summary = gr.Markdown()
            words = gr.HighlightedText(label="Words", color_map={"check": "#f59e0b"}, show_legend=False)
            table = gr.Dataframe(headers=["Word", "Expected", "Heard", "Op", "GOP", "Tip"],
                                 datatype=["str"] * 6, label="Flagged sounds", wrap=True,
                                 column_widths=["14%", "11%", "11%", "8%", "8%", "48%"])
            ipa_view = gr.Markdown()
    btn.click(run, [text, audio, ipa, threshold], [summary, words, table, ipa_view])

if __name__ == "__main__":
    demo.launch()
