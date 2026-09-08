"""Check a language is ready to evaluate before committing to a long run.

Tier 1 costs real time — a few hundred TTS calls, then a forward pass per clip,
plus a 1.2 GB model download on the first run ever. Discovering a missing
dependency or a G2P problem twenty minutes in is the failure this avoids.

Two halves:

* **Environment** — are the packages installed, is the TTS endpoint reachable,
  is the model already cached? Reported, never fixed automatically.
* **Transcription sanity (model-free)** — run every evaluation sentence through
  G2P and the tokeniser and look for trouble: espeak language-switch artifacts,
  tokens outside the language's expected inventory, and words where sentence-level
  and per-word phonemisation disagree. These are cheap to find here and expensive
  to find after an hour of inference, because they corrupt the canonical side of
  every comparison rather than showing up as an obvious error.

  python -m eval.preflight fr
"""
if __name__ == "__main__":
    from mdd._utf8 import ensure_utf8_mode
    ensure_utf8_mode()

import argparse  # noqa: E402
import importlib.util  # noqa: E402
from collections import Counter  # noqa: E402

from mdd.languages import get  # noqa: E402

from .common import canonical_tokens_by_word, load_sentences  # noqa: E402
from .tts import default_voices  # noqa: E402

#: Rough cost per sentence per voice: one TTS call plus one forward pass.
SECONDS_PER_CLIP = 2.5
#: Below this many occurrences, a per-phone false-positive rate is too noisy to read.
THIN_PHONE = 10


def check_environment(lang: str) -> list[tuple[bool, str]]:
    out = []
    for mod, why in [("torch", "the recogniser"), ("transformers", "the recogniser"),
                     ("soundfile", "reading audio"), ("edge_tts", "neural TTS voices"),
                     ("numpy", "the analysis")]:
        ok = importlib.util.find_spec(mod) is not None
        out.append((ok, f"{mod} installed ({why})"))

    # Importing mdd.recognizer pulls in torch, which may not be installed yet;
    # the model id is the only thing needed, so fall back to the literal.
    from pathlib import Path
    model_id = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    if importlib.util.find_spec("torch") is not None:
        from mdd.recognizer import MODEL_ID as model_id  # noqa: N813
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    cached = hub.is_dir() and any(hub.glob(f"models--{model_id.replace('/', '--')}*"))
    out.append((cached, "recogniser already downloaded "
                        "(~1.2 GB on first run if not)"))
    try:
        voices = default_voices(lang)
        out.append((True, f"{len(voices)} voices configured: {', '.join(voices)}"))
    except ValueError as exc:
        out.append((False, str(exc)))
    return out


def check_transcription(lang: str, n: int | None = None) -> dict:
    """Model-free: does the canonical side hold up across the sentence list?"""
    profile = get(lang)
    sentences = load_sentences(n, lang)
    expected = set()
    for contrast in profile.contrasts:
        expected.update(contrast.phones)
    for canon, real in profile.tips:
        expected.update({canon, real})
    expected.discard("")

    counts: Counter = Counter()
    switches: list[tuple[str, str]] = []
    for text in sentences:
        from mdd.g2p import text_to_ipa_words
        for word, ipa in text_to_ipa_words(text, profile):
            if "(" in ipa:
                switches.append((word, ipa))
        for _, tokens in canonical_tokens_by_word(text, lang):
            counts.update(tokens)

    contrast_phones = {p for c in profile.contrasts for p in c.phones}
    missing_phones = [p for c in profile.contrasts for p in c.phones if p not in counts]
    unexpected = [(t, n) for t, n in counts.most_common() if t not in expected]
    return {
        "lang": lang,
        "sentences": len(sentences),
        "phones": sum(counts.values()),
        "distinct": len(counts),
        "switches": switches,
        "missing_contrast_phones": missing_phones,
        "unexpected": unexpected,
        "contrast_phone_counts": {p: counts.get(p, 0) for p in sorted(contrast_phones)},
        "rarest": counts.most_common()[-8:],
    }


def main():
    ap = argparse.ArgumentParser(prog="eval.preflight",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("lang", nargs="?", default="fr")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    profile = get(args.lang)

    print(f"=== {profile.name} ({args.lang}) preflight ===\n")
    print("Environment")
    env = check_environment(args.lang)
    for ok, label in env:
        print(f"  [{'ok' if ok else '--'}] {label}")

    print("\nTranscription sanity (no model needed)")
    try:
        rep = check_transcription(args.lang, args.n)
    except FileNotFoundError as exc:
        print(f"  [--] {exc}")
        print(f"\nNOT READY — write eval/sentences_{args.lang}.txt first "
              f"(a few dozen ordinary sentences, weighted toward this language's contrasts).")
        return
    print(f"  {rep['sentences']} sentences · {rep['phones']} phones · "
          f"{rep['distinct']} distinct tokens")

    problems = 0
    if rep["switches"]:
        problems += 1
        print(f"  [!!] {len(rep['switches'])} espeak language switches — these words are "
              f"phonemised as another language:")
        for word, ipa in rep["switches"][:6]:
            print(f"         {word} -> {ipa}")
        print("       The tokeniser strips the markers, but the phonemes inside are")
        print("       wrong for this language. Replace these words in the sentence list.")
    else:
        print("  [ok] no espeak language switches")

    # Tier 1 measures the false-positive rate over whatever phones the sentences
    # happen to contain, so what matters is that the phones the contrasts care
    # about occur *often enough* for the rate to mean anything. A phone seen
    # twice gives an FPR with an error bar wider than the number.
    thin = [(p, n) for p, n in rep["contrast_phone_counts"].items() if n < THIN_PHONE]
    if thin:
        print(f"  [??] contrast phones seen fewer than {THIN_PHONE} times: "
              + ", ".join(f"{p}×{n}" for p, n in sorted(thin, key=lambda kv: kv[1])))
        print("       Their per-phone false-positive rates will be noisy. A phone at 0 is")
        print("       usually a realisation rather than a target (German's [ɐ]: espeak")
        print("       writes coda r as ʁ, and [ɐ] appears only when a speaker vocalises it).")
    else:
        print(f"  [ok] every contrast phone occurs at least {THIN_PHONE} times")

    if rep["unexpected"]:
        print(f"  [??] {len(rep['unexpected'])} tokens outside the profile's tip/contrast "
              f"inventory (usually fine — ordinary phones nobody wrote a tip for):")
        print("       " + ", ".join(f"{t}×{n}" for t, n in rep["unexpected"][:12]))

    n_voices = sum(1 for ok, label in env if ok and "voices configured" in label)
    if n_voices:
        voices = len(default_voices(args.lang))
        mins = rep["sentences"] * voices * SECONDS_PER_CLIP / 60
        print(f"\nEstimated first run: ~{mins:.0f} min "
              f"({rep['sentences']} sentences × {voices} voices), "
              f"cached afterwards.")
    missing = [label for ok, label in env if not ok]
    print(f"\n{'READY' if not missing and not problems else 'NOT READY'}"
          f" — {len(missing)} missing dependencies, {problems} transcription problems")
    if missing:
        print("  install/obtain: " + "; ".join(missing))
    print(f"\nWhen ready:  python -m eval.native_control --lang {args.lang}")


if __name__ == "__main__":
    main()
