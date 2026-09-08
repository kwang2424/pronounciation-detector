"""Shared helpers: sentence list, on-disk caches, threshold sweep."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
THRESHOLDS = [-8.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0]


DEFAULT_LANG = "de"


def load_sentences(n: int | None = None, lang: str = DEFAULT_LANG) -> list[str]:
    path = ROOT / f"sentences_{lang}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"no evaluation sentences for {lang!r} — expected {path.name}. "
            f"Tier 1 needs a few dozen ordinary sentences in the target language.")
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    lines = [l for l in lines if l and not l.startswith("#")]
    return lines[:n] if n else lines


def results_name(name: str, lang: str = DEFAULT_LANG) -> str:
    """German keeps the original paths so its committed baseline stays comparable;
    other languages get a subdirectory."""
    return name if lang == DEFAULT_LANG else f"{lang}/{name}"


def key(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


_rec = None


def recognizer():
    global _rec
    if _rec is None:
        from mdd.recognizer import PhoneRecognizer
        _rec = PhoneRecognizer()
    return _rec


def analyse_cached(text: str, wav_path: Path, json_path: Path,
                   lang: str = DEFAULT_LANG) -> dict:
    """Run the pipeline once per clip; later threshold sweeps reuse the raw per-phone GOPs.
    The cache file name carries the pipeline version so rule changes trigger a re-run."""
    from mdd.pipeline import VERSION, analyse
    json_path = json_path.with_suffix(f".v{VERSION}.json")
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            json_path.unlink()   # truncated by an interrupted run; recompute
    rep = analyse(text, str(wav_path), recognizer(), lang=lang)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    return rep


def is_flagged(p: dict, tau: float) -> bool:
    """Same rule as mdd.pipeline, re-evaluated at an arbitrary GOP threshold."""
    from mdd.pipeline import INS_MIN_PROB
    if p["op"] == "ins":
        return p.get("conf") is None or p["conf"] >= INS_MIN_PROB
    return p["op"] != "match" and (p["gop"] is None or p["gop"] < tau)


def canonical_tokens_by_word(text: str, lang: str = DEFAULT_LANG) -> list[tuple[str, list[str]]]:
    from mdd.g2p import text_to_ipa_words
    from mdd.languages import get
    from mdd.normalize import tokenize
    profile = get(lang)
    return [(w, tokenize(ipa, profile)) for w, ipa in text_to_ipa_words(text, profile)]


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def write_results(name: str, data: dict, markdown: str) -> Path:
    (RESULTS / name).parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    path = RESULTS / f"{name}.md"
    path.write_text(markdown, encoding="utf-8")
    return path
