"""Spaced review of the words you mispronounce.

A production report on its own is forgotten the moment you type the next
sentence. The spacing effect is among the most robust findings in learning
research (Cepeda et al. 2006): practice spread over growing intervals is
retained far better than the same practice massed together. So a word the
scorer flags joins a review queue, and comes back at increasing gaps each time
you say it cleanly — ten minutes, then a day, three days, a week, about two
weeks, about a month — and drops back to the start whenever it is flagged again.
A word said cleanly at its longest gap is counted as learned and leaves the queue.

Three rules keep it honest about what the scorer can tell:

- **"noisy" flags neither add nor fail a word.** A substitution the recogniser
  also makes on native speech (German d→t, or ö heard as [ɛ]) is weak evidence,
  so it should not send a word back to day one. The same goes for an inserted
  sound, which the native control produced 26 times.
- **Saying a word before it is due does not advance it.** The gap is the point;
  a clean repeat a minute after a failure shows short-term memory, not learning.
- **Review is by sentence, not by lone word.** Each item remembers the sentence it
  was flagged in: sounds change with their neighbours, and a word said alone is
  said more carefully than in speech.

Every analysed attempt also logs which expected sounds occurred and which were
flagged, so `trouble_sounds` can report rates ("ö: 4 of 6") rather than raw
counts, which would simply favour the most common sounds.

Stored at ~/.mdd/review.json (override with `$MDD_REVIEW`), written atomically; an
unreadable file is reported and treated as empty, as for perception progress.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .progress import write_json_atomic
from .reliability import reliability

SCHEMA_VERSION = 1
MINUTE, DAY = 60.0, 86400.0
#: Gap before the next review after 0, 1, 2 ... clean reviews in a row.
INTERVALS = (10 * MINUTE, 1 * DAY, 3 * DAY, 7 * DAY, 16 * DAY, 35 * DAY)
#: Attempts kept for sound statistics; older ones are dropped.
MAX_ATTEMPTS = 500
SOUND_WINDOW = 30 * DAY
#: A sound must have come up in this many separate attempts before a rate for it
#: is shown: one sentence with three ich-Laute is one attempt, not three.
MIN_SOUND_ATTEMPTS = 3


def default_path() -> Path:
    env = os.environ.get("MDD_REVIEW")
    return Path(env) if env else Path.home() / ".mdd" / "review.json"


def word_key(word: str) -> str:
    return word.strip(".,;:!?\"'()«»„“”").casefold()


def gap(seconds: float) -> str:
    """'10 minutes', '1 day', '16 days'."""
    if seconds < DAY:
        n, unit = max(1, round(seconds / MINUTE)), "minute"
        if n >= 60:
            n, unit = round(n / 60), "hour"
    else:
        n, unit = round(seconds / DAY), "day"
    return f"{n} {unit}{'s' if n != 1 else ''}"


@dataclass
class Item:
    word: str
    sentence: str
    box: int = 0
    due: float = 0.0
    flagged: int = 0
    clean: int = 0
    learned: bool = False
    #: "expected→heard" -> times flagged that way.
    errors: dict[str, int] = field(default_factory=dict)

    def is_due(self, now: float) -> bool:
        return not self.learned and self.due <= now

    @property
    def worst(self) -> str | None:
        return max(self.errors, key=self.errors.get) if self.errors else None


@dataclass
class Outcome:
    """What one attempt did to the queue, for telling the learner."""
    added: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    advanced: list[tuple[str, float]] = field(default_factory=list)
    learned: list[str] = field(default_factory=list)
    early: list[str] = field(default_factory=list)
    noisy: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.added:
            parts.append("added to review: " + ", ".join(f"*{w}*" for w in self.added))
        if self.failed:
            parts.append("back to the start: " + ", ".join(f"*{w}*" for w in self.failed))
        for w, secs in self.advanced:
            parts.append(f"*{w}* ✓ — next review in {gap(secs)}")
        if self.learned:
            parts.append("learned 🎉: " + ", ".join(f"*{w}*" for w in self.learned))
        if self.early:
            parts.append(", ".join(f"*{w}*" for w in self.early) +
                         " clean, but not due yet, so the schedule is unchanged")
        if self.noisy:
            parts.append(", ".join(f"*{w}*" for w in self.noisy) +
                         " only had noisy flags — not counted either way")
        return "**Review:** " + " · ".join(parts) if parts else ""


class Review:
    def __init__(self, path: Path | str | None = None, data: dict | None = None,
                 load_error: str | None = None):
        self.path = Path(path) if path is not None else default_path()
        self._data = data if data is not None else {"version": SCHEMA_VERSION,
                                                    "languages": {}}
        self.load_error = load_error

    @classmethod
    def load(cls, path: Path | str | None = None) -> Review:
        p = Path(path) if path is not None else default_path()
        if not p.exists():
            return cls(p)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return cls(p, load_error=f"could not read {p}: {exc}")
        if not isinstance(data, dict) or data.get("version") != SCHEMA_VERSION:
            return cls(p, load_error=f"{p} is not a version-{SCHEMA_VERSION} review file")
        return cls(p, data=data)

    def save(self) -> Path:
        return write_json_atomic(self.path, self._data)

    # ---------------------------------------------------------------- storage
    def _lang(self, lang: str) -> dict:
        return self._data.setdefault("languages", {}).setdefault(
            lang, {"items": {}, "attempts": []})

    def items(self, lang: str) -> dict[str, Item]:
        return {k: Item(**v) for k, v in self._lang(lang)["items"].items()}

    def _put(self, lang: str, key: str, item: Item) -> None:
        self._lang(lang)["items"][key] = item.__dict__

    # ---------------------------------------------------------------- writing
    def record(self, lang: str, sentence: str, phones: list[dict],
               now: float | None = None) -> Outcome:
        """Update the queue and sound log from one analysed attempt.

        `phones` is `analyse()["phones"]`. Only call this for real recordings: a
        dry run with typed IPA is not evidence about your pronunciation.
        """
        now = time.time() if now is None else now
        items = self.items(lang)
        out = Outcome()

        # Grouped by normalised word, so "Die ... die" is one word said twice and
        # cannot advance twice in one attempt.
        by_word: dict[str, list[dict]] = {}
        shown: dict[str, str] = {}
        seen, flagged = Counter(), Counter()
        for p in phones:
            key = word_key(p.get("word") or "")
            if key:
                by_word.setdefault(key, []).append(p)
                shown.setdefault(key, p["word"].strip(".,;:!?\"'()«»„“”"))
            if p.get("canonical"):
                seen[p["canonical"]] += 1
            if p.get("flagged") and not self._noisy(lang, p) and p.get("canonical"):
                flagged[p["canonical"]] += 1

        for key, ps in by_word.items():
            word = shown[key]
            flags = [p for p in ps if p.get("flagged")]
            strong = [p for p in flags if not self._noisy(lang, p)]
            item = items.get(key)
            if strong:
                if item is None:
                    item = Item(word=word, sentence=sentence)
                    out.added.append(word)
                else:
                    out.failed.append(word)
                item.sentence, item.box, item.learned = sentence, 0, False
                item.due = now + INTERVALS[0]
                item.flagged += 1
                for p in strong:
                    err = f"{p.get('canonical') or '∅'}→{p.get('realized') or '∅'}"
                    item.errors[err] = item.errors.get(err, 0) + 1
                self._put(lang, key, item)
            elif flags:
                if item is not None and not item.learned:
                    out.noisy.append(word)
            elif item is not None and not item.learned:
                if not item.is_due(now):
                    out.early.append(word)
                    continue
                item.clean += 1
                item.box += 1
                if item.box >= len(INTERVALS):
                    item.learned = True
                    out.learned.append(word)
                else:
                    item.due = now + INTERVALS[item.box]
                    out.advanced.append((word, INTERVALS[item.box]))
                self._put(lang, key, item)

        attempts = self._lang(lang)["attempts"]
        attempts.append({"at": now, "sentence": sentence,
                         "seen": dict(seen), "flagged": dict(flagged)})
        del attempts[:-MAX_ATTEMPTS]
        return out

    @staticmethod
    def _noisy(lang: str, p: dict) -> bool:
        # An inserted sound has no expected phone to judge it by, and the native
        # control produced 26 false insertions, so it is weak evidence too.
        if not p.get("canonical"):
            return True
        return reliability(p["canonical"], lang, p.get("realized")).level == "noisy"

    # ---------------------------------------------------------------- reading
    def due(self, lang: str, now: float | None = None) -> list[Item]:
        now = time.time() if now is None else now
        return sorted((i for i in self.items(lang).values() if i.is_due(now)),
                      key=lambda i: i.due)

    def next_review(self, lang: str, now: float | None = None) -> tuple[str, list[str]] | None:
        """The sentence covering the most due words (oldest first on a tie)."""
        due = self.due(lang, now)
        if not due:
            return None
        groups: dict[str, list[Item]] = {}
        for item in due:
            groups.setdefault(item.sentence, []).append(item)
        sentence, group = max(groups.items(), key=lambda kv: (len(kv[1]), -kv[1][0].due))
        return sentence, [i.word for i in group]

    def upcoming(self, lang: str, now: float | None = None) -> float | None:
        """Seconds until the next item falls due, if none are due now."""
        now = time.time() if now is None else now
        waits = [i.due - now for i in self.items(lang).values() if not i.learned]
        return min(waits) if waits and min(waits) > 0 else None

    def counts(self, lang: str, now: float | None = None) -> dict[str, int]:
        now = time.time() if now is None else now
        items = self.items(lang).values()
        return {"due": sum(i.is_due(now) for i in items),
                "tracked": sum(not i.learned for i in items),
                "learned": sum(i.learned for i in items)}

    def trouble_sounds(self, lang: str, now: float | None = None,
                       window: float = SOUND_WINDOW) -> list[tuple[str, int, int]]:
        """(phone, times flagged, times said) over the window, worst rate first."""
        now = time.time() if now is None else now
        seen, flagged, attempts = Counter(), Counter(), Counter()
        for a in self._lang(lang)["attempts"]:
            if now - a.get("at", 0) <= window:
                seen.update(a.get("seen") or {})
                flagged.update(a.get("flagged") or {})
                attempts.update((a.get("seen") or {}).keys())
        rows = [(ph, flagged[ph], n) for ph, n in seen.items()
                if attempts[ph] >= MIN_SOUND_ATTEMPTS and flagged[ph]]
        return sorted(rows, key=lambda r: (-r[1] / r[2], -r[1]))

    def attempts(self, lang: str) -> int:
        return len(self._lang(lang)["attempts"])

    def times_said(self, lang: str) -> Counter:
        """How often each sentence has been recorded, to vary suggestions."""
        return Counter(a.get("sentence") for a in self._lang(lang)["attempts"])
