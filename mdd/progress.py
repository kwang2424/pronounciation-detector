"""Durable perception-training history: what you've practised and how it's going.

Without this a session's accuracy vanishes when the app closes, which hides the
one thing worth knowing — whether a contrast improved over weeks. HVPT gains
accumulate across distributed sessions (Logan, Lively & Pisoni ran 15 sessions
over three weeks), so the unit of progress is the training block, not the sitting.

Sessions are the single source of truth and are keyed by id, so a session saved
repeatedly — the app saves after every answer, since a training session is
normally closed rather than formally ended — overwrites its own record instead of
adding to it. Lifetime totals are derived by summing, never accumulated in place,
which is what keeps repeated saves from inflating your history.

Stored as one JSON file, written atomically (temp file + replace) so an
interrupted write cannot leave a half-file behind. A corrupt or unreadable store
is never fatal: it is reported and treated as empty, because losing practice
history should not stop you practising.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 2
#: Keep the file bounded. Trimmed sessions are folded into `archived` totals, so
#: lifetime figures stay correct even once the per-session detail is gone.
MAX_SESSIONS = 200
_CONFUSION_SEP = "|"


def default_path() -> Path:
    """`$MDD_PROGRESS` if set, else ~/.mdd/progress.json (works on Windows too)."""
    env = os.environ.get("MDD_PROGRESS")
    return Path(env) if env else Path.home() / ".mdd" / "progress.json"


@dataclass
class ContrastHistory:
    seen: int = 0
    correct: int = 0
    confusions: dict[tuple[str, str], int] = field(default_factory=dict)
    last_practiced: float | None = None

    @property
    def accuracy(self) -> float:
        return self.correct / self.seen if self.seen else 0.0


@dataclass
class SessionRecord:
    id: str
    at: float
    trials: int
    correct: int
    #: contrast id -> (seen, correct) within that session
    contrasts: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.trials if self.trials else 0.0


def _split(key: str) -> tuple[str, str]:
    target, _, picked = key.partition(_CONFUSION_SEP)
    return target, picked


class Progress:
    """Per-language perception history, loaded from and saved to a JSON file."""

    def __init__(self, path: Path | str | None = None, data: dict | None = None,
                 load_error: str | None = None):
        self.path = Path(path) if path is not None else default_path()
        self._data: dict = data if data is not None else {"version": SCHEMA_VERSION,
                                                          "languages": {}}
        #: Set when an existing file could not be read; the store is empty but usable.
        self.load_error = load_error

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: Path | str | None = None) -> Progress:
        p = Path(path) if path is not None else default_path()
        if not p.exists():
            return cls(p)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return cls(p, load_error=f"could not read {p}: {exc}")
        if not isinstance(data, dict) or "languages" not in data:
            return cls(p, load_error=f"{p} is not a progress file")
        version = data.get("version")
        if version != SCHEMA_VERSION:
            # An unknown version is left on disk untouched rather than rewritten
            # in a shape its writer may not expect.
            return cls(p, load_error=(f"{p} has schema version {version!r}, "
                                      f"expected {SCHEMA_VERSION}"))
        return cls(p, data=data)

    # ---------------------------------------------------------------- reading
    def _lang(self, lang: str) -> dict:
        return self._data.setdefault("languages", {}).setdefault(
            lang, {"sessions": [], "archived": {}})

    def history(self, lang: str) -> dict[str, ContrastHistory]:
        """Lifetime per-contrast totals, summed over sessions plus archived totals."""
        out: dict[str, ContrastHistory] = {}
        entry = self._lang(lang)

        for cid, raw in (entry.get("archived") or {}).items():
            h = out.setdefault(cid, ContrastHistory())
            h.seen += int(raw.get("seen", 0))
            h.correct += int(raw.get("correct", 0))
            for key, n in (raw.get("confusions") or {}).items():
                h.confusions[_split(key)] = h.confusions.get(_split(key), 0) + int(n)
            at = raw.get("last_practiced")
            if at is not None and (h.last_practiced is None or at > h.last_practiced):
                h.last_practiced = at

        for rec in entry.get("sessions", []):
            at = rec.get("at", 0.0)
            for cid, c in (rec.get("contrasts") or {}).items():
                h = out.setdefault(cid, ContrastHistory())
                h.seen += int(c.get("seen", 0))
                h.correct += int(c.get("correct", 0))
                for key, n in (c.get("confusions") or {}).items():
                    h.confusions[_split(key)] = h.confusions.get(_split(key), 0) + int(n)
                if h.last_practiced is None or at > h.last_practiced:
                    h.last_practiced = at
        return out

    def sessions(self, lang: str) -> list[SessionRecord]:
        out = []
        for raw in self._lang(lang).get("sessions", []):
            contrasts = {cid: (int(c.get("seen", 0)), int(c.get("correct", 0)))
                         for cid, c in (raw.get("contrasts") or {}).items()}
            out.append(SessionRecord(id=raw.get("id", ""), at=raw.get("at", 0.0),
                                     trials=int(raw.get("trials", 0)),
                                     correct=int(raw.get("correct", 0)),
                                     contrasts=contrasts))
        return out

    def is_empty(self, lang: str | None = None) -> bool:
        if lang is None:
            return not self._data.get("languages")
        entry = self._lang(lang)
        return not entry.get("sessions") and not entry.get("archived")

    # ---------------------------------------------------------------- writing
    def record_session(self, lang: str, report: dict, session_id: str,
                       now: float | None = None) -> None:
        """Upsert one session's `Session.report()` by id.

        Idempotent: calling it repeatedly for the same `session_id` replaces that
        session's record rather than adding to it, so saving after every answer
        does not inflate lifetime totals.
        """
        contrasts = report.get("contrasts") or {}
        now = time.time() if now is None else now
        rows: dict[str, dict] = {}
        trials = correct = 0
        for cid, c in contrasts.items():
            seen, ok = int(c.get("seen", 0)), int(c.get("correct", 0))
            if not seen:
                continue
            trials += seen
            correct += ok
            row = {"seen": seen, "correct": ok}
            worst = c.get("worst_confusion")
            if worst:
                row["confusions"] = {
                    f"{worst['target']}{_CONFUSION_SEP}{worst['picked']}": int(worst.get("count", 0))
                }
            rows[cid] = row
        if not rows:
            return

        entry = self._lang(lang)
        sessions = entry.setdefault("sessions", [])
        record = {"id": session_id, "at": now, "trials": trials,
                  "correct": correct, "contrasts": rows}
        for i, existing in enumerate(sessions):
            if existing.get("id") == session_id:
                record["at"] = existing.get("at", now)   # keep the session's start time
                sessions[i] = record
                break
        else:
            sessions.append(record)

        if len(sessions) > MAX_SESSIONS:
            self._archive(entry, sessions[:-MAX_SESSIONS])
            del sessions[:-MAX_SESSIONS]

    @staticmethod
    def _archive(entry: dict, old: list[dict]) -> None:
        """Fold trimmed sessions into running totals so lifetime figures survive."""
        archived = entry.setdefault("archived", {})
        for rec in old:
            for cid, c in (rec.get("contrasts") or {}).items():
                row = archived.setdefault(cid, {"seen": 0, "correct": 0, "confusions": {}})
                row["seen"] += int(c.get("seen", 0))
                row["correct"] += int(c.get("correct", 0))
                for key, n in (c.get("confusions") or {}).items():
                    conf = row.setdefault("confusions", {})
                    conf[key] = int(conf.get(key, 0)) + int(n)
                at = rec.get("at")
                if at is not None and at > row.get("last_practiced", 0):
                    row["last_practiced"] = at

    def save(self) -> Path:
        """Atomic write: temp file in the same directory, then replace."""
        self._data["version"] = SCHEMA_VERSION
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".progress-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return self.path
