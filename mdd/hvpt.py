"""High Variability Phonetic Training: forced-choice perception drills.

The design follows the training paradigm that has held up since Logan, Lively &
Pisoni (1991): identify a word from a minimal set, hear it from many talkers,
get immediate right/wrong feedback, and repeat over short sessions. Bradlow et
al. (1997) showed the perceptual gains transfer to production without any
production practice, which is why this belongs next to the pronunciation
scorer rather than inside it.

Three properties of the paradigm the implementation preserves:

* **Variability.** Talker rotates every trial; the same word is never heard
  twice in a row from the same voice. Training on one talker produces gains
  that do not generalise.
* **Immediate feedback.** Each answer is scored at once and the correct item is
  available for replay.
* **Desirable difficulty.** A 2-down-1-up staircase widens the answer set after
  two correct and narrows it after one wrong, holding the learner near ~70%
  accuracy — hard enough to force discrimination, not so hard it is guessing.

Contrasts are validated before use (`mdd.validate`), so trials are never built
over pairs the synthesiser renders identically.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .languages import Contrast, LanguageProfile, get
from .synth import TALKERS, Talker

MIN_CHOICES = 2
MAX_CHOICES = 4
#: Consecutive correct answers before difficulty increases (2-down-1-up).
STEP_UP_AFTER = 2


@dataclass
class Trial:
    contrast_id: str
    target: str
    choices: tuple[str, ...]
    talker: Talker
    lang: str

    @property
    def answer_index(self) -> int:
        return self.choices.index(self.target)

    def is_correct(self, choice: str | int) -> bool:
        if isinstance(choice, int):
            return choice == self.answer_index
        return choice == self.target

    def audio(self):
        """(sample_rate, samples) for the stimulus."""
        from . import synth

        return synth.synthesize(self.target, self.lang, self.talker)

    def render(self, path):
        from . import synth

        return synth.render_to_file(self.target, self.lang, path, self.talker)


@dataclass
class ContrastStats:
    seen: int = 0
    correct: int = 0
    #: Per-answer confusion counts: what the learner picked when wrong.
    confusions: dict[tuple[str, str], int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.seen if self.seen else 0.0

    def worst_confusion(self) -> tuple[tuple[str, str], int] | None:
        if not self.confusions:
            return None
        return max(self.confusions.items(), key=lambda kv: kv[1])


class Session:
    """A perception training session over one language's usable contrasts."""

    def __init__(self, lang: LanguageProfile | str | None = None,
                 contrast_ids: list[str] | None = None,
                 seed: int | None = None,
                 validate: bool = True,
                 audio_check: bool = False):
        """`audio_check` adds the (slow, stochastic) acoustic gate on top of the
        deterministic transcription gate. Off by default so the set of available
        contrasts is stable run to run; `python -m mdd.validate` runs both."""
        self.profile = lang if isinstance(lang, LanguageProfile) else get(lang)
        self.rng = random.Random(seed)
        self.stats: dict[str, ContrastStats] = {}
        self.history: list[tuple[Trial, bool]] = []
        self._streak = 0
        self._n_choices = MIN_CHOICES
        self._last_talker: Talker | None = None
        self.skipped: dict[str, str] = {}

        if not self.profile.hvpt_ready:
            raise PerceptionUnavailable(
                self.profile.hvpt_caveat
                or f"perception training is not enabled for {self.profile.name}")

        wanted = [c for c in self.profile.contrasts
                  if contrast_ids is None or c.id in contrast_ids]
        self.pools: dict[str, list[tuple[str, ...]]] = {}
        for contrast in wanted:
            groups = self._usable_groups(contrast, validate, audio_check)
            if groups:
                self.pools[contrast.id] = groups
            else:
                self.skipped[contrast.id] = "no pairs the synthesiser renders distinctly"
        if not self.pools:
            raise PerceptionUnavailable(
                f"no usable contrasts for {self.profile.name}: {self.skipped}")

    def _usable_groups(self, contrast: Contrast, validate: bool,
                       audio_check: bool) -> list[tuple[str, ...]]:
        if not validate:
            return [g for g in contrast.pairs if len(g) >= MIN_CHOICES]
        from .validate import check_contrast

        report = check_contrast(contrast, self.profile, audio=audio_check)
        ok = {(c.word_a, c.word_b) for c in report.usable_pairs}
        groups = []
        for group in contrast.pairs:
            # Keep the words in a set that are separable from every other word
            # kept alongside them, so a 3-way set never contains a merged pair.
            keep: list[str] = []
            for w in group:
                if all(((w, k) in ok or (k, w) in ok) for k in keep):
                    keep.append(w)
            if len(keep) >= MIN_CHOICES:
                groups.append(tuple(keep))
        return groups

    @property
    def contrast_ids(self) -> list[str]:
        return list(self.pools)

    def _pick_talker(self) -> Talker:
        options = [t for t in TALKERS if t != self._last_talker] or list(TALKERS)
        talker = self.rng.choice(options)
        self._last_talker = talker
        return talker

    def next_trial(self, contrast_id: str | None = None) -> Trial:
        if contrast_id is None:
            contrast_id = self._weakest_contrast()
        groups = self.pools[contrast_id]
        # Prefer a set large enough for the current difficulty.
        big = [g for g in groups if len(g) >= self._n_choices]
        group = self.rng.choice(big or groups)
        n = min(self._n_choices, len(group))
        choices = self.rng.sample(list(group), n)
        target = self.rng.choice(choices)
        self.rng.shuffle(choices)
        return Trial(contrast_id, target, tuple(choices), self._pick_talker(),
                     self.profile.code)

    def _weakest_contrast(self) -> str:
        """Spend trials where accuracy is lowest, sampling unseen contrasts first."""
        unseen = [c for c in self.pools if c not in self.stats]
        if unseen:
            return self.rng.choice(unseen)
        return min(self.pools, key=lambda c: (self.stats[c].accuracy, -self.stats[c].seen))

    def record(self, trial: Trial, choice: str | int) -> bool:
        correct = trial.is_correct(choice)
        st = self.stats.setdefault(trial.contrast_id, ContrastStats())
        st.seen += 1
        if correct:
            st.correct += 1
            self._streak += 1
            if self._streak >= STEP_UP_AFTER:
                self._streak = 0
                self._n_choices = min(MAX_CHOICES, self._n_choices + 1)
        else:
            picked = trial.choices[choice] if isinstance(choice, int) else choice
            key = (trial.target, picked)
            st.confusions[key] = st.confusions.get(key, 0) + 1
            self._streak = 0
            self._n_choices = max(MIN_CHOICES, self._n_choices - 1)
        self.history.append((trial, correct))
        return correct

    @property
    def difficulty(self) -> int:
        return self._n_choices

    def accuracy(self) -> float:
        seen = sum(s.seen for s in self.stats.values())
        return sum(s.correct for s in self.stats.values()) / seen if seen else 0.0

    def report(self) -> dict:
        out = {
            "language": self.profile.name,
            "lang": self.profile.code,
            "trials": len(self.history),
            "accuracy": self.accuracy(),
            "difficulty": self._n_choices,
            "contrasts": {},
            "skipped": dict(self.skipped),
        }
        for cid, st in self.stats.items():
            contrast = self.profile.contrast(cid)
            entry = {
                "label": contrast.label,
                "seen": st.seen,
                "correct": st.correct,
                "accuracy": st.accuracy,
                "tip": contrast.tip,
            }
            worst = st.worst_confusion()
            if worst:
                (target, picked), n = worst
                entry["worst_confusion"] = {"target": target, "picked": picked, "count": n}
            out["contrasts"][cid] = entry
        return out


class PerceptionUnavailable(RuntimeError):
    """No contrast in this language can currently produce honest trials."""
