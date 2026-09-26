"""Clip hygiene shared by everything that plays synthesised speech to a learner.

Three separate complaints from one training session traced back to clip edges: a
word heard as starting with a plosive it does not have ("haute", whose h is
silent), a short word heard as truncated ("rue" sounding like "re"), and nasal
vowels heard as cut off at the end ("sans", "son") — the tail being exactly where
French carries nasality. Padding keeps the tail; short fades stop an abrupt edge
reading as a burst.
"""

#: Silence added around each clip. Neural TTS trims hard on isolated words.
PAD_HEAD_MS = 60
PAD_TAIL_MS = 220
#: Fade at each end; long enough to kill the transient, too short to touch speech.
FADE_MS = 12


def polish(data, sr: int):
    """Pad and fade a clip; returns int16 samples."""
    import numpy as np

    x = np.asarray(data)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float64)

    fade = max(1, int(sr * FADE_MS / 1000))
    if x.size > 2 * fade:
        x[:fade] *= np.linspace(0.0, 1.0, fade)
        x[-fade:] *= np.linspace(1.0, 0.0, fade)
    head = np.zeros(int(sr * PAD_HEAD_MS / 1000))
    tail = np.zeros(int(sr * PAD_TAIL_MS / 1000))
    return np.concatenate([head, x, tail]).astype(np.int16)
