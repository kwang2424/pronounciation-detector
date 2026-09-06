# Native negative control

510 native clips across 6 voices (8830 canonical phones from natural voices). Every flag is a false positive.

Threshold τ is the GOP cutoff: a phone is flagged only if the free decode disagrees **and** its GOP is below τ. Less negative τ = stricter = more flags.

## Phone-level false-positive rate

| τ | natural voices | all voices | de-DE-KatjaNeural | de-DE-ConradNeural | de-DE-AmalaNeural | de-DE-KillianNeural | Microsoft Hedda Desktop | espeak |
|---|---|---|---|---|---|---|---|---|
| -8 | 0.3% | 0.7% | 0.2% | 0.5% | 0.2% | 0.4% | 0.3% | 2.7% |
| -6 | 0.4% | 1.1% | 0.2% | 0.6% | 0.4% | 0.5% | 0.4% | 4.2% |
| -4 | 0.9% | 1.8% | 0.6% | 1.1% | 1.1% | 0.8% | 0.7% | 6.5% |
| -3 | 1.2% | 2.3% | 0.8% | 1.3% | 1.6% | 1.1% | 1.1% | 8.2% |
| -2 | 1.6% | 3.1% | 1.4% | 1.9% | 2.2% | 1.2% | 1.5% | 10.5% |
| -1.5 | 2.0% | 3.6% | 1.7% | 2.4% | 2.8% | 1.6% | 1.5% | 11.7% |
| -1 | 2.6% | 4.4% | 2.2% | 3.0% | 3.2% | 2.3% | 2.2% | 13.5% |
| -0.5 | 3.4% | 5.5% | 2.9% | 3.8% | 4.4% | 2.9% | 3.2% | 15.8% |
| 0 | 4.3% | 6.6% | 3.9% | 4.8% | 5.2% | 3.7% | 3.9% | 18.1% |

## Sentences with at least one flag

| τ | natural voices | de-DE-KatjaNeural | de-DE-ConradNeural | de-DE-AmalaNeural | de-DE-KillianNeural | Microsoft Hedda Desktop | espeak |
|---|---|---|---|---|---|---|---|
| -8 | 5.5% | 3.9% | 8.8% | 3.9% | 5.9% | 4.9% | 32.4% |
| -6 | 7.1% | 3.9% | 9.8% | 6.9% | 7.8% | 6.9% | 48.0% |
| -4 | 14.1% | 10.8% | 15.7% | 17.6% | 13.7% | 12.7% | 64.7% |
| -3 | 18.2% | 14.7% | 18.6% | 22.5% | 17.6% | 17.6% | 72.5% |
| -2 | 23.5% | 20.6% | 25.5% | 27.5% | 19.6% | 24.5% | 78.4% |
| -1.5 | 26.3% | 23.5% | 28.4% | 33.3% | 21.6% | 24.5% | 80.4% |
| -1 | 32.7% | 27.5% | 34.3% | 39.2% | 32.4% | 30.4% | 83.3% |
| -0.5 | 41.6% | 36.3% | 44.1% | 48.0% | 37.3% | 42.2% | 84.3% |
| 0 | 49.0% | 46.1% | 51.0% | 54.9% | 46.1% | 47.1% | 89.2% |

## Alignment disagreement before the GOP gate

Share of canonical phones where the free decode disagrees at all (what the FPR would be with no GOP gate).

| voice | disagreement |
|---|---|
| edge:de-DE-KatjaNeural | 5.2% |
| edge:de-DE-ConradNeural | 5.8% |
| edge:de-DE-AmalaNeural | 6.4% |
| edge:de-DE-KillianNeural | 4.9% |
| sapi:Microsoft Hedda Desktop | 5.0% |
| espeak | 20.9% |

## Recommended threshold

Largest τ with natural-voice FPR below 5%: **τ = 0** (FPR 4.3%).

## Most falsely flagged phones (natural voices, τ = -2)

| canonical | false flags | occurrences | rate | heard as |
|---|---|---|---|---|
| (insertion) | 26 | 0 | n/a | t ×12, ə ×6, j ×4 |
| eː | 14 | 195 | 7.2% | iː ×12, ɛ ×2 |
| d | 11 | 395 | 2.8% | t ×10, ∅ ×1 |
| ʊ | 10 | 115 | 8.7% | uː ×10 |
| ə | 9 | 640 | 1.4% | ɛ ×6, ∅ ×1, a ×1 |
| l | 7 | 285 | 2.5% | ∅ ×6, ə ×1 |
| ʁ | 7 | 590 | 1.2% | ∅ ×6, aː ×1 |
| ɛ | 6 | 285 | 2.1% | iː ×3, eː ×2, ɪ ×1 |
| œ | 6 | 35 | 17.1% | ɛ ×6 |
| ç | 5 | 240 | 2.1% | ∅ ×2, ʃ ×1, ɕ ×1 |
| iː | 5 | 340 | 1.5% | ɪ ×5 |
| ɜ | 4 | 130 | 3.1% | ə ×2, ɐ ×1, a ×1 |
| ɔ | 4 | 75 | 5.3% | oː ×4 |
| a | 4 | 310 | 1.3% | aɪ ×4 |
| b | 4 | 195 | 2.1% | v ×2, ɡ ×1, d ×1 |
| t | 3 | 770 | 0.4% | ∅ ×3 |
| n | 3 | 740 | 0.4% | m ×2, ∅ ×1 |
| ɪ | 2 | 475 | 0.4% | iː ×2 |
| pf | 2 | 5 | 40.0% | f ×2 |
| aː | 2 | 135 | 1.5% | ∅ ×1, aɪ ×1 |
