# Native negative control

510 native clips across 6 voices (8830 canonical phones from natural voices). Every flag is a false positive.

Threshold τ is the GOP cutoff: a phone is flagged only if the free decode disagrees **and** its GOP is below τ. Less negative τ = stricter = more flags.

## Phone-level false-positive rate

| τ | natural voices | all voices | de-DE-KatjaNeural | de-DE-ConradNeural | de-DE-AmalaNeural | de-DE-KillianNeural | Microsoft Hedda Desktop | espeak |
|---|---|---|---|---|---|---|---|---|
| -8 | 0.9% | 1.7% | 0.9% | 0.8% | 1.0% | 0.7% | 1.0% | 5.7% |
| -6 | 1.3% | 2.4% | 1.2% | 1.2% | 1.4% | 1.2% | 1.2% | 8.2% |
| -4 | 2.0% | 3.5% | 2.0% | 2.2% | 2.4% | 1.8% | 1.9% | 10.7% |
| -3 | 2.6% | 4.3% | 2.4% | 2.5% | 3.2% | 2.3% | 2.5% | 12.9% |
| -2 | 3.3% | 5.3% | 3.1% | 3.5% | 4.0% | 2.7% | 3.1% | 15.3% |
| -1.5 | 3.7% | 5.9% | 3.7% | 4.0% | 4.6% | 3.2% | 3.2% | 16.6% |
| -1 | 4.5% | 6.9% | 4.2% | 4.9% | 5.2% | 4.1% | 4.2% | 18.5% |
| -0.5 | 5.5% | 8.1% | 5.2% | 5.6% | 6.6% | 4.8% | 5.4% | 20.9% |
| 0 | 6.5% | 9.2% | 6.3% | 6.7% | 7.5% | 5.6% | 6.1% | 23.2% |

## Sentences with at least one flag

| τ | natural voices | de-DE-KatjaNeural | de-DE-ConradNeural | de-DE-AmalaNeural | de-DE-KillianNeural | Microsoft Hedda Desktop | espeak |
|---|---|---|---|---|---|---|---|
| -8 | 14.3% | 14.7% | 13.7% | 14.7% | 10.8% | 17.6% | 63.7% |
| -6 | 20.8% | 20.6% | 21.6% | 21.6% | 18.6% | 21.6% | 80.4% |
| -4 | 31.4% | 31.4% | 31.4% | 34.3% | 28.4% | 31.4% | 88.2% |
| -3 | 35.9% | 35.3% | 35.3% | 37.3% | 35.3% | 36.3% | 89.2% |
| -2 | 42.5% | 44.1% | 44.1% | 45.1% | 38.2% | 41.2% | 90.2% |
| -1.5 | 44.9% | 48.0% | 45.1% | 49.0% | 41.2% | 41.2% | 91.2% |
| -1 | 51.6% | 51.0% | 51.0% | 55.9% | 50.0% | 50.0% | 92.2% |
| -0.5 | 58.2% | 57.8% | 56.9% | 61.8% | 57.8% | 56.9% | 93.1% |
| 0 | 63.5% | 64.7% | 64.7% | 67.6% | 61.8% | 58.8% | 96.1% |

## Alignment disagreement before the GOP gate

Share of canonical phones where the free decode disagrees at all (what the FPR would be with no GOP gate).

| voice | disagreement |
|---|---|
| edge:de-DE-KatjaNeural | 7.1% |
| edge:de-DE-ConradNeural | 7.6% |
| edge:de-DE-AmalaNeural | 8.4% |
| edge:de-DE-KillianNeural | 6.6% |
| sapi:Microsoft Hedda Desktop | 6.8% |
| espeak | 24.2% |

## Recommended threshold

Largest τ with natural-voice FPR below 5%: **τ = -1** (FPR 4.5%).

## Most falsely flagged phones (natural voices, τ = -2)

| canonical | false flags | occurrences | rate | heard as |
|---|---|---|---|---|
| (insertion) | 54 | 0 | n/a | t ×15, ə ×8, ɪ ×7 |
| aː | 50 | 135 | 37.0% | a ×48, ∅ ×1, aɪ ×1 |
| ʏ | 35 | 35 | 100.0% | y ×31, ə ×3, ɪ ×1 |
| ɛː | 25 | 25 | 100.0% | ɛ ×16, eː ×9 |
| ʁ | 17 | 590 | 2.9% | ∅ ×14, aː ×1, ɜ ×1 |
| eː | 13 | 170 | 7.6% | iː ×12, ɛ ×1 |
| d | 11 | 395 | 2.8% | t ×10, ∅ ×1 |
| ʊ | 10 | 115 | 8.7% | uː ×10 |
| ə | 9 | 640 | 1.4% | ɛ ×6, ∅ ×1, a ×1 |
| l | 7 | 285 | 2.5% | ∅ ×6, ə ×1 |
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
