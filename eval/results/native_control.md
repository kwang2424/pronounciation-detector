# Native negative control

408 native clips across 5 voices (7064 canonical phones from natural voices). Every flag is a false positive.

Threshold τ is the GOP cutoff: a phone is flagged only if the free decode disagrees **and** its GOP is below τ. Less negative τ = stricter = more flags.

## Phone-level false-positive rate

| τ | natural voices | all voices | de-DE-KatjaNeural | de-DE-ConradNeural | de-DE-AmalaNeural | de-DE-KillianNeural | espeak |
|---|---|---|---|---|---|---|---|
| -8 | 0.6% | 1.1% | 0.3% | 0.8% | 0.3% | 0.9% | 3.3% |
| -6 | 0.6% | 1.4% | 0.3% | 0.8% | 0.3% | 0.9% | 4.6% |
| -4 | 1.0% | 2.2% | 0.8% | 1.1% | 0.7% | 1.2% | 7.2% |
| -3 | 1.4% | 2.8% | 1.1% | 1.7% | 1.0% | 1.6% | 8.7% |
| -2 | 2.0% | 3.8% | 1.5% | 2.7% | 1.3% | 2.6% | 11.1% |
| -1.5 | 2.4% | 4.5% | 1.8% | 3.0% | 1.8% | 3.1% | 12.9% |
| -1 | 3.1% | 5.3% | 2.4% | 3.5% | 2.4% | 3.9% | 14.2% |
| -0.5 | 4.0% | 6.5% | 3.2% | 4.8% | 3.3% | 4.9% | 16.4% |
| 0 | 5.0% | 7.8% | 4.3% | 5.7% | 4.2% | 5.8% | 18.7% |

## Sentences with at least one flag

| τ | natural voices | de-DE-KatjaNeural | de-DE-ConradNeural | de-DE-AmalaNeural | de-DE-KillianNeural | espeak |
|---|---|---|---|---|---|---|
| -8 | 9.3% | 4.9% | 12.7% | 4.9% | 14.7% | 40.2% |
| -6 | 10.0% | 5.9% | 13.7% | 5.9% | 14.7% | 53.9% |
| -4 | 16.2% | 13.7% | 18.6% | 11.8% | 20.6% | 70.6% |
| -3 | 21.6% | 16.7% | 25.5% | 17.6% | 26.5% | 75.5% |
| -2 | 28.4% | 22.5% | 34.3% | 21.6% | 35.3% | 81.4% |
| -1.5 | 32.1% | 25.5% | 37.3% | 26.5% | 39.2% | 84.3% |
| -1 | 37.7% | 33.3% | 39.2% | 34.3% | 44.1% | 85.3% |
| -0.5 | 46.6% | 41.2% | 49.0% | 44.1% | 52.0% | 86.3% |
| 0 | 58.1% | 55.9% | 59.8% | 54.9% | 61.8% | 88.2% |

## Alignment disagreement before the GOP gate

Share of canonical phones where the free decode disagrees at all (what the FPR would be with no GOP gate).

| voice | disagreement |
|---|---|
| edge:de-DE-KatjaNeural | 4.9% |
| edge:de-DE-ConradNeural | 6.8% |
| edge:de-DE-AmalaNeural | 4.9% |
| edge:de-DE-KillianNeural | 6.9% |
| espeak | 21.2% |

## Recommended threshold

Largest τ with natural-voice FPR below 5%: **τ = -0.5** (FPR 4.0%).

By the **median voice** (robust to one bad talker): **τ = -0.5** (median FPR 4.0%).

Per-voice disagreement ranges 4.9%–6.9%. The voices agree closely, so pooling is safe.

## Most falsely flagged phones (natural voices, τ = -1)

| canonical | false flags | occurrences | rate | heard as |
|---|---|---|---|---|
| (insertion) | 31 | 0 | n/a | t ×13, ʁ ×8, ə ×4 |
| eː | 28 | 156 | 17.9% | iː ×23, ɛ ×5 |
| œ | 22 | 28 | 78.6% | ɛ ×10, ə ×8, a ×2 |
| ɔʏ | 18 | 56 | 32.1% | ɔ ×9, o ×4, aɪ ×3 |
| ə | 17 | 512 | 3.3% | ɛ ×5, eː ×4, e ×2 |
| yː | 12 | 44 | 27.3% | uː ×10, y ×2 |
| d | 11 | 316 | 3.5% | t ×8, ∅ ×3 |
| ʊ | 11 | 92 | 12.0% | uː ×8, u ×3 |
| l | 8 | 228 | 3.5% | ∅ ×8 |
| ɜ | 8 | 104 | 7.7% | ə ×6, a ×2 |
| t | 6 | 616 | 1.0% | ∅ ×6 |
| ɔ | 6 | 60 | 10.0% | oː ×4, o ×2 |
| ɛ | 5 | 228 | 2.2% | eː ×5 |
| n | 4 | 592 | 0.7% | m ×4 |
| ʁ | 4 | 472 | 0.8% | ∅ ×4 |
| b | 4 | 156 | 2.6% | d ×2, p ×2 |
| a | 3 | 248 | 1.2% | aɪ ×2, aː ×1 |
| y | 3 | 28 | 10.7% | ə ×3 |
| ɪ | 2 | 380 | 0.5% | iː ×2 |
| iː | 2 | 272 | 0.7% | ɪ ×2 |
