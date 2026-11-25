# Native negative control

408 native clips across 5 voices (7064 canonical phones from natural voices). Every flag is a false positive.

Threshold τ is the GOP cutoff: a phone is flagged only if the free decode disagrees **and** its GOP is below τ. Less negative τ = stricter = more flags.

## Phone-level false-positive rate

| τ | natural voices | all voices | de-DE-KatjaNeural | de-DE-ConradNeural | de-DE-AmalaNeural | de-DE-KillianNeural | espeak |
|---|---|---|---|---|---|---|---|
| -8 | 0.6% | 1.1% | 0.3% | 0.8% | 0.3% | 0.9% | 3.3% |
| -6 | 0.6% | 1.4% | 0.3% | 0.8% | 0.3% | 0.9% | 4.6% |
| -4 | 0.9% | 2.2% | 0.7% | 1.1% | 0.6% | 1.2% | 7.1% |
| -3 | 1.3% | 2.8% | 1.0% | 1.7% | 1.0% | 1.6% | 8.6% |
| -2 | 1.8% | 3.5% | 1.4% | 2.3% | 1.2% | 2.2% | 10.7% |
| -1.5 | 2.1% | 4.1% | 1.6% | 2.6% | 1.6% | 2.7% | 12.2% |
| -1 | 2.6% | 4.8% | 2.0% | 3.1% | 2.1% | 3.3% | 13.4% |
| -0.5 | 3.4% | 5.8% | 2.7% | 4.0% | 2.6% | 4.2% | 15.3% |
| 0 | 4.0% | 6.8% | 3.3% | 4.7% | 3.2% | 4.9% | 17.7% |

## Sentences with at least one flag

| τ | natural voices | de-DE-KatjaNeural | de-DE-ConradNeural | de-DE-AmalaNeural | de-DE-KillianNeural | espeak |
|---|---|---|---|---|---|---|
| -8 | 9.3% | 4.9% | 12.7% | 4.9% | 14.7% | 40.2% |
| -6 | 10.0% | 5.9% | 13.7% | 5.9% | 14.7% | 53.9% |
| -4 | 15.4% | 11.8% | 18.6% | 10.8% | 20.6% | 69.6% |
| -3 | 21.1% | 15.7% | 25.5% | 16.7% | 26.5% | 74.5% |
| -2 | 26.5% | 21.6% | 32.4% | 19.6% | 32.4% | 80.4% |
| -1.5 | 30.9% | 24.5% | 35.3% | 25.5% | 38.2% | 82.4% |
| -1 | 34.6% | 28.4% | 37.3% | 31.4% | 41.2% | 83.3% |
| -0.5 | 38.7% | 34.3% | 41.2% | 35.3% | 44.1% | 85.3% |
| 0 | 45.8% | 42.2% | 48.0% | 41.2% | 52.0% | 87.3% |

## Alignment disagreement before the GOP gate

Share of canonical phones where the free decode disagrees at all (what the FPR would be with no GOP gate).

| voice | disagreement |
|---|---|
| edge:de-DE-KatjaNeural | 4.9% |
| edge:de-DE-ConradNeural | 6.7% |
| edge:de-DE-AmalaNeural | 4.9% |
| edge:de-DE-KillianNeural | 6.9% |
| espeak | 21.1% |

## Recommended threshold

Largest τ with natural-voice FPR below 5%: **τ = 0** (FPR 4.0%).

By the **median voice** (robust to one bad talker): **τ = 0** (median FPR 4.0%).

Per-voice disagreement ranges 4.9%–6.9%. The voices agree closely, so pooling is safe.

## Most falsely flagged phones (natural voices, τ = -1)

| canonical | false flags | occurrences | rate | heard as |
|---|---|---|---|---|
| (insertion) | 31 | 0 | n/a | t ×13, ʁ ×8, ə ×4 |
| eː | 28 | 156 | 17.9% | iː ×23, ɛ ×5 |
| ə | 17 | 512 | 3.3% | ɛ ×5, eː ×4, e ×2 |
| œ | 12 | 28 | 42.9% | ɛ ×6, a ×2, ə ×2 |
| d | 11 | 316 | 3.5% | t ×8, ∅ ×3 |
| ʊ | 11 | 92 | 12.0% | uː ×8, u ×3 |
| l | 8 | 228 | 3.5% | ∅ ×8 |
| ɜ | 8 | 104 | 7.7% | ə ×6, a ×2 |
| t | 6 | 616 | 1.0% | ∅ ×6 |
| ɔ | 6 | 60 | 10.0% | oː ×4, o ×2 |
| yː | 6 | 44 | 13.6% | uː ×4, y ×2 |
| ɛ | 5 | 228 | 2.2% | eː ×5 |
| n | 4 | 592 | 0.7% | m ×4 |
| ʁ | 4 | 472 | 0.8% | ∅ ×4 |
| b | 4 | 156 | 2.6% | d ×2, p ×2 |
| a | 3 | 248 | 1.2% | aɪ ×2, aː ×1 |
| ç | 3 | 192 | 1.6% | ∅ ×3 |
| y | 3 | 28 | 10.7% | ə ×3 |
| ɪ | 2 | 380 | 0.5% | iː ×2 |
| iː | 2 | 272 | 0.7% | ɪ ×2 |
