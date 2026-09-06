# Synthetic positives

157 injected errors across 21 error types, synthesised with espeak-ng from modified phoneme strings. Recall = the injected phone was flagged at τ. Diagnosis = the alignment named the substitute exactly (independent of τ). Control FP = the same position was flagged in the unmodified synthesis, i.e. a false alarm espeak's audio causes on its own.

| error | n | recall@-6 | recall@-4 | recall@-2 | recall@-1 | recall@0 | diagnosis | control FP@-2 | heard as |
|---|---|---|---|---|---|---|---|---|---|
| ü_long→uː | 8 | 37.5% | 75.0% | 87.5% | 100.0% | 100.0% | 100.0% | 37.5% | uː ×8 |
| ü_short→ʊ | 7 | 28.6% | 71.4% | 100.0% | 100.0% | 100.0% | 57.1% | 14.3% | ʊ ×4, uː ×2, oː ×1 |
| ö_long→oː | 5 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | oː ×5 |
| ö_short→ɔ | 7 | 28.6% | 100.0% | 100.0% | 100.0% | 100.0% | 85.7% | 57.1% | ɔ ×6, oː ×1 |
| ich→k | 8 | 12.5% | 37.5% | 62.5% | 62.5% | 87.5% | 75.0% | 0.0% | k ×6, (match) ×1, None ×1 |
| ich→sch | 8 | 12.5% | 25.0% | 25.0% | 25.0% | 25.0% | 25.0% | 0.0% | (match) ×6, ʃ ×2 |
| ach→k | 8 | 0.0% | 75.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | k ×8 |
| ach→h | 8 | 25.0% | 75.0% | 87.5% | 87.5% | 87.5% | 0.0% | 0.0% | f ×5, None ×1, v ×1 |
| z→voiced_z | 8 | 50.0% | 100.0% | 100.0% | 100.0% | 100.0% | 37.5% | 0.0% | z ×3, f ×1, None ×1 |
| w→english_w | 8 | 0.0% | 25.0% | 62.5% | 62.5% | 75.0% | 50.0% | 25.0% | w ×4, (match) ×2, m ×1 |
| v→voiced | 8 | 12.5% | 37.5% | 37.5% | 50.0% | 50.0% | 37.5% | 0.0% | (match) ×4, v ×3, m ×1 |
| st_initial→s | 8 | 0.0% | 25.0% | 37.5% | 50.0% | 62.5% | 50.0% | 0.0% | s ×4, (match) ×2, None ×1 |
| final_t→d | 8 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | (match) ×7, None ×1 |
| final_k→ɡ | 8 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | (match) ×8 |
| final_p→b | 2 | 0.0% | 0.0% | 0.0% | 0.0% | 50.0% | 50.0% | 0.0% | (match) ×1, b ×1 |
| long_a→short | 8 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | (match) ×8 |
| schwa→eː | 8 | 0.0% | 0.0% | 25.0% | 37.5% | 37.5% | 25.0% | 0.0% | (match) ×5, eː ×2, ɪ ×1 |
| schwa_dropped | 8 | 25.0% | 50.0% | 62.5% | 75.0% | 87.5% | 75.0% | 0.0% | ∅ ×6, n ×1, t ×1 |
| ei→iː | 8 | 25.0% | 87.5% | 87.5% | 87.5% | 100.0% | 87.5% | 0.0% | iː ×7, l ×1 |
| eu→uː | 8 | 0.0% | 75.0% | 100.0% | 100.0% | 100.0% | 62.5% | 0.0% | uː ×5, l ×1, h ×1 |
| ng→ng+ɡ | 8 | 62.5% | 75.0% | 75.0% | 75.0% | 75.0% | 50.0% | 25.0% | +k ×3, (match) ×2, ɡ ×1 |
| **all** | 157 | 15.9% | 49.7% | 60.5% | 63.7% | 68.2% | 49.7% | 7.6% | |

Skipped 1 sentences whose sentence-level espeak phonemes differ from per-word G2P:

- Ich lerne seit einem Jahr Deutsch.: sentence-level espeak phonemes differ from per-word G2P
