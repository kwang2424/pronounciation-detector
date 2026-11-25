"""wav2vec2 phone recogniser + CTC forced alignment + GOP scores.

Model: facebook/wav2vec2-xlsr-53-espeak-cv-ft (CTC over espeak IPA, multilingual).
"""
from dataclasses import dataclass
import unicodedata

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
FRAME_SEC = 0.02  # wav2vec2 stride


@dataclass
class Segment:
    token: str
    t_start: float
    t_end: float
    gop: float


class PhoneRecognizer:
    supports_logit_mask = True

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device).eval()
        self.vocab = {v: k for k, v in self.processor.tokenizer.get_vocab().items()}
        self.blank = self.processor.tokenizer.pad_token_id
        self._mask_cache: dict[frozenset[str], torch.Tensor | None] = {}

    @staticmethod
    def load_audio(path: str, sr: int = 16000) -> np.ndarray:
        wav, file_sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if file_sr != sr:
            wav = F.resample(torch.from_numpy(wav), file_sr, sr).numpy()
        return wav

    def _logit_mask(self, allowed_tokens: frozenset[str] | set[str]) -> torch.Tensor | None:
        key = frozenset(allowed_tokens)
        cache = getattr(self, "_mask_cache", None)
        if cache is None:
            cache = {}
            self._mask_cache = cache
        if key in cache:
            return cache[key]
        tok2id = self.processor.tokenizer.get_vocab()
        matched = [tok2id[t] for t in key if t in tok2id]
        if not matched:
            cache[key] = None
            return None
        mask = torch.full((len(tok2id),), float("-inf"))
        mask[self.blank] = 0.0
        mask[matched] = 0.0
        cache[key] = mask
        return mask

    @torch.no_grad()
    def log_probs(
        self,
        wav: np.ndarray,
        allowed_tokens: frozenset[str] | set[str] | None = None,
    ) -> torch.Tensor:
        inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0].cpu()    # (T, V)
        if allowed_tokens:
            mask = self._logit_mask(allowed_tokens)
            if mask is not None and mask.shape[0] == logits.shape[-1]:
                logits = logits + mask
        return torch.log_softmax(logits, dim=-1)

    def greedy_spikes(self, logp: torch.Tensor) -> list[tuple[str, float]]:
        """Greedy CTC decode as (token, confidence) pairs, confidence being the highest
        posterior probability the token reached in its run of frames."""
        ids = logp.argmax(-1).tolist()
        out, prev = [], None
        for t, i in enumerate(ids):
            if i == self.blank:
                prev = None
                continue
            p = float(logp[t, i].exp())
            if i == prev:
                out[-1] = (out[-1][0], max(out[-1][1], p))
            else:
                out.append((self.vocab[i], p))
            prev = i
        return out

    def greedy_decode(self, logp: torch.Tensor) -> str:
        return "".join(tok for tok, _ in self.greedy_spikes(logp))

    def gop(self, logp: torch.Tensor, canonical_tokens: list[str]) -> list[Segment]:
        """Force-align the canonical token sequence and compute a GOP score per token.

        GOP(p) = mean_t [ logP(p|x_t) - max_q logP(q|x_t) ] over frames assigned to p.
        Multi-character tokens absent from the vocab are expanded into constituent
        vocab sub-tokens before forced alignment and pooled back to each token.
        """
        if not canonical_tokens or logp.shape[0] == 0:
            return []
        tok2id = self.processor.tokenizer.get_vocab()
        unk_id = self.processor.tokenizer.unk_token_id
        expanded_ids: list[int] = []
        group_lens: list[int] = []
        for t in canonical_tokens:
            sub = _expand_token(t, tok2id, unk_id)
            expanded_ids.extend(sub)
            group_lens.append(len(sub))

        min_frames = len(expanded_ids) + sum(
            1 for a, b in zip(expanded_ids, expanded_ids[1:]) if a == b
        )
        if logp.shape[0] < min_frames:
            return []

        targets = torch.tensor([expanded_ids], dtype=torch.int32)
        alignment, _ = F.forced_align(logp.unsqueeze(0), targets, blank=self.blank)
        best = logp.max(-1).values
        spans = F.merge_tokens(alignment[0], best.exp(), blank=self.blank)
        segs: list[Segment] = []
        offset = 0
        for tok, k in zip(canonical_tokens, group_lens):
            sub_spans = spans[offset:offset + k]
            sub_ids = expanded_ids[offset:offset + k]
            offset += k
            s, e = sub_spans[0].start, sub_spans[-1].end
            frame_scores = [
                logp[sp.start:sp.end, tid] - best[sp.start:sp.end]
                for sp, tid in zip(sub_spans, sub_ids)
                if sp.end > sp.start
            ]
            mean_gop = float(torch.cat(frame_scores).mean()) if frame_scores else 0.0
            segs.append(Segment(tok, s * FRAME_SEC, e * FRAME_SEC, mean_gop))
        return segs


_VOCAB_ALIASES: dict[str, tuple[str, ...]] = {
    "ɔʏ": ("ɔø", "ɔɪ"),   # espeak-trained XLSR-53 vocab contains ɔø/ɔɪ, not ɔʏ
    "ʏ": ("y",),
}


def _expand_token(tok: str, tok2id: dict[str, int], unk_id: int) -> list[int]:
    """Expand a canonical token into vocab IDs, keeping multi-char constituents."""
    if tok in tok2id:
        return [tok2id[tok]]
    nfc = unicodedata.normalize("NFC", tok)
    if nfc in tok2id:
        return [tok2id[nfc]]
    for alias in _VOCAB_ALIASES.get(nfc, ()):
        if alias in tok2id:
            return [tok2id[alias]]
    out: list[int] = []
    i = 0
    while i < len(tok):
        hit = next((j for j in range(len(tok), i, -1) if tok[i:j] in tok2id), None)
        if hit is not None:
            out.append(tok2id[tok[i:hit]])
            i = hit
        else:
            if tok[i] != "ː" and not unicodedata.combining(tok[i]):
                ch_alias = next((a for a in _VOCAB_ALIASES.get(tok[i], ()) if a in tok2id), None)
                if ch_alias is not None:
                    out.append(tok2id[ch_alias])
            i += 1
    return out or [unk_id]
