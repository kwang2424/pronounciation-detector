"""wav2vec2 phone recogniser + CTC forced alignment + GOP scores.

Model: facebook/wav2vec2-xlsr-53-espeak-cv-ft (CTC over espeak IPA, multilingual).
"""
from dataclasses import dataclass

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
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device).eval()
        self.vocab = {v: k for k, v in self.processor.tokenizer.get_vocab().items()}
        self.blank = self.processor.tokenizer.pad_token_id

    @staticmethod
    def load_audio(path: str, sr: int = 16000) -> np.ndarray:
        wav, file_sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if file_sr != sr:
            wav = F.resample(torch.from_numpy(wav), file_sr, sr).numpy()
        return wav

    @torch.no_grad()
    def log_probs(self, wav: np.ndarray) -> torch.Tensor:
        inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0]          # (T, V)
        return torch.log_softmax(logits, dim=-1).cpu()

    def greedy_decode(self, logp: torch.Tensor) -> str:
        ids = logp.argmax(-1).tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and i != self.blank:
                out.append(self.vocab[i])
            prev = i
        return "".join(out)

    def gop(self, logp: torch.Tensor, canonical_tokens: list[str]) -> list[Segment]:
        """Force-align the canonical token sequence and compute a GOP score per token.

        GOP(p) = mean_t [ logP(p|x_t) - max_q logP(q|x_t) ] over frames assigned to p.
        Tokens absent from the model vocab are approximated by their first character.
        """
        tok2id = self.processor.tokenizer.get_vocab()
        ids = []
        for t in canonical_tokens:
            ids.append(tok2id.get(t, tok2id.get(t[0], self.processor.tokenizer.unk_token_id)))
        targets = torch.tensor([ids], dtype=torch.int32)
        alignment, _ = F.forced_align(logp.unsqueeze(0), targets, blank=self.blank)
        spans = F.merge_tokens(alignment[0], logp.max(-1).values.exp(), blank=self.blank)
        best = logp.max(-1).values
        segs = []
        for tok, span in zip(canonical_tokens, spans):
            s, e = span.start, span.end
            frames = logp[s:e, ids[len(segs)]] - best[s:e]
            segs.append(Segment(tok, s * FRAME_SEC, e * FRAME_SEC, float(frames.mean()) if e > s else 0.0))
        return segs
