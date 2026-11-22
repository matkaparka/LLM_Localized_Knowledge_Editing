"""EC-11 same-information retrieval (WikiBigEdit's RAG baseline and two gated variants).

Memory: one record per edit, (update question, answer). Retriever: a sentence encoder (all-mpnet-base-v2 in the runs)
with mean pooling over the attention mask and L2 normalisation, as Sentence-Transformers does; the query is the test
question itself. Retrieved records are written in the format shots' own format, 'Q: ...\\nA: ...\\n', after the
format shots, the most similar record last (next to the test question).

Arms:
  rag:always:k   always the top k records (WikiBigEdit's setting is k = 2)
  rag:k:tau      the top k among records with cosine >= tau (nothing when none reaches it)
  ragp:m         the top record among those with cosine >= its own threshold max(floor, record's largest cosine
                 over the public probe questions + m): KT's per-entry calibration applied to retrieval
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


class MeanPoolEncoder:
    def __init__(self, model, tok, device, max_len: int = 384, batch: int = 64):
        self.model, self.tok, self.device, self.max_len, self.batch = model, tok, device, int(max_len), int(batch)
        self.model.eval()
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    def _ids(self, text: str) -> list[int]:
        try:
            ids = self.tok.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_len)
        except TypeError:          # minimal test tokenizers
            ids = self.tok.encode(text)[:self.max_len]
        return ids

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        outs = []
        for s in range(0, len(texts), self.batch):
            ids = [self._ids(t) for t in texts[s:s + self.batch]]
            width = max(map(len, ids))
            x = torch.full((len(ids), width), self.pad, dtype=torch.long)
            mask = torch.zeros((len(ids), width), dtype=torch.long)
            for i, sq in enumerate(ids):
                x[i, :len(sq)] = torch.tensor(sq, dtype=torch.long)
                mask[i, :len(sq)] = 1
            x, mask = x.to(self.device), mask.to(self.device)
            h = self.model(input_ids=x, attention_mask=mask).last_hidden_state.float()
            m = mask.unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
            outs.append(F.normalize(emb, dim=-1))
        if not outs:
            return torch.empty((0, 0), device=self.device)
        return torch.cat(outs, 0)


def parse(arm: str) -> dict:
    kind, _, rest = arm.partition(':')
    if kind == 'rag':
        a, _, b = rest.partition(':')
        if a == 'always':
            return dict(sub='always', k=int(b), tau=None)
        return dict(sub='gated', k=int(a), tau=float(b))
    if kind == 'ragp':
        return dict(sub='probe', k=1, margin=float(rest))
    raise ValueError(f'not a retrieval arm: {arm}')


def record_thresholds(rec_emb: torch.Tensor, probe_emb: torch.Tensor, margin: float, floor: float) -> torch.Tensor:
    pmax = (probe_emb @ rec_emb.T).max(0).values if probe_emb.shape[0] else torch.full((rec_emb.shape[0],), -1.0,
                                                                                       device=rec_emb.device)
    return torch.clamp(pmax + margin, min=floor)


@torch.no_grad()
def retrieve(sim: torch.Tensor, arm: str, rec_thr: torch.Tensor | None = None) -> list[list[int]]:
    """Per query, the chosen record indices, most similar first. sim: queries x records (cosine)."""
    p = parse(arm)
    if sim.shape[1] == 0:
        return [[] for _ in range(sim.shape[0])]
    k = min(p['k'], sim.shape[1])
    if p['sub'] == 'always':
        return torch.topk(sim, k, dim=1).indices.tolist()
    if p['sub'] == 'gated':
        allowed = sim >= p['tau']
    else:
        if rec_thr is None:
            raise ValueError('ragp needs record thresholds')
        allowed = sim >= rec_thr.unsqueeze(0)
    masked = torch.where(allowed, sim, torch.full_like(sim, -9.0))
    vals, idx = torch.topk(masked, k, dim=1)
    return [[int(i) for v, i in zip(vr, ir) if v > -9.0] for vr, ir in zip(vals.tolist(), idx.tolist())]


def context_with(shots: str, records: list[tuple[str, str]], picked: list[int]) -> str:
    """Format shots, then the picked records with the most similar last."""
    return shots + ''.join(f'Q: {records[i][0].strip()}\nA: {records[i][1].strip()}\n' for i in reversed(picked))
