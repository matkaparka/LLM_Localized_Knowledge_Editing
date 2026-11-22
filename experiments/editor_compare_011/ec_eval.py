"""EC-11 evaluation: fixed-shape greedy generation and the final-step item sets.

Fixed shapes. Every prompt goes into the smallest width bucket that holds it, is left padded to that width, and is
decoded in a batch of exactly `batch` rows (missing rows are a fixed dummy prompt that counts as finished from the
start). Matrix products and attention then see the same shapes whatever the other rows hold, so a row's tokens depend
only on its own prompt and on what the arm changes for it. A method that does not touch an item therefore gives the
base model's answer bit for bit, and damage has no batch-noise floor (checked on the GPU in the smoke run).

Decoding is SO-08's (so_model.Writable._greedy): explicit position ids from the mask, KV cache, stop at a newline
token or EOS, at most max_new tokens; the text is everything before the first newline, stripped.

Firing logs. Hooks (the KT routing hook, the GRACE hook) append (forward index, width, rows, positions, entries)
records to a list owned by this module while a batch runs; `relevant()` keeps the records that can change a row's answer: prompt
positions that are not padding, and decode inputs before the row's last generated token.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from so_data import norm, shots_text
from so_model import fmt_prompt


@dataclass
class GenOut:
    text: str
    n_gen: int
    fired: tuple            # sorted (entry, region) pairs; region 0 context, 1 question, 2 decode


class FixedShapeGen:
    """Greedy generation in fixed-shape batches. `loggers` are objects with a `log` attribute (list or None) and a
    `_fwd` counter, e.g. the KT LoggedEditor and the GRACE hook; their records are parsed per row."""

    def __init__(self, wm, widths=(96, 128, 192, 256, 384), batch: int = 64, max_new: int = 24,
                 dummy: str = 'Q: ?\nA:'):
        self.wm, self.widths, self.batch, self.max_new = wm, tuple(sorted(widths)), int(batch), int(max_new)
        self.dummy = wm.tok.encode(dummy, add_special_tokens=False)
        if len(self.dummy) > self.widths[0]:
            raise ValueError('dummy prompt longer than the smallest width')
        self.loggers: list = []

    def width(self, n: int) -> int:
        for w in self.widths:
            if n <= w:
                return w
        raise ValueError(f'prompt of {n} tokens exceeds the largest width {self.widths[-1]}')

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], nctx: list[int] | None = None) -> list[GenOut]:
        """prompts: token ids per item; nctx: context tokens per item (for the firing regions)."""
        nctx = nctx or [0] * len(prompts)
        out: list[GenOut | None] = [None] * len(prompts)
        by_w: dict[int, list[int]] = {}
        for i, ids in enumerate(prompts):
            by_w.setdefault(self.width(len(ids)), []).append(i)
        for w in sorted(by_w):
            idx = by_w[w]
            for s in range(0, len(idx), self.batch):
                part = idx[s:s + self.batch]
                res = self._batch([prompts[i] for i in part], w, [nctx[i] for i in part])
                for i, r in zip(part, res):
                    out[i] = r
        return out

    def _batch(self, seqs, width, nctx):
        wm = self.wm
        n_real = len(seqs)
        rows = list(seqs) + [self.dummy] * (self.batch - n_real)
        x = torch.full((self.batch, width), wm.pad, dtype=torch.long)
        mask = torch.zeros((self.batch, width), dtype=torch.long)
        for r, s in enumerate(rows):
            x[r, width - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[r, width - len(s):] = 1
        x, mask = x.to(wm.device), mask.to(wm.device)
        for lg in self.loggers:
            lg.log, lg._fwd = [], 0
        try:
            gen = self._greedy(x, mask, n_real)
        finally:
            logs = []
            for lg in self.loggers:
                logs.append(lg.log)
                lg.log = None
        lens = [len(s) for s in seqs]
        gl = [len(g) for g in gen]
        fired = [set() for _ in seqs]
        for log in logs:
            for r, pair in relevant(log or [], width, lens, nctx, gl):
                fired[r].add(pair)
        return [GenOut(wm.tok.decode(g, skip_special_tokens=True).split('\n', 1)[0].strip(), len(g),
                       tuple(sorted(f))) for g, f in zip(gen, fired)]

    def _greedy(self, x, mask, n_real):
        wm = self.wm
        n = x.shape[0]
        pos = (mask.cumsum(1) - 1).clamp_min(0)
        out = wm.model(input_ids=x, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
        wm.costs['forward'] += 1
        wm.costs['prompt_tokens'] += float(mask[:n_real].sum())
        cache, logits, last = out.past_key_values, out.logits[:, -1].float(), pos[:, -1]
        gen = [[] for _ in range(n_real)]
        done = torch.zeros(n, dtype=torch.bool)
        done[n_real:] = True
        for step in range(self.max_new):
            nxt = torch.log_softmax(logits, -1).argmax(-1)
            nxt_cpu = nxt.cpu()
            for i in range(n_real):
                if not done[i]:
                    gen[i].append(int(nxt_cpu[i]))
            done |= wm.stop[nxt_cpu]
            if bool(done.all()) or step + 1 == self.max_new:
                break
            mask = torch.cat([mask, torch.ones((n, 1), dtype=mask.dtype, device=mask.device)], 1)
            last = last + 1
            out = wm.model(input_ids=nxt[:, None], attention_mask=mask, position_ids=last[:, None],
                           past_key_values=cache, use_cache=True)
            wm.costs['forward'] += 1
            cache, logits = out.past_key_values, out.logits[:, -1].float()
        wm.costs['generated_tokens'] += sum(map(len, gen))
        return gen


def relevant(log, width: int, lens, nctx, gen_lens):
    """(row, (entry, region)) for logged firings that can change a real row's answer. Forward 0 is the prompt forward
    (left padded to `width`): positions before the row's first token are padding; then nctx context tokens (region 0)
    and the question (region 1). Decode forward k >= 1 feeds generated token k-1 and predicts token k, so it matters
    only if k <= gen_lens[r] - 1 (region 2). Rows beyond len(lens) are dummies."""
    out = []
    for fwd, w, rows, pos, ents in log:
        if fwd == 0 and w != width:
            raise RuntimeError(f'prompt forward width {w} != {width}')
        for r, p, e in zip(rows, pos, ents):
            if r >= len(lens):
                continue
            if fwd == 0:
                start = width - lens[r]
                if p < start:
                    continue
                region = 0 if p < start + nctx[r] else 1
            else:
                if fwd > gen_lens[r] - 1:
                    continue
                region = 2
            out.append((r, (int(e), region)))
    return out


# ---- items ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    kind: str          # 'reph', 'pers', 'upd' or 'loc'
    question: str
    valid: frozenset   # normalised answers that count as right
    key: str           # edit uid, or the locality question


def edit_items(stream, index, step: int) -> list[Item]:
    """Every edit's rephrase, persona and original wording, scored against the answers valid at `step`."""
    items = []
    for kind, field in (('reph', 'rephrase'), ('pers', 'personas'), ('upd', 'update')):
        for e in stream.edits:
            q = getattr(e, field)
            if q:
                items.append(Item(kind, q, frozenset(index.valid(e.key, step)), e.uid))
    return items


def loc_items(pairs) -> list[Item]:
    return [Item('loc', q, frozenset({norm(a)}), q) for q, a in pairs]


def score(items: list[Item], outs: list[GenOut]) -> tuple[str, str, dict]:
    """ok bits, fired bits (any relevant firing) and the normalised answers of wrong locality items."""
    ok, fired, wrong = [], [], {}
    for i, (it, o) in enumerate(zip(items, outs)):
        a = norm(o.text)
        good = a in it.valid
        ok.append('1' if good else '0')
        fired.append('1' if o.fired else '0')
        if it.kind == 'loc' and not good:
            wrong[i] = a
    return ''.join(ok), ''.join(fired), wrong


def plain_prompts(tok, items: list[Item], context: str):
    """Token ids of fmt_prompt(question, context) and the context length, as SO-08's generator encodes them."""
    nctx = len(tok.encode(context, add_special_tokens=False)) if context else 0
    return [tok.encode(fmt_prompt(it.question, context), add_special_tokens=False) for it in items], [nctx] * len(items)


def stream_context(stream) -> str:
    return shots_text(stream.shots)
