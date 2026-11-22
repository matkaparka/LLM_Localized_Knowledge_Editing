"""SO-08 model layer: a frozen causal LM with stacked low-rank writes.

Each target nn.Linear becomes StackLoRALinear: y = base(x) + x Ac^T Bc^T + scale * x A^T B^T.
(Ac, Bc) is the stack of committed writes (coefficients folded into Bc); (A, B) is the write in progress.
- With nothing committed and no write in progress the forward is exactly the base forward.
- commit(coef) adds coef times the new effective update, i.e. it interpolates in effective-ΔW space between the
  state before the write and the fully written state; shrink(beta) scales every committed update toward the
  original weights. Both are exact; no factor averaging.
- snapshot()/restore() copy the whole write state, so branches and rollbacks are exact.
Gradient checkpointing and KV caching both work (unlike pause2's WritableCausalLM).
"""
from __future__ import annotations

import random
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F


def fmt_prompt(question: str, context: str = '') -> str:
    return f'{context}Q: {question.strip()}\nA:'


def fmt_target(answer: str) -> str:
    return f' {answer.strip()}\n'


class StackLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, scale: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        out_f, in_f = base.weight.shape
        dev = base.weight.device
        self.rank, self.scale, self.active = rank, scale, False
        self.A = nn.Parameter(torch.zeros(rank, in_f, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=torch.float32))
        self.register_buffer('Ac', torch.zeros(0, in_f, device=dev, dtype=torch.float32), persistent=False)
        self.register_buffer('Bc', torch.zeros(out_f, 0, device=dev, dtype=torch.float32), persistent=False)

    def forward(self, x):
        y = self.base(x)
        if self.Ac.shape[0] == 0 and not self.active:
            return y
        cdt = torch.float64 if x.dtype == torch.float64 else torch.float32
        xc = x.to(cdt)
        d = None
        if self.Ac.shape[0]:
            d = (xc @ self.Ac.to(cdt).T) @ self.Bc.to(cdt).T
        if self.active:
            w = ((xc @ self.A.to(cdt).T) @ self.B.to(cdt).T) * self.scale
            d = w if d is None else d + w
        return y + d.to(y.dtype)

    def delta(self) -> torch.Tensor:
        """Effective ΔW of this site (committed plus in-progress), float32."""
        d = self.Bc @ self.Ac if self.Ac.shape[0] else torch.zeros_like(self.base.weight, dtype=torch.float32)
        if self.active:
            d = d + self.scale * (self.B.detach() @ self.A.detach())
        return d


class Writable:
    """Wraps a Hugging Face causal LM. Only the in-progress factors (A, B) are ever trained."""

    def __init__(self, model, tok, sites, rank=4, scale=1.0, grad_ckpt=False):
        self.model, self.tok = model, tok
        for p in model.parameters():
            p.requires_grad_(False)
        self.mods: dict[str, StackLoRALinear] = {}
        for name in sites:
            parent_name, _, child = name.rpartition('.')
            parent = model.get_submodule(parent_name)
            base = getattr(parent, child)
            if not isinstance(base, nn.Linear):
                raise TypeError(f'{name} is {type(base).__name__}, not nn.Linear')
            mod = StackLoRALinear(base, rank, scale)
            setattr(parent, child, mod)
            self.mods[name] = mod
        if not self.mods:
            raise ValueError('no write sites')
        self.device = next(model.parameters()).device
        self.eos = tok.eos_token_id
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else self.eos
        vocab = model.get_output_embeddings().weight.shape[0]
        stop = torch.zeros(vocab, dtype=torch.bool)
        for i in range(vocab):
            try:
                s = tok.decode([i], skip_special_tokens=False)
            except Exception:  # ids beyond the tokenizer vocabulary
                s = ''
            stop[i] = ('\n' in s) or i == self.eos
        self.stop = stop
        if grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.costs = defaultdict(float)
        model.eval()

    # ---- write state -------------------------------------------------------------------------------
    def snapshot(self):
        return {n: (m.Ac.clone(), m.Bc.clone(), m.A.detach().clone(), m.B.detach().clone(), m.active)
                for n, m in self.mods.items()}

    def empty_state(self):
        return {n: (torch.zeros_like(m.Ac[:0]), torch.zeros_like(m.Bc[:, :0]), torch.zeros_like(m.A.detach()),
                    torch.zeros_like(m.B.detach()), False) for n, m in self.mods.items()}

    @torch.no_grad()
    def restore(self, snap):
        for n, m in self.mods.items():
            ac, bc, a, b, act = snap[n]
            m.Ac, m.Bc = ac.clone(), bc.clone()
            m.A.copy_(a)
            m.B.copy_(b)
            m.active = act

    @torch.no_grad()
    def start_write(self, seed: int):
        for i, m in enumerate(self.mods.values()):
            g = torch.Generator().manual_seed((seed * 1_000_003 + i) % (2 ** 63 - 1))
            a = torch.randn(m.A.shape, generator=g, dtype=torch.float32) / m.A.shape[1] ** 0.5
            m.A.copy_(a.to(m.A.device))
            m.B.zero_()
            m.active = True

    @torch.no_grad()
    def commit(self, coef: float = 1.0):
        for m in self.mods.values():
            if m.active:
                m.Ac = torch.cat([m.Ac, m.A.detach().clone()], 0)
                m.Bc = torch.cat([m.Bc, (coef * m.scale) * m.B.detach().clone()], 1)
            m.A.zero_()
            m.B.zero_()
            m.active = False

    @torch.no_grad()
    def discard(self):
        for m in self.mods.values():
            m.A.zero_()
            m.B.zero_()
            m.active = False

    @torch.no_grad()
    def shrink(self, beta: float):
        for m in self.mods.values():
            m.Bc = m.Bc * beta

    def delta_norm(self) -> float:
        return float(sum(m.delta().double().pow(2).sum() for m in self.mods.values()) ** 0.5)

    def temporarily(self, snap):
        wm = self

        class _Ctx:
            def __enter__(self_):
                self_.saved = wm.snapshot()
                wm.restore(snap)

            def __exit__(self_, *exc):
                wm.restore(self_.saved)
                return False
        return _Ctx()

    # ---- training -----------------------------------------------------------------------------------
    def _decoder(self):
        dec = getattr(self.model, 'model', None)
        return dec if dec is not None else self.model.get_decoder()

    def _pack(self, items):
        seqs, spans = [], []
        for prompt, target, _ in items:
            p = self.tok.encode(prompt, add_special_tokens=False)
            t = self.tok.encode(target, add_special_tokens=False)
            if not t:
                raise ValueError('empty target')
            seqs.append(p + t)
            spans.append((len(p), len(p) + len(t)))
        width = max(map(len, seqs))
        x = torch.full((len(seqs), width), self.pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1
        w = torch.tensor([float(i[2]) for i in items], dtype=torch.float32)
        return x.to(self.device), mask.to(self.device), spans, w.to(self.device)

    def _per_example_ce(self, items):
        x, mask, spans, w = self._pack(items)
        hidden = self._decoder()(input_ids=x, attention_mask=mask, use_cache=False).last_hidden_state
        rows, cols, owner = [], [], []
        for i, (a, b) in enumerate(spans):
            for j in range(a, b):
                rows.append(i)
                cols.append(j - 1)
                owner.append(i)
        rows_t = torch.tensor(rows, device=self.device)
        cols_t = torch.tensor(cols, device=self.device)
        target = x[rows_t, cols_t + 1]
        logits = self.model.get_output_embeddings()(hidden[rows_t, cols_t]).float()
        ce = F.cross_entropy(logits, target, reduction='none')
        own = torch.tensor(owner, device=self.device)
        total = torch.zeros(len(spans), device=self.device, dtype=ce.dtype).index_add(0, own, ce)
        count = torch.zeros(len(spans), device=self.device, dtype=ce.dtype).index_add(0, own, torch.ones_like(ce))
        self.costs['train_tokens' if torch.is_grad_enabled() else 'score_tokens'] += float(mask.sum())
        return total / count, w

    def fit(self, items, epochs: int, lr: float, batch_size: int, seed: int) -> list[float]:
        """Train the in-progress factors on (prompt, target, weight) items. Call start_write first."""
        if not items:
            return []
        if not any(m.active for m in self.mods.values()):
            raise RuntimeError('start_write() before fit()')
        params = [p for m in self.mods.values() for p in (m.A, m.B)]
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=lr)
        rng = random.Random(seed)
        losses = []
        self.model.train()
        try:
            for _ in range(epochs):
                order = list(range(len(items)))
                rng.shuffle(order)
                for s in range(0, len(order), batch_size):
                    per, w = self._per_example_ce([items[j] for j in order[s:s + batch_size]])
                    loss = (per * w).sum() / w.sum()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.detach()))
                    self.costs['backward'] += 1
        finally:
            self.model.eval()
            for p in params:
                p.requires_grad_(False)
                p.grad = None
        return losses

    @torch.no_grad()
    def mean_loss(self, pairs, batch_size: int = 64) -> float | None:
        """Mean per-example answer cross-entropy of (question, answer) pairs; no training."""
        if not pairs:
            return None
        vals = []
        for s in range(0, len(pairs), batch_size):
            items = [(fmt_prompt(q), fmt_target(a), 1.0) for q, a in pairs[s:s + batch_size]]
            per, _ = self._per_example_ce(items)
            vals += per.tolist()
            self.costs['forward'] += 1
        return sum(vals) / len(vals)

    # ---- generation ---------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, questions, contexts=None, max_new: int = 24, batch_size: int = 64, use_cache: bool = True):
        """Greedy answers: list of (text before the first newline, summed log-prob, generated token count)."""
        out = []
        contexts = contexts or [''] * len(questions)
        for s in range(0, len(questions), batch_size):
            ids = [self.tok.encode(fmt_prompt(q, c), add_special_tokens=False)
                   for q, c in zip(questions[s:s + batch_size], contexts[s:s + batch_size])]
            out += self._greedy(ids, max_new) if use_cache else self._greedy_nocache(ids, max_new)
        return out

    def _left_pad(self, ids):
        width = max(map(len, ids))
        x = torch.full((len(ids), width), self.pad, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for i, s in enumerate(ids):
            x[i, width - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[i, width - len(s):] = 1
        return x.to(self.device), mask.to(self.device)

    def _finish(self, gen, lp):
        texts = [self.tok.decode(g, skip_special_tokens=True).split('\n', 1)[0].strip() for g in gen]
        self.costs['generated_tokens'] += sum(map(len, gen))
        return [(t, float(v), len(g)) for t, v, g in zip(texts, lp, gen)]

    def _greedy(self, ids, max_new):
        n = len(ids)
        x, mask = self._left_pad(ids)
        pos = (mask.cumsum(1) - 1).clamp_min(0)
        out = self.model(input_ids=x, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
        self.costs['forward'] += 1
        self.costs['prompt_tokens'] += float(mask.sum())
        cache, logits, last = out.past_key_values, out.logits[:, -1].float(), pos[:, -1]
        gen, lp = [[] for _ in range(n)], [0.0] * n
        done = torch.zeros(n, dtype=torch.bool)
        for step in range(max_new):
            logp = torch.log_softmax(logits, -1)
            nxt = logp.argmax(-1)
            nxt_cpu = nxt.cpu()
            for i in range(n):
                if not done[i]:
                    gen[i].append(int(nxt_cpu[i]))
                    lp[i] += float(logp[i, nxt_cpu[i]])
            done |= self.stop[nxt_cpu]
            if bool(done.all()) or step + 1 == max_new:
                break
            mask = torch.cat([mask, torch.ones((n, 1), dtype=mask.dtype, device=mask.device)], 1)
            last = last + 1
            out = self.model(input_ids=nxt[:, None], attention_mask=mask, position_ids=last[:, None],
                             past_key_values=cache, use_cache=True)
            self.costs['forward'] += 1
            cache, logits = out.past_key_values, out.logits[:, -1].float()
        return self._finish(gen, lp)

    def _greedy_nocache(self, ids, max_new):
        """Reference decoder: full recomputation every step (tests only)."""
        seqs = [list(s) for s in ids]
        n = len(seqs)
        gen, lp = [[] for _ in range(n)], [0.0] * n
        done = torch.zeros(n, dtype=torch.bool)
        for step in range(max_new):
            x, mask = self._left_pad(seqs)
            pos = (mask.cumsum(1) - 1).clamp_min(0)
            logits = self.model(input_ids=x, attention_mask=mask, position_ids=pos, use_cache=False).logits[:, -1]
            logp = torch.log_softmax(logits.float(), -1)
            nxt = logp.argmax(-1).cpu()
            for i in range(n):
                seqs[i].append(int(nxt[i]))
                if not done[i]:
                    gen[i].append(int(nxt[i]))
                    lp[i] += float(logp[i, nxt[i]])
            done |= self.stop[nxt]
            if bool(done.all()):
                break
        return self._finish(gen, lp)
