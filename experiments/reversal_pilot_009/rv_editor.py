"""RV-09 model layer: SO-08's Writable plus a key-based local editor (MEMIT family).

EditWritable adds to SO-08's Writable (so_model.py, unchanged):
- editor state (`extra`, e.g. the accumulated keys of earlier edits) that snapshot()/restore() carry, so branches
  stay exact;
- commit_factors(): append an exact update dW = B A to a site's committed stack;
- optional gradient-norm clipping in fit() (SO-08 round 2 found unclipped Adam 3e-3 unstable on larger writes).

LocalEditor writes a batch of facts (question, answer, subject) into the MLP down projections of a few
early-middle layers, as in MEMIT (Meng et al., 2023):
1. target vectors: for each fact, an offset added to the last edit layer's output at the subject's last token is
   optimised so that the model produces the answer, averaged over context templates, with a KL term on
   "{subject} is a" and weight decay; z = initial output + offset;
2. for each edit layer in order: keys k = down_proj inputs at the subject's last token (averaged over templates),
   residual share r = (z - current output) / layers left, and a least-squares update that maps the keys to their
   residual share while keeping generic keys (covariance C0 of public text, weight lam) and, with protect_prev,
   the keys of earlier edits (accumulated C_p) unchanged:
     mode 'memit':  dW = R K^T (lam C0 + K K^T + C_p)^-1
     mode 'alpha':  dW^T = (P (K K^T + C_p) + l2 I)^-1 P K R^T, P the projector onto C0's small-eigenvalue space
                    (AlphaEdit, Fang et al., 2025);
3. dW is committed as the exact factor pair B = R (out x n), A = (M^-1 K)^T (n x in).
Write randomness: the random context prefixes are sampled from the base model with the write seed.
"""
from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

SO08 = Path(__file__).resolve().parents[1] / 'self_observation_008'
if str(SO08) not in sys.path:
    sys.path.insert(0, str(SO08))

from so_model import Writable, fmt_prompt, fmt_target  # noqa: E402

START_WORDS = ('The', 'Therefore', 'Because', 'I', 'You')


class StopForward(Exception):
    """Raised by a hook once the needed activations are captured, to skip the rest of the forward."""


class EditWritable(Writable):
    EXTRA = '__extra__'

    def __init__(self, model, tok, sites, rank=4, scale=1.0, grad_ckpt=False, clip=None):
        super().__init__(model, tok, sites, rank=rank, scale=scale, grad_ckpt=grad_ckpt)
        self.extra: dict[str, torch.Tensor] = {}
        self.clip = clip

    # ---- state ------------------------------------------------------------------------------------------
    def snapshot(self):
        snap = super().snapshot()
        snap[self.EXTRA] = {k: v.clone() for k, v in self.extra.items()}
        return snap

    def empty_state(self):
        snap = super().empty_state()
        snap[self.EXTRA] = {}
        return snap

    @torch.no_grad()
    def restore(self, snap):
        super().restore(snap)
        self.extra = {k: v.clone() for k, v in snap.get(self.EXTRA, {}).items()}

    @torch.no_grad()
    def commit_factors(self, name: str, a: torch.Tensor, b: torch.Tensor):
        """Append dW = b @ a to the committed stack of site `name` (a: r x in, b: out x r), exactly."""
        m = self.mods[name]
        if a.dim() != 2 or b.dim() != 2 or a.shape[1] != m.Ac.shape[1] or b.shape[0] != m.Bc.shape[0] \
                or a.shape[0] != b.shape[1]:
            raise ValueError(f'factor shapes {tuple(a.shape)}, {tuple(b.shape)} do not fit {name}')
        m.Ac = torch.cat([m.Ac, a.to(m.Ac.device, torch.float32)], 0)
        m.Bc = torch.cat([m.Bc, b.to(m.Bc.device, torch.float32)], 1)

    def committed_rank(self) -> dict:
        return {n: int(m.Ac.shape[0]) for n, m in self.mods.items()}

    # ---- training with optional clipping -------------------------------------------------------------------
    def fit(self, items, epochs: int, lr: float, batch_size: int, seed: int) -> list[float]:
        """As Writable.fit; with clip set, the gradient norm of the write is clipped before every step."""
        if self.clip is None:
            return super().fit(items, epochs, lr, batch_size, seed)
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
                    torch.nn.utils.clip_grad_norm_(params, self.clip)
                    opt.step()
                    losses.append(float(loss.detach()))
                    self.costs['backward'] += 1
        finally:
            self.model.eval()
            for p in params:
                p.requires_grad_(False)
                p.grad = None
        return losses


# ---- token lookup ----------------------------------------------------------------------------------------
def _encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


def lookup(tok, text: str, subject: str | None, start: int = 0) -> tuple[list[int], int, bool]:
    """Token ids of `text` and the index of the token holding the last character of the last occurrence of
    `subject` in text[start:] (exact case, then case-insensitive). Without a match (found=False), the last
    non-space character of the question stands in: the text before the final '\\nA:' if there is one, else all of
    text[start:]. Ids are tok.encode(text), as the SO-08 generator uses."""
    ids = _encode(tok, text)
    pos = text.rfind(subject, start) if subject else -1
    if pos < 0 and subject:
        pos = text.lower().rfind(subject.lower(), start)
    found = pos >= 0
    if found:
        last = pos + len(subject) - 1
    else:
        q_end = text.rfind('\nA:')
        body = text[start:q_end] if q_end >= start else text[start:]
        last = start + len(body.rstrip()) - 1
    try:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    except TypeError:
        enc = None
    if enc is not None and 'offset_mapping' in enc:
        if list(enc['input_ids']) != list(ids):
            raise RuntimeError('tokenizer call and encode() disagree')
        idx = None
        for i, (a, b) in enumerate(enc['offset_mapping']):
            if a <= last < b:
                idx = i
        if idx is None:  # the character sits in a gap of the offsets: take the last token starting before it
            idx = max(i for i, (a, _) in enumerate(enc['offset_mapping']) if a <= last)
    else:
        idx = len(_encode(tok, text[:last + 1])) - 1
    return ids, idx, found


def _qa_question_start(context: str) -> int:
    return len(context)


# ---- the editor -------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LocalCfg:
    layers: tuple = (4, 5, 6, 7, 8)
    mode: str = 'memit'
    lam: float = 15000.0
    protect_prev: bool = True
    null_rel: float = 1e-2
    l2: float = 10.0
    v_steps: int = 25
    v_lr: float = 0.5
    v_wd: float = 0.5
    clamp: float = 0.75
    kl: float = 0.0625
    stop_loss: float = 0.05
    n_rand: int = 2
    rand_len: int = 10
    use_context: bool = True
    chunk: int = 8
    key_batch: int = 64
    site: str = 'model.layers.{i}.mlp.down_proj'
    block: str = 'model.layers.{i}'

    tau: float = 0.9            # route: centred-cosine threshold for a key match
    massive: float = 200.0      # route: positions whose block-output norm exceeds this never match (first token)
    probe_margin: float | None = None  # route: per-entry threshold max(tau, max sim over public probe tokens + margin)

    def __post_init__(self):
        if self.mode not in ('memit', 'alpha', 'route'):
            raise ValueError(f'unknown local mode {self.mode}')
        if list(self.layers) != sorted(set(self.layers)) or not self.layers:
            raise ValueError('layers must be strictly increasing and non-empty')


@dataclass
class Fact:
    question: str
    answer: str
    subject: str | None


class LocalEditor:
    """MEMIT-family editor. C0 (and P for 'alpha') are fixed after prepare(); per-state memory (C_p) lives in
    wm.extra, so it follows snapshots."""

    def __init__(self, cfg: LocalCfg):
        self.cfg = cfg
        self.c0: dict[int, torch.Tensor] = {}
        self.proj: dict[int, torch.Tensor] = {}
        self.mu: torch.Tensor | None = None
        self.prep_info: dict = {}
        self._templates: dict = {}
        self.route_off = False
        self.hits = 0
        self._handle = None
        self.probe_states: torch.Tensor | None = None
        self.probe_info: dict = {}

    @torch.no_grad()
    def set_probes(self, wm: EditWritable, questions, context: str, batch_size: int = 64) -> dict:
        """Centred unit raw block outputs (base state, routing off) at every non-massive position of the public
        format context and of each public unlabelled probe question asked after it. Entries get thresholds above
        their largest similarity to these positions (probe_margin), so they cannot fire on them."""
        cfg = self.cfg
        block = self._block(wm, cfg.layers[-1])
        saved = wm.snapshot()
        wm.restore(wm.empty_state())
        n_ctx = len(_encode(wm.tok, context)) if context else 0
        texts = [fmt_prompt(q, context) for q in questions]
        if context:
            texts.append(context)
        chunks, cur = [], {}

        def hook(mod, inp, out):
            h = (out[0] if isinstance(out, tuple) else out).float()
            ok = cur['m'].clone()
            ok[:, 0] = False
            if cur['skip_ctx']:
                ok[:, :n_ctx] = False
            ok &= h.norm(dim=-1) < cfg.massive
            chunks.append(F.normalize(h[ok] - self.mu, dim=-1))
            raise StopForward
        handle = block.register_forward_hook(hook)
        off, self.route_off = self.route_off, True
        try:
            for s in range(0, len(texts), batch_size):
                part = texts[s:s + batch_size]
                x, mask = self._pad([_encode(wm.tok, t) for t in part], wm.pad)
                cur['m'] = mask.bool().to(wm.device)
                cur['skip_ctx'] = not (context and part[-1] == context and s + len(part) == len(texts))
                try:
                    wm._decoder()(input_ids=x.to(wm.device), attention_mask=mask.to(wm.device), use_cache=False)
                except StopForward:
                    pass
        finally:
            handle.remove()
            self.route_off = off
            wm.restore(saved)
        self.probe_states = torch.cat(chunks, 0)
        self.probe_info = dict(questions=len(questions), positions=int(self.probe_states.shape[0]))
        return self.probe_info

    # ---- key-routed residual memory (mode 'route') ----------------------------------------------------------
    def attach(self, wm: EditWritable):
        """Install the routing hook on the last edit layer's block (mode 'route'). Entries live in wm.extra
        ('route_k': centred unit keys, 'route_v': offsets), so snapshots carry them. At every position whose
        centred block output has cosine >= tau with some key (and whose norm is not massive), the offset of the
        most similar key (newest on ties) is added. With no entries the hook returns the output unchanged."""
        if self.cfg.mode != 'route' or self._handle is not None:
            return
        block = self._block(wm, self.cfg.layers[-1])

        def hook(mod, inp, out):
            k = wm.extra.get('route_k')
            if self.route_off or k is None or k.shape[0] == 0:
                return out
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            hc = F.normalize(hf - self.mu, dim=-1)
            sims = hc @ k.T
            thr = wm.extra.get('route_t')
            if thr is not None:  # per-entry thresholds: only entries above their own threshold compete
                sims = torch.where(sims >= thr, sims, torch.full_like(sims, -2.0))
            best, rev = sims.flip(-1).max(-1)
            idx = k.shape[0] - 1 - rev
            hit = (best >= (self.cfg.tau if thr is None else -1.0)) & (hf.norm(dim=-1) < self.cfg.massive)
            if not bool(hit.any()):
                return out
            self.hits += int(hit.sum())
            h2 = h + (wm.extra['route_v'][idx] * hit.unsqueeze(-1)).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        self._handle = block.register_forward_hook(hook)

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @torch.no_grad()
    def block_keys(self, wm: EditWritable, facts: list[Fact], templates: list[str]) -> torch.Tensor:
        """Raw last-edit-layer block outputs at the subject's last token, mean over templates (routing off)."""
        cfg = self.cfg
        block = self._block(wm, cfg.layers[-1])
        jobs = [(i, *self._prompt(wm, tmpl, f)[:2]) for i, f in enumerate(facts) for tmpl in templates]
        d = wm.model.get_output_embeddings().weight.shape[1]
        ksum = torch.zeros(len(facts), d, device=wm.device, dtype=torch.float64)
        kcnt = torch.zeros(len(facts), device=wm.device, dtype=torch.float64)
        cur = {}

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            ksum.index_add_(0, cur['req'], h[cur['rows'], cur['look']].to(torch.float64))
            kcnt.index_add_(0, cur['req'], torch.ones(len(cur['req']), device=h.device, dtype=torch.float64))
            raise StopForward
        handle = block.register_forward_hook(hook)
        off, self.route_off = self.route_off, True
        try:
            for s in range(0, len(jobs), cfg.key_batch):
                part = jobs[s:s + cfg.key_batch]
                x, mask = self._pad([p[1] for p in part], wm.pad)
                cur['rows'] = torch.arange(len(part), device=wm.device)
                cur['look'] = torch.tensor([p[2] for p in part], device=wm.device)
                cur['req'] = torch.tensor([p[0] for p in part], device=wm.device)
                try:
                    wm._decoder()(input_ids=x.to(wm.device), attention_mask=mask.to(wm.device), use_cache=False)
                except StopForward:
                    pass
        finally:
            handle.remove()
            self.route_off = off
        return (ksum / kcnt.clamp_min(1).unsqueeze(1)).float()

    @torch.no_grad()
    def _prepare_route(self, wm: EditWritable, texts, batch_size: int = 64):
        """mu = mean raw block output over the corpus positions after the first whose norm is not massive."""
        cfg = self.cfg
        block = self._block(wm, cfg.layers[-1])
        d = wm.model.get_output_embeddings().weight.shape[1]
        acc = torch.zeros(d, device=wm.device, dtype=torch.float64)
        cnt, cur = 0, {}

        def hook(mod, inp, out):
            nonlocal cnt
            h = (out[0] if isinstance(out, tuple) else out).float()
            ok = cur['m'].clone()
            ok[:, 0] = False
            ok &= h.norm(dim=-1) < cfg.massive
            acc.add_(h[ok].sum(0).to(torch.float64))
            cnt += int(ok.sum())
            raise StopForward
        handle = block.register_forward_hook(hook)
        off, self.route_off = self.route_off, True
        try:
            for s in range(0, len(texts), batch_size):
                x, mask = self._pad([_encode(wm.tok, t) for t in texts[s:s + batch_size]], wm.pad)
                cur['m'] = mask.bool().to(wm.device)
                try:
                    wm._decoder()(input_ids=x.to(wm.device), attention_mask=mask.to(wm.device), use_cache=False)
                except StopForward:
                    pass
        finally:
            handle.remove()
            self.route_off = off
        self.mu = (acc / max(cnt, 1)).float()
        return dict(tokens=cnt, mu_norm=float(self.mu.norm()))

    def site(self, layer: int) -> str:
        return self.cfg.site.format(i=layer)

    def _block(self, wm, layer: int):
        return wm.model.get_submodule(self.cfg.block.format(i=layer))

    # ---- generic statistics -------------------------------------------------------------------------------
    @torch.no_grad()
    def prepare(self, wm: EditWritable, texts, batch_size: int = 64):
        """C0[l] = mean k k^T over every token of `texts` (public text), on the base state. For 'alpha' also the
        projector onto the eigen-directions of C0 with eigenvalue below null_rel times the mean eigenvalue."""
        cfg = self.cfg
        t0 = time.time()
        saved = wm.snapshot()
        wm.restore(wm.empty_state())
        if cfg.mode == 'route':
            try:
                info = dict(texts=len(texts), route=self._prepare_route(wm, texts, batch_size))
            finally:
                wm.restore(saved)
            info['seconds'] = time.time() - t0
            self.prep_info = info
            self.c0 = {cfg.layers[-1]: None}  # marks prepared
            self.attach(wm)
            return info
        sums = {l: torch.zeros(wm.mods[self.site(l)].base.in_features, wm.mods[self.site(l)].base.in_features,
                               dtype=torch.float64, device=wm.device) for l in cfg.layers}
        count, cur_mask = 0, {}
        hooks = []

        def make_hook(l):
            def hook(mod, inp):
                x = inp[0]
                m = cur_mask['m']
                v = x[m].float()
                sums[l].add_((v.T @ v).to(torch.float64))
                if l == cfg.layers[-1]:
                    raise StopForward
            return hook
        for l in cfg.layers:
            hooks.append(wm.mods[self.site(l)].register_forward_pre_hook(make_hook(l)))
        try:
            for s in range(0, len(texts), batch_size):
                ids = [_encode(wm.tok, t) for t in texts[s:s + batch_size]]
                width = max(map(len, ids))
                x = torch.full((len(ids), width), wm.pad, dtype=torch.long)
                mask = torch.zeros((len(ids), width), dtype=torch.long)
                for i, seq in enumerate(ids):
                    x[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                    mask[i, :len(seq)] = 1
                x, mask = x.to(wm.device), mask.to(wm.device)
                cur_mask['m'] = mask.bool()
                count += int(mask.sum())
                try:
                    wm._decoder()(input_ids=x, attention_mask=mask, use_cache=False)
                except StopForward:
                    pass
        finally:
            for hk in hooks:
                hk.remove()
            wm.restore(saved)
        info = dict(texts=len(texts), tokens=count, seconds=None, layers={})
        for l in cfg.layers:
            c = sums[l] / max(count, 1)
            self.c0[l] = c
            ev = torch.linalg.eigvalsh(c)
            rec = dict(trace=float(ev.sum()), max_eig=float(ev[-1]), mean_eig=float(ev.mean()),
                       min_eig=float(ev[0]))
            if cfg.mode == 'alpha':
                vals, vecs = torch.linalg.eigh(c)
                keep = vals < cfg.null_rel * vals.mean()
                u = vecs[:, keep]
                self.proj[l] = u @ u.T
                rec.update(null_dim=int(keep.sum()), null_energy=float(vals[keep].sum() / vals.sum()))
            info['layers'][l] = rec
        info['seconds'] = time.time() - t0
        self.prep_info = info
        return info

    # ---- context templates -------------------------------------------------------------------------------
    @torch.no_grad()
    def random_prefixes(self, wm: EditWritable, seed: int) -> list[str]:
        """n_rand short prefixes sampled (top-5) from the base model with this seed, as MEMIT's context templates."""
        cfg = self.cfg
        if cfg.n_rand <= 0:
            return []
        key = (seed, cfg.n_rand, cfg.rand_len)
        if key in self._templates:
            return list(self._templates[key])
        rng = random.Random(seed * 7919 + 17)
        g = torch.Generator().manual_seed(seed * 104_729 + 3)
        saved = wm.snapshot()
        wm.restore(wm.empty_state())
        out = []
        try:
            for _ in range(cfg.n_rand):
                ids = _encode(wm.tok, rng.choice(START_WORDS))
                for _ in range(cfg.rand_len):
                    x = torch.tensor([ids], dtype=torch.long, device=wm.device)
                    logits = wm.model(input_ids=x, use_cache=False).logits[0, -1].float().cpu()
                    top = torch.topk(logits, 5)
                    p = torch.softmax(top.values, -1)
                    ids.append(int(top.indices[int(torch.multinomial(p, 1, generator=g))]))
                text = wm.tok.decode(ids, skip_special_tokens=True)
                text = ' '.join(text.replace('{', ' ').replace('}', ' ').split())
                out.append(text.rstrip('.') + '. ')
        finally:
            wm.restore(saved)
        self._templates[key] = tuple(out)
        return out

    def templates(self, wm, seed: int, context: str) -> list[str]:
        t = ['']
        if self.cfg.use_context and context:
            t.append(context)
        return t + self.random_prefixes(wm, seed)

    # ---- sequences ---------------------------------------------------------------------------------------
    @staticmethod
    def _pad(seqs, pad):
        width = max(map(len, seqs))
        x = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1
        return x, mask

    def _prompt(self, wm, template: str, fact: Fact):
        text = fmt_prompt(fact.question, template)
        return lookup(wm.tok, text, fact.subject, _qa_question_start(template))

    # ---- step 1: target vectors ----------------------------------------------------------------------------
    def _parts(self, wm, chunk: list[Fact], templates: list[str]):
        """One batch per template (rewriting prompts with the answer appended) plus one of KL prompts, so that
        sequences in a batch have similar lengths. Part 0 is the bare template."""
        parts = []
        for j, tmpl in enumerate(templates):
            seqs, look, req, spans = [], [], [], []
            for i, f in enumerate(chunk):
                t_ids = _encode(wm.tok, fmt_target(f.answer))
                p_ids, idx, _ = self._prompt(wm, tmpl, f)
                seqs.append(p_ids + t_ids)
                look.append(idx)
                req.append(i)
                spans.append((len(p_ids), len(p_ids) + len(t_ids)))
            parts.append(dict(kind='rw', seqs=seqs, look=look, req=req, spans=spans))
        if self.cfg.kl > 0:
            seqs, look, req = [], [], []
            for i, f in enumerate(chunk):
                if f.subject:
                    ids, idx, _ = lookup(wm.tok, f'{f.subject} is a', f.subject, 0)
                    seqs.append(ids)
                    look.append(idx)
                    req.append(i)
            if seqs:
                parts.append(dict(kind='kl', seqs=seqs, look=look, req=req, spans=None))
        for p in parts:
            x, mask = self._pad(p['seqs'], wm.pad)
            p['x'], p['mask'] = x.to(wm.device), mask.to(wm.device)
            p['rows'] = torch.arange(len(p['seqs']), device=wm.device)
            p['look_t'] = torch.tensor(p['look'], device=wm.device)
            p['req_t'] = torch.tensor(p['req'], device=wm.device)
            if p['kind'] == 'rw':
                tr, tc = [], []
                for r, (a, b) in enumerate(p['spans']):
                    for j in range(a, b):
                        tr.append(r)
                        tc.append(j - 1)  # the prediction for token j sits at position j - 1
                p['tr'] = torch.tensor(tr, device=wm.device)
                p['tc'] = torch.tensor(tc, device=wm.device)
                p['tgt'] = p['x'][p['tr'], p['tc'] + 1]
        return parts

    def targets(self, wm: EditWritable, facts: list[Fact], templates: list[str]) -> tuple[torch.Tensor, dict]:
        """Batched MEMIT target optimisation. The facts' offsets are independent (the loss is a sum over facts), so
        a chunk is optimised jointly; each part's loss is backpropagated on its own (gradients accumulate), which
        keeps only one part's activations in memory. Facts whose total loss falls below stop_loss are frozen."""
        cfg = self.cfg
        block = self._block(wm, cfg.layers[-1])
        head = wm.model.get_output_embeddings()
        d = head.weight.shape[1]
        m = len(templates)
        zs, rows_info, deltas = [], [], []
        for c0 in range(0, len(facts), cfg.chunk):
            chunk = facts[c0:c0 + cfg.chunk]
            n = len(chunk)
            parts = self._parts(wm, chunk, templates)
            delta = torch.zeros(n, d, device=wm.device, dtype=torch.float32, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=cfg.v_lr)
            state = dict(h_init=None, part=None)

            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                p = state['part']
                if state['h_init'] is None:  # first forward is the bare part with delta = 0
                    hb = torch.zeros(n, h.shape[-1], device=h.device, dtype=torch.float32)
                    hb[p['req_t']] = h[p['rows'], p['look_t']].detach().float()
                    state['h_init'] = hb
                add = torch.zeros(h.shape, device=h.device, dtype=torch.float32)
                add = add.index_put((p['rows'], p['look_t']), delta[p['req_t']], accumulate=True)
                h2 = h + add.to(h.dtype)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            handle = block.register_forward_hook(hook)
            active = torch.ones(n, dtype=torch.bool, device=wm.device)
            kl_init = None
            last = None
            try:
                for it in range(cfg.v_steps):
                    final = it == cfg.v_steps - 1
                    opt.zero_grad(set_to_none=True)
                    nll_tot = torch.zeros(n, device=wm.device)
                    kl_tot = torch.zeros(n, device=wm.device)
                    for p in parts:
                        state['part'] = p
                        hidden = wm._decoder()(input_ids=p['x'], attention_mask=p['mask'],
                                               use_cache=False).last_hidden_state
                        if p['kind'] == 'rw':
                            ce = F.cross_entropy(head(hidden[p['tr'], p['tc']]).float(), p['tgt'], reduction='none')
                            per = torch.zeros(len(p['seqs']), device=wm.device).index_add(0, p['tr'], ce)
                            cnt = torch.zeros(len(p['seqs']), device=wm.device).index_add(0, p['tr'],
                                                                                         torch.ones_like(ce))
                            each = torch.zeros(n, device=wm.device).index_add(0, p['req_t'], per / cnt) / m
                            nll_tot += each.detach()
                        else:
                            lp = torch.log_softmax(head(hidden[p['rows'], p['look_t']]).float(), -1)
                            if kl_init is None:
                                kl_init = lp.detach()
                            kl_each = (lp.exp() * (lp - kl_init)).sum(-1)
                            each = torch.zeros(n, device=wm.device).index_add(0, p['req_t'], kl_each)
                            kl_tot += each.detach()
                            each = cfg.kl * each
                        if not final:
                            each.sum().backward()
                        del hidden
                    h0n = state['h_init'].norm(dim=1).clamp_min(1e-6)
                    wd = cfg.v_wd * delta.norm(dim=1) / h0n ** 2
                    if not final:
                        wd.sum().backward()
                    loss_each = nll_tot + cfg.kl * kl_tot + wd.detach()
                    last = dict(nll=nll_tot, kl=kl_tot, loss=loss_each)
                    active &= loss_each >= cfg.stop_loss
                    if final or not bool(active.any()):
                        break
                    with torch.no_grad():
                        frozen = delta.detach().clone()
                        delta.grad[~active] = 0
                    opt.step()
                    with torch.no_grad():
                        delta[~active] = frozen[~active]
                        max_norm = cfg.clamp * h0n
                        nrm = delta.norm(dim=1)
                        over = nrm > max_norm
                        if bool(over.any()):
                            delta[over] = delta[over] * (max_norm[over] / nrm[over]).unsqueeze(1)
                    wm.costs['edit_backward'] += len(parts)
            finally:
                handle.remove()
            z = (state['h_init'] + delta.detach()).float()
            zs.append(z)
            deltas.append(delta.detach().float().clone())
            for i in range(n):
                rows_info.append(dict(nll=float(last['nll'][i]), kl=float(last['kl'][i]), loss=float(last['loss'][i]),
                                      delta_rel=float(delta.detach()[i].norm()
                                                      / state['h_init'][i].norm().clamp_min(1e-6))))
            del delta, opt, parts
        return torch.cat(zs, 0), dict(per_fact=rows_info, delta=torch.cat(deltas, 0))

    # ---- step 2: keys and current outputs ----------------------------------------------------------------
    @torch.no_grad()
    def keys_and_outputs(self, wm: EditWritable, facts: list[Fact], templates: list[str], layer: int):
        """Keys at `layer` (down_proj input at the subject's last token, mean over templates) and the last edit
        layer's output at the same token in the bare template."""
        cfg = self.cfg
        zl = cfg.layers[-1]
        site = wm.mods[self.site(layer)]
        block = self._block(wm, zl)
        jobs = []
        for i, f in enumerate(facts):
            for j, tmpl in enumerate(templates):
                ids, idx, _ = self._prompt(wm, tmpl, f)
                jobs.append((i, j, ids, idx))
        d_in = site.base.in_features
        d_out = wm.model.get_output_embeddings().weight.shape[1]
        ksum = torch.zeros(len(facts), d_in, device=wm.device, dtype=torch.float64)
        kcnt = torch.zeros(len(facts), device=wm.device, dtype=torch.float64)
        zcur = torch.zeros(len(facts), d_out, device=wm.device, dtype=torch.float32)
        cur = {}

        def key_hook(mod, inp):
            x = inp[0]
            v = x[cur['rows'], cur['look']].to(torch.float64)
            ksum.index_add_(0, cur['req'], v)
            kcnt.index_add_(0, cur['req'], torch.ones(len(cur['req']), device=x.device, dtype=torch.float64))

        def out_hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            b = cur['bare']
            if len(b):
                zcur[cur['req'][b]] = h[cur['rows'][b], cur['look'][b]].float()
            raise StopForward
        h1 = site.register_forward_pre_hook(key_hook)
        h2 = block.register_forward_hook(out_hook)
        try:
            for s in range(0, len(jobs), cfg.key_batch):
                part = jobs[s:s + cfg.key_batch]
                x, mask = self._pad([p[2] for p in part], wm.pad)
                cur['rows'] = torch.arange(len(part), device=wm.device)
                cur['look'] = torch.tensor([p[3] for p in part], device=wm.device)
                cur['req'] = torch.tensor([p[0] for p in part], device=wm.device)
                cur['bare'] = torch.tensor([r for r, p in enumerate(part) if p[1] == 0], device=wm.device,
                                           dtype=torch.long)
                try:
                    wm._decoder()(input_ids=x.to(wm.device), attention_mask=mask.to(wm.device), use_cache=False)
                except StopForward:
                    pass
                wm.costs['edit_forward'] += 1
        finally:
            h1.remove()
            h2.remove()
        return (ksum / kcnt.clamp_min(1).unsqueeze(1)).float(), zcur

    # ---- step 3: the update --------------------------------------------------------------------------------
    def solve(self, layer: int, keys: torch.Tensor, resid: torch.Tensor, cp: torch.Tensor | None):
        """keys: n x d_in, resid: n x d_out. Returns factors (A: n x d_in, B: d_out x n) with dW = B A."""
        cfg = self.cfg
        kc = keys.to(torch.float64).T          # d_in x n
        rc = resid.to(torch.float64).T         # d_out x n
        kk = kc @ kc.T
        if cp is not None:
            kk = kk + cp.to(torch.float64)
        if cfg.mode == 'memit':
            m = cfg.lam * self.c0[layer] + kk
            x = torch.linalg.solve(m, kc)       # d_in x n
        else:
            p = self.proj[layer]
            m = p @ kk + cfg.l2 * torch.eye(kk.shape[0], dtype=torch.float64, device=kk.device)
            x = torch.linalg.solve(m, p @ kc)
        return x.T.float(), rc.float()

    def edit(self, wm: EditWritable, facts: list[Fact], context: str, seed: int) -> dict:
        """Write `facts` into the current state. Returns diagnostics; wm.extra carries the accumulated keys."""
        cfg = self.cfg
        if not self.c0:
            raise RuntimeError('prepare() before edit()')
        t0 = time.time()
        facts = [f for f in facts if f.answer and f.answer.strip()]
        rec = dict(n=len(facts), layers=list(cfg.layers), mode=cfg.mode)
        if not facts:
            return rec
        templates = self.templates(wm, seed, context)
        n_found = sum(1 for f in facts if self._prompt(wm, '', f)[2])
        if cfg.mode == 'route':
            return self._edit_route(wm, facts, templates, rec, n_found, t0)
        z, zinfo = self.targets(wm, facts, templates)
        keys_all, upd = {}, {}
        for i, l in enumerate(cfg.layers):
            keys, zcur = self.keys_and_outputs(wm, facts, templates, l)
            resid = (z - zcur) / (len(cfg.layers) - i)
            cp = wm.extra.get(f'Cp{l}') if cfg.protect_prev else None
            a, b = self.solve(l, keys, resid, cp)
            wm.commit_factors(self.site(l), a, b)
            keys_all[l] = keys
            upd[l] = dict(resid_norm=float(resid.norm(dim=1).mean()), update_fro=float((b @ a).norm()),
                          key_norm=float(keys.norm(dim=1).mean()))
        _, zfin = self.keys_and_outputs(wm, facts, templates, cfg.layers[-1])
        if cfg.protect_prev:
            for l, keys in keys_all.items():
                k = keys.to(torch.float64)
                prev = wm.extra.get(f'Cp{l}')
                add = (k.T @ k).float()
                wm.extra[f'Cp{l}'] = add if prev is None else prev + add
        losses = [r['loss'] for r in zinfo['per_fact']]
        nlls = [r['nll'] for r in zinfo['per_fact']]
        rec.update(templates=len(templates), subject_found=n_found, z_loss_mean=sum(losses) / len(losses),
                   z_nll_mean=sum(nlls) / len(nlls), z_nll_max=max(nlls),
                   z_converged=sum(1 for v in nlls if v < 0.5) / len(nlls),
                   z_gap_after=float(((z - zfin).norm(dim=1) / (z.norm(dim=1).clamp_min(1e-6))).mean()),
                   delta_rel_mean=sum(r['delta_rel'] for r in zinfo['per_fact']) / len(facts),
                   layers_info=upd, seconds=time.time() - t0)
        return rec

    def _edit_route(self, wm: EditWritable, facts: list[Fact], templates: list[str], rec: dict, n_found: int,
                    t0: float) -> dict:
        """Mode 'route': offsets optimised on the raw state (routing off), keyed by the centred unit block output
        at the subject's last token (mean over templates), appended to the entries in wm.extra."""
        off, self.route_off = self.route_off, True
        try:
            _, zinfo = self.targets(wm, facts, templates)
            keys = self.block_keys(wm, facts, templates)
        finally:
            self.route_off = off
        kc = F.normalize(keys - self.mu, dim=-1)
        vals = zinfo['delta']
        old_k, old_v = wm.extra.get('route_k'), wm.extra.get('route_v')
        wm.extra['route_k'] = kc if old_k is None else torch.cat([old_k, kc], 0)
        wm.extra['route_v'] = vals if old_v is None else torch.cat([old_v, vals], 0)
        if self.cfg.probe_margin is not None:
            if self.probe_states is None:
                raise RuntimeError('probe_margin needs set_probes()')
            pmax = torch.full((kc.shape[0],), -1.0, device=kc.device)
            for s in range(0, self.probe_states.shape[0], 8192):
                pmax = torch.maximum(pmax, (self.probe_states[s:s + 8192] @ kc.T).max(0).values)
            thr = torch.clamp(pmax + self.cfg.probe_margin, min=self.cfg.tau)
            old_t = wm.extra.get('route_t')
            wm.extra['route_t'] = thr if old_t is None else torch.cat([old_t, thr], 0)
            rec['raised_thresholds'] = int((thr > self.cfg.tau).sum())
            rec['threshold_mean'] = float(thr.mean())
        near = None
        if old_k is not None and old_k.shape[0]:
            near = float(((kc @ old_k.T).max(1).values >= self.cfg.tau).float().mean())
        nlls = [r['nll'] for r in zinfo['per_fact']]
        losses = [r['loss'] for r in zinfo['per_fact']]
        rec.update(templates=len(templates), subject_found=n_found, z_loss_mean=sum(losses) / len(losses),
                   z_nll_mean=sum(nlls) / len(nlls), z_nll_max=max(nlls),
                   z_converged=sum(1 for v in nlls if v < 0.5) / len(nlls),
                   delta_rel_mean=sum(r['delta_rel'] for r in zinfo['per_fact']) / len(facts),
                   entries=int(wm.extra['route_k'].shape[0]), new_keys_matching_old=near,
                   seconds=time.time() - t0)
        return rec
