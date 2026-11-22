"""EC-11 GRACE-IV: GRACE (Hartvigsen et al., NeurIPS 2023) as EasyEdit runs it on decoder-only models, with every
edit's value trained once and independently.

Adaptor at `model.layers[l].mlp.down_proj` (EasyEdit's site for Llama/Qwen):
- key: the module's input at the query position, Euclidean distance (fp32, exact differences);
- query position: the last position of every forward, i.e. the prompt's last token in the prompt forward (left
  padding) and the new token in each decode forward (EasyEdit's replace_last with a KV cache behaves this way);
- the nearest key fires when its distance is at most its radius; its value replaces the module's output there.

Codebook (EasyEdit's edit(), in stream order): with eps_init the initial radius and i* the nearest key at distance d,
  d > eps_init + eps[i*]           add (key, value, eps_init)
  else, different answer           add the key, then eps[i*] = d/2 - 1e-5 and the new key's eps = d/2
  else, same answer                eps[i*] = d if d > eps[i*] (coverage); no key is added
"Same answer" compares normalised answers (EasyEdit compares label-tensor means, and prepends labels where it
appends keys; both are fixed here).

Values: EasyEdit trains the value of the chosen key for n_iter Adam steps (lr edit_lr) on the edit's teacher-forced
loss, from a cold start U[0, 1). Here every edit's value is trained that way once, in batches (values are separate
parameters and Adam is element-wise, so batching does not change them). Deviation: when an edit only expands an
existing key, EasyEdit would continue training that key's value on the new edit; here the old value is kept and the
new edit's value is discarded. Expansion counts are recorded.

Keys and probe keys are read from the evaluation's own fixed-shape prompt forward (same width bucket, padding and batch
layout), so an edit's original question sits at distance 0 from its key at evaluation time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from rv_editor import StopForward

SITE = 'model.layers.{l}.mlp.down_proj'


def site(wm, layer: int):
    return wm.model.get_submodule(SITE.format(l=layer))


def cdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a.float(), b.float(), compute_mode='donot_use_mm_for_euclid_dist')


# ---- keys from the fixed-shape prompt forward ----------------------------------------------------------------
@torch.no_grad()
def fixed_keys(gen, layer: int, prompts: list[list[int]]) -> torch.Tensor:
    """down_proj input at the last position of each prompt's fixed-shape prompt forward (FixedShapeGen layout)."""
    wm = gen.wm
    mod = site(wm, layer)
    d_in = mod.base.in_features if hasattr(mod, 'base') else mod.in_features
    out = torch.zeros(len(prompts), d_in, device=wm.device, dtype=torch.float32)
    by_w: dict[int, list[int]] = {}
    for i, ids in enumerate(prompts):
        by_w.setdefault(gen.width(len(ids)), []).append(i)
    cur = {}

    def hook(m, inp):
        cur['k'] = inp[0][:, -1].float()
        raise StopForward
    h = mod.register_forward_pre_hook(hook)
    try:
        for w in sorted(by_w):
            idx = by_w[w]
            for s in range(0, len(idx), gen.batch):
                part = idx[s:s + gen.batch]
                rows = [prompts[i] for i in part] + [gen.dummy] * (gen.batch - len(part))
                x = torch.full((gen.batch, w), wm.pad, dtype=torch.long)
                mask = torch.zeros((gen.batch, w), dtype=torch.long)
                for r, sq in enumerate(rows):
                    x[r, w - len(sq):] = torch.tensor(sq, dtype=torch.long)
                    mask[r, w - len(sq):] = 1
                x, mask = x.to(wm.device), mask.to(wm.device)
                pos = (mask.cumsum(1) - 1).clamp_min(0)
                try:
                    wm.model(input_ids=x, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
                except StopForward:
                    pass
                out[torch.tensor(part, device=wm.device)] = cur['k'][:len(part)]
    finally:
        h.remove()
    return out


def probe_scale(keys: torch.Tensor) -> float:
    """Median over probe keys of the distance to the nearest other probe key."""
    d = cdist(keys, keys)
    d.fill_diagonal_(float('inf'))
    return float(d.min(1).values.median())


# ---- value training ------------------------------------------------------------------------------------------
@dataclass
class Values:
    values: torch.Tensor                 # E x d_out
    loss_first: list = field(default_factory=list)
    loss_last: list = field(default_factory=list)


def train_values(wm, layer: int, prompts: list[list[int]], targets: list[list[int]], n_iter: int, lr: float,
                 seed: int, chunk: int = 32) -> Values:
    """Each edit's replacement value at its prompt's last token: cold start U[0, 1), n_iter Adam steps on the mean
    answer cross-entropy (teacher forced), nothing else trained. Right-padded batches of `chunk` edits."""
    mod = site(wm, layer)
    head = wm.model.get_output_embeddings()
    d_out = mod.base.out_features if hasattr(mod, 'base') else mod.out_features
    vals = torch.zeros(len(prompts), d_out, device=wm.device, dtype=torch.float32)
    first, last = [0.0] * len(prompts), [0.0] * len(prompts)
    for c0 in range(0, len(prompts), chunk):
        part = list(range(c0, min(c0 + chunk, len(prompts))))
        seqs = [prompts[i] + targets[i] for i in part]
        width = max(map(len, seqs))
        x = torch.full((len(part), width), wm.pad, dtype=torch.long)
        mask = torch.zeros((len(part), width), dtype=torch.long)
        for r, sq in enumerate(seqs):
            x[r, :len(sq)] = torch.tensor(sq, dtype=torch.long)
            mask[r, :len(sq)] = 1
        x, mask = x.to(wm.device), mask.to(wm.device)
        rows = torch.arange(len(part), device=wm.device)
        look = torch.tensor([len(prompts[i]) - 1 for i in part], device=wm.device)
        tr, tc = [], []
        for r, i in enumerate(part):
            a = len(prompts[i])
            for j in range(a, a + len(targets[i])):
                tr.append(r)
                tc.append(j - 1)
        tr_t, tc_t = torch.tensor(tr, device=wm.device), torch.tensor(tc, device=wm.device)
        tgt = x[tr_t, tc_t + 1]
        cnt = torch.zeros(len(part), device=wm.device).index_add(0, tr_t, torch.ones(len(tr), device=wm.device))
        g = torch.Generator().manual_seed(int(seed) * 1_000_003 + layer * 7919 + c0)
        v = torch.rand(len(part), d_out, generator=g, dtype=torch.float32).to(wm.device).requires_grad_(True)
        opt = torch.optim.Adam([v], lr=lr)

        def hook(m, inp, out):
            o = out.clone()
            o[rows, look] = v.to(o.dtype)
            return o
        h = mod.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                for it in range(n_iter + 1):
                    hidden = wm._decoder()(input_ids=x, attention_mask=mask, use_cache=False).last_hidden_state
                    ce = F.cross_entropy(head(hidden[tr_t, tc_t]).float(), tgt, reduction='none')
                    per = torch.zeros(len(part), device=wm.device).index_add(0, tr_t, ce) / cnt
                    per_d = per.detach().tolist()
                    if it == 0:
                        for r, i in enumerate(part):
                            first[i] = per_d[r]
                    if it == n_iter:           # the loss after the last step; no further update
                        for r, i in enumerate(part):
                            last[i] = per_d[r]
                        break
                    opt.zero_grad(set_to_none=True)
                    per.sum().backward()
                    opt.step()
                    wm.costs['edit_backward'] += 1
        finally:
            h.remove()
        vals[part[0]:part[-1] + 1] = v.detach()
        del v, opt, hidden
    return Values(vals, first, last)


# ---- codebook -------------------------------------------------------------------------------------------------
@dataclass
class Codebook:
    keys: torch.Tensor       # K x d_in
    values: torch.Tensor     # K x d_out
    eps: torch.Tensor        # K
    owner: list              # edit index of each key
    added: int = 0
    splits: int = 0
    expansions: int = 0


def build_codebook(keys: torch.Tensor, values: torch.Tensor, labels: list, eps_init: float) -> Codebook:
    """EasyEdit's codebook updates over the edits in order (see the module docstring)."""
    kept, eps, owner = [], [], []
    splits = expansions = 0
    for e in range(keys.shape[0]):
        q = keys[e:e + 1]
        if not kept:
            kept.append(e)
            eps.append(float(eps_init))
            owner.append(e)
            continue
        d = cdist(q, keys[torch.tensor(kept, device=keys.device)])[0]
        dmin, i = float(d.min()), int(d.argmin())
        if dmin > eps_init + eps[i]:
            kept.append(e)
            eps.append(float(eps_init))
            owner.append(e)
        elif labels[e] != labels[owner[i]]:
            kept.append(e)
            eps.append(dmin / 2)
            owner.append(e)
            eps[i] = dmin / 2 - 1e-5
            splits += 1
        else:
            if dmin > eps[i]:
                eps[i] = dmin
            expansions += 1
    idx = torch.tensor(kept, device=keys.device)
    return Codebook(keys[idx].clone(), values[idx].clone(), torch.tensor(eps, device=keys.device), owner,
                    added=len(kept), splits=splits, expansions=expansions)


# ---- inference hook --------------------------------------------------------------------------------------------
class GraceHook:
    """Forward hook on the adaptor site. With a codebook set, the last position of every forward is checked against
    the nearest key and replaced by its value when within the radius. Logs (fwd, width, rows, positions, owners)
    while `log` is a list."""

    def __init__(self):
        self.cb: Codebook | None = None
        self.layer: int | None = None
        self.log: list | None = None
        self._fwd = 0
        self._handle = None
        self.hits = 0

    def attach(self, wm, layer: int):
        self.detach()
        self.layer = layer

        def hook(m, inp, out):
            fwd = self._fwd
            self._fwd += 1
            cb = self.cb
            if cb is None or cb.keys.shape[0] == 0:
                return out
            q = inp[0][:, -1].float()
            d = cdist(q, cb.keys)
            dmin, j = d.min(1)
            hit = dmin <= cb.eps[j]
            if not bool(hit.any()):
                return out
            rows = torch.nonzero(hit, as_tuple=True)[0]
            self.hits += int(rows.numel())
            if self.log is not None:
                own = [cb.owner[k] for k in j[rows].tolist()]
                self.log.append((fwd, int(out.shape[1]), rows.tolist(), [int(out.shape[1]) - 1] * len(own), own))
            o = out.clone()
            o[rows, -1] = cb.values[j[rows]].to(o.dtype)
            return o
        self._handle = site(wm, layer).register_forward_hook(hook)

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
