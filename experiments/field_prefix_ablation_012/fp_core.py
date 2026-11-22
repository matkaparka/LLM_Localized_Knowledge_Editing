"""FP-12 core: anchors, prompt-position blocking, per-row thresholds, threshold arms and batched interventions.

The writer is RV-09's route-mode LocalEditor through KT-10's LoggedEditor; FPEditor only adds
  - explicit anchors: for a (question, span text) pair in `anchors`, the key and the offset are taken at the token
    holding the span's last character instead of at rv_editor.lookup's position (the corrected row, PLAN section 2);
  - per-row thresholds: wm.extra['route_t'] may be a (rows, 1, entries) tensor; it broadcasts against the routing
    similarities exactly as the (entries,) vector does, so a row with the same vector gives the same result;
  - blocking: wm.extra['route_block'], a (rows, width) bool tensor, switches off the residual at those positions of
    the prompt forward (forward 0 of a fixed-shape batch). Decode forwards are never blocked. Similarities, the choice
    of the best entry and everything else are unchanged; only the addition is masked.
The routing computation is the parent's, line for line, otherwise.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fp_data  # noqa: E402  (import paths for EC-11, KT-10, RV-09, SO-08)
from ec_eval import FixedShapeGen, GenOut, relevant  # noqa: E402
from kt_core import OFF, LoggedEditor  # noqa: E402
from rv_editor import StopForward, _encode  # noqa: E402
from so_data import norm  # noqa: E402
from so_model import fmt_prompt  # noqa: E402
from so_policy import derive  # noqa: E402

MASKS = ('on', 'off', 'strict')


# ---- the editor -------------------------------------------------------------------------------------------------
class FPEditor(LoggedEditor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.anchors: dict[tuple[str, str], tuple[int, int]] = {}

    def _prompt(self, wm, template: str, fact):
        key = (fact.question, fact.subject)
        if key not in self.anchors:
            return super()._prompt(wm, template, fact)
        text = fmt_prompt(fact.question, template)
        q = fact.question.strip()
        off = len(template) + len(fp_data.Q_PREFIX)
        if text[off:off + len(q)] != q:
            raise RuntimeError('prompt layout changed')
        a, b = self.anchors[key]
        if not 0 <= a < b <= len(q):
            raise ValueError(f'span {(a, b)} outside the question')
        return _encode(wm.tok, text), fp_data.token_at(wm.tok, text, off + b - 1), True

    def attach(self, wm):
        if self.cfg.mode != 'route' or self._handle is not None:
            return
        block = self._block(wm, self.cfg.layers[-1])

        def hook(mod, inp, out):
            fwd = self._fwd
            self._fwd += 1
            k = wm.extra.get('route_k')
            if self.route_off or k is None or k.shape[0] == 0:
                return out
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            hc = F.normalize(hf - self.mu, dim=-1)
            sims = hc @ k.T
            thr = wm.extra.get('route_t')
            if thr is not None:  # per-entry thresholds, (E,) or per row (B, 1, E)
                sims = torch.where(sims >= thr, sims, torch.full_like(sims, -2.0))
            best, rev = sims.flip(-1).max(-1)
            idx = k.shape[0] - 1 - rev
            hit = (best >= (self.cfg.tau if thr is None else -1.0)) & (hf.norm(dim=-1) < self.cfg.massive)
            blk = wm.extra.get('route_block')
            if blk is not None and fwd == 0:
                if tuple(blk.shape) != tuple(hit.shape):
                    raise RuntimeError(f'block mask {tuple(blk.shape)} != {tuple(hit.shape)}')
                hit = hit & ~blk
            if not bool(hit.any()):
                return out
            self.hits += int(hit.sum())
            if self.log is not None:
                b, p = torch.nonzero(hit, as_tuple=True)
                self.log.append((fwd, int(h.shape[1]), b.tolist(), p.tolist(), idx[b, p].tolist()))
            h2 = h + (wm.extra['route_v'][idx] * hit.unsqueeze(-1)).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        self._handle = block.register_forward_hook(hook)


# ---- prompt positions ----------------------------------------------------------------------------------------------
def marker_tokens(tok, text: str, ctx_len: int) -> list[int]:
    """Indices of the tokens of fmt_prompt(question, ctx) that lie entirely inside the fixed 'Q: ' and '\\nA:' marks."""
    q0, q1 = ctx_len, ctx_len + len(fp_data.Q_PREFIX)
    a0, a1 = len(text) - 3, len(text)
    if not text.endswith('\nA:') or text[q0:q1] != fp_data.Q_PREFIX:
        raise RuntimeError('prompt layout changed')
    offs = fp_data.offsets(tok, text)
    if offs is None:  # character tokenizer (CPU tests): one token per character
        if len(tok.encode(text, add_special_tokens=False)) != len(text):
            raise RuntimeError('no offsets and not a character tokenizer')
        offs = [(i, i + 1) for i in range(len(text))]
    return [i for i, (a, b) in enumerate(offs) if b > a and ((q0 <= a and b <= q1) or (a0 <= a and b <= a1))]


def blocked_positions(kind: str, nctx: int, markers: list[int]) -> tuple[int, ...]:
    """Prompt positions (relative to the prompt's first token) where the residual is not added."""
    if kind == 'on':
        return ()
    if kind == 'off':
        return tuple(range(nctx))
    if kind == 'strict':
        return tuple(sorted(set(range(nctx)) | set(markers)))
    raise ValueError(kind)


# ---- generation with per-row thresholds and blocking -----------------------------------------------------------------
@dataclass
class Row:
    ids: list            # prompt token ids
    nctx: int            # context tokens at the start of the prompt
    thr: int             # row of the threshold matrix
    block: tuple = ()    # blocked prompt positions, relative to the prompt start


class FPGen(FixedShapeGen):
    """EC-11's fixed-shape generator; every row carries its own threshold vector and blocked prompt positions."""

    def __init__(self, wm, editor: FPEditor, widths, batch: int, max_new: int, dummy: str = 'Q: ?\nA:'):
        super().__init__(wm, widths, batch, max_new, dummy)
        self.editor = editor
        self.loggers = [editor]

    @torch.no_grad()
    def run(self, rows: list[Row], thr: torch.Tensor) -> list[GenOut]:
        """thr: (n_vectors, entries) on the device; wm.extra must hold the table's route_k / route_v."""
        out: list[GenOut | None] = [None] * len(rows)
        by_w: dict[int, list[int]] = {}
        for i, r in enumerate(rows):
            by_w.setdefault(self.width(len(r.ids)), []).append(i)
        for w in sorted(by_w):
            idx = by_w[w]
            for s in range(0, len(idx), self.batch):
                part = idx[s:s + self.batch]
                for i, g in zip(part, self._rows_batch([rows[i] for i in part], w, thr)):
                    out[i] = g
        return out

    def _rows_batch(self, rows: list[Row], width: int, thr: torch.Tensor):
        wm = self.wm
        n_real = len(rows)
        seqs = [r.ids for r in rows] + [self.dummy] * (self.batch - n_real)
        x = torch.full((self.batch, width), wm.pad, dtype=torch.long)
        mask = torch.zeros((self.batch, width), dtype=torch.long)
        blk = torch.zeros((self.batch, width), dtype=torch.bool)
        for r, s in enumerate(seqs):
            start = width - len(s)
            x[r, start:] = torch.tensor(s, dtype=torch.long)
            mask[r, start:] = 1
            if r < n_real and rows[r].block:
                blk[r, [start + p for p in rows[r].block]] = True
        x, mask = x.to(wm.device), mask.to(wm.device)
        sel = torch.tensor([r.thr for r in rows], device=thr.device)
        t = torch.full((self.batch, thr.shape[1]), OFF, device=thr.device)
        t[:n_real] = thr[sel]
        saved_t, saved_b = wm.extra.get('route_t'), wm.extra.get('route_block')
        wm.extra['route_t'] = t.unsqueeze(1)
        wm.extra['route_block'] = blk.to(wm.device)
        self.editor.log, self.editor._fwd = [], 0
        try:
            gen = self._greedy(x, mask, n_real)
        finally:
            log, self.editor.log = self.editor.log, None
            wm.extra['route_t'], wm.extra['route_block'] = saved_t, saved_b
            if saved_b is None:
                wm.extra.pop('route_block', None)
        lens = [len(r.ids) for r in rows]
        gl = [len(g) for g in gen]
        fired = [set() for _ in rows]
        for r, pair in relevant(log or [], width, lens, [r.nctx for r in rows], gl):
            fired[r].add(pair)
        return [GenOut(wm.tok.decode(g, skip_special_tokens=True).split('\n', 1)[0].strip(), len(g),
                       tuple(sorted(f))) for g, f in zip(gen, fired)]


# ---- tables and threshold arms -----------------------------------------------------------------------------------
@dataclass
class Table:
    """One anchor row's entries (all steps), in the write order of KT-10's build_entries."""
    name: str
    keys: torch.Tensor
    offsets: torch.Tensor
    matched: torch.Tensor          # the writer's own thresholds (KT-10's public positions)
    ends: list                     # entries after each step
    step: list                     # write step of each entry
    uids: list
    changed: list = field(default_factory=list)   # entries whose anchor differs from the orig row

    @property
    def n(self) -> int:
        return int(self.keys.shape[0])

    def install(self, wm):
        wm.extra = {'route_k': self.keys, 'route_v': self.offsets, 'route_t': self.matched}

    def digest(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for t in (self.keys, self.offsets, self.matched):
            h.update(t.detach().float().cpu().contiguous().numpy().tobytes())
        h.update(repr((self.ends, self.step, self.uids)).encode())
        return h.hexdigest()


def permutation(stream_name: str, step: int, p: int, n: int) -> list[int]:
    order = list(range(n))
    random.Random(derive('fp12-perm', stream_name, step, p)).shuffle(order)
    return order


def shuffled(matched: torch.Tensor, ends: list, active: torch.Tensor, stream_name: str, p: int) -> torch.Tensor:
    """matched permuted within each write batch among its active entries (fixed permutation p); others unchanged."""
    out, start = matched.clone(), 0
    for t, end in enumerate(ends):
        idx = [j for j in range(start, end) if bool(active[j])]
        if len(idx) > 1:
            perm = permutation(stream_name, t, p, len(idx))
            src = torch.tensor([idx[k] for k in perm], device=matched.device)
            out[torch.tensor(idx, device=matched.device)] = matched[src]
        start = end
    return out


def arm_vector(arm: str, matched: torch.Tensor, ends: list, active: torch.Tensor, stream_name: str) -> torch.Tensor:
    """Threshold of every entry under one arm at the last step; inactive (refused) entries never fire."""
    kind, _, val = arm.partition(':')
    if kind == 'none':
        v = torch.full_like(matched, OFF)
    elif kind == 'matched':
        v = matched.clone()
    elif kind == 'shuffled':
        v = shuffled(matched, ends, active, stream_name, int(val))
    elif kind == 'uniform':
        v = torch.full_like(matched, float(val))
    else:
        raise ValueError(f'unknown arm {arm}')
    return torch.where(active, v, torch.full_like(v, OFF))


def at_step(v: torch.Tensor, ends: list, step: int) -> torch.Tensor:
    """The vector as installed after `step`: later entries do not exist yet."""
    out = v.clone()
    out[ends[step]:] = OFF
    return out


def multisets_equal(a: torch.Tensor, b: torch.Tensor, ends: list) -> bool:
    start = 0
    for end in ends:
        if not torch.equal(torch.sort(a[start:end]).values, torch.sort(b[start:end]).values):
            return False
        start = end
    return True


# ---- strict calibration ------------------------------------------------------------------------------------------
@torch.no_grad()
def strict_probe_states(wm, editor, questions, context: str, batch_size: int = 64) -> torch.Tensor:
    """Centred unit block outputs (base state, routing off) at the probe questions' own positions only: no format
    context, no fixed 'Q: ' / '\\nA:' tokens, no first token, no massive positions."""
    cfg = editor.cfg
    block = editor._block(wm, cfg.layers[-1])
    saved = wm.snapshot()
    wm.restore(wm.empty_state())
    n_ctx = len(_encode(wm.tok, context)) if context else 0
    texts = [fmt_prompt(q, context) for q in questions]
    chunks, cur = [], {}

    def hook(mod, inp, out):
        h = (out[0] if isinstance(out, tuple) else out).float()
        ok = cur['m'].clone()
        ok[:, 0] = False
        ok[:, :n_ctx] = False
        for r, ms in enumerate(cur['markers']):
            if ms:
                ok[r, ms] = False
        ok &= h.norm(dim=-1) < cfg.massive
        chunks.append(F.normalize(h[ok] - editor.mu, dim=-1))
        raise StopForward
    handle = block.register_forward_hook(hook)
    off, editor.route_off = editor.route_off, True
    try:
        for s in range(0, len(texts), batch_size):
            part = texts[s:s + batch_size]
            x, mask = editor._pad([_encode(wm.tok, t) for t in part], wm.pad)
            cur['m'] = mask.bool().to(wm.device)
            cur['markers'] = [marker_tokens(wm.tok, t, len(context)) for t in part]
            try:
                wm._decoder()(input_ids=x.to(wm.device), attention_mask=mask.to(wm.device), use_cache=False)
            except StopForward:
                pass
    finally:
        handle.remove()
        editor.route_off = off
        wm.restore(saved)
    return torch.cat(chunks, 0)


@torch.no_grad()
def max_sim(states: torch.Tensor, keys: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    m = torch.full((keys.shape[0],), -1.0, device=keys.device)
    for s in range(0, states.shape[0], chunk):
        m = torch.maximum(m, (states[s:s + chunk] @ keys.T).max(0).values)
    return m


# ---- batched per-key interventions -----------------------------------------------------------------------------------
@torch.no_grad()
def intervene(gen: FPGen, thr_vec: torch.Tensor, items: list[Row], damaged: list[int], fired: dict,
              max_single: int, max_rounds: int = 6) -> list[dict]:
    """KT-10's kt_core.intervene definitions, batched over items with per-row thresholds (rows are independent in
    the fixed-shape layout, so the order and the batch mates do not matter). items: the arm's rows at T (their thr
    field is ignored); damaged: item indices right under none, wrong under the arm; fired: item -> [(entry, region)].
    Every run uses the item's own prompt and blocking.
      rerun    the arm's thresholds; must reproduce the wrong answer
      closure  switch off the entries that fired; continue with whatever fires next until nothing fires
      alone    each fired entry with every other entry off (when at most max_single fired)
      single   each fired entry switched off, the others as in the arm (same condition)"""
    vecs, rows, tags = [], [], []

    def add(i, v, tag):
        vecs.append(v)
        r = items[i]
        rows.append(Row(r.ids, r.nctx, len(vecs) - 1, r.block))
        tags.append(tag)

    recs = {}
    for i in damaged:
        f = sorted({e for e, _ in fired.get(i, [])})
        recs[i] = dict(i=i, fired=f, hits=[list(x) for x in fired.get(i, [])])
        if not f:
            continue
        add(i, thr_vec, (i, 'rerun', None))
        if len(f) <= max_single:
            for e in f:
                only = torch.full_like(thr_vec, OFF)
                only[e] = thr_vec[e]
                add(i, only, (i, 'alone', e))
                one = thr_vec.clone()
                one[e] = OFF
                add(i, one, (i, 'single', e))
    if rows:
        outs = gen.run(rows, torch.stack(vecs))
        for (i, kind, e), o in zip(tags, outs):
            if kind == 'rerun':
                recs[i]['rerun'] = norm(o.text)
            else:
                recs[i].setdefault(kind, []).append([e, norm(o.text)])
    # closure, one round at a time over all open items
    closed = {i: set(recs[i]['fired']) for i in damaged if recs[i]['fired']}
    open_ = sorted(closed)
    rounds = {i: 0 for i in open_}
    final = {}
    while open_:
        vecs, rows = [], []
        for i in open_:
            v = thr_vec.clone()
            v[sorted(closed[i])] = OFF
            vecs.append(v)
            r = items[i]
            rows.append(Row(r.ids, r.nctx, len(vecs) - 1, r.block))
        outs = gen.run(rows, torch.stack(vecs))
        nxt = []
        for i, o in zip(open_, outs):
            rounds[i] += 1
            now = {e for e, _ in o.fired}
            new = now - closed[i]
            final[i] = (norm(o.text), now)
            if now and new and rounds[i] < max_rounds:
                closed[i] |= new
                nxt.append(i)
        open_ = nxt
    for i in closed:
        ans, now = final[i]
        recs[i]['closure'] = dict(entries=sorted(closed[i]), rounds=rounds[i], answer=ans, complete=not now)
    return [recs[i] for i in damaged]
