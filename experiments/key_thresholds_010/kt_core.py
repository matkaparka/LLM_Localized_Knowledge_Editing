"""KT-10 core: write a stream's routed entries once, then evaluate the same entries under different thresholds.

RV-09's route-mode editor (rv_editor.LocalEditor, mode 'route') never changes the weights. Each entry is a centred
unit key at layer 8 plus an offset, both computed on the base state with routing off, so a stream's entries depend
only on its facts and write seeds, not on earlier entries or on their thresholds. KT-10 therefore writes every
step's batch once (RV-09's `local` writer, write seed 0) and evaluates each threshold arm by installing the same keys
and offsets with that arm's threshold vector. Base, questions, batches and greedy decoding are shared by all arms.

LoggedEditor is LocalEditor with a routing hook that can also record which entry fired at which position; the
routing computation is the parent's, line for line. generate_logged() runs SO-08's generator batch by batch (the
same batches as a single call) and keeps the firings that can influence an answer: real prompt positions (format
context or question) and decode inputs before the last generated token.

Threshold arms (all clamp at the same floor tau and use the same margin as RV-09):
  none           2.0 for every entry: nothing fires (the base model under the evaluation's batching)
  matched        RV-09: max(tau, entry's largest similarity over all public probe positions + margin)
  shuffled:p     the matched thresholds of each write batch, permuted within that batch (fixed permutation p)
  uniform:x      x for every entry
  ctx_only       as matched, but only the format-context positions count
  probes:n       as matched, but only the context and the first n probe questions count
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
RV9 = HERE.parent / 'reversal_pilot_009'
if str(RV9) not in sys.path:
    sys.path.insert(0, str(RV9))

from rv_editor import EditWritable, LocalCfg, LocalEditor, StopForward, _encode  # noqa: E402,F401  (SO-08 path)
from rv_policy import SubjView, write_seed  # noqa: E402
from so_data import Stream, norm, shots_text  # noqa: E402
from so_model import fmt_prompt  # noqa: E402
from so_policy import EvalCfg, derive, eval_items  # noqa: E402

OFF = 2.0            # a threshold no centred cosine can reach
REGIONS = ('ctx', 'q', 'gen')


# ---- the editor with a logging routing hook ---------------------------------------------------------------------
class LoggedEditor(LocalEditor):
    """LocalEditor (route mode) whose hook can log firings. While `log` is a list, every forward through the routing
    layer appends (forward index, sequence width, rows, positions, entries) for the positions where an entry fired.
    `_fwd` counts forwards since the caller last reset it (0 = the prompt forward of a generate() batch)."""

    def __init__(self, cfg: LocalCfg):
        super().__init__(cfg)
        self.log: list | None = None
        self._fwd = 0

    def attach(self, wm: EditWritable):
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
            if thr is not None:  # per-entry thresholds: only entries above their own threshold compete
                sims = torch.where(sims >= thr, sims, torch.full_like(sims, -2.0))
            best, rev = sims.flip(-1).max(-1)
            idx = k.shape[0] - 1 - rev
            hit = (best >= (self.cfg.tau if thr is None else -1.0)) & (hf.norm(dim=-1) < self.cfg.massive)
            if not bool(hit.any()):
                return out
            self.hits += int(hit.sum())
            if self.log is not None:
                b, p = torch.nonzero(hit, as_tuple=True)
                self.log.append((fwd, int(h.shape[1]), b.tolist(), p.tolist(), idx[b, p].tolist()))
            h2 = h + (wm.extra['route_v'][idx] * hit.unsqueeze(-1)).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        self._handle = block.register_forward_hook(hook)


# ---- logged generation and evaluation ---------------------------------------------------------------------------
@torch.no_grad()
def generate_logged(wm: EditWritable, editor: LoggedEditor, questions, contexts, max_new: int, batch_size: int):
    """wm.generate() one batch at a time (the same batches as one call over all questions). Returns the predictions
    and, per question, the sorted (entry, region) pairs that fired where they can change its answer: region 0 is a
    format-context token, 1 a question token (both in the prompt forward, left padding excluded), 2 a decode input
    before the question's last generated token."""
    preds, hits = [], []
    for s in range(0, len(questions), batch_size):
        qs, cs = questions[s:s + batch_size], contexts[s:s + batch_size]
        lens = [len(_encode(wm.tok, fmt_prompt(q, c))) for q, c in zip(qs, cs)]
        nctx = [len(_encode(wm.tok, c)) if c else 0 for c in cs]
        editor.log, editor._fwd = [], 0
        try:
            out = wm.generate(qs, contexts=cs, max_new=max_new, batch_size=batch_size)
        finally:
            log, editor.log = editor.log, None
        preds += out
        hits += relevant_hits(log, lens, nctx, [g for _, _, g in out])
    return preds, hits


def relevant_hits(log, lens, nctx, gen_lens) -> list[list[tuple[int, int]]]:
    """Parse one batch's firing log. Prompt forward (index 0, left padded to max(lens)): a position before the
    row's first token is padding (dropped); then nctx[r] context tokens (region 0) and question tokens (region 1).
    Decode forward k >= 1 feeds the row's generated token k-1 and predicts token k, so it can matter only if
    k <= gen_lens[r] - 1 (region 2)."""
    width = max(lens)
    per = [set() for _ in lens]
    for fwd, w, rows, pos, ents in log:
        if fwd == 0 and w != width:
            raise RuntimeError(f'prompt forward width {w} != {width}')
        for r, p, e in zip(rows, pos, ents):
            if fwd == 0:
                start = width - lens[r]
                if p < start:
                    continue                          # left padding
                region = 0 if p < start + nctx[r] else 1
            else:
                if fwd > gen_lens[r] - 1:
                    continue                          # after this row's last generated token
                region = 2
            per[r].add((int(e), region))
    return [sorted(x) for x in per]


@torch.no_grad()
def evaluate_logged(wm: EditWritable, editor: LoggedEditor, stream: Stream, index, step: int, cfg: EvalCfg,
                    base_loc: dict | None = None):
    """so_policy.evaluate (same items, batches, scoring and summary fields) plus the firings. Instead of the item
    list it returns `ok` (one '0'/'1' per item, in eval_items order), `hits` {item index: [[entry, region], ...]}
    and `pred` {item index: normalised answer} for items with firings and for wrong locality items. Also returns the
    item list itself."""
    items = eval_items(stream, index, step, 0, cfg)
    ctx = shots_text(stream.shots)
    preds, hits = generate_logged(wm, editor, [q for _, q, _, _ in items], [ctx] * len(items), cfg.max_new,
                                  cfg.gen_batch)
    by = {'cur': [], 'ret': [], 'loc': [], 'upd': []}
    loc_now, bits, hit_items, pred = {}, [], {}, {}
    for i, ((comp, q, valid, key), (text, _, _)) in enumerate(zip(items, preds)):
        ok = norm(text) in valid
        by[comp].append(ok)
        bits.append('1' if ok else '0')
        if comp == 'loc':
            loc_now[q] = (ok, norm(text))
        if hits[i]:
            hit_items[i] = [list(x) for x in hits[i]]
        if hits[i] or (comp == 'loc' and not ok):
            pred[i] = norm(text)
    acc = {k: (sum(v) / len(v) if v else None) for k, v in by.items()}
    present = [acc[k] for k in ('cur', 'ret', 'loc') if acc[k] is not None]
    out = dict(step=step, n={k: len(v) for k, v in by.items()}, acc=acc,
               composite=sum(present) / len(present) if present else None, ok=''.join(bits), hits=hit_items,
               pred=pred)
    if base_loc is not None:
        common = [q for q in loc_now if q in base_loc]
        out['loc_right_to_wrong'] = sum(1 for q in common if base_loc[q][0] and not loc_now[q][0])
        out['loc_wrong_to_right'] = sum(1 for q in common if not base_loc[q][0] and loc_now[q][0])
        out['loc_agree_base'] = (sum(1 for q in common if base_loc[q][1] == loc_now[q][1]) / len(common)
                                 if common else None)
    return out, items


# ---- entries ----------------------------------------------------------------------------------------------------
@dataclass
class Entries:
    keys: torch.Tensor        # E x d centred unit keys
    offsets: torch.Tensor     # E x d offsets
    matched: torch.Tensor     # E, the editor's own (RV-09) thresholds
    ends: list                # number of entries after each step's write
    step: list                # write step of each entry
    meta: list                # per entry: uid, step, subject, answer, literal-subject flags
    records: list             # the editor's diagnostics per step

    @property
    def n(self) -> int:
        return int(self.keys.shape[0])


def _count(wm) -> int:
    k = wm.extra.get('route_k')
    return 0 if k is None else int(k.shape[0])


def build_entries(wm: EditWritable, editor: LocalEditor, stream: Stream, seed: int) -> Entries:
    """RV-09's `local` writer over the whole stream from the empty state: step t writes the batch arriving at t with
    write seed write_seed(seed, stream, t), as rv_policy.trajectory does. editor.probe_states must be set (the
    matched thresholds come from it)."""
    if editor.cfg.mode != 'route':
        raise ValueError('KT-10 needs the route mode')
    wm.restore(wm.empty_state())
    by_uid = {e.uid: e for e in stream.edits}
    ends, step, meta, records = [], [], [], []
    for t in range(stream.steps):
        view = SubjView(stream, t)
        pairs = [(f, b[3]) for f, b in zip(view.batch_facts(), view._batch) if f.answer and f.answer.strip()]
        before = _count(wm)
        rec = editor.edit(wm, [f for f, _ in pairs], view.context(), write_seed(seed, stream, t)) if pairs \
            else dict(n=0)
        after = _count(wm)
        if after - before != len(pairs):
            raise RuntimeError(f'step {t}: {after - before} entries for {len(pairs)} facts')
        for f, uid in pairs:
            e = by_uid[uid]
            subj = (f.subject or '').casefold()
            meta.append(dict(uid=uid, step=t, subject=f.subject, answer=norm(f.answer),
                             subj_in_rephrase=bool(subj) and subj in (e.rephrase or '').casefold(),
                             subj_in_personas=bool(subj) and subj in (e.personas or '').casefold()))
            step.append(t)
        ends.append(after)
        records.append(rec)
    x = wm.extra
    return Entries(x['route_k'].clone(), x['route_v'].clone(), x['route_t'].clone(), ends, step, meta, records)


def install(wm: EditWritable, ent: Entries, step: int, thr: torch.Tensor):
    """The entries written up to `step`, with thresholds thr (a vector over all entries)."""
    n = ent.ends[step]
    wm.extra = {'route_k': ent.keys[:n], 'route_v': ent.offsets[:n], 'route_t': thr[:n]}


# ---- probe positions and per-entry maxima -------------------------------------------------------------------------
@torch.no_grad()
def probe_positions(wm: EditWritable, editor: LocalEditor, questions, context: str, batch_size: int = 64):
    """The positions and states of LocalEditor.set_probes (same masks, same order), plus the owner of every position:
    the index of its probe question, or len(questions) for a format-context token. Base state, routing off."""
    cfg = editor.cfg
    block = editor._block(wm, cfg.layers[-1])
    saved = wm.snapshot()
    wm.restore(wm.empty_state())
    nq = len(questions)
    n_ctx = len(_encode(wm.tok, context)) if context else 0
    texts = [fmt_prompt(q, context) for q in questions]
    owner = list(range(nq))
    if context:
        texts.append(context)
        owner.append(nq)
    chunks, labels, cur = [], [], {}

    def hook(mod, inp, out):
        h = (out[0] if isinstance(out, tuple) else out).float()
        ok = cur['m'].clone()
        ok[:, 0] = False
        if cur['skip_ctx']:
            ok[:, :n_ctx] = False
        ok &= h.norm(dim=-1) < cfg.massive
        chunks.append(F.normalize(h[ok] - editor.mu, dim=-1))
        lab = cur['own'].unsqueeze(1).expand(ok.shape).clone()
        if n_ctx:
            lab[:, :n_ctx] = nq
        labels.append(lab[ok])
        raise StopForward
    handle = block.register_forward_hook(hook)
    off, editor.route_off = editor.route_off, True
    try:
        for s in range(0, len(texts), batch_size):
            part = texts[s:s + batch_size]
            x, mask = editor._pad([_encode(wm.tok, t) for t in part], wm.pad)
            cur['m'] = mask.bool().to(wm.device)
            cur['own'] = torch.tensor(owner[s:s + batch_size], device=wm.device)
            cur['skip_ctx'] = not (context and part[-1] == context and s + len(part) == len(texts))
            try:
                wm._decoder()(input_ids=x.to(wm.device), attention_mask=mask.to(wm.device), use_cache=False)
            except StopForward:
                pass
    finally:
        handle.remove()
        editor.route_off = off
        wm.restore(saved)
    return torch.cat(chunks, 0), torch.cat(labels, 0)


@torch.no_grad()
def probe_max(states: torch.Tensor, labels: torch.Tensor, keys: torch.Tensor, n_rows: int, chunk: int = 8192):
    """n_rows x E: each entry's largest similarity over the positions of every owner (probe question or context)."""
    m = torch.full((n_rows, keys.shape[0]), -1.0, device=keys.device)
    for s in range(0, states.shape[0], chunk):
        sims = states[s:s + chunk] @ keys.T
        m.scatter_reduce_(0, labels[s:s + chunk].unsqueeze(1).expand_as(sims), sims, reduce='amax', include_self=True)
    return m


@torch.no_grad()
def hidden_max(wm: EditWritable, editor: LocalEditor, questions, context: str, keys: torch.Tensor,
               batch_size: int = 64):
    """len(questions) x E: each entry's largest similarity over each question's own positions (format context, first
    token and massive positions excluded; the context's maxima are the context row of probe_max). Base state,
    routing off, right padding as in set_probes."""
    cfg = editor.cfg
    block = editor._block(wm, cfg.layers[-1])
    n_ctx = len(_encode(wm.tok, context)) if context else 0
    rows, cur = [], {}

    def hook(mod, inp, out):
        h = (out[0] if isinstance(out, tuple) else out).float()
        ok = cur['m'].clone()
        ok[:, 0] = False
        ok[:, :n_ctx] = False
        ok &= h.norm(dim=-1) < cfg.massive
        sims = F.normalize(h - editor.mu, dim=-1) @ keys.T
        sims = torch.where(ok.unsqueeze(-1), sims, torch.full_like(sims, -1.0))
        rows.append(sims.max(1).values)
        raise StopForward
    handle = block.register_forward_hook(hook)
    off, editor.route_off = editor.route_off, True
    try:
        for s in range(0, len(questions), batch_size):
            part = [fmt_prompt(q, context) for q in questions[s:s + batch_size]]
            x, mask = editor._pad([_encode(wm.tok, t) for t in part], wm.pad)
            cur['m'] = mask.bool().to(wm.device)
            try:
                wm._decoder()(input_ids=x.to(wm.device), attention_mask=mask.to(wm.device), use_cache=False)
            except StopForward:
                pass
    finally:
        handle.remove()
        editor.route_off = off
    return torch.cat(rows, 0) if rows else torch.empty((0, keys.shape[0]), device=keys.device)


# ---- threshold arms ---------------------------------------------------------------------------------------------
def permutation(stream_name: str, step: int, p: int, n: int) -> list[int]:
    order = list(range(n))
    random.Random(derive('kt10-perm', stream_name, step, p)).shuffle(order)
    return order


def thresholds(arm: str, ent: Entries, pmax: torch.Tensor, nq: int, cfg: LocalCfg, stream_name: str) -> torch.Tensor:
    """The threshold of every entry under `arm` (see the module docstring); pmax is probe_max with nq probe rows
    followed by the context row."""
    tau, margin = cfg.tau, cfg.probe_margin
    dev = ent.keys.device
    if arm == 'none':
        return torch.full((ent.n,), OFF, device=dev)
    if arm == 'matched':
        return ent.matched.clone()
    kind, _, val = arm.partition(':')
    if kind == 'uniform':
        return torch.full((ent.n,), float(val), device=dev)
    if kind == 'ctx_only':
        return torch.clamp(pmax[nq] + margin, min=tau)
    if kind == 'probes':
        n = int(val)
        if not 0 < n <= nq:
            raise ValueError(f'{arm}: {nq} probe questions')
        return torch.clamp(torch.maximum(pmax[:n].max(0).values, pmax[nq]) + margin, min=tau)
    if kind == 'shuffled':
        out, start = ent.matched.clone(), 0
        for t, end in enumerate(ent.ends):
            n = end - start
            if n:
                idx = torch.tensor(permutation(stream_name, t, int(val), n), device=dev) + start
                out[start:end] = ent.matched[idx]
            start = end
        return out
    raise ValueError(f'unknown arm {arm}')


# ---- per-key intervention ---------------------------------------------------------------------------------------
@torch.no_grad()
def intervene(wm: EditWritable, editor: LoggedEditor, ent: Entries, step: int, thr: torch.Tensor, items,
              context: str, ev_arm: dict, ev_none: dict, cfg: EvalCfg, max_single: int,
              max_rounds: int = 6) -> list[dict]:
    """For every locality item that is right with nothing firing and wrong under this arm (same step, same batches),
    rerun its own evaluation batch with only the thresholds changed:
      rerun    the arm's thresholds; must reproduce the wrong answer
      closure  switch off the entries that fired on it. Only the best entry above its threshold acts at a position,
               so a runner-up can then take over: switch off whatever fires next as well, until nothing fires on the
               item (at most max_rounds). Records the set, the rounds and the final answer
      alone    each entry that fired under the arm, with every other entry off: does it break the item by itself?
      single   each entry that fired under the arm switched off, the others as in the arm: is the item right again?
    alone and single are run when at most max_single entries fired under the arm."""
    bs = cfg.gen_batch
    qs = [q for _, q, _, _ in items]
    out = []

    def run(i, t, logged=False):
        b0 = (i // bs) * bs
        part = qs[b0:b0 + bs]
        install(wm, ent, step, t)
        if logged:
            preds, hits = generate_logged(wm, editor, part, [context] * len(part), cfg.max_new, bs)
            text, fired = preds[i - b0][0], sorted({e for e, _ in hits[i - b0]})
        else:
            text, fired = wm.generate(part, contexts=[context] * len(part), max_new=cfg.max_new,
                                      batch_size=bs)[i - b0][0], None
        return [int(norm(text) in items[i][2]), norm(text)], fired
    try:
        for i, (comp, q, valid, key) in enumerate(items):
            if comp != 'loc' or ev_none['ok'][i] != '1' or ev_arm['ok'][i] != '0':
                continue
            hits = ev_arm['hits'].get(i, [])
            fired = sorted({e for e, _ in hits})
            rec = dict(i=i, question=q, pred=ev_arm['pred'].get(i), fired=fired,
                       regions=sorted({r for _, r in hits}), hits=[list(h) for h in hits])
            if fired:
                rec['rerun'] = run(i, thr)[0]
                closed, rounds, new, ans = set(fired), 0, set(), None
                while rounds < max_rounds:
                    rounds += 1
                    off = thr.clone()
                    off[sorted(closed)] = OFF
                    ans, now = run(i, off, logged=True)
                    new = set(now) - closed
                    if not now:
                        break
                    closed |= new
                    if not new:   # cannot happen: switched-off entries never fire
                        break
                rec['closure'] = dict(entries=sorted(closed), rounds=rounds, answer=ans, complete=not new and not now)
                if len(fired) <= max_single:
                    alone, single = [], []
                    for e in fired:
                        only = torch.full_like(thr, OFF)
                        only[e] = thr[e]
                        alone.append([e, *run(i, only)[0]])
                        one = thr.clone()
                        one[e] = OFF
                        single.append([e, *run(i, one)[0]])
                    rec['alone'], rec['single'] = alone, single
            out.append(rec)
    finally:
        install(wm, ent, step, thr)
    return out


# ---- prompt-level firing on the whole hidden candidate pool --------------------------------------------------------
@torch.no_grad()
def hidden_fire(hmax: torch.Tensor, ctx_row: torch.Tensor, thr: torch.Tensor, known: torch.Tensor) -> dict:
    """Share of hidden questions on which some entry reaches its threshold at a prompt position (question tokens or
    the shared format context), over all candidates and over the base-known ones."""
    eff = torch.maximum(hmax, ctx_row.unsqueeze(0))
    fire = (eff >= thr.unsqueeze(0)).any(1)
    return dict(all=float(fire.float().mean()), known=float(fire[known].float().mean()) if bool(known.any()) else None,
                n_all=int(fire.numel()), n_known=int(known.sum()))
