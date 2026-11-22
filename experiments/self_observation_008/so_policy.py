"""SO-08 candidate writes, readouts and evaluation.

Strategies only receive a PublicView (labels of arrived edits, unlabelled probe questions, the public format
shots). Evaluation reads the private Stream and the FactIndex, generates freely (no answer prefix) after the same
format shots and scores normalised exact match against the answers in force at that step.

Every generated answer (evaluation, readouts, self-labels) is preceded by the stream's format shots; training
prompts are plain "Q: ...\nA:". The pilot showed that without shots the base model answers in sentences, which
confounds answer format with knowledge.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from so_data import FactIndex, PublicView, Stream, h, norm, sample, shots_text
from so_model import Writable, fmt_prompt, fmt_target

KINDS = ('skip', 'plain', 'replay', 'merge', 'shrink', 'selfdistill', 'rollback', 'selfdistill_rel',
         'teacherdistill_rel', 'selfprobe_rel', 'baseprobe_rel', 'replay_self', 'replay_base')
# Round 2 (2026-11-19). All four protect the batch relations' locality probes; they differ only in who labels them
# (current self or the original, unwritten model) and in what protects the arrived edits:
#   selfprobe_rel:w / baseprobe_rel:w  probes labelled by self / base (weight w) + the batch relations' arrived
#                                      edits with their stored labels (weight 1)
#   replay_self:k / replay_base:k      replay of k stored labels (as replay:k) + probes labelled by self / base
#                                      (weight 1)
PROBE_ONLY = ('selfprobe_rel', 'baseprobe_rel', 'replay_self', 'replay_base')
FROM_BASE = ('teacherdistill_rel', 'baseprobe_rel', 'replay_base')


@dataclass(frozen=True)
class Action:
    """kind plus one strength: k (replay, replay_self, replay_base), coef (merge), beta (shrink), weight
    (selfdistill, selfdistill_rel, teacherdistill_rel, selfprobe_rel, baseprobe_rel), loss-increase threshold
    (rollback)."""
    kind: str
    value: float = 0.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f'unknown action kind {self.kind}')

    @property
    def label(self) -> str:
        return self.kind if self.kind in ('skip', 'plain') else f'{self.kind}:{self.value:g}'


def parse_action(spec) -> Action:
    if isinstance(spec, str):
        kind, _, v = spec.partition(':')
        return Action(kind, float(v) if v else 0.0)
    return Action(spec['kind'], float(spec.get('value', 0.0)))


@dataclass(frozen=True)
class WriteCfg:
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 32
    accept_k: int = 64
    distill_k: int = 64
    max_new: int = 24
    gen_batch: int = 64


def derive(*parts) -> int:
    return h('so08-seed', *parts) % (2 ** 31 - 1)


def execute(wm: Writable, view: PublicView, action: Action, cfg: WriteCfg, seed: int) -> dict:
    """Run one candidate write at this decision state and return its public diagnostics.

    Public diagnostics (training loss, loss on a fixed acceptance set of arrived labels before and after) use only
    labels the strategy legitimately holds; every candidate at the same state sees the same acceptance set.
    """
    batch = view.batch()
    accept = view.acceptance(cfg.accept_k)
    rec = dict(action=action.label, n_batch=len(batch), wrote=False, rolled_back=False, n_replay=0, n_history=0,
               n_distill=0)
    before = wm.mean_loss(accept)
    rec['accept_before'] = before
    if action.kind == 'skip' or not batch:
        rec.update(accept_after=before, train_loss_last=None)
        return rec
    items = [(fmt_prompt(q), fmt_target(a), 1.0) for q, a, _ in batch]
    now = {norm(q) for q, _, _ in batch}
    if action.kind in ('replay', 'replay_self', 'replay_base'):
        rep = view.replay(int(action.value), seed)
        items += [(fmt_prompt(q), fmt_target(a), 1.0) for q, a in rep]
        rec['n_replay'] = len(rep)
    if action.kind in ('selfprobe_rel', 'baseprobe_rel'):  # the batch relations' arrived edits, stored labels
        hist = [(q, a) for q, a in view.history_items_for(view.batch_relations(), cfg.distill_k) if norm(q) not in now]
        items += [(fmt_prompt(q), fmt_target(a), 1.0) for q, a in hist]
        rec['n_history'] = len(hist)
    if action.kind in ('selfdistill', 'selfdistill_rel', 'teacherdistill_rel') + PROBE_ONLY:
        if action.kind == 'selfdistill':
            pool = view.probe_questions(cfg.distill_k) + view.history_questions(cfg.distill_k)
        elif action.kind in PROBE_ONLY:  # locality probes only: arrived edits are protected by stored labels
            pool = view.probe_questions_for(view.batch_relations(), cfg.distill_k)
        else:  # protect answers where this write is most likely to spill: the batch's relations
            rels = view.batch_relations()
            pool = view.probe_questions_for(rels, cfg.distill_k) + view.history_questions_for(rels, cfg.distill_k)
        probes = [q for q in pool if norm(q) not in now]
        ctx = view.context()
        if action.kind in FROM_BASE:  # labels from the original (unwritten) model, not the current self
            with wm.temporarily(wm.empty_state()):
                outs = wm.generate(probes, contexts=[ctx] * len(probes), max_new=cfg.max_new,
                                   batch_size=cfg.gen_batch)
        else:
            outs = wm.generate(probes, contexts=[ctx] * len(probes), max_new=cfg.max_new,
                               batch_size=cfg.gen_batch)  # pre-write self-labels
        weight = 1.0 if action.kind in ('replay_self', 'replay_base') else action.value
        distill = [(fmt_prompt(q), fmt_target(t), weight) for q, (t, _, _) in zip(probes, outs) if t]
        items += distill
        rec['n_distill'] = len(distill)
    snap = wm.snapshot()
    wm.start_write(seed)
    losses = wm.fit(items, cfg.epochs, cfg.lr, cfg.batch_size, seed)
    wm.commit(action.value if action.kind == 'merge' else 1.0)
    if action.kind == 'shrink':
        wm.shrink(action.value)
    after = wm.mean_loss(accept)
    rec.update(wrote=True, train_loss_first=losses[0] if losses else None,
               train_loss_last=losses[-1] if losses else None, accept_after=after)
    if action.kind == 'rollback' and before is not None and after is not None and after - before > action.value:
        wm.restore(snap)
        rec.update(rolled_back=True, wrote=False, accept_after=before)
    return rec


@dataclass(frozen=True)
class EvalCfg:
    max_cur: int = 100       # fixed evaluation panel of each step's edits
    ret_per_step: int = 50   # how much of every earlier panel is re-asked later (rephrase and persona wording)
    max_loc: int = 256       # locality items (after screening: facts the base model answers correctly)
    max_new: int = 24
    gen_batch: int = 64


def panel(stream: Stream, src: int, cfg: EvalCfg):
    """The fixed, deterministic evaluation panel of the edits of step `src`."""
    return sample([e for e in stream.edits if e.step == src and e.rephrase], cfg.max_cur, stream.name, src, 'panel')


def eval_items(stream: Stream, index: FactIndex, step: int, lag: int, cfg: EvalCfg):
    """Hidden evaluation questions at `step`, identical for every branch at that step.

    cur: rephrase wording of the panel just written; upd: its original wording (reported only); ret: the first
    ret_per_step items of every earlier panel, in rephrase and persona wording, so each edit can be followed from
    right after its write to later steps; loc: the locality pool. Keys identify items across steps.
    """
    src = step - lag
    items = []
    if src >= 0:
        for e in panel(stream, src, cfg):
            items.append(('cur', e.rephrase, index.valid(e.key, step), f'{e.uid}|r'))
            items.append(('upd', e.update, index.valid(e.key, step), f'{e.uid}|u'))
    for s in range(max(src, 0)):
        for e in panel(stream, s, cfg)[:cfg.ret_per_step]:
            items.append(('ret', e.rephrase, index.valid(e.key, step), f'{e.uid}|r'))
            if e.personas:
                items.append(('ret', e.personas, index.valid(e.key, step), f'{e.uid}|p'))
    for q, a in stream.loc_hidden[:cfg.max_loc]:
        items.append(('loc', q, {norm(a)}, q))
    return items


def evaluate(wm: Writable, stream: Stream, index: FactIndex, step: int, lag: int, cfg: EvalCfg,
             base_loc: dict | None = None) -> dict:
    items = eval_items(stream, index, step, lag, cfg)
    ctx = shots_text(stream.shots)
    preds = wm.generate([q for _, q, _, _ in items], contexts=[ctx] * len(items), max_new=cfg.max_new,
                        batch_size=cfg.gen_batch)
    hits = {'cur': [], 'ret': [], 'loc': [], 'upd': []}
    loc_now, per_item = {}, []
    for (comp, q, valid, key), (text, _, _) in zip(items, preds):
        ok = norm(text) in valid
        hits[comp].append(ok)
        per_item.append([comp, key, int(ok)])
        if comp == 'loc':
            loc_now[q] = (ok, norm(text))
    acc = {k: (sum(v) / len(v) if v else None) for k, v in hits.items()}
    present = [acc[k] for k in ('cur', 'ret', 'loc') if acc[k] is not None]
    out = dict(step=step, n={k: len(v) for k, v in hits.items()}, acc=acc,
               composite=sum(present) / len(present) if present else None, items=per_item)
    if base_loc is not None:
        common = [q for q in loc_now if q in base_loc]
        out['loc_right_to_wrong'] = sum(1 for q in common if base_loc[q][0] and not loc_now[q][0])
        out['loc_wrong_to_right'] = sum(1 for q in common if not base_loc[q][0] and loc_now[q][0])
        out['loc_agree_base'] = (sum(1 for q in common if base_loc[q][1] == loc_now[q][1]) / len(common)
                                 if common else None)
    return out


def base_locality(wm: Writable, stream: Stream, cfg: EvalCfg) -> dict:
    """Per locality question: (correct, normalised answer) under the current state (call on the base state)."""
    qs = [q for q, _ in stream.loc_hidden[:cfg.max_loc]]
    ctx = shots_text(stream.shots)
    preds = wm.generate(qs, contexts=[ctx] * len(qs), max_new=cfg.max_new, batch_size=cfg.gen_batch)
    gold = {q: norm(a) for q, a in stream.loc_hidden[:cfg.max_loc]}
    return {q: (norm(t) == gold[q], norm(t)) for q, (t, _, _) in zip(qs, preds)}


def screen_locality(wm: Writable, stream: Stream, cfg: EvalCfg) -> tuple[Stream, dict]:
    """Keep the locality candidates the base model answers correctly (with the format shots), up to max_loc.

    Damage to prior knowledge is only measurable on facts the model knew. Candidates are taken in their stored
    (hash) order, so the selection is deterministic. Uses the base state; restores nothing else.
    """
    wm.restore(wm.empty_state())
    ctx = shots_text(stream.shots)
    cands = list(stream.loc_hidden)
    preds = wm.generate([q for q, _ in cands], contexts=[ctx] * len(cands), max_new=cfg.max_new,
                        batch_size=cfg.gen_batch)
    known = [(q, a) for (q, a), (t, _, _) in zip(cands, preds) if norm(t) == norm(a)]
    kept = tuple(known[:cfg.max_loc])
    return dataclasses.replace(stream, loc_hidden=kept), dict(candidates=len(cands), known=len(known),
                                                               kept=len(kept))


@dataclass(frozen=True)
class ReadoutCfg:
    probe_k: int = 64
    cache_k: int = 64


def readout_sets(view: PublicView, cfg: ReadoutCfg):
    """The questions every readout is taken on at this decision state: probe questions (shared by the current,
    lagged and frozen-teacher readouts) and the label cache (same item count)."""
    return view.probe_questions(cfg.probe_k), view.cache(cfg.cache_k)


def outputs(wm: Writable, questions, max_new: int, batch_size: int, context: str = '') -> list[str]:
    if not questions:
        return []
    return [norm(t) for t, _, _ in wm.generate(questions, contexts=[context] * len(questions), max_new=max_new,
                                               batch_size=batch_size)]
