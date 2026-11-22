"""RV-09 data, writers, trajectories and branches.

Data: SO-08's entity-grouped WikiBigEdit streams (so_data.build_streams with the config_run_002 settings) are
rebuilt unchanged; RV-09 caps every step's batch, takes the generic-key corpus from unused rows of buckets no RV-09
family uses, and draws independent futures from unused rows of a family's own bucket.

Writers (all receive the same arrived labels; replay writers also hold stored labels of earlier arrivals):
  lora_plain           SO-08 'plain' LoRA write (all-layer down_proj), optional clipping
  lora_replay:k        SO-08 'replay:k'
  local                LocalEditor on the current batch
  local_replay:k       LocalEditor on the current batch plus k stored labels of earlier arrivals (same stratified
                       choice as SO-08 replay)
Evaluation is SO-08's (so_policy.evaluate / screen_locality / base_locality), unchanged.
"""
from __future__ import annotations

import dataclasses
import time
from collections import defaultdict
from dataclasses import dataclass, field

from rv_editor import EditWritable, Fact, LocalEditor  # noqa: F401  (sets up the SO-08 path)
from so_data import Edit, FactIndex, PublicView, Stream, h, norm, sample, shots_text
from so_policy import Action, EvalCfg, WriteCfg, base_locality, derive, evaluate, execute, screen_locality

WRITER_KINDS = ('lora_plain', 'lora_replay', 'local', 'local_replay')
N_BUCKETS = 10
NAMES = ['P0', 'P1'] + [f'D{i}' for i in range(8)]


# ---- data ---------------------------------------------------------------------------------------------------
def bucket_of(subject_id: str) -> int:
    return h('so08-entity', subject_id) % N_BUCKETS


def cap_stream(stream: Stream, cap: int | None) -> Stream:
    """At most `cap` edits per step, chosen by a fixed hash; keeps file order."""
    if not cap:
        return stream
    keep = []
    for t in range(stream.steps):
        at = [e for e in stream.edits if e.step == t]
        keep += sample(at, cap, stream.name, t, 'rv09-cap')
    keep.sort(key=lambda e: (e.step, e.uid))
    return dataclasses.replace(stream, edits=tuple(keep))


def pool_texts(streams) -> set:
    """Every normalised question text any SO-08 stream or pool uses (edit wordings, probes, hidden candidates,
    format shots)."""
    out = set()
    for s in streams:
        for e in s.edits:
            out.update(norm(t) for t in (e.update, e.rephrase, e.personas) if t)
        out.update(norm(q) for q in s.loc_probe)
        out.update(norm(q) for q, _ in s.loc_hidden)
        out.update(norm(q) for q, _ in s.shots)
    return out


def spare_rows(edits, streams, half_of: dict, half: str = 'dev'):
    """Rows of the half that no SO-08 stream uses and whose texts (including their own locality question) touch no
    stream or pool text."""
    used = {e.uid for s in streams for e in s.edits}
    taken = pool_texts(streams)
    out = []
    for e in edits:
        if half_of.get(e.relation_id) != half or e.uid in used:
            continue
        texts = [norm(t) for t in (e.update, e.rephrase, e.personas, e.loc) if t]
        if any(t in taken for t in texts):
            continue
        out.append(e)
    return out


def corpus(spare, buckets, n: int) -> list[str]:
    """Generic-key corpus: up to n distinct 'Q: <update question>\\nA:' prompts from spare rows of `buckets`."""
    seen, qs = set(), []
    for e in sorted((e for e in spare if bucket_of(e.subject_id) in buckets), key=lambda e: h('rv09-c0', e.uid)):
        k = norm(e.update)
        if k and k not in seen:
            seen.add(k)
            qs.append(f'Q: {e.update.strip()}\nA:')
        if len(qs) >= n:
            break
    return qs


def futures(stream: Stream, spare, family_bucket: int, t: int, n_futures: int, n_per_step: int,
            seed_name: str = 'rv09-future') -> list[Stream]:
    """Independent futures after decision step t for one family: each keeps the stream's edits of steps <= t and
    gets, for every later step, up to n_per_step spare rows of the family's bucket from that dated window.

    Futures are subject-disjoint from one another and from every subject of the stream; rows need a rephrase. All
    futures keep the stream's name, so history panels and write seeds stay identical across futures."""
    subjects = {e.subject_id for e in stream.edits}
    cand = [e for e in spare if bucket_of(e.subject_id) == family_bucket and e.step > t and e.rephrase
            and e.subject_id not in subjects]
    by = defaultdict(lambda: defaultdict(list))
    for e in cand:
        j = h(seed_name, stream.name, t, e.subject_id) % n_futures
        by[j][e.step].append(e)
    hist = [e for e in stream.edits if e.step <= t]
    out = []
    for j in range(n_futures):
        fut = []
        for tau in range(t + 1, stream.steps):
            rows = sorted(by[j][tau], key=lambda e: h(seed_name, 'order', e.uid))
            seen, picked = set(), []
            for e in rows:  # one row per key
                if e.key not in seen:
                    seen.add(e.key)
                    picked.append(e)
                if len(picked) >= n_per_step:
                    break
            fut += picked
        edits = sorted(hist + fut, key=lambda e: (e.step, e.uid))
        out.append(dataclasses.replace(stream, edits=tuple(edits)))
    return out


# ---- public view with subjects ---------------------------------------------------------------------------------
class SubjView(PublicView):
    """SO-08's PublicView plus the subject string of every arrived edit (public with its label)."""

    def __init__(self, stream: Stream, step: int, lag: int = 0):
        super().__init__(stream, step, lag)
        src = step - lag
        self._subj = {e.uid: e.subject for e in stream.edits if e.step <= src}

    def batch_facts(self) -> list[Fact]:
        return [Fact(q, a, self._subj.get(uid)) for q, a, _, uid in self._batch]

    def replay_facts(self, k: int, seed: int) -> list[Fact]:
        """The same stratified choice as PublicView.replay(k, seed), with subjects."""
        by_step = defaultdict(list)
        for q, a, _, uid, st in self.latest_history():
            by_step[st].append((h('so08-replay', seed, uid), q, a, uid))
        for v in by_step.values():
            v.sort()
        out, order = [], sorted(by_step)
        while len(out) < k and any(by_step[s] for s in order):
            for s in order:
                if by_step[s] and len(out) < k:
                    _, q, a, uid = by_step[s].pop(0)
                    out.append(Fact(q, a, self._subj.get(uid)))
        return out


# ---- writers ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Writer:
    kind: str
    k: int = 0

    def __post_init__(self):
        if self.kind not in WRITER_KINDS:
            raise ValueError(f'unknown writer {self.kind}')

    @property
    def label(self) -> str:
        return f'{self.kind}:{self.k}' if self.kind in ('lora_replay', 'local_replay') else self.kind

    @property
    def local(self) -> bool:
        return self.kind.startswith('local')


def parse_writer(spec: str) -> Writer:
    kind, _, v = spec.partition(':')
    return Writer(kind, int(v) if v else 0)


def write(wm: EditWritable, editor: LocalEditor | None, view: SubjView, writer: Writer, wcfg: WriteCfg,
          seed: int) -> dict:
    """One write of the batch arriving at view.step. Returns public diagnostics (no hidden evaluation)."""
    t0 = time.time()
    if writer.kind == 'lora_plain':
        rec = execute(wm, view, Action('plain'), wcfg, seed)
    elif writer.kind == 'lora_replay':
        rec = execute(wm, view, Action('replay', float(writer.k)), wcfg, seed)
    else:
        if editor is None:
            raise ValueError('local writer without an editor')
        facts = view.batch_facts()
        n_rep = 0
        if writer.kind == 'local_replay':
            now = {norm(f.question) for f in facts}
            rep = [f for f in view.replay_facts(writer.k, seed) if norm(f.question) not in now]
            n_rep = len(rep)
            facts = facts + rep
        rec = dict(action=writer.label, n_batch=len(view.batch()), n_replay=n_rep, wrote=bool(facts))
        if facts:
            rec['edit'] = editor.edit(wm, facts, view.context(), seed)
    rec['action'] = writer.label
    rec['write_seconds'] = time.time() - t0
    return rec


def write_seed(seed: int, stream: Stream, step: int) -> int:
    """Same derivation as SO-08 (so_branch._write_seed)."""
    return derive(seed, stream.name, step, 'write')


# ---- trajectories ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RunCfg:
    write: WriteCfg = field(default_factory=WriteCfg)
    ev: EvalCfg = field(default_factory=EvalCfg)


def trajectory(wm: EditWritable, editor: LocalEditor | None, stream: Stream, index: FactIndex, writer: Writer,
               cfg: RunCfg, seed: int, base_loc: dict, capture=(), evaluate_steps: bool = True,
               max_steps: int | None = None, start: dict | None = None, t0: int = 0, seed_fn=None) -> dict:
    """The writer as a fixed policy from step t0 (empty state, or `start`) to the end, with SO-08 evaluation after
    every write. Captured states (before the write of a capture step) are returned as snapshots. Write seeds are
    write_seed(seed, stream, t) unless seed_fn(t) is given."""
    wm.restore(start if start is not None else wm.empty_state())
    steps, captures = [], {}
    end = stream.steps if max_steps is None else min(max_steps, stream.steps)
    for t in range(t0, end):
        if t in capture:
            captures[t] = wm.snapshot()
        view = SubjView(stream, t)
        ws = seed_fn(t) if seed_fn is not None else write_seed(seed, stream, t)
        rec = write(wm, editor, view, writer, cfg.write, ws)
        ev = evaluate(wm, stream, index, t, 0, cfg.ev, base_loc) if evaluate_steps else None
        steps.append(dict(step=t, exec=rec, eval=ev, delta_norm=wm.delta_norm()))
    return dict(stream=stream.name, seed=seed, writer=writer.label, steps=steps, captures=captures)


def value(steps) -> float | None:
    """Remaining-episode value: mean SO-08 composite over the given steps (decision step included)."""
    vals = [s['eval']['composite'] for s in steps if s['eval'] and s['eval']['composite'] is not None]
    return sum(vals) / len(vals) if vals else None


def decision_seed(stream: Stream, t: int) -> int:
    """Write seed of the decision-step write: the same for both actions and every state of a family (common random
    numbers); state differences then come only from the states."""
    return derive('rv09-decision', stream.name, t)


def continuation_seed(stream: Stream, t: int, future: int, tau: int) -> int:
    """Seed of the default writer at step tau > t under future `future`: shared by both actions and all states."""
    return derive('rv09-continuation', stream.name, t, future, tau)


def branch_pair(wm: EditWritable, editor: LocalEditor, stream_futures: list[Stream], index: FactIndex,
                state: dict, t: int, actions: tuple[Writer, Writer], default: Writer, cfg: RunCfg,
                base_loc: dict) -> dict:
    """From one captured state before step t: each action writes the step-t batch (decision seed), then the default
    writer continues to the end under each future (continuation seed of that future). Data, seeds and evaluation
    items are shared by both actions, so their difference is paired.

    Returns per action: the decision-step record and evaluation, and per future the continuation's evaluations and
    the remaining-episode value (decision-step evaluation included)."""
    out = {}
    stream0 = stream_futures[0]
    for a in actions:
        wm.restore(state)
        rec = write(wm, editor, SubjView(stream0, t), a, cfg.write, decision_seed(stream0, t))
        after = wm.snapshot()
        # the decision-step evaluation depends only on edits up to t, identical in every future
        ev_t = evaluate(wm, stream0, index, t, 0, cfg.ev, base_loc)
        per_future = []
        for j, fs in enumerate(stream_futures):
            cont = trajectory(wm, editor, fs, index, default, cfg, 0, base_loc, start=after, t0=t + 1,
                              seed_fn=lambda tau, j=j, fs=fs: continuation_seed(fs, t, j, tau))
            steps = [dict(step=t, eval=ev_t)] + cont['steps']
            per_future.append(dict(future=j, value=value(steps), steps=cont['steps']))
        out[a.label] = dict(exec=rec, eval_decision=ev_t, futures=per_future)
        del after
    wm.restore(state)
    return out


def screen(wm: EditWritable, stream: Stream, ev: EvalCfg):
    """SO-08 locality screening (base-known facts only) and the base model's locality answers."""
    s, info = screen_locality(wm, stream, ev)
    return s, info, base_locality(wm, s, ev)
