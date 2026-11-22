"""SO-08 trajectories, decision-state captures, branches and P0.

Common random numbers: at a decision step every candidate uses the same write seed, and the continuation after
it uses exactly the seeds of the default trajectory. So the branch whose action equals the default policy
reproduces the default trajectory, which the tests check.

Continuation (RunCfg.continuation): 'default' (round 1) lets a candidate act at the decision step only and then
follows the default policy; 'persistent' (round 2) lets the candidate act at the decision step and at every later
step, so a branch compares policies from the decision step on rather than one write followed by shared damage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from so_data import FactIndex, PublicView, Stream, norm, sample, shots_text
from so_model import Writable
from so_policy import (Action, EvalCfg, ReadoutCfg, WriteCfg, base_locality, derive, evaluate, execute, outputs,
                       readout_sets, screen_locality)


@dataclass(frozen=True)
class RunCfg:
    lag: int = 0
    write: WriteCfg = field(default_factory=WriteCfg)
    ev: EvalCfg = field(default_factory=EvalCfg)
    ro: ReadoutCfg = field(default_factory=ReadoutCfg)
    default: Action = Action('plain')
    continuation: str = 'default'

    def __post_init__(self):
        if self.continuation not in ('default', 'persistent'):
            raise ValueError(f'unknown continuation {self.continuation}')


def _write_seed(seed, stream: Stream, step: int) -> int:
    return derive(seed, stream.name, step, 'write')


def run_default(wm: Writable, stream: Stream, index: FactIndex, cfg: RunCfg, seed: int, capture=(),
                evaluate_steps: bool = True, max_steps: int | None = None) -> dict:
    """Default policy from the empty state over all steps.

    At each step, before writing, the current self-readout is taken on the probe questions and the label-cache
    questions; at capture steps the full write state is kept for branching. The lagged readout of step t is the
    current readout of step t-1 (the frozen teacher at t=0). All readouts use the stream's format shots.
    """
    wm.restore(wm.empty_state())
    ctx = shots_text(stream.shots)
    probe_q = PublicView(stream, 0, cfg.lag).probe_questions(cfg.ro.probe_k)
    teacher = outputs(wm, probe_q, cfg.ev.max_new, cfg.ev.gen_batch, ctx)
    base_loc = base_locality(wm, stream, cfg.ev)
    lagged, captures, steps = teacher, {}, []
    for t in range(stream.steps if max_steps is None else min(max_steps, stream.steps)):
        view = PublicView(stream, t, cfg.lag)
        probe_q, cache = readout_sets(view, cfg.ro)
        current = outputs(wm, probe_q + [q for q, _ in cache], cfg.ev.max_new, cfg.ev.gen_batch, ctx)
        if t in capture:
            captures[t] = dict(step=t, snapshot=wm.snapshot(), probe_q=probe_q, cache=cache, current=current,
                               lagged=lagged, teacher=teacher)
        rec = execute(wm, view, cfg.default, cfg.write, _write_seed(seed, stream, t))
        ev = evaluate(wm, stream, index, t, cfg.lag, cfg.ev, base_loc) if evaluate_steps else None
        steps.append(dict(step=t, exec=rec, eval=ev, delta_norm=wm.delta_norm()))
        lagged = current[:len(probe_q)]
    return dict(stream=stream.name, seed=seed, teacher=teacher, base_loc=base_loc, steps=steps, captures=captures)


def branch(wm: Writable, stream: Stream, index: FactIndex, cfg: RunCfg, cap: dict, menu, seed: int,
           base_loc: dict) -> list[dict]:
    """Every candidate from the captured state to the end of the stream; after the decision step the branch follows
    the default policy or, with continuation='persistent', keeps executing the candidate.

    value = mean composite over the remaining steps (the decision step included): the remaining-episode return,
    not a fixed short horizon. post = the trial-written model's answers on the probe and cache questions after the
    decision-step write. later = the public diagnostics of the continuation writes.
    """
    t = cap['step']
    ctx = shots_text(stream.shots)
    cache_q = [q for q, _ in cap['cache']]
    out = []
    for a in menu:
        follow = a if cfg.continuation == 'persistent' else cfg.default
        wm.restore(cap['snapshot'])
        rec = execute(wm, PublicView(stream, t, cfg.lag), a, cfg.write, _write_seed(seed, stream, t))
        post = outputs(wm, cap['probe_q'] + cache_q, cfg.ev.max_new, cfg.ev.gen_batch, ctx)
        comps, later = [evaluate(wm, stream, index, t, cfg.lag, cfg.ev, base_loc)], []
        for tau in range(t + 1, stream.steps):
            later.append(execute(wm, PublicView(stream, tau, cfg.lag), follow, cfg.write,
                                 _write_seed(seed, stream, tau)))
            comps.append(evaluate(wm, stream, index, tau, cfg.lag, cfg.ev, base_loc))
        vals = [c['composite'] for c in comps if c['composite'] is not None]
        out.append(dict(action=a.label, continuation=cfg.continuation, exec=rec, later=later, post=post, comps=comps,
                        value=sum(vals) / len(vals) if vals else None))
    wm.restore(cap['snapshot'])
    return out


def public_capture(cap: dict) -> dict:
    """A capture without its tensors, for JSON."""
    return {k: v for k, v in cap.items() if k != 'snapshot'}


def capability(wm: Writable, stream: Stream, index: FactIndex, cfg: RunCfg) -> dict:
    """P0 capability on the empty (base) state: no information, format shots only, and the fact in context."""
    wm.restore(wm.empty_state())
    edits = sample([e for e in stream.edits if e.rephrase], cfg.ev.max_cur, stream.name, 'p0-capability')
    shots = shots_text(stream.shots)

    def acc(contexts):
        preds = wm.generate([e.rephrase for e in edits], contexts=contexts, max_new=cfg.ev.max_new,
                            batch_size=cfg.ev.gen_batch)
        ok = [norm(t) in index.valid(e.key, e.step) for e, (t, _, _) in zip(edits, preds)]
        return sum(ok) / len(ok) if ok else None

    return dict(n=len(edits), no_info=acc([''] * len(edits)), shots_only=acc([shots] * len(edits)),
                fact_in_context=acc([f'{shots}Fact: {e.update} {e.answer}\n' for e in edits]))


def p0_stream(wm: Writable, stream: Stream, index: FactIndex, cfg: RunCfg, seed: int, menu, repair_steps) -> dict:
    stream, screening = screen_locality(wm, stream, cfg.ev)
    cap = capability(wm, stream, index, cfg)
    traj = run_default(wm, stream, index, cfg, seed, capture=set(repair_steps))
    repairs = {t: branch(wm, stream, index, cfg, traj['captures'][t], menu, seed, traj['base_loc'])
               for t in repair_steps}
    traj['captures'] = {t: public_capture(c) for t, c in traj['captures'].items()}
    return dict(stream=stream.name, seed=seed, locality_screening=screening, capability=cap, trajectory=traj,
                repairs=repairs)


def matched_stream(wm: Writable, stream: Stream, index: FactIndex, cfg: RunCfg, seed: int, menu, steps) -> dict:
    """One seed of the matched-readout experiment: default trajectory plus all candidates at each capture."""
    stream, screening = screen_locality(wm, stream, cfg.ev)
    traj = run_default(wm, stream, index, cfg, seed, capture=set(steps))
    states = []
    for t in sorted(steps):
        cap = traj['captures'][t]
        states.append(dict(public_capture(cap), branches=branch(wm, stream, index, cfg, cap, menu, seed,
                                                                 traj['base_loc'])))
    traj['captures'] = {}
    return dict(stream=stream.name, seed=seed, locality_screening=screening, trajectory=traj, states=states)
