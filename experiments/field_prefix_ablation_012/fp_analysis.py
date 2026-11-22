"""FP-12 analysis: per-arm measures, the uniform-threshold rule, the four-cell decomposition, two-level paired
bootstrap intervals, the descriptive gates and the per-key intervention categories (PLAN sections 4-7).

Records come from fp_run.run_stream. Item kinds: cur / upd (each step's panel right after its write), ret (earlier
panels at T), loc (base-known hidden facts at T), cov (every edit's rephrasing at T).
"""
from __future__ import annotations

import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fp_data  # noqa: E402,F401  (import paths)
from kt_analysis import name_relation  # noqa: E402

CELLS = ('orig/on', 'orig/off', 'corr/on', 'corr/off')
TOL = 0.02


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ---- per stream and arm ----------------------------------------------------------------------------------------------
def kinds(rec):
    return [c for c, _, _ in rec['items']]


def base_loc(rec) -> list[int]:
    none = rec['arms']['none']['ok']
    return [i for i, k in enumerate(kinds(rec)) if k == 'loc' and none[i] == '1']


def wrong_vec(rec, arm: str, idx: list[int]) -> np.ndarray:
    ok = rec['arms'][arm]['ok']
    return np.array([ok[i] == '0' for i in idx], dtype=np.float64)


def arm_metrics(rec, arm: str) -> dict:
    a = rec['arms'][arm]
    ks = kinds(rec)
    steps = [t for _, _, t in rec['items']]
    none = rec['arms']['none']['ok']
    base = base_loc(rec)
    cur_by_t, upd_by_t = {}, {}
    for i, (k, t) in enumerate(zip(ks, steps)):
        if k == 'cur':
            cur_by_t.setdefault(t, []).append(a['ok'][i] == '1')
        elif k == 'upd':
            upd_by_t.setdefault(t, []).append(a['ok'][i] == '1')
    ret = [a['ok'][i] == '1' for i, k in enumerate(ks) if k == 'ret']
    cov = [a['ok'][i] == '1' for i, k in enumerate(ks) if k == 'cov']
    learn = [a['ok'][i] == '1' for i, k in enumerate(ks) if k == 'cur' and none[i] == '0']
    loc_reg = [int(a['reg'][i]) for i in base]
    n_ent = rec['tables']['n']
    return dict(
        damage=sum(a['ok'][i] == '0' for i in base) / len(base) if base else None,
        damaged=sum(a['ok'][i] == '0' for i in base), known=len(base),
        Q=mean([sum(v) / len(v) for v in cur_by_t.values() if v]),
        R=sum(ret) / len(ret) if ret else None,
        coverage=sum(cov) / len(cov) if cov else None,
        update=mean([sum(v) / len(v) for v in upd_by_t.values() if v]),
        learned=sum(learn) / len(learn) if learn else None,
        fired_loc=sum(r > 0 for r in loc_reg) / len(loc_reg) if loc_reg else None,
        fired_loc_ctx=sum(bool(r & 1) for r in loc_reg) / len(loc_reg) if loc_reg else None,
        fired_loc_q=sum(bool(r & 2) for r in loc_reg) / len(loc_reg) if loc_reg else None,
        fired_loc_gen=sum(bool(r & 4) for r in loc_reg) / len(loc_reg) if loc_reg else None,
        refusal=(len(rec['tables']['refused']) / n_ent) if not arm.startswith('bridge:') else 0.0)


def shuffles(rec, prefix: str) -> list[str]:
    return sorted(a for a in rec['arms'] if a.startswith(prefix + ':shuffled:'))


def delta_items(rec, prefix: str, ref: str | None = None) -> np.ndarray:
    """Per base-known item: mean over the shuffles of 'wrong' minus 'wrong' under prefix:matched (or under ref)."""
    idx = base_loc(rec)
    sh = shuffles(rec, prefix)
    m = wrong_vec(rec, f'{prefix}:matched', idx)
    comp = np.mean([wrong_vec(rec, a, idx) for a in sh], axis=0) if ref is None else wrong_vec(rec, ref, idx)
    return comp - m


def quality_items(rec, prefix: str, kind: str, ref: str | None = None) -> np.ndarray:
    """Per item of `kind` (cur for Q, ret for R): matched right minus the comparison arm right, i.e. the mean over
    the shuffles (mechanism gate) or the arm `ref` (practical gate: the selected uniform threshold). Q is a mean of
    per-step accuracies; with (near-)equal panels per step the pooled item mean is used for its interval."""
    ks = kinds(rec)
    idx = [i for i, k in enumerate(ks) if k == kind]
    ok = lambda arm: np.array([rec['arms'][arm]['ok'][i] == '1' for i in idx], dtype=np.float64)  # noqa: E731
    comp = ok(ref) if ref else np.mean([ok(a) for a in shuffles(rec, prefix)], axis=0)
    return ok(f'{prefix}:matched') - comp


# ---- bootstrap ---------------------------------------------------------------------------------------------------------
def boot(per_stream: dict, reps: int, seed: int) -> tuple[float, float] | None:
    """Two-level paired bootstrap of an equal-stream mean: streams with replacement, then items within each drawn
    stream (independently per draw). per_stream: name -> item vector. 95% percentile interval."""
    names = sorted(k for k, v in per_stream.items() if len(v))
    if not names:
        return None
    rng = np.random.default_rng(seed)
    k = len(names)
    draws = rng.integers(0, k, size=(reps, k))
    means = np.zeros((reps, k))
    for si, n in enumerate(names):
        v = np.asarray(per_stream[n], dtype=np.float64)
        slots = np.argwhere(draws == si)
        for c in range(0, len(slots), 2048):
            part = slots[c:c + 2048]
            ix = rng.integers(0, len(v), size=(len(part), len(v)))
            means[part[:, 0], part[:, 1]] = v[ix].mean(1)
    stat = means.mean(1)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(lo), float(hi)


def summary(per_stream: dict, reps: int, seed: int) -> dict:
    vals = {n: float(np.mean(v)) for n, v in per_stream.items() if len(v)}
    return dict(mean=mean(list(vals.values())), ci=boot(per_stream, reps, seed), per_stream=vals,
                positive=sum(v > 0 for v in vals.values()), n=len(vals))


# ---- uniform rule ------------------------------------------------------------------------------------------------------
def tune(recs: dict, cfg: dict) -> dict:
    """Per cell, over the dev streams: among uniform points with Q_U >= Q_M - 0.02 and R_U >= R_M - 0.02 the lowest mean
    damage (ties: higher Q, then lower threshold); else the smallest largest shortfall, flagged."""
    out = {}
    grid = [f'uniform:{x:.2f}' for x in cfg['arms']['uniform']]
    for cell in CELLS:
        M = {k: mean([arm_metrics(r, f'{cell}:matched')[k] for r in recs.values()]) for k in ('Q', 'R', 'damage')}
        pts = []
        for g in grid:
            m = {k: mean([arm_metrics(r, f'{cell}:{g}')[k] for r in recs.values()]) for k in ('Q', 'R', 'damage')}
            short = max(M['Q'] - TOL - m['Q'], M['R'] - TOL - m['R'])
            pts.append((g, m, short))
        ok = [p for p in pts if p[2] <= 1e-12]
        if ok:
            best = min(ok, key=lambda p: (p[1]['damage'], -p[1]['Q'], float(p[0].split(':')[1])))
            flag = 'met'
        else:
            best = min(pts, key=lambda p: (p[2], -p[1]['Q'], float(p[0].split(':')[1])))
            flag = 'dev gate not met'
        out[cell] = dict(arm=best[0], flag=flag, matched=M,
                         grid={p[0]: dict(p[1], shortfall=p[2]) for p in pts})
    return out


# ---- gates ---------------------------------------------------------------------------------------------------------
def gate(d: dict, q: dict, r: dict) -> dict:
    """Descriptive mechanism gate (guide v2 section 5.2) from the damage-reduction and quality summaries."""
    n = d['n'] or 0
    need = math.ceil(0.75 * n) if n else None
    passed = (d['mean'] is not None and d['mean'] >= TOL and d['ci'] is not None and d['ci'][0] > 0
              and d['positive'] >= (need or 0) and q['mean'] is not None and q['mean'] >= -TOL
              and r['mean'] is not None and r['mean'] >= -TOL)
    return dict(passed=bool(passed), mean_delta=d['mean'], ci=d['ci'], positive=d['positive'], needed=need,
                q_diff=q['mean'], r_diff=r['mean'])


# ---- interventions -------------------------------------------------------------------------------------------------------
def categorise(rec, arm: str) -> dict:
    ivs = rec['interventions'].get(arm)
    if ivs is None:
        return dict(available=False)
    row = arm.split('/')[0]
    changed = set(rec['tables']['changed'])
    entity = rec['tables']['entity']
    qtext, answers = rec['loc_questions'], rec['loc_answers']
    wrong = rec['arms'][arm]['wrong_loc']
    per_item, culprit_rows = Counter(), Counter()
    rerun_mismatch = closure_right = single_recovers = 0
    for r in ivs:
        i = str(r['i'])
        if not r['fired']:
            per_item['no firing'] += 1
            continue
        if r.get('rerun') != wrong.get(i):
            rerun_mismatch += 1
        if r.get('closure') and r['closure']['answer'] == answers[i]:
            closure_right += 1
        if 'alone' not in r:
            per_item['too many fired'] += 1
            continue
        if any(ans == answers[i] for _, ans in r.get('single', [])):
            single_recovers += 1
        culprits = [e for e, ans in r['alone'] if ans != answers[i]]   # breaks the fact on its own
        if not culprits:
            per_item['no single culprit'] += 1
            continue
        cats = set()
        for e in culprits:
            regs = {g for x, g in r['hits'] if x == e}
            if regs == {0}:
                c = 'shared prefix'
            elif row == 'orig' and e in changed:
                c = 'bad anchor'
            elif name_relation(qtext.get(i, ''), entity[e]) in ('same name', 'near name'):
                c = 'near name'
            else:
                c = 'other'
            cats.add(c)
            culprit_rows[c] += 1
        per_item[cats.pop() if len(cats) == 1 else 'several kinds'] += 1
    return dict(available=True, damaged=len(ivs), per_item=dict(per_item), culprits=dict(culprit_rows),
                rerun_mismatch=rerun_mismatch, closure_right=closure_right, single_recovers=single_recovers)


# ---- the whole analysis -----------------------------------------------------------------------------------------------
def analyze(recs: dict, cfg: dict, dev: dict | None = None) -> dict:
    reps, seed = int(cfg['rule']['bootstrap']), int(cfg['rule']['bootstrap_seed'])
    names = sorted(recs)
    sel = cfg.get('selected_uniform') or {}
    arms_all = sorted(next(iter(recs.values()))['arms'])
    table = {a: {n: arm_metrics(recs[n], a) for n in names} for a in arms_all}
    means = {a: {k: mean([v[k] for v in table[a].values()]) for k in next(iter(table[a].values()))}
             for a in arms_all}

    d_items = {p: {n: delta_items(recs[n], p) for n in names} for p in CELLS + ('bridge', 'strict')}
    deltas = {p: summary(d_items[p], reps, seed) for p in d_items}
    du = {}
    for cell in CELLS:
        if cell in sel:
            ref = f'{cell}:{sel[cell]["arm"]}'
            du[cell] = summary({n: delta_items(recs[n], cell, ref) for n in names}, reps, seed)
    qd = {p: summary({n: quality_items(recs[n], p, 'cur') for n in names}, reps, seed) for p in d_items}
    rd = {p: summary({n: quality_items(recs[n], p, 'ret') for n in names}, reps, seed) for p in d_items}

    def contrast(fn):
        return summary({n: fn(n) for n in names}, reps, seed)
    dec = dict(
        corr_off=deltas['corr/off'],
        orig_on_minus_corr_off=contrast(lambda n: d_items['orig/on'][n] - d_items['corr/off'][n]),
        field_on=contrast(lambda n: d_items['orig/on'][n] - d_items['corr/on'][n]),
        field_off=contrast(lambda n: d_items['orig/off'][n] - d_items['corr/off'][n]),
        mask_orig=contrast(lambda n: d_items['orig/on'][n] - d_items['orig/off'][n]),
        mask_corr=contrast(lambda n: d_items['corr/on'][n] - d_items['corr/off'][n]),
        interaction=contrast(lambda n: d_items['orig/on'][n] - d_items['corr/on'][n] - d_items['orig/off'][n]
                             + d_items['corr/off'][n]),
        bridge_minus_orig_on=contrast(lambda n: d_items['bridge'][n] - d_items['orig/on'][n]),
        strict=deltas['strict'])
    without_d7 = [n for n in names if n != 'D7']
    desc_no_d7 = {p: mean([float(np.mean(d_items[p][n])) for n in without_d7]) for p in d_items} \
        if len(without_d7) < len(names) else None
    loo = {p: {n: mean([float(np.mean(d_items[p][m])) for m in names if m != n]) for n in names} for p in d_items}
    gates = {p: gate(deltas[p], qd[p], rd[p]) for p in d_items}
    # practical gate: the same rule, with Q and R compared against the selected uniform arm (analysis fix of
    # 2026-11-21, after the run: the first version compared them against the shuffles)
    qdu = {c: summary({n: quality_items(recs[n], c, 'cur', f'{c}:{sel[c]["arm"]}') for n in names}, reps, seed)
           for c in du}
    rdu = {c: summary({n: quality_items(recs[n], c, 'ret', f'{c}:{sel[c]["arm"]}') for n in names}, reps, seed)
           for c in du}
    practical = {c: gate(du[c], qdu[c], rdu[c]) for c in du}
    ivs = {a: {n: categorise(recs[n], a) for n in names} for a in recs[names[0]]['interventions']}
    checks = {n: dict(none_known_all_right=recs[n]['checks']['none_known_all_right'],
                      unfired_max=max(recs[n]['checks']['unfired_differs_from_none'].values()),
                      reversed_same=recs[n]['checks']['determinism']['loc_reversed_same'],
                      matched_recompute=recs[n]['checks']['matched_recompute_max_abs_diff'],
                      corr_unchanged_bitwise=recs[n]['checks'].get('corr_unchanged_bitwise'),
                      shuffle_multisets_equal=recs[n]['checks']['shuffle_multisets_equal'],
                      refused_never_active=recs[n]['checks']['refused_never_active'],
                      changed=len(recs[n]['tables']['changed']), refused=len(recs[n]['tables']['refused']),
                      entries=recs[n]['tables']['n'])
              for n in names}
    headline = dict(delta_corr_off=dec['corr_off'], orig_on_minus_corr_off=dec['orig_on_minus_corr_off'],
                    delta_orig_on=deltas['orig/on'], interaction=dec['interaction'])
    return dict(streams=names, headline=headline, deltas=deltas, practical=du, quality_q=qd, quality_r=rd,
                practical_quality_q=qdu, practical_quality_r=rdu,
                decomposition=dec, gates=gates, practical_gates=practical, without_d7=desc_no_d7,
                leave_one_out=loo, arms=table, arm_means=means, selected_uniform=sel, interventions=ivs,
                checks=checks, dev_tuning=tune(dev, cfg) if dev else None)
