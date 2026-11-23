"""RC-15 / BR-14 analysis.

Per arm: FP-12's measures (fp_analysis.arm_metrics: D, Q, R, coverage, update, learned, firing). Per cell P:
  delta           mean_p D(P:shuffled:p) - D(P:matched)                      (FP-12's Δ)
  delta_active    mean_p D(P:active_shuffled:p) - D(P:matched)               (banned entries held in place)
  delta_bans      mean_p D(P:shuffled:p) - mean_p D(P:active_shuffled:p)     (what moving the bans adds)
  factorial over {bans kept, bans at the floor} x {per-key, flat floor} (tau = the floor, 0.90 unless set):
    total         D(uniform:tau) - D(matched)
    ban_flat      D(uniform:tau) - D(ban_uniform:tau)        banning, flat thresholds
    ban_cont      D(noban_matched) - D(matched)              banning, per-key thresholds
    cont_ban      D(ban_uniform:tau) - D(matched)            per-key vs flat, bans kept
    cont_noban    D(uniform:tau) - D(noban_matched)          per-key vs flat, no bans
    interaction   ban_flat - ban_cont
  practical       D(comparator) - D(matched) for every comparator in config['practical'][P] (the dev-selected
                  uniform and ban_uniform levels), with Q and R compared against the same comparator
Intervals: FP-12's two-level paired bootstrap (streams, then items within streams; streams weighted equally).
Gates: FP-12's descriptive rule (mean >= 2 pp, interval above 0, >= 3/4 streams positive, Q and R not more than 2 pp
below) for delta, delta_active and each practical comparison. The selection rule for uniform and ban_uniform levels
is FP-12's (fp_analysis.tune) over the chosen grid. Culprits of the per-key interventions are split by whether the
entry is banned under matched. With a reference run (FP-12's test records) the shared arms are compared item by item.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rc_core  # noqa: E402,F401  (import paths)
import fp_analysis as fa  # noqa: E402

TOL = 0.02


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def arms_of(rec, cell: str, kind: str) -> list[str]:
    pre = f'{cell}:{kind}:'
    return sorted((a for a in rec['arms'] if a.startswith(pre)), key=lambda a: int(a[len(pre):]))


def has(rec, *arms) -> bool:
    return all(a in rec['arms'] for a in arms)


def floor_of(rec) -> float:
    return float((rec.get('options') or {}).get('tau', 0.9))


def lvl(x: float) -> str:
    return f'{x:.2f}'


# ---- item vectors --------------------------------------------------------------------------------------------------
def d_diff(rec, a: str | list, b: str | list) -> np.ndarray:
    """Per base-known item: wrong under a (or the mean over a list of arms) minus wrong under b."""
    idx = fa.base_loc(rec)

    def w(x):
        if isinstance(x, list):
            return np.mean([fa.wrong_vec(rec, y, idx) for y in x], axis=0) if x else None
        return fa.wrong_vec(rec, x, idx)
    return w(a) - w(b)


def q_diff(rec, kind: str, a: str, b: str | list) -> np.ndarray:
    """Per item of `kind` ('cur' for Q, 'ret' for R): right under a minus right under b (or the mean over a list)."""
    ks = fa.kinds(rec)
    idx = [i for i, k in enumerate(ks) if k == kind]

    def ok(x):
        if isinstance(x, list):
            return np.mean([ok(y) for y in x], axis=0)
        return np.array([rec['arms'][x]['ok'][i] == '1' for i in idx], dtype=np.float64)
    return ok(a) - ok(b)


# ---- selection rule ------------------------------------------------------------------------------------------------
def select(recs: dict, cell: str, kind: str, levels: list[float]) -> dict:
    """FP-12's rule over the grid kind:level: among points with Q >= Q_M - 0.02 and R >= R_M - 0.02 the lowest mean
    damage (ties: higher Q, then lower level); else the smallest largest shortfall, flagged 'dev gate not met'."""
    M = {k: mean([fa.arm_metrics(r, f'{cell}:matched')[k] for r in recs.values()]) for k in ('Q', 'R', 'damage')}
    pts = []
    for x in levels:
        g = f'{kind}:{lvl(x)}'
        m = {k: mean([fa.arm_metrics(r, f'{cell}:{g}')[k] for r in recs.values()]) for k in ('Q', 'R', 'damage')}
        short = max(M['Q'] - TOL - m['Q'], M['R'] - TOL - m['R'])
        pts.append((g, float(x), m, short))
    ok = [p for p in pts if p[3] <= 1e-12]
    if ok:
        best, flag = min(ok, key=lambda p: (p[2]['damage'], -p[2]['Q'], p[1])), 'met'
    else:
        best, flag = min(pts, key=lambda p: (p[3], -p[2]['Q'], p[1])), 'dev gate not met'
    return dict(arm=best[0], level=best[1], flag=flag, matched=M, grid={p[0]: dict(p[2], shortfall=p[3]) for p in pts})


def tune(recs: dict, cfg: dict) -> dict:
    t = cfg['tune']
    return {cell: {kind: select(recs, cell, kind, t['levels']) for kind in t['kinds']} for cell in t['cells']}


# ---- culprits by ban status --------------------------------------------------------------------------------------------
def culprits_by_ban(rec, arm: str) -> dict | None:
    ivs = rec['interventions'].get(arm)
    if ivs is None:
        return None
    cell = arm.split(':', 1)[0]
    row = cell.split('/')[0]
    bans = set(rec['bans'][row])
    answers = rec['loc_answers']
    per, ents = Counter(), Counter()
    for r in ivs:
        i = str(r['i'])
        if not r['fired']:
            per['no firing'] += 1
            continue
        if 'alone' not in r:
            per['too many fired'] += 1
            continue
        cul = [e for e, ans in r['alone'] if ans != answers[i]]
        if not cul:
            per['no single culprit'] += 1
            continue
        k = {('banned' if e in bans else 'active') for e in cul}
        per['banned' if k == {'banned'} else 'active' if k == {'active'} else 'both'] += 1
        for e in cul:
            ents[e] += 1
    ent = rec['tables'].get('entity') or []
    top = [[e, n, e in bans, ent[e] if e < len(ent) else None] for e, n in ents.most_common(8)]
    return dict(damaged=len(ivs), per_item=dict(per), top=top)


# ---- the analysis ------------------------------------------------------------------------------------------------------
def gate(d: dict, q: dict, r: dict) -> dict:
    return fa.gate(d, q, r)


def analyze(recs: dict, cfg: dict, dev: dict | None = None, ref: dict | None = None) -> dict:
    reps, seed = int(cfg['rule']['bootstrap']), int(cfg['rule']['bootstrap_seed'])
    names = sorted(recs)
    first = recs[names[0]]
    cells = first['cells']
    arms_all = sorted(first['arms'])
    table = {a: {n: fa.arm_metrics(recs[n], a) for n in names} for a in arms_all}
    means = {a: {k: mean([v[k] for v in table[a].values()]) for k in next(iter(table[a].values()))}
             for a in arms_all}

    def S(fn, pick=None):
        use = [n for n in names if pick is None or pick(recs[n])]
        return fa.summary({n: fn(recs[n]) for n in use}, reps, seed)

    out = {}
    for P in cells:
        res = {}
        sh, ac = arms_of(first, P, 'shuffled'), arms_of(first, P, 'active_shuffled')
        M = f'{P}:matched'
        if sh:
            res['delta'] = S(lambda r: d_diff(r, arms_of(r, P, 'shuffled'), M))
            res['delta_q'] = S(lambda r: q_diff(r, 'cur', M, arms_of(r, P, 'shuffled')))
            res['delta_r'] = S(lambda r: q_diff(r, 'ret', M, arms_of(r, P, 'shuffled')))
            res['delta_gate'] = gate(res['delta'], res['delta_q'], res['delta_r'])
        if ac:
            res['delta_active'] = S(lambda r: d_diff(r, arms_of(r, P, 'active_shuffled'), M))
            res['delta_active_q'] = S(lambda r: q_diff(r, 'cur', M, arms_of(r, P, 'active_shuffled')))
            res['delta_active_r'] = S(lambda r: q_diff(r, 'ret', M, arms_of(r, P, 'active_shuffled')))
            res['delta_active_gate'] = gate(res['delta_active'], res['delta_active_q'], res['delta_active_r'])
        if sh and ac:
            res['delta_bans'] = S(lambda r: d_diff(r, arms_of(r, P, 'shuffled'), arms_of(r, P, 'active_shuffled')))
        tau = lvl(floor_of(first))
        U, B, N = f'{P}:uniform:{tau}', f'{P}:ban_uniform:{tau}', f'{P}:noban_matched'
        fac = {}
        for key, a, b in (('total', U, M), ('ban_flat', U, B), ('ban_cont', N, M), ('cont_ban', B, M),
                          ('cont_noban', U, N)):
            if has(first, a, b):
                fac[key] = S(lambda r, a=a, b=b: d_diff(r, a, b))
        if has(first, U, B, N, M):
            fac['interaction'] = S(lambda r: d_diff(r, U, B) - d_diff(r, N, M))
        res['factorial'] = fac
        prac = {}
        for comp in (cfg.get('practical') or {}).get(P, []):
            C = f'{P}:{comp}'
            if not has(first, C):
                continue
            du = S(lambda r, C=C: d_diff(r, C, M))
            qd = S(lambda r, C=C: q_diff(r, 'cur', M, C))
            rd = S(lambda r, C=C: q_diff(r, 'ret', M, C))
            prac[comp] = dict(delta=du, q=qd, r=rd, gate=gate(du, qd, rd))
        res['practical'] = prac
        front = {}
        for kind in ('uniform', 'ban_uniform'):
            pts = sorted({a.split(':', 2)[2] for a in first['arms'] if a.startswith(f'{P}:{kind}:')}, key=float)
            front[kind] = {x: {k: means[f'{P}:{kind}:{x}'][k] for k in ('damage', 'Q', 'R', 'coverage', 'update')}
                           for x in pts}
        res['frontier'] = front
        row = P.split('/')[0]
        res['bans'] = {n: dict(banned=len(recs[n]['bans'][row]), entries=recs[n]['tables']['n'],
                               mid=sum(1 for x in recs[n]['thresholds'][row] if floor_of(recs[n]) + 1e-9 < x <= 1.0),
                               floor=sum(1 for x in recs[n]['thresholds'][row] if abs(x - floor_of(recs[n])) < 1e-6))
                       for n in names}
        res['culprits'] = {a.split(':', 1)[1]: {n: culprits_by_ban(recs[n], a) for n in names}
                           for a in first['interventions'] if a.startswith(P + ':')}
        res['culprit_categories'] = {a.split(':', 1)[1]: {n: fa.categorise(recs[n], a) for n in names}
                                     for a in first['interventions'] if a.startswith(P + ':') and '/' in P}
        loo = {}
        if sh:
            for n in names:
                loo[n] = mean([float(np.mean(d_diff(recs[m], arms_of(recs[m], P, 'shuffled'), M)))
                               for m in names if m != n and len(fa.base_loc(recs[m]))])
        res['delta_leave_one_out'] = loo
        out[P] = res

    checks = {}
    for n in names:
        c = recs[n]['checks']
        checks[n] = dict(none_known_all_right=c['none_known_all_right'],
                         unfired_max=max(c['unfired_differs_from_none'].values()),
                         reversed_same=c['determinism']['loc_reversed_same'],
                         matched_recompute=c.get('matched_recompute_max_abs_diff'),
                         corr_unchanged_bitwise=c.get('corr_unchanged_bitwise'),
                         shuffle_multisets_equal=c['shuffle_multisets_equal'],
                         active_shuffle_bans_fixed=c['active_shuffle_bans_fixed'],
                         ban_uniform_bans_off=c['ban_uniform_bans_off'],
                         refused_never_active=c['refused_never_active'],
                         blocked=c.get('blocked_cells_never_fire_in_context', c.get('blocked_never_fire_in_context')),
                         known=recs[n]['screening']['known'], entries=recs[n]['tables']['n'],
                         changed=len(recs[n]['tables']['changed']), refused=len(recs[n]['tables']['refused']),
                         seconds=recs[n].get('seconds'))
    headline = {P: {k: (out[P][k]['mean'], out[P][k]['ci'], out[P][k]['positive'], out[P][k]['n'])
                    for k in ('delta', 'delta_active', 'delta_bans') if k in out[P]} for P in cells}
    return dict(streams=names, cells=cells, options=first.get('options'), norm_info={n: recs[n].get('norm_info')
                                                                                   for n in names},
                headline=headline, cells_result=out, arms=table, arm_means=means, checks=checks,
                selected=cfg.get('selected'), practical_comparators=cfg.get('practical'),
                dev_tuning=tune(dev, cfg) if dev and cfg.get('tune') else None,
                reproduction=reproduce(recs, ref) if ref else None)


# ---- reproduction against an earlier run ---------------------------------------------------------------------------------
def reproduce(recs: dict, ref: dict) -> dict:
    """For streams in both runs: are the tables bit-identical (digests), and do arms with the same name give the same
    per-item outcomes and the same wrong answers?"""
    out = {}
    for n in sorted(set(recs) & set(ref)):
        a, b = recs[n], ref[n]
        tabs = {r: a['tables'].get(r) == b['tables'].get(r) for r in ('orig', 'corr', 'xr') if r in a['tables']}
        same, diff = [], []
        for arm in sorted(set(a['arms']) & set(b['arms'])):
            x, y = a['arms'][arm], b['arms'][arm]
            if a['items'] != b['items']:
                diff.append(arm)
                continue
            (same if x['ok'] == y['ok'] and x['wrong_loc'] == y['wrong_loc'] else diff).append(arm)
        out[n] = dict(tables_identical=tabs, items_identical=a['items'] == b['items'], shared_arms=len(same) + len(diff),
                      identical_arms=len(same), differing_arms=diff[:20])
    return out


def sig(x: float | None) -> str:
    return '—' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{100 * x:+.2f}'
