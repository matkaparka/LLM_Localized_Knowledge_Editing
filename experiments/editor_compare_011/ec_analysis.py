"""EC-11 analysis (CPU; numpy). Tuning on the dev records and the pre-registered verdicts on the test records.

Per stream and arm, after the last write:
  coverage   share of the edits' rephrase questions answered with a valid answer (the primary measure)
  damage     share of the base-known locality facts answered wrong (right under `none` by construction)
  also       persona and original-wording accuracy, coverage on rephrases that do not contain the subject
             verbatim, firing / retrieval rates
Stream means weight streams equally.

Tuning (PLAN section 5): per family, among the arms with mean dev damage <= cap, the highest mean dev coverage (ties:
lower damage, then the earlier arm in the config); with none under the cap, the lowest-damage arm, flagged.

Contrasts (PLAN section 6): delta = coverage(reference) - coverage(opponent), stream means of item-paired
differences; two-level bootstrap (streams with replacement, then items with replacement within each drawn stream; the
same item draws for every arm, so arms are paired). Verdicts by the frozen rule in `verdict`.
"""
from __future__ import annotations

import math
import statistics

import numpy as np

from ec_arms import family, subfamily

KINDS = ('reph', 'pers', 'upd', 'loc')


# ---- per stream --------------------------------------------------------------------------------------------------
def arrays(rec: dict) -> dict:
    """Bit arrays per arm and kind, plus the rephrase items' literal flags and write steps."""
    kinds = [k for k, _ in rec['items']]
    idx = {k: [i for i, x in enumerate(kinds) if x == k] for k in KINDS}
    edit_of = {e['uid']: e for e in rec['edits']}
    reph_keys = [rec['items'][i][1] for i in idx['reph']]
    literal = np.array([bool(edit_of[u]['literal']) for u in reph_keys], dtype=bool)
    step = np.array([int(edit_of[u]['step']) for u in reph_keys], dtype=int)
    arms = {}
    for a, v in rec['arms'].items():
        ok = np.frombuffer(v['ok'].encode('ascii'), dtype=np.uint8) - ord('0')
        fired = np.frombuffer(v['fired'].encode('ascii'), dtype=np.uint8) - ord('0')
        arms[a] = {k: ok[idx[k]] for k in KINDS} | {f'fired_{k}': fired[idx[k]] for k in KINDS}
    return dict(arms=arms, literal=literal, step=step, n={k: len(v) for k, v in idx.items()})


def _mean(x) -> float | None:
    return float(np.mean(x)) if len(x) else None


def metrics(arr: dict, arm: str) -> dict:
    a = arr['arms'][arm]
    lit = arr['literal']
    out = dict(coverage=_mean(a['reph']), personas=_mean(a['pers']), update=_mean(a['upd']),
               damage=None if not len(a['loc']) else 1.0 - float(np.mean(a['loc'])),
               coverage_literal=_mean(a['reph'][lit]), coverage_nonliteral=_mean(a['reph'][~lit]),
               fired_loc=_mean(a['fired_loc']), fired_reph=_mean(a['fired_reph']))
    steps = sorted(set(arr['step'].tolist()))
    out['coverage_by_step'] = [_mean(a['reph'][arr['step'] == t]) for t in steps]
    return out


def table(recs: dict, arms: list[str]) -> dict:
    """{arm: {stream: metrics} + 'mean': stream means}"""
    arrs = {s: arrays(r) for s, r in recs.items()}
    out = {}
    for a in arms:
        per = {s: metrics(arr, a) for s, arr in arrs.items() if a in arr['arms']}
        keys = ('coverage', 'personas', 'update', 'damage', 'coverage_literal', 'coverage_nonliteral', 'fired_loc',
                'fired_reph')
        mean = {k: statistics.fmean([m[k] for m in per.values() if m[k] is not None])
                if any(m[k] is not None for m in per.values()) else None for k in keys}
        out[a] = dict(per_stream=per, mean=mean, streams=len(per))
    return out


# ---- tuning -------------------------------------------------------------------------------------------------------
def select(arms: list[str], tab: dict, cap: float) -> tuple[str, bool]:
    """The tuning rule over `arms` (in config order). Returns (arm, met_cap)."""
    under = [a for a in arms if tab[a]['mean']['damage'] is not None and tab[a]['mean']['damage'] <= cap]
    if under:
        best = max(under, key=lambda a: (tab[a]['mean']['coverage'], -tab[a]['mean']['damage'], -arms.index(a)))
        return best, True
    best = min(arms, key=lambda a: (tab[a]['mean']['damage'], -tab[a]['mean']['coverage'], arms.index(a)))
    return best, False


def tune(dev_recs: dict, cfg: dict) -> dict:
    arms = list(cfg['arms']['dev'])
    tab = table(dev_recs, arms)
    cap = float(cfg['objective']['cap'])
    selected, notes = {}, {}
    groups = {}
    for a in arms:
        groups.setdefault(family(a), []).append(a)
        sub = subfamily(a)
        if sub in ('rag_gated', 'ragp'):
            groups.setdefault(sub, []).append(a)
    for g, members in groups.items():
        if g == 'none':
            continue
        arm, met = select(members, tab, cap)
        selected[g] = arm
        notes[g] = dict(met_cap=met, coverage=tab[arm]['mean']['coverage'], damage=tab[arm]['mean']['damage'])
    return dict(selected=selected, notes=notes,
                table={a: dict(tab[a]['mean'], per_stream={s: dict(coverage=m['coverage'], damage=m['damage'])
                                                           for s, m in tab[a]['per_stream'].items()})
                       for a in arms})


# ---- bootstrap ----------------------------------------------------------------------------------------------------
def boot_means(mats: list[np.ndarray], choice: np.ndarray, rng, chunk: int = 256) -> np.ndarray:
    """Two-level bootstrap means. mats[s]: arms x items (0/1) of stream s; choice: B x S drawn stream indices.
    Every (replicate, slot) gets its own item draw, shared by all arms. Returns arms x B (mean over slots)."""
    A = mats[0].shape[0]
    B, S = choice.shape
    M = np.zeros((A, B, S), dtype=np.float64)
    for s, mat in enumerate(mats):
        slots = np.argwhere(choice == s)
        n = mat.shape[1]
        if n == 0:
            M[:, slots[:, 0], slots[:, 1]] = np.nan
            continue
        for c0 in range(0, len(slots), chunk):
            sl = slots[c0:c0 + chunk]
            idx = rng.integers(0, n, size=(len(sl), n))
            M[:, sl[:, 0], sl[:, 1]] = mat[:, idx].mean(-1)
    return M.mean(-1)


def bootstrap(recs: dict, arms: list[str], B: int, seed: int) -> dict:
    """Bootstrap replicates of coverage and damage for every arm (paired across arms)."""
    names = sorted(recs)
    arrs = [arrays(recs[s]) for s in names]
    rng = np.random.default_rng(seed)
    choice = rng.integers(0, len(names), size=(B, len(names)))
    cov = boot_means([np.stack([a['arms'][x]['reph'] for x in arms]).astype(np.float32) for a in arrs], choice, rng)
    loc = boot_means([np.stack([a['arms'][x]['loc'] for x in arms]).astype(np.float32) for a in arrs], choice, rng)
    return dict(arms=arms, coverage=cov, damage=1.0 - loc, streams=names)


def ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    lo, hi = np.quantile(x, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


# ---- verdicts -----------------------------------------------------------------------------------------------------
def verdict(mean: float, lo: float, hi: float, pos: int, neg: int, d_ref: float, d_opp: float, obj: dict) -> str:
    c, tol, guard = float(obj['cap']), float(obj['cap_tol']), float(obj['guard_tol'])
    dq, k = float(obj['delta_q']), int(obj['min_positive_streams'])
    ref_ok, opp_ok = d_ref <= c + tol, d_opp <= c + tol
    if mean >= dq and lo > 0 and pos >= k:
        if ref_ok and d_ref <= d_opp + guard:
            return 'reference ahead'
        return 'undetermined: reference ahead on coverage, but over the damage cap or the guard'
    if mean <= -dq and hi < 0 and neg >= k:
        if opp_ok and d_opp <= d_ref + guard:
            return 'opponent ahead'
        return 'undetermined: opponent ahead on coverage, but over the damage cap or the guard'
    if lo > -dq and hi < dq:
        if ref_ok and opp_ok:
            return 'equivalent'
        return 'undetermined: equivalent coverage, but a method is over the damage cap'
    return 'undetermined: interval too wide'


def contrast(tab: dict, bs: dict, ref: str, opp: str, obj: dict) -> dict:
    i, j = bs['arms'].index(ref), bs['arms'].index(opp)
    diff = bs['coverage'][i] - bs['coverage'][j]
    per = {s: tab[ref]['per_stream'][s]['coverage'] - tab[opp]['per_stream'][s]['coverage'] for s in bs['streams']}
    mean = statistics.fmean(per.values())
    lo, hi = ci(diff)
    pos, neg = sum(v > 0 for v in per.values()), sum(v < 0 for v in per.values())
    d_ref, d_opp = tab[ref]['mean']['damage'], tab[opp]['mean']['damage']
    dd = bs['damage'][i] - bs['damage'][j]
    return dict(reference=ref, opponent=opp, delta=mean, ci=[lo, hi], per_stream=per, positive=pos, negative=neg,
                damage_reference=d_ref, damage_opponent=d_opp,
                damage_reference_ci=list(ci(bs['damage'][i])), damage_opponent_ci=list(ci(bs['damage'][j])),
                damage_diff_ci=list(ci(dd)),
                coverage_reference=tab[ref]['mean']['coverage'], coverage_opponent=tab[opp]['mean']['coverage'],
                verdict=verdict(mean, lo, hi, pos, neg, d_ref, d_opp, obj))


# ---- frontier -----------------------------------------------------------------------------------------------------
def frontier_value(points, x: float):
    """Coverage at damage x on the Pareto frontier of (damage, coverage) points, linearly interpolated; None below
    the frontier's lowest damage; the best coverage beyond its highest."""
    pts = sorted(p for p in points if p[0] is not None and p[1] is not None and not math.isnan(p[0]))
    front, best = [], -math.inf
    for d, c in pts:
        if c > best:
            front.append((d, c))
            best = c
    if not front or x < front[0][0]:
        return None
    for (d0, c0), (d1, c1) in zip(front, front[1:]):
        if d0 <= x <= d1:
            return c0 + (c1 - c0) * (x - d0) / (d1 - d0) if d1 > d0 else max(c0, c1)
    return front[-1][1]


def frontiers(tab: dict, bs: dict, groups: dict, caps, ref_group: str = 'kt') -> dict:
    out = {}
    arms = bs['arms']
    for x in caps:
        row = {}
        point = {g: frontier_value([(tab[a]['mean']['damage'], tab[a]['mean']['coverage']) for a in members], x)
                 for g, members in groups.items()}
        reps = {g: [] for g in groups}
        for b in range(bs['coverage'].shape[1]):
            for g, members in groups.items():
                idx = [arms.index(a) for a in members]
                reps[g].append(frontier_value([(bs['damage'][k, b], bs['coverage'][k, b]) for k in idx], x))
        for g in groups:
            r = dict(point=point[g])
            if g != ref_group:
                pairs = [(u, v) for u, v in zip(reps[ref_group], reps[g]) if u is not None and v is not None]
                if pairs:
                    d = np.array([u - v for u, v in pairs])
                    r.update(delta_vs_kt=None if point[g] is None or point[ref_group] is None
                             else point[ref_group] - point[g], delta_ci=list(ci(d)),
                             replicates_used=len(pairs))
            row[g] = r
        out[str(x)] = row
    return out


# ---- the whole analysis ---------------------------------------------------------------------------------------------
def analyze(recs: dict, cfg: dict, dev_recs: dict | None = None) -> dict:
    obj = cfg['objective']
    arms = list(cfg['arms']['test'])
    tab = table(recs, arms)
    bs = bootstrap(recs, arms, int(obj['bootstrap']), int(obj['bootstrap_seed']))
    sel = cfg['selected']
    prim, sec = {}, {}
    ref = sel['kt']
    for g in ('rag', 'grace', 'hyg'):
        prim[f'kt_vs_{g}'] = contrast(tab, bs, ref, sel[g], obj)
    sec['kt_vs_rad'] = contrast(tab, bs, ref, sel['rad'], obj)
    sec['ragp_vs_rag_gated'] = contrast(tab, bs, sel['ragp'], sel['rag_gated'], obj)
    leads = all(prim[f'kt_vs_{g}']['verdict'] == 'reference ahead' for g in ('rag', 'grace'))
    overall = dict(kt_leads_strong_opponents=leads,
                   kt_beats_cleaning=prim['kt_vs_hyg']['verdict'] == 'reference ahead',
                   p1=prim['kt_vs_rag']['verdict'], p2=prim['kt_vs_grace']['verdict'],
                   p3=prim['kt_vs_hyg']['verdict'])
    groups = {}
    for a in arms:
        if family(a) != 'none':
            groups.setdefault(family(a), []).append(a)
    front = frontiers(tab, bs, groups, obj['frontier_caps'])
    cap_ok = {a: dict(damage=tab[a]['mean']['damage'], meets_cap=tab[a]['mean']['damage'] is not None and
                      tab[a]['mean']['damage'] <= obj['cap'] + obj['cap_tol'], damage_ci=list(ci(bs['damage'][i])))
              for i, a in enumerate(arms)}
    checks = {s: r['checks'] for s, r in recs.items()}
    out = dict(selected=sel, primary=prim, secondary=sec, overall=overall, frontier=front, cap=cap_ok,
               arms={a: tab[a]['mean'] for a in arms},
               per_stream={a: {s: {k: m[k] for k in ('coverage', 'damage', 'personas', 'update',
                                                      'coverage_nonliteral', 'fired_loc')}
                               for s, m in tab[a]['per_stream'].items()} for a in arms},
               coverage_by_step={a: {s: m['coverage_by_step'] for s, m in tab[a]['per_stream'].items()}
                                 for a in set(sel.values()) | {'none'} if a in tab},
               checks=checks, costs=costs(recs, cfg), cost_claims=cost_claims(prim, recs, cfg))
    if dev_recs is not None:
        out['dev_tuning'] = tune(dev_recs, dict(cfg, arms=dict(cfg['arms'], dev=cfg['arms'].get('dev', arms))))
    return out


def costs(recs: dict, cfg: dict) -> dict:
    """Write seconds per edit by method, storage per edit, serving time (timing stream)."""
    rows = {}
    for s, r in recs.items():
        n = len(r['edits'])
        clk = r['clock']
        row = dict(edits=n, kt_seconds_per_edit=(clk['kt_entries'] + clk['kt_table']) / max(1, n))
        for layer, g in (r.get('grace') or {}).items():
            row[f'grace{layer}_seconds_per_edit'] = g['seconds'] / max(1, n)
        if r.get('rag'):
            row['rag_seconds_per_edit'] = r['rag']['write_seconds'] / max(1, n)
        row['storage'] = r['storage']
        rows[s] = row
    timing = next((r['timing'] for r in recs.values() if r.get('timing')), None)
    return dict(per_stream=rows, timing=timing)


def cost_claims(prim: dict, recs: dict, cfg: dict) -> dict:
    """Where a primary contrast is 'equivalent': is the reference's serving overhead at most half the opponent's, with
    an absolute gap of at least 10% of the base time (batch 64)?"""
    timing = next((r['timing'] for r in recs.values() if r.get('timing')), None)
    out = {}
    if not timing or 'none' not in timing:
        return out
    base = timing['none']['batch64']['seconds_per_question']
    for name, c in prim.items():
        if c['verdict'] != 'equivalent':
            continue
        r, o = timing.get(c['reference']), timing.get(c['opponent'])
        if not r or not o:
            out[name] = 'no timing for these arms'
            continue
        ov_r = r['batch64']['seconds_per_question'] - base
        ov_o = o['batch64']['seconds_per_question'] - base
        ratio = ov_r / ov_o if ov_o > 0 else math.inf
        out[name] = dict(overhead_reference=ov_r, overhead_opponent=ov_o, base=base, ratio=ratio,
                         reference_cheaper=ratio <= 0.5 and (ov_o - ov_r) >= 0.1 * base,
                         opponent_cheaper=(ov_o / ov_r if ov_r > 0 else math.inf) <= 0.5 and (ov_r - ov_o) >= 0.1 * base)
    return out
