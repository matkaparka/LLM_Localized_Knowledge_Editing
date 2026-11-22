"""RV-09 offline analysis (CPU only).

A1: per family and writer, learning (original-wording accuracy right after each write), rephrase accuracy,
retention of earlier edits, end-of-stream damage to base-known facts, forgetting, and the stream value (mean SO-08
composite over all steps). The frozen gate picks feasible writers, the default and the second action, or stops.

A2: for each family (one public record H = family x decision step) and each state s (write seed of the history),
D[s, j] = value(a) - value(b) under future j. Futures are split into discovery and verification halves.
- cross-fitted selection gain: the state-aware choice sign(mean_disc D[s]) against the H-only choice
  sign(mean over s of mean_disc D[s]), scored on verification futures; then the halves swap roles; averaged.
  This estimates V(state) - V(H), the most any readout of the state could add (no readout cost).
- in-sample bound: (mean_s |mu_s| - |mean_s mu_s|) / 2 with mu_s over all futures; biased upward by noise.
- reversal: states of the same H whose discovery means have opposite signs, with the verification means
  repeating both signs.
Statistics are over families (the independent units); steps, futures and states are not counted as samples.
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

_SO08 = str(Path(__file__).resolve().parents[1] / 'self_observation_008')
if _SO08 not in sys.path:
    sys.path.insert(0, _SO08)

from so_analysis import edit_forgetting, locality_damage  # noqa: E402


# ---- A1 ---------------------------------------------------------------------------------------------------------
def a1_row(rec: dict) -> dict:
    steps = rec['trajectory']['steps']
    evals = [s['eval'] for s in steps]
    upd = [e['acc']['upd'] for e in evals if e['acc']['upd'] is not None]
    cur = [e['acc']['cur'] for e in evals if e['acc']['cur'] is not None]
    comp = [e['composite'] for e in evals if e['composite'] is not None]
    ld = locality_damage(evals[-1], rec['base_loc'])
    ef = edit_forgetting(evals)
    return dict(family=rec['family'], writer=rec['writer'], seed=rec['seed'],
                learn=sum(upd) / len(upd) if upd else None, rephrase=sum(cur) / len(cur) if cur else None,
                retention_final=evals[-1]['acc']['ret'], locality_final=evals[-1]['acc']['loc'],
                damage=ld['fraction'], damage_n=ld['right_to_wrong'], base_right=ld['base_right'],
                forgetting=ef['fraction'], value=sum(comp) / len(comp) if comp else None,
                seconds=rec.get('seconds'), write_seconds=sum(s['exec'].get('write_seconds', 0) for s in steps))


def a1_table(records) -> list[dict]:
    return [a1_row(r) for r in records]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def a1_gate(rows: list[dict], gate: dict) -> dict:
    """Frozen A1 rule.

    feasible writer: in every family, learn >= learn_min and end-of-stream damage <= damage_max (seed means).
    default d: gate['default'] if feasible, else the feasible writer with the highest mean value.
    second b: the feasible writer other than d with the highest mean value.
    stop if no feasible writer, if only one, or if one of d, b beats the other by more than delta in every family
    (the same fixed write is always best). Otherwise A2 runs with actions (d, b) and default d.
    """
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r['writer']][r['family']].append(r)
    families = sorted({r['family'] for r in rows})
    summary = {}
    for w, fams in by.items():
        per = {}
        for f in families:
            rs = fams.get(f, [])
            per[f] = dict(learn=_mean([r['learn'] for r in rs]), damage=_mean([r['damage'] for r in rs]),
                          value=_mean([r['value'] for r in rs]), rephrase=_mean([r['rephrase'] for r in rs]),
                          forgetting=_mean([r['forgetting'] for r in rs]), n=len(rs))
        ok = all(per[f]['n'] and per[f]['learn'] is not None and per[f]['learn'] >= gate['learn_min']
                 and per[f]['damage'] is not None and per[f]['damage'] <= gate['damage_max'] for f in families)
        summary[w] = dict(per_family=per, feasible=ok, value=_mean([per[f]['value'] for f in families]),
                          learn=_mean([per[f]['learn'] for f in families]),
                          damage=_mean([per[f]['damage'] for f in families]))
    feasible = sorted((w for w, s in summary.items() if s['feasible']), key=lambda w: -summary[w]['value'])
    out = dict(rule=gate, families=families, summary=summary, feasible=feasible, go=False)
    if not feasible:
        out['reason'] = 'no feasible writer'
        return out
    d = gate['default'] if gate['default'] in feasible else feasible[0]
    others = [w for w in feasible if w != d]
    out['default'] = d
    if not others:
        out['reason'] = f'only one feasible writer ({d}): fixed write'
        return out
    b = others[0]
    out['second'] = b
    diffs = {f: summary[d]['per_family'][f]['value'] - summary[b]['per_family'][f]['value'] for f in families}
    out['value_diff_default_minus_second'] = diffs
    delta = gate['delta']
    if all(x > delta for x in diffs.values()) or all(x < -delta for x in diffs.values()):
        out['reason'] = 'the same fixed write is best in every family by more than delta'
        return out
    out['go'] = True
    out['reason'] = 'two feasible writers without uniform dominance: run A2'
    return out


# ---- A2 ---------------------------------------------------------------------------------------------------------
def d_matrix(records, a: str, b: str) -> dict:
    """{family: {seed: [D per future]}} from A2 records (one record per family and state seed)."""
    out = defaultdict(dict)
    for rec in records:
        va = [x['value'] for x in rec['branches'][a]['futures']]
        vb = [x['value'] for x in rec['branches'][b]['futures']]
        out[rec['family']][rec['seed']] = [x - y for x, y in zip(va, vb)]
    return out


def _gain(mu_sel: list[float], mu_eval: list[float]) -> float:
    """Selection gain of choosing per state by sign(mu_sel[s]) over choosing one action for all states by the sign
    of mean(mu_sel), scored with mu_eval."""
    pick_h = 1.0 if sum(mu_sel) / len(mu_sel) > 0 else 0.0
    return sum(((1.0 if m > 0 else 0.0) - pick_h) * e for m, e in zip(mu_sel, mu_eval)) / len(mu_sel)


def family_estimates(dmat: dict, disc: list[int], ver: list[int]) -> dict:
    seeds = sorted(dmat)
    mu_d = [sum(dmat[s][j] for j in disc) / len(disc) for s in seeds]
    mu_v = [sum(dmat[s][j] for j in ver) / len(ver) for s in seeds]
    allf = sorted(set(disc) | set(ver))
    mu_a = [sum(dmat[s][j] for j in allf) / len(allf) for s in seeds]
    cross = (_gain(mu_d, mu_v) + _gain(mu_v, mu_d)) / 2
    in_sample = (sum(abs(m) for m in mu_a) / len(mu_a) - abs(sum(mu_a) / len(mu_a))) / 2
    pos_d = [s for s, m in zip(seeds, mu_d) if m > 0]
    neg_d = [s for s, m in zip(seeds, mu_d) if m < 0]
    candidate = bool(pos_d and neg_d)
    confirmed = candidate and any(mu_v[seeds.index(s)] > 0 for s in pos_d) and \
        any(mu_v[seeds.index(s)] < 0 for s in neg_d)
    # noise: spread of D over futures within a state, against the spread of state means
    within = [_sd(dmat[s]) for s in seeds]
    return dict(seeds=seeds, mu_disc=mu_d, mu_ver=mu_v, mu_all=mu_a, mean_D=sum(mu_a) / len(mu_a),
                cross_fitted_gain=cross, in_sample_bound=in_sample, reversal_candidate=candidate,
                reversal_confirmed=confirmed, sd_within_state=_mean(within), sd_between_states=_sd(mu_a))


def _sd(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def bootstrap(values: list[float], reps: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (float('nan'), float('nan'))
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * (reps - 1))], means[int(0.975 * (reps - 1))]


def a2_verdict(records, a: str, b: str, rule: dict) -> dict:
    """Frozen A2 rule (delta = minimal meaningful end-of-stream improvement).

    stop ('fixed write'): the upper 95% family-bootstrap bound of the in-sample bound is below delta (even the
      noise-inflated oracle gain cannot reach delta);
    go ('reversal worth a readout test', i.e. section 5C may follow in a new run): the cross-fitted gain is at least
      delta on average, its family-bootstrap lower bound is above 0, and at least half of the families show a
      confirmed reversal;
    otherwise 'undetermined'.
    """
    dm = d_matrix(records, a, b)
    fam = {f: family_estimates(v, rule['discovery'], rule['verification']) for f, v in sorted(dm.items())}
    cross = [e['cross_fitted_gain'] for e in fam.values()]
    bound = [e['in_sample_bound'] for e in fam.values()]
    out = dict(actions=[a, b], rule=rule, families=fam,
               cross_fitted_gain=_mean(cross), cross_fitted_ci=bootstrap(cross),
               in_sample_bound=_mean(bound), in_sample_ci=bootstrap(bound),
               mean_D=_mean([e['mean_D'] for e in fam.values()]),
               confirmed_reversals=sum(e['reversal_confirmed'] for e in fam.values()),
               candidate_reversals=sum(e['reversal_candidate'] for e in fam.values()))
    delta = rule['delta']
    if out['in_sample_ci'][1] < delta:
        out['verdict'] = 'fixed write: even the noise-inflated oracle gain is reliably below delta'
    elif out['cross_fitted_gain'] >= delta and out['cross_fitted_ci'][0] > 0 \
            and out['confirmed_reversals'] * 2 >= len(fam):
        out['verdict'] = 'repeatable reversal with a meaningful gain: a readout test (section 5C) may follow'
    else:
        out['verdict'] = 'undetermined'
    return out
