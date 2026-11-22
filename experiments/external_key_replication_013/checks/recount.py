"""XR-13 independent recount: standard library only, no shared analysis code.

Recomputes from the stream records each family's damage D under every arm, Delta (shuffles - matched) and Delta_U
(selected uniform - matched) on the families with at least MIN_KNOWN base-known old facts, the Q and R differences,
the leave-one-family-out means, the concentration of damage under uniform 0.90, the uniform-threshold choice on the dev
records, both gates (with its own two-level bootstrap), and compares everything with the analysis.

  python checks/recount.py TEST_RUN DEV_RUN OUT.json
"""
from __future__ import annotations

import gzip
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

MIN_KNOWN = 30
TOL = 0.02


def records(run: Path) -> dict:
    out = {}
    for p in sorted(run.glob('stream_*.json.gz')):
        with gzip.open(p, 'rb') as g:
            out[p.name[7:-8]] = json.loads(g.read().decode('utf-8'))
    return out


def base(rec):
    none = rec['arms']['none']['ok']
    return [i for i, (k, _, _) in enumerate(rec['items']) if k == 'loc' and none[i] == '1']


def damage(rec, arm, b):
    ok = rec['arms'][arm]['ok']
    return sum(1 for i in b if ok[i] == '0') / len(b) if b else None


def q_r(rec, arm):
    ok = rec['arms'][arm]['ok']
    by, ret = {}, []
    for i, (k, _, t) in enumerate(rec['items']):
        if k == 'cur':
            by.setdefault(t, []).append(ok[i] == '1')
        elif k == 'ret':
            ret.append(ok[i] == '1')
    return sum(sum(v) / len(v) for v in by.values()) / len(by), sum(ret) / len(ret)


def shuffles(rec):
    return sorted(a for a in rec['arms'] if a.startswith('xr:shuffled:'))


def wrong_items(rec, arm, b):
    ok = rec['arms'][arm]['ok']
    return [1.0 if ok[i] == '0' else 0.0 for i in b]


def delta_items(rec, b, ref=None):
    m = wrong_items(rec, 'xr:matched', b)
    if ref:
        comp = wrong_items(rec, ref, b)
    else:
        sh = [wrong_items(rec, a, b) for a in shuffles(rec)]
        comp = [sum(col) / len(col) for col in zip(*sh)]
    return [c - x for c, x in zip(comp, m)]


def quality_items(rec, kind, ref=None):
    idx = [i for i, (k, _, _) in enumerate(rec['items']) if k == kind]
    ok = lambda arm: [1.0 if rec['arms'][arm]['ok'][i] == '1' else 0.0 for i in idx]  # noqa: E731
    if ref:
        comp = ok(ref)
    else:
        sh = [ok(a) for a in shuffles(rec)]
        comp = [sum(col) / len(col) for col in zip(*sh)]
    return [x - c for x, c in zip(ok('xr:matched'), comp)]


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def boot(per: dict, reps: int, seed: int):
    rng = random.Random(seed)
    names = sorted(per)
    stats = []
    for _ in range(reps):
        tot = 0.0
        for _ in names:
            v = per[rng.choice(names)]
            tot += sum(rng.choices(v, k=len(v))) / len(v)
        stats.append(tot / len(names))
    stats.sort()
    return [stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]]


def gate(per_d: dict, per_q: dict, per_r: dict, ci) -> dict:
    vals = {n: avg(v) for n, v in per_d.items()}
    mean_d = avg(list(vals.values()))
    pos = sum(v > 0 for v in vals.values())
    need = math.ceil(0.75 * len(vals))
    q = avg([avg(v) for v in per_q.values()])
    r = avg([avg(v) for v in per_r.values()])
    return dict(passed=bool(mean_d >= TOL and ci[0] > 0 and pos >= need and q >= -TOL and r >= -TOL),
                mean_delta=mean_d, ci=ci, positive=pos, needed=need, q_diff=q, r_diff=r)


def tune(dev: dict, grid: list) -> str:
    def mean_of(arm):
        ds, qs, rs = [], [], []
        for rec in dev.values():
            ds.append(damage(rec, arm, base(rec)))
            q, r = q_r(rec, arm)
            qs.append(q)
            rs.append(r)
        return avg(ds), avg(qs), avg(rs)
    _, qm, rm = mean_of('xr:matched')
    pts = []
    for x in grid:
        arm = f'uniform:{x:.2f}'
        d, q, r = mean_of(f'xr:{arm}')
        pts.append((arm, d, q, r, max(qm - TOL - q, rm - TOL - r)))
    ok = [p for p in pts if p[4] <= 1e-12]
    best = (min(ok, key=lambda p: (p[1], -p[2], float(p[0].split(':')[1]))) if ok else
            min(pts, key=lambda p: (p[4], -p[2], float(p[0].split(':')[1]))))
    return best[0]


def concentration(rec, arm='xr:uniform:0.90', top_frac=0.01, min_items=5):
    ivs = rec['interventions'].get(arm, [])
    answers = rec['loc_answers']
    culprits = {r['i']: [e for e, ans in r['alone'] if ans != answers[str(r['i'])]] for r in ivs if 'alone' in r}
    counts = Counter(e for cs in culprits.values() for e in cs)
    k = max(1, math.ceil(top_frac * rec['tables']['n']))
    top = [e for e, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:k]]
    hit = sum(1 for cs in culprits.values() if set(cs) & set(top))
    return dict(damaged=len(ivs), top_k=k, numerator=hit,
                share=(hit / len(ivs)) if len(ivs) >= min_items else None)


def main(test_run: str, dev_run: str, out_path: str):
    test, dev = records(Path(test_run)), records(Path(dev_run))
    an = json.loads((Path(test_run) / 'analysis.json').read_text(encoding='utf-8'))
    cfg = json.loads((Path(test_run) / 'config.json').read_text(encoding='utf-8'))
    sel = cfg['selected_uniform']['arm']
    ref = f'xr:{sel}'
    diffs = []
    fam = {}
    for n, rec in test.items():
        b = base(rec)
        fam[n] = dict(known=len(b), damage={a: damage(rec, a, b) for a in rec['arms']},
                      shuffled_mean=avg([damage(rec, a, b) for a in shuffles(rec)]), measured=len(b) >= MIN_KNOWN)
        q, r = q_r(rec, 'xr:matched')
        fam[n].update(Q_matched=q, R_matched=r)
    keep = sorted(n for n in fam if fam[n]['measured'])
    no_damage = sorted(n for n in fam if not fam[n]['measured'])
    per_d = {n: delta_items(test[n], base(test[n])) for n in keep}
    per_du = {n: delta_items(test[n], base(test[n]), ref) for n in keep}
    per_q = {n: quality_items(test[n], 'cur') for n in test}
    per_r = {n: quality_items(test[n], 'ret') for n in test}
    per_qu = {n: quality_items(test[n], 'cur', ref) for n in test}
    per_ru = {n: quality_items(test[n], 'ret', ref) for n in test}
    for n in keep:
        fam[n]['delta'] = avg(per_d[n])
        fam[n]['delta_u'] = avg(per_du[n])
        diffs += [abs(fam[n]['delta'] - an['delta']['per_stream'][n]),
                  abs(fam[n]['delta_u'] - an['delta_uniform']['per_stream'][n])]
    for n in test:
        for a, v in fam[n]['damage'].items():
            if v is not None:
                diffs.append(abs(v - an['arms'][a][n]['damage']))
        diffs += [abs(avg(per_q[n]) - an['quality_q']['per_stream'][n]), abs(avg(per_r[n]) - an['quality_r']['per_stream'][n]),
                  abs(avg(per_qu[n]) - an['practical_quality_q']['per_stream'][n]),
                  abs(avg(per_ru[n]) - an['practical_quality_r']['per_stream'][n])]
    means = dict(delta=avg([fam[n]['delta'] for n in keep]), delta_u=avg([fam[n]['delta_u'] for n in keep]),
                 q=avg([avg(v) for v in per_q.values()]), r=avg([avg(v) for v in per_r.values()]),
                 q_u=avg([avg(v) for v in per_qu.values()]), r_u=avg([avg(v) for v in per_ru.values()]),
                 shuffled_damage=avg([fam[n]['shuffled_mean'] for n in keep]),
                 matched_damage=avg([fam[n]['damage']['xr:matched'] for n in keep]))
    diffs += [abs(means['delta'] - an['delta']['mean']), abs(means['delta_u'] - an['delta_uniform']['mean']),
              abs(means['q'] - an['quality_q']['mean']), abs(means['r'] - an['quality_r']['mean'])]
    loo = {n: avg([fam[m]['delta'] for m in keep if m != n]) for n in keep}
    diffs += [abs(loo[n] - an['leave_one_out'][n]) for n in keep]
    conc = {n: concentration(test[n]) for n in test}
    conc_same = all(conc[n]['numerator'] == an['concentration'][n]['numerator'] and
                    conc[n]['damaged'] == an['concentration'][n]['damaged'] for n in test)
    reps = 2000
    ci_d, ci_du = boot(per_d, reps, 0), boot(per_du, reps, 0)
    mech = gate(per_d, per_q, per_r, ci_d)
    prac = gate(per_du, per_qu, per_ru, ci_du)
    boot_cmp = dict(delta=dict(recount=ci_d, analysis=an['delta']['ci']),
                    delta_u=dict(recount=ci_du, analysis=an['delta_uniform']['ci']))
    for v in boot_cmp.values():
        v['max_diff'] = max(abs(v['recount'][0] - v['analysis'][0]), abs(v['recount'][1] - v['analysis'][1]))
    tuned = tune(dev, cfg['arms']['uniform'])
    out = dict(families=sorted(test), measured=keep, no_damage_measurement=no_damage, per_family=fam, means=means,
               leave_one_out=loo, concentration=conc, concentration_matches=conc_same,
               mechanism_gate=mech, practical_gate=prac,
               gates_match=dict(mechanism=mech['passed'] == an['mechanism_gate']['passed'],
                                practical=prac['passed'] == an['practical_gate']['passed']),
               no_damage_matches=no_damage == sorted(an['no_damage_measurement']),
               bootstrap=boot_cmp, bootstrap_reps=reps, tuning_selected=tuned, tuning_matches_config=tuned == sel,
               max_abs_diff=max(diffs))
    Path(out_path).write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(max_abs_diff=out['max_abs_diff'], tuning_matches_config=out['tuning_matches_config'],
                          gates_match=out['gates_match'], concentration_matches=conc_same,
                          no_damage_matches=out['no_damage_matches'],
                          bootstrap={k: v['max_diff'] for k, v in boot_cmp.items()}), indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
