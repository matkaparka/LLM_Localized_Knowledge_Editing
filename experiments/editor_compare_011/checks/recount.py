"""EC-11 independent recount (standard library only; shares no code with ec_analysis.py).

  python checks/recount.py TEST_RUN DEV_RUN OUT.json

From the raw stream records it recomputes, with its own code:
- every arm's coverage (rephrase items) and damage (locality items) per stream and as stream means;
- the dev tuning choice of every family and sub-family (PLAN section 5), compared with config_test.json;
- the contrasts' point estimates (per-stream deltas, mean, positive streams, damages);
- a separate two-level bootstrap (2,000 replicates, its own random stream) for the primary contrasts' intervals,
  which should agree with the analysis to about 0.005;
and reports the largest differences against analysis.json.
"""
from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path


def load(run: Path) -> dict:
    out = {}
    for p in sorted(run.glob('stream_*.json.gz')):
        with gzip.open(p, 'rb') as g:
            out[p.name[7:-8]] = json.loads(g.read().decode('utf-8'))
    return out


def fam(arm):
    k = arm.split(':')[0]
    return {'kthyg': 'kt', 'ragp': 'rag'}.get(k, k)


def sub(arm):
    if arm.startswith('ragp:'):
        return 'ragp'
    if arm.startswith('rag:') and not arm.startswith('rag:always'):
        return 'rag_gated'
    return None


def bits(rec, arm, kind):
    kinds = [k for k, _ in rec['items']]
    s = rec['arms'][arm]['ok']
    return [int(s[i]) for i, k in enumerate(kinds) if k == kind]


def stream_metrics(recs, arm):
    per = {}
    for name, r in recs.items():
        reph, loc = bits(r, arm, 'reph'), bits(r, arm, 'loc')
        per[name] = (sum(reph) / len(reph), 1 - sum(loc) / len(loc))
    n = len(per)
    return per, (sum(v[0] for v in per.values()) / n, sum(v[1] for v in per.values()) / n)


def tune(dev, arms, cap):
    means = {a: stream_metrics(dev, a)[1] for a in arms}
    groups = {}
    for a in arms:
        if a == 'none':
            continue
        groups.setdefault(fam(a), []).append(a)
        if sub(a):
            groups.setdefault(sub(a), []).append(a)
    out = {}
    for g, members in groups.items():
        under = [a for a in members if means[a][1] <= cap]
        if under:
            best = under[0]
            for a in under[1:]:
                c, d = means[a]
                bc, bd = means[best]
                if c > bc or (c == bc and d < bd):
                    best = a
        else:
            best = members[0]
            for a in members[1:]:
                c, d = means[a]
                bc, bd = means[best]
                if d < bd or (d == bd and c > bc):
                    best = a
        out[g] = best
    return out


def boot(recs, ref, opp, B=2000, seed=12345):
    rng = random.Random(seed)
    names = sorted(recs)
    data = {s: (bits(recs[s], ref, 'reph'), bits(recs[s], opp, 'reph')) for s in names}
    reps = []
    for _ in range(B):
        tot = 0.0
        for _ in names:
            s = names[rng.randrange(len(names))]
            a, b = data[s]
            n = len(a)
            idx = [rng.randrange(n) for _ in range(n)]
            tot += sum(a[i] - b[i] for i in idx) / n
        reps.append(tot / len(names))
    reps.sort()
    return reps[int(0.025 * B)], reps[int(0.975 * B) - 1]


def main(test_run: str, dev_run: str, out: str):
    test_run, dev_run = Path(test_run), Path(dev_run)
    cfg = json.loads((test_run / 'config.json').read_text(encoding='utf-8'))
    res = json.loads((test_run / 'analysis.json').read_text(encoding='utf-8'))
    test, dev = load(test_run), load(dev_run)
    rep = {}
    worst = 0.0
    for a in cfg['arms']['test']:
        per, (c, d) = stream_metrics(test, a)
        m = res['arms'][a]
        worst = max(worst, abs(c - m['coverage']), abs(d - m['damage']))
        rep.setdefault('arms', {})[a] = dict(coverage=c, damage=d)
    rep['max_abs_diff_arm_means'] = worst
    sel = tune(dev, list(cfg['arms']['dev']), float(cfg['objective']['cap']))
    rep['tuning_recount'] = sel
    rep['tuning_matches_config'] = all(cfg['selected'].get(g) == a for g, a in sel.items()) and \
        set(sel) == set(cfg['selected'])
    cons = {**res['primary'], **res['secondary']}
    cdiff = 0.0
    rep['contrasts'] = {}
    for name, c in cons.items():
        pr, _ = stream_metrics(test, c['reference'])
        po, _ = stream_metrics(test, c['opponent'])
        deltas = {s: pr[s][0] - po[s][0] for s in pr}
        mean = sum(deltas.values()) / len(deltas)
        cdiff = max(cdiff, abs(mean - c['delta']), *(abs(deltas[s] - c['per_stream'][s]) for s in deltas))
        row = dict(delta=mean, positive=sum(v > 0 for v in deltas.values()), per_stream=deltas)
        if name in res['primary']:
            lo, hi = boot(test, c['reference'], c['opponent'])
            row.update(ci_recount=[lo, hi], ci_analysis=c['ci'],
                       ci_max_abs_diff=max(abs(lo - c['ci'][0]), abs(hi - c['ci'][1])))
        rep['contrasts'][name] = row
    rep['max_abs_diff_contrasts'] = cdiff
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps({k: rep[k] for k in ('max_abs_diff_arm_means', 'tuning_matches_config', 'max_abs_diff_contrasts')}))
    for name, row in rep['contrasts'].items():
        if 'ci_recount' in row:
            print(name, 'delta', round(row['delta'], 4), 'ci recount', [round(x, 4) for x in row['ci_recount']],
                  'analysis', [round(x, 4) for x in row['ci_analysis']])


if __name__ == '__main__':
    main(*sys.argv[1:4])
