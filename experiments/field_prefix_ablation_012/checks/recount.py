"""FP-12 independent recount: standard library only, no shared analysis code.

Recomputes from the stream records every arm's damage D, Q and R, the damage reduction of every cell, the
decomposition, the uniform-threshold choice on the dev records, and a two-level bootstrap of the two headline numbers;
then compares them with the analysis.

  python checks/recount.py TEST_RUN DEV_RUN OUT.json
"""
from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

CELLS = ('orig/on', 'orig/off', 'corr/on', 'corr/off')


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
    return sum(1 for i in b if ok[i] == '0') / len(b)


def q_r(rec, arm):
    ok = rec['arms'][arm]['ok']
    by = {}
    ret = []
    for i, (k, _, t) in enumerate(rec['items']):
        if k == 'cur':
            by.setdefault(t, []).append(ok[i] == '1')
        elif k == 'ret':
            ret.append(ok[i] == '1')
    q = sum(sum(v) / len(v) for v in by.values()) / len(by)
    return q, sum(ret) / len(ret)


def shuffles(rec, prefix):
    return sorted(a for a in rec['arms'] if a.startswith(prefix + ':shuffled:'))


def delta(rec, prefix, b):
    sh = shuffles(rec, prefix)
    return sum(damage(rec, a, b) for a in sh) / len(sh) - damage(rec, f'{prefix}:matched', b)


def item_delta(rec, prefix, b):
    sh = shuffles(rec, prefix)
    m = rec['arms'][f'{prefix}:matched']['ok']
    oks = [rec['arms'][a]['ok'] for a in sh]
    return [sum(o[i] == '0' for o in oks) / len(oks) - (m[i] == '0') for i in b]


def tune(dev: dict, grid: list) -> dict:
    out = {}
    for cell in CELLS:
        def mean_of(arm):
            ds, qs, rs = [], [], []
            for rec in dev.values():
                b = base(rec)
                ds.append(damage(rec, arm, b))
                q, r = q_r(rec, arm)
                qs.append(q)
                rs.append(r)
            return sum(ds) / len(ds), sum(qs) / len(qs), sum(rs) / len(rs)
        _, qm, rm = mean_of(f'{cell}:matched')
        pts = []
        for x in grid:
            arm = f'uniform:{x:.2f}'
            d, q, r = mean_of(f'{cell}:{arm}')
            pts.append((arm, d, q, r, max(qm - 0.02 - q, rm - 0.02 - r)))
        ok = [p for p in pts if p[4] <= 1e-12]
        if ok:
            best = min(ok, key=lambda p: (p[1], -p[2], float(p[0].split(':')[1])))
        else:
            best = min(pts, key=lambda p: (p[4], -p[2], float(p[0].split(':')[1])))
        out[cell] = best[0]
    return out


def boot(per: dict, reps: int, seed: int):
    rng = random.Random(seed)
    names = sorted(per)
    stats = []
    for _ in range(reps):
        tot = 0.0
        for _ in names:
            v = per[rng.choice(names)]
            tot += sum(v[rng.randrange(len(v))] for _ in range(len(v))) / len(v)
        stats.append(tot / len(names))
    stats.sort()
    return stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]


def main(test_run: str, dev_run: str, out_path: str):
    test, dev = records(Path(test_run)), records(Path(dev_run))
    an = json.loads((Path(test_run) / 'analysis.json').read_text(encoding='utf-8'))
    cfg = json.loads((Path(test_run) / 'config.json').read_text(encoding='utf-8'))
    res, diffs = {}, []
    for n, rec in test.items():
        b = base(rec)
        d = {p: delta(rec, p, b) for p in CELLS + ('bridge', 'strict')}
        res[n] = dict(known=len(b), delta=d,
                      orig_on_minus_corr_off=d['orig/on'] - d['corr/off'],
                      interaction=d['orig/on'] - d['corr/on'] - d['orig/off'] + d['corr/off'],
                      damage={a: damage(rec, a, b) for a in rec['arms']})
        for p, v in d.items():
            diffs.append(abs(v - an['deltas'][p]['per_stream'][n]))
    means = {p: sum(res[n]['delta'][p] for n in res) / len(res) for p in CELLS + ('bridge', 'strict')}
    for p, v in means.items():
        diffs.append(abs(v - an['deltas'][p]['mean']))
    inter = sum(r['interaction'] for r in res.values()) / len(res)
    diffs.append(abs(inter - an['decomposition']['interaction']['mean']))
    sel = tune(dev, cfg['arms']['uniform'])
    sel_ok = all(sel[c] == cfg['selected_uniform'][c]['arm'] for c in CELLS)
    reps = 2000
    head = {}
    for key, fn in (('delta_corr_off', lambda rec, b: item_delta(rec, 'corr/off', b)),
                    ('orig_on_minus_corr_off', lambda rec, b: [x - y for x, y in
                                                               zip(item_delta(rec, 'orig/on', b),
                                                                   item_delta(rec, 'corr/off', b))])):
        per = {n: fn(rec, base(rec)) for n, rec in test.items()}
        ci = boot(per, reps, 0)
        ref = an['headline'][key]['ci']
        head[key] = dict(recount_ci=ci, analysis_ci=ref, max_ci_diff=max(abs(ci[0] - ref[0]), abs(ci[1] - ref[1])))
    out = dict(streams=sorted(res), per_stream=res, means=means, interaction=inter, max_abs_diff=max(diffs),
               tuning_selected=sel, tuning_matches_config=sel_ok, bootstrap=head, bootstrap_reps=reps)
    Path(out_path).write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(max_abs_diff=out['max_abs_diff'], tuning_matches_config=sel_ok,
                          bootstrap={k: v['max_ci_diff'] for k, v in head.items()}), indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
