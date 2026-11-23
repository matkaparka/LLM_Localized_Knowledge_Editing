"""Independent recount of RC-15 / BR-14 results from the raw stream records: standard library only, no code shared with
rc_analysis.py or fp_analysis.py.

  python checks/recount.py TEST_RUN [DEV_RUN] OUT_JSON

Recomputes, per cell and stream, the damage D of every arm (base-known locality items: right under none), the stream
means of delta (shuffles), delta_active (active-only shuffles), delta_bans, the ban x continuous factorial and the
practical comparisons, and compares them with TEST_RUN/analysis.json (max absolute difference). With DEV_RUN and a
'tune' block in the config, it re-applies the selection rule to the dev records and checks the config's selections.
"""
import gzip
import json
import sys
from pathlib import Path

TOL = 0.02


def load(run):
    out = {}
    for p in sorted(Path(run).glob('stream_*.json.gz')):
        with gzip.open(p, 'rb') as g:
            r = json.loads(g.read().decode('utf-8'))
        out[r['stream']] = r
    return out


def base_items(r):
    none = r['arms']['none']['ok']
    return [i for i, it in enumerate(r['items']) if it[0] == 'loc' and none[i] == '1']


def D(r, arm):
    idx = base_items(r)
    if not idx:
        return None
    ok = r['arms'][arm]['ok']
    return sum(1 for i in idx if ok[i] == '0') / len(idx)


def Q(r, arm):
    ok = r['arms'][arm]['ok']
    by = {}
    for i, it in enumerate(r['items']):
        if it[0] == 'cur':
            by.setdefault(it[2], []).append(ok[i] == '1')
    vals = [sum(v) / len(v) for v in by.values() if v]
    return sum(vals) / len(vals) if vals else None


def R(r, arm):
    ok = r['arms'][arm]['ok']
    v = [ok[i] == '1' for i, it in enumerate(r['items']) if it[0] == 'ret']
    return sum(v) / len(v) if v else None


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def family(r, cell, kind):
    pre = f'{cell}:{kind}:'
    return [a for a in r['arms'] if a.startswith(pre)]


def stream_mean(recs, fn):
    vals = [fn(r) for r in recs.values()]
    return avg(vals)


def contrasts(recs, cell, tau, practical):
    out = {}
    M = f'{cell}:matched'

    def dmean(arms, r):
        v = [D(r, a) for a in arms]
        return None if not v or v[0] is None else sum(v) / len(v)

    first = next(iter(recs.values()))
    if family(first, cell, 'shuffled'):
        out['delta'] = stream_mean(recs, lambda r: None if D(r, M) is None else dmean(family(r, cell, 'shuffled'), r) - D(r, M))
    if family(first, cell, 'active_shuffled'):
        out['delta_active'] = stream_mean(
            recs, lambda r: None if D(r, M) is None else dmean(family(r, cell, 'active_shuffled'), r) - D(r, M))
    if 'delta' in out and 'delta_active' in out:
        out['delta_bans'] = stream_mean(recs, lambda r: None if D(r, M) is None else
                                        dmean(family(r, cell, 'shuffled'), r) - dmean(family(r, cell, 'active_shuffled'), r))
    U, B, N = f'{cell}:uniform:{tau}', f'{cell}:ban_uniform:{tau}', f'{cell}:noban_matched'
    pairs = dict(total=(U, M), ban_flat=(U, B), ban_cont=(N, M), cont_ban=(B, M), cont_noban=(U, N))
    for k, (a, b) in pairs.items():
        if a in first['arms'] and b in first['arms']:
            out['factorial.' + k] = stream_mean(recs, lambda r, a=a, b=b: None if D(r, a) is None else D(r, a) - D(r, b))
    for comp in practical.get(cell, []):
        C = f'{cell}:{comp}'
        if C in first['arms']:
            out['practical.' + comp] = stream_mean(recs, lambda r, C=C: None if D(r, C) is None else D(r, C) - D(r, M))
    return out


def select(recs, cell, kind, levels):
    M = {k: avg([f(r, f'{cell}:matched') for r in recs.values()]) for k, f in (('Q', Q), ('R', R), ('D', D))}
    pts = []
    for x in levels:
        a = f'{cell}:{kind}:{x:.2f}'
        m = {k: avg([f(r, a) for r in recs.values()]) for k, f in (('Q', Q), ('R', R), ('D', D))}
        short = max(M['Q'] - TOL - m['Q'], M['R'] - TOL - m['R'])
        pts.append((x, m, short))
    ok = [p for p in pts if p[2] <= 1e-12]
    if ok:
        return min(ok, key=lambda p: (p[1]['D'], -p[1]['Q'], p[0]))[0]
    return min(pts, key=lambda p: (p[2], -p[1]['Q'], p[0]))[0]


def main(test_run, dev_run, out_json):
    recs = load(test_run)
    cfg = json.loads((Path(test_run) / 'config.json').read_text(encoding='utf-8'))
    ana = json.loads((Path(test_run) / 'analysis.json').read_text(encoding='utf-8'))
    first = next(iter(recs.values()))
    tau = f"{float(first.get('options', {}).get('tau', 0.9)):.2f}"
    res, diffs = {}, []
    for cell in first['cells']:
        mine = contrasts(recs, cell, tau, cfg.get('practical') or {})
        theirs = ana['cells_result'][cell]
        for k, v in mine.items():
            if k.startswith('factorial.'):
                t = theirs['factorial'][k.split('.', 1)[1]]['mean']
            elif k.startswith('practical.'):
                t = theirs['practical'][k.split('.', 1)[1]]['delta']['mean']
            else:
                t = theirs[k]['mean']
            if v is not None and t is not None:
                diffs.append(abs(v - t))
        res[cell] = mine
    tuning = None
    if dev_run and cfg.get('tune') and cfg.get('selected'):
        dev = load(dev_run)
        t = cfg['tune']
        tuning = all(abs(select(dev, c, k, t['levels']) - cfg['selected'][c][k]['level']) < 1e-12
                     for c in t['cells'] for k in t['kinds'])
    out = dict(recount=res, max_abs_diff=max(diffs) if diffs else 0.0, compared=len(diffs),
               tuning_matches_config=tuning)
    Path(out_json).write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(max_abs_diff=out['max_abs_diff'], compared=out['compared'], tuning=tuning)))


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) == 2:
        main(a[0], None, a[1])
    else:
        main(a[0], a[1], a[2])
