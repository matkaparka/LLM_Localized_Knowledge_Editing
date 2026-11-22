"""Independent recount of KT-10's primary numbers from the per-stream records (does not import kt_analysis).

For every stream: damage at the last step relative to 'none', mean rephrase accuracy and last-step retention for
every arm, then H1 (mean over shuffled arms minus matched) and H2 (tuned uniform minus matched) per stream and on
average, compared with RUN_DIR/analysis.json. Writes checks/recount_<run>.json.

  python checks/recount.py RUN_DIR [EXTRA_RUN_DIR ...]     (streams rerun elsewhere, e.g. run_003_D7)
"""
import gzip
import json
import sys
from pathlib import Path


def load(*runs: Path):
    out = {}
    for p in sorted(q for run in runs for q in run.glob('stream_*.json.gz')):
        with gzip.open(p, 'rb') as g:
            out[p.name[7:-8]] = json.loads(g.read().decode('utf-8'))
    return out


def per_arm(rec):
    T = rec['steps'] - 1
    items = rec['items'][T]
    none_ok = rec['arms']['none'][T]['ok']
    base = [i for i in range(len(items)) if items[i][0] == 'loc' and none_ok[i] == '1']
    res = {}
    for arm, st in rec['arms'].items():
        ok = st[T]['ok']
        cur = []
        for t in range(rec['steps']):
            its, o = rec['items'][t], st[t]['ok']
            n = [o[i] == '1' for i in range(len(its)) if its[i][0] == 'cur']
            if n:
                cur.append(sum(n) / len(n))
        ret = [ok[i] == '1' for i in range(len(items)) if items[i][0] == 'ret']
        res[arm] = dict(damage=sum(ok[i] == '0' for i in base) / len(base), coverage=sum(cur) / len(cur),
                        retention=sum(ret) / len(ret) if ret else None, n=len(base))
    return res


def main(run: Path, *extra: Path):
    recs = load(run, *extra)
    ana = json.loads((run / 'analysis.json').read_text(encoding='utf-8'))
    tuned = ana['tuning']['tuned']
    out, diffs = {}, []
    for role in ('dev', 'test'):
        names = ana['streams'][role]
        rows = {}
        for s in names:
            m = per_arm(recs[s])
            sh = [a for a in m if a.startswith('shuffled:')]
            rows[s] = dict(matched=m['matched']['damage'], shuffled=sum(m[a]['damage'] for a in sh) / len(sh),
                           tuned=m[tuned]['damage'], h1=sum(m[a]['damage'] for a in sh) / len(sh) - m['matched']['damage'],
                           h2=m[tuned]['damage'] - m['matched']['damage'],
                           cov_h1=m['matched']['coverage'] - sum(m[a]['coverage'] for a in sh) / len(sh),
                           arms={a: m[a] for a in sorted(m)})
        h1 = sum(r['h1'] for r in rows.values()) / len(rows)
        h2 = sum(r['h2'] for r in rows.values()) / len(rows)
        a1 = ana[role]['h1_matched_vs_shuffled']
        a2 = ana[role]['h2_matched_vs_tuned_uniform']
        diffs += [abs(h1 - a1['mean_delta']), abs(h2 - a2['mean_delta'])]
        for r in a1['rows']:
            diffs.append(abs(rows[r['stream']]['h1'] - r['delta']))
            diffs.append(abs(rows[r['stream']]['cov_h1'] - r['coverage_diff']))
        for a, v in ana[role]['arms'].items():
            mine = sum(rows[s]['arms'][a]['damage'] for s in names) / len(names)
            diffs.append(abs(mine - v['damage']))
        out[role] = dict(h1_mean=h1, h2_mean=h2, analysis_h1=a1['mean_delta'], analysis_h2=a2['mean_delta'],
                         per_stream={s: {k: v for k, v in r.items() if k != 'arms'} for s, r in rows.items()},
                         arms={a: {k: sum(rows[s]['arms'][a][k] for s in names) / len(names)
                                   for k in ('damage', 'coverage')} for a in rows[names[0]]['arms']})
    out['tuned'] = tuned
    out['max_abs_diff_vs_analysis'] = max(diffs)
    path = Path(__file__).resolve().parent / f'recount_{run.name}.json'
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'max abs diff vs analysis: {out["max_abs_diff_vs_analysis"]:.3g}; written {path.name}')


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:]))
