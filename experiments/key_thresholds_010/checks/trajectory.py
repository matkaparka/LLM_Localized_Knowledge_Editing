"""Descriptive (not part of the pre-registered analysis): damage relative to 'none' and rephrase accuracy at every
step, per stream, for none-relative damage curves. Arms: matched, the mean of the shuffled arms, every uniform arm,
ctx_only and the probe-count arms. Writes checks/trajectory_<run>.json and prints the test-stream means.

  python checks/trajectory.py RUN_DIR [EXTRA_RUN_DIR ...]
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


def curves(rec):
    res = {}
    for arm, st in rec['arms'].items():
        dmg, cur = [], []
        for t in range(rec['steps']):
            items, ok, none = rec['items'][t], st[t]['ok'], rec['arms']['none'][t]['ok']
            base = [i for i in range(len(items)) if items[i][0] == 'loc' and none[i] == '1']
            dmg.append(sum(ok[i] == '0' for i in base) / len(base) if base else None)
            cur.append(st[t]['acc']['cur'])
        res[arm] = dict(damage=dmg, rephrase=cur)
    sh = [a for a in res if a.startswith('shuffled:')]
    if sh:
        res['shuffled_mean'] = {k: [sum(res[a][k][t] for a in sh) / len(sh) for t in range(rec['steps'])]
                                for k in ('damage', 'rephrase')}
    return res


def main(run: Path, *extra: Path):
    recs = load(run, *extra)
    cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    out = {s: curves(r) for s, r in recs.items()}
    path = Path(__file__).resolve().parent / f'trajectory_{run.name}.json'
    path.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    for role in ('dev', 'test'):
        names = [s for s in cfg[role] if s in out]
        if not names:
            continue
        print(role, names)
        for arm in ('none', 'matched', 'shuffled_mean', 'uniform:0.90', 'ctx_only', 'probes:250'):
            if arm not in out[names[0]]:
                continue
            steps = len(out[names[0]][arm]['damage'])
            m = [sum(out[s][arm]['damage'][t] for s in names) / len(names) for t in range(steps)]
            print(f'  {arm:14s} damage by step: ' + ' '.join(f'{100 * x:5.2f}' for x in m))
    print('written', path.name)


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:]))
