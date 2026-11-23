"""Review Exp-5a (CPU only, existing records): do the strict exact-match damage counts overstate errors, and does a
lenient matcher change the FP-12 / XR-13 contrasts?

Lenient rule (automatic, not a human audit): a base-known fact that an arm answers wrongly under exact match still
counts as right when the gold answer's words appear, in order, as whole words inside the arm's answer (e.g. 'mit'
inside 'the mit license'), or when XR-13's licence canonicaliser maps both strings to the same licence.
Recomputes, per stream, D under both rules for every arm, then the stream means of Δ (shuffles - matched) and of
D(uniform:0.90) - D(matched). Every lenient flip under matched is listed for inspection.

  python -I checks/exp5a_relaxed_match.py OUT_JSON
"""
import gzip
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXP = HERE.parent
sys.path.insert(0, str(EXP / 'external_key_replication_013'))
import xr_events  # noqa: E402

RUNS = {'fp12': (EXP / 'field_prefix_ablation_012' / 'runs' / 'test_001', ('orig/on', 'corr/off')),
        'xr13': (EXP / 'external_key_replication_013' / 'data' / 'host' / 'runs' / 'test_001', ('xr',))}


def words(s):
    return re.findall(r'[a-z0-9]+', (s or '').lower())


def lenient_right(gold, ans, licences):
    g, a = words(gold), words(ans)
    if g and any(a[i:i + len(g)] == g for i in range(len(a) - len(g) + 1)):
        return True
    return licences and xr_events.canon_licence(ans).lower() == xr_events.canon_licence(gold).lower()


def main(out_json):
    res = {}
    for src, (run, cells) in RUNS.items():
        recs = {}
        for p in sorted(run.glob('stream_*.json.gz')):
            r = json.loads(gzip.open(p).read().decode('utf-8'))
            recs[r['stream']] = r
        per = {}
        flips = []
        for n, r in recs.items():
            none = r['arms']['none']['ok']
            base = [i for i, it in enumerate(r['items']) if it[0] == 'loc' and none[i] == '1']
            if not base:
                continue
            gold = r['loc_answers']
            row = {}
            for cell in cells:
                def D(arm, lenient):
                    a = r['arms'][f'{cell}:{arm}']
                    bad = 0
                    for i in base:
                        if a['ok'][i] == '0':
                            if lenient and lenient_right(gold[str(i)], a['wrong_loc'].get(str(i)), src == 'xr13'):
                                continue
                            bad += 1
                    return bad / len(base)
                for lenient in (False, True):
                    k = 'lenient' if lenient else 'strict'
                    sh = [D(f'shuffled:{p}', lenient) for p in range(8)]
                    m = D('matched', lenient)
                    row[f'{cell}|{k}'] = dict(matched=m, shuffled=sum(sh) / 8, delta=sum(sh) / 8 - m,
                                              uniform_090=D('uniform:0.90', lenient),
                                              delta_u090=D('uniform:0.90', lenient) - m)
                a = r['arms'][f'{cell}:matched']
                for i in base:
                    if a['ok'][i] == '0' and lenient_right(gold[str(i)], a['wrong_loc'].get(str(i)), src == 'xr13'):
                        flips.append(dict(stream=n, cell=cell, gold=gold[str(i)], answer=a['wrong_loc'].get(str(i))))
            per[n] = row
        keys = sorted({k for v in per.values() for k in v})
        means = {k: {m: sum(v[k][m] for v in per.values()) / len(per) for m in ('matched', 'shuffled', 'delta',
                                                                                 'uniform_090', 'delta_u090')}
                 for k in keys}
        res[src] = dict(streams=sorted(per), per_stream=per, means=means, matched_flips=flips)
        for k, v in means.items():
            print(src, k, {m: round(100 * x, 2) for m, x in v.items()})
        print(src, 'lenient flips under matched:', len(flips))
    Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
