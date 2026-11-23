"""Review Exp-0B (CPU only, existing records): threshold strata and culprit attribution in FP-12 and XR-13.

For every test stream (FP-12 D4-D7, the four cells; XR-13's eight registries) and its matched thresholds:
  strata       entries at the floor (0.90), in the middle (0.90, 1.00] and above 1 (never fire under matched)
  culprits     per stored intervention arm (uniform:0.90, matched, shuffled:0): every damaged item whose answer one
               entry breaks on its own ('alone' run), with that entry's matched stratum; per item: banned only, middle
               only, floor only, or mixed; plus the entries that break the most items
Also KT-10's probe-count arms (probes:250 ... matched) on its test streams, from its analysis.json.

  python -I checks/exp0b_existing_records.py OUT_JSON
"""
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXP = HERE.parent
FP_RUN = EXP / 'field_prefix_ablation_012' / 'runs' / 'test_001'
XR_RUN = EXP / 'external_key_replication_013' / 'data' / 'host' / 'runs' / 'test_001'
KT_ANA = EXP / 'key_thresholds_010' / 'runs' / 'run_002' / 'analysis.json'
FLOOR = 0.90


def load(run):
    out = {}
    for p in sorted(Path(run).glob('stream_*.json.gz')):
        r = json.loads(gzip.open(p).read().decode('utf-8'))
        out[r['stream']] = r
    return out


def stratum(t):
    if t > 1.0:
        return 'banned'
    if t > FLOOR + 1e-6:
        return 'middle'
    return 'floor'


def strata(thr):
    c = Counter(stratum(t) for t in thr)
    n = len(thr)
    return dict(n=n, floor=c['floor'], middle=c['middle'], banned=c['banned'],
                banned_share=c['banned'] / n if n else None, middle_share=c['middle'] / n if n else None)


def culprits(rec, arm, thr):
    ivs = rec['interventions'].get(arm)
    if ivs is None:
        return None
    answers = rec['loc_answers']
    ent = rec['tables'].get('entity') or []
    per, by_stratum, top = Counter(), Counter(), Counter()
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
        ks = {stratum(thr[e]) for e in cul}
        per[ks.pop() + ' only' if len(ks) == 1 else 'mixed'] += 1
        for e in cul:
            by_stratum[stratum(thr[e])] += 1
            top[e] += 1
    return dict(damaged=len(ivs), per_item=dict(per), culprit_entries_by_stratum=dict(by_stratum),
                top=[dict(entry=e, items=n, threshold=thr[e], stratum=stratum(thr[e]),
                          entity=ent[e] if e < len(ent) else None) for e, n in top.most_common(5)])


def main(out_json):
    res = dict(fp12={}, xr13={}, kt10_probes=None)
    for name, rec in load(FP_RUN).items():
        row = {}
        for cell in ('orig/on', 'orig/off', 'corr/on', 'corr/off'):
            tab = cell.split('/')[0]
            thr = rec['thresholds'][tab]
            row[cell] = dict(strata=strata(thr),
                             culprits={a: culprits(rec, f'{cell}:{a}', thr)
                                       for a in ('uniform:0.90', 'matched', 'shuffled:0')})
        row['strict'] = dict(strata=strata(rec['thresholds']['strict']))
        res['fp12'][name] = row
    for name, rec in load(XR_RUN).items():
        thr = rec['thresholds']['matched']
        res['xr13'][name] = dict(known=rec['screening']['known'], strata=strata(thr),
                                 culprits={a: culprits(rec, f'xr:{a}', thr)
                                           for a in ('uniform:0.90', 'matched', 'shuffled:0')})
    if KT_ANA.exists():
        kt = json.loads(KT_ANA.read_text(encoding='utf-8'))
        arms = kt.get('test', {}).get('arms') or {}
        res['kt10_probes'] = {k: arms[k] for k in ('probes:250', 'probes:500', 'probes:1000', 'matched', 'ctx_only')
                              if k in arms}
    Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    tot = Counter()
    for name, row in res['fp12'].items():
        for cell in ('orig/on', 'corr/off'):
            for a in ('uniform:0.90', 'shuffled:0'):
                c = row[cell]['culprits'][a]
                for k, v in c['per_item'].items():
                    tot[(cell, a, k)] += v
    for k in sorted(tot):
        print('FP', k, tot[k])
    xt = Counter()
    for name, row in res['xr13'].items():
        for a in ('uniform:0.90', 'shuffled:0'):
            c = row['culprits'][a]
            if c:
                for k, v in c['per_item'].items():
                    xt[(a, k)] += v
    for k in sorted(xt):
        print('XR', k, xt[k])


if __name__ == '__main__':
    main(sys.argv[1])
