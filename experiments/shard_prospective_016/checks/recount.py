"""Independent recount of SP-16 results from the stream records: standard library only, no code shared with
sp_core.py or sp_analysis.py (both matchers are re-implemented here).

  python checks/recount.py RUN OUT_JSON

Recomputes, per matcher and stream: baseline-correct and damaged protected questions, the sufficient slots from the
single-slot replays, hits of the risk and frequency rankings at every budget and the random expectation
1 - C(N-m,k)/C(N,k); the pooled rates and differences; and, per arm, pooled damage and coverage. Compares them with
RUN/analysis.json and reports the largest absolute difference.
"""
import gzip
import json
import math
import re
import sys
import unicodedata
from pathlib import Path


def norm_main(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())


def main_match(gen, ref):
    g, r = norm_main(gen), norm_main(ref)
    return bool(r) and g.startswith(r) and (len(g) == len(r) or not g[len(r)].isalnum())


def norm_author(s):
    t = re.sub(r'\s+', ' ', s.strip().lower())
    while t and t[-1] in '.,!?;:\'"':
        t = t[:-1]
    return t


def author_match(gen, ref):
    t = norm_author(ref)
    return bool(t) and t in norm_author(gen)


MATCH = {'main': main_match, 'author': author_match}


def rnd(N, k, m):
    if m <= 0:
        return 0.0
    if N - m < k:
        return 1.0
    return 1.0 - math.comb(N - m, k) / math.comb(N, k)


def recount(run):
    run = Path(run)
    cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    recs = [json.loads(gzip.open(p).read().decode('utf-8')) for p in sorted(run.glob('stream_*.json.gz'))]
    out = {}
    for name, match in MATCH.items():
        res = {'budgets': {}, 'arms': {}}
        tot = {k: dict(d=0, h=0, f=0, r=0.0) for k in cfg['budgets']}
        for rec in recs:
            prot = rec['items']['protect']
            none = [match(t, it['target']) for t, it in zip(rec['arms']['none']['protect']['text'], prot)]
            nat = [match(t, it['target']) for t, it in zip(rec['arms']['native']['protect']['text'], prot)]
            dam = [i for i in range(len(prot)) if none[i] and not nat[i]]
            suff = {i: set() for i in dam}
            for r in rec['replays']:
                if r['item'] in suff and not match(r['text'], prot[r['item']]['target']):
                    suff[r['item']].add(r['slot'])
            for k in cfg['budgets']:
                top, ftop = set(rec['seal']['ranking'][:k]), set(rec['seal']['freq_ranking'][:k])
                t = tot[k]
                t['d'] += len(dam)
                t['h'] += sum(1 for s in suff.values() if s & top)
                t['f'] += sum(1 for s in suff.values() if s & ftop)
                t['r'] += sum(rnd(rec['n_slots'], k, len(s)) for s in suff.values())
        for k, t in tot.items():
            d = t['d']
            res['budgets'][str(k)] = dict(damaged=d, hits=t['h'], freq_hits=t['f'], random_expected=t['r'],
                                          rate_risk=t['h'] / d if d else None,
                                          diff_random=(t['h'] - t['r']) / d if d else None,
                                          diff_freq=(t['h'] - t['f']) / d if d else None)
        for arm in cfg['arms']:
            if arm == 'none':
                continue
            base = dam = 0
            cov = {'edit': [0, 0], 'para': [0, 0]}
            for rec in recs:
                prot = rec['items']['protect']
                none = [match(t, it['target']) for t, it in zip(rec['arms']['none']['protect']['text'], prot)]
                a = [match(t, it['target']) for t, it in zip(rec['arms'][arm]['protect']['text'], prot)]
                base += sum(none)
                dam += sum(1 for x, y in zip(none, a) if x and not y)
                for kind in cov:
                    ok = [match(t, it['target']) for t, it in zip(rec['arms'][arm][kind]['text'], rec['items'][kind])]
                    cov[kind][0] += sum(ok)
                    cov[kind][1] += len(ok)
            res['arms'][arm] = dict(damage=dam / base if base else None,
                                    edit_coverage=cov['edit'][0] / cov['edit'][1] if cov['edit'][1] else None,
                                    para_coverage=cov['para'][0] / cov['para'][1] if cov['para'][1] else None)
        out[name] = res
    return out


def compare(rc, an):
    diffs = []
    for name in rc:
        for k, b in rc[name]['budgets'].items():
            a = an[name]['budgets'][k]
            for key in b:
                x, y = b[key], a[key]
                if x is None or y is None:
                    diffs.append(0.0 if x is None and y is None else float('inf'))
                else:
                    diffs.append(abs(float(x) - float(y)))
        for arm, b in rc[name]['arms'].items():
            p = an[name]['arms'][arm]['pooled']
            for key in b:
                x, y = b[key], p[key]
                diffs.append(0.0 if x is None and y is None else abs(float(x) - float(y)) if None not in (x, y)
                             else float('inf'))
    return max(diffs) if diffs else 0.0


def main(run, out):
    rc = recount(run)
    an_path = Path(run) / 'analysis.json'
    if an_path.exists():
        rc['max_abs_diff_vs_analysis'] = compare(rc, json.loads(an_path.read_text(encoding='utf-8')))
    Path(out).write_text(json.dumps(rc, indent=1) + '\n', encoding='utf-8')
    print('max abs diff vs analysis:', rc.get('max_abs_diff_vs_analysis'))


if __name__ == '__main__':
    main(*sys.argv[1:])
