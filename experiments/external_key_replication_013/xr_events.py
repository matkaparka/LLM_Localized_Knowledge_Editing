"""XR-13 events: from collected version histories to monthly batches of edits, stable old facts and spare entities.

For one family and a window of 8 calendar months:
- state of an entity at the end of a month: its highest version (not a pre-release, not yanked or unlisted) published by
  then, with that version's runtime requirement and licence where the registry records them;
- an edit in month m: the state's version differs from the end of month m-1 ('latest version'), or its runtime
  requirement does ('runtime requirement'); an entity contributes at most one edit per attribute and month (the state
  at the month's end);
- per month at most `per_batch` edits: every runtime-requirement change first (up to `spec_quota`), then version
  changes chosen by a fixed hash; entities are drawn from the family's `max_rank` most popular;
- pools: with --pool test or dev, only the entities whose fixed hash falls in that half (so the dev and test streams of
  one family never share an entity); --pool all keeps every entity;
- stable old facts: the licence (canonical spelling, see canon_licence) of entities with no edit in the window whose
  licence is the same at the start and at the end of the window; at most enough of them that `min_spares`
  entities stay free for unlabelled questions;
- spares: the remaining entities, for the public probes and the routing-centre corpus.

  python xr_events.py ENTITIES.jsonl.gz OUT.json --family F --start 2026-03 [--months 8] [--per-batch 125]
                      [--pool all|test|dev] [--min-spares 2000]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

PRE = re.compile(r'(alpha|beta|rc|preview|nightly|insider|canary|snapshot|dev|pre)', re.I)


def vkey(v: str):
    """A total order for version strings across ecosystems: PEP 440 when it parses, else natural sort."""
    try:
        from packaging.version import Version
        return (0, Version(v.lstrip('vV')), ())
    except Exception:
        parts = re.findall(r'\d+|[A-Za-z]+', v)
        return (1, None, tuple((0, int(p)) if p.isdigit() else (1, p.lower()) for p in parts))


def sort_key(v: str):
    k = vkey(v)
    return (k[0], k[1] if k[1] is not None else 0, k[2]) if k[0] == 0 else (k[0], 0, k[2])


def month_end(ym: str) -> str:
    y, m = map(int, ym.split('-'))
    y2, m2 = (y + 1, 1) if m == 12 else (y, m + 1)
    return f'{y2:04d}-{m2:02d}-01T00:00:00Z'


def months_from(start: str, n: int) -> list[str]:
    y, m = map(int, start.split('-'))
    out = []
    for _ in range(n):
        out.append(f'{y:04d}-{m:02d}')
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def prev_month(ym: str) -> str:
    y, m = map(int, ym.split('-'))
    return f'{y - 1:04d}-12' if m == 1 else f'{y:04d}-{m - 1:02d}'


def state(versions: list[dict], until: str):
    ok = [v for v in versions if v.get('t') and v['t'] < until and not v.get('pre') and not v.get('yanked')
          and not PRE.search(v['v'])]
    if not ok:
        return None
    best = max(ok, key=lambda v: sort_key(v['v']))
    return dict(v=best['v'], t=best['t'], spec=best.get('spec'), license=best.get('license'))


def h(*parts) -> str:
    return hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()


def canon_licence(x: str | None) -> str | None:
    """Fixed answer spelling: strip CRAN's '+ file LICENSE' / '| file LICENSE' and a trailing ' License'."""
    if not x:
        return None
    y = re.sub(r'\s*[+|]\s*file LICEN[CS]E\s*$', '', x.strip(), flags=re.I)
    y = re.sub(r'\s+License$', '', y)
    return y.strip() or None


def in_pool(family: str, entity: str, pool: str) -> bool:
    if pool == 'all':
        return True
    half = int(h('xr13-pool', family, entity)[:8], 16) % 2
    return half == (0 if pool == 'test' else 1)


def build(entities: list[dict], family: str, start: str, months: int, per_batch: int, spec_quota: int,
          max_rank: int, pool: str = 'all', min_spares: int = 2000) -> dict:
    window = months_from(start, months)
    ents = [e for e in entities if not e.get('error') and e.get('versions') and (e.get('rank') or 0) < max_rank
            and in_pool(family, e['entity'], pool)]
    batches, edited = [], set()
    for step, ym in enumerate(window):
        cand_v, cand_s = [], []
        for e in ents:
            a, b = state(e['versions'], month_end(prev_month(ym))), state(e['versions'], month_end(ym))
            if b is None:
                continue
            if a is None or a['v'] != b['v']:
                cand_v.append(dict(entity=e['entity'], attribute='latest version', value=b['v'], old=a and a['v'],
                                   time=b['t'], rank=e.get('rank')))
            if a is not None and b.get('spec') and a.get('spec') and b['spec'] != a['spec']:
                cand_s.append(dict(entity=e['entity'], attribute='runtime requirement', value=b['spec'],
                                   old=a['spec'], time=b['t'], rank=e.get('rank')))
        cand_s.sort(key=lambda x: h(family, ym, 'spec', x['entity']))
        cand_v.sort(key=lambda x: h(family, ym, 'version', x['entity']))
        pick = cand_s[:spec_quota]
        pick += cand_v[:per_batch - len(pick)]
        for x in pick:
            x['step'] = step
            x['month'] = ym
            x['uid'] = f'{family}:{ym}:{x["attribute"][:4]}:{x["entity"]}'
            edited.add(x['entity'])
        batches.append(dict(month=ym, n=len(pick), candidates=dict(version=len(cand_v), spec=len(cand_s)), edits=pick))
    first, last = month_end(prev_month(window[0])), month_end(window[-1])
    candidates, spares = [], []
    for e in ents:
        if e['entity'] in edited:
            continue
        a, b = state(e['versions'], first), state(e['versions'], last)
        lic_a = canon_licence((a or {}).get('license'))
        lic_b = canon_licence((b or {}).get('license'))
        if lic_a is None and lic_b is None:
            lic = canon_licence(e.get('license_latest'))
        else:
            lic = lic_b if lic_a == lic_b else None
        if lic and b is not None:
            candidates.append(dict(entity=e['entity'], attribute='licence', value=lic, rank=e.get('rank')))
        else:
            spares.append(dict(entity=e['entity'], rank=e.get('rank')))
    candidates.sort(key=lambda f: (f['rank'] or 0, f['entity']))
    room = max(0, len(candidates) + len(spares) - min_spares)
    stable = candidates[:room]
    spares += [dict(entity=f['entity'], rank=f['rank']) for f in candidates[room:]]
    return dict(family=family, pool=pool, window=window, per_batch=per_batch, spec_quota=spec_quota,
                max_rank=max_rank, n_entities=len(ents), batches=batches, stable=stable, spares=spares,
                edits=sum(b['n'] for b in batches))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('entities')
    ap.add_argument('out')
    ap.add_argument('--family', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--months', type=int, default=8)
    ap.add_argument('--per-batch', type=int, default=125)
    ap.add_argument('--spec-quota', type=int, default=25)
    ap.add_argument('--max-rank', type=int, default=12000)
    ap.add_argument('--pool', default='all', choices=('all', 'test', 'dev'))
    ap.add_argument('--min-spares', type=int, default=2000)
    a = ap.parse_args(argv)
    with gzip.open(a.entities, 'rt', encoding='utf-8') as g:
        ents = [json.loads(line) for line in g]
    seen, uniq = set(), []
    for e in ents:                     # a resumed collection can hold an entity twice; keep the first record
        if e.get('entity') not in seen:
            seen.add(e.get('entity'))
            uniq.append(e)
    res = build(uniq, a.family, a.start, a.months, a.per_batch, a.spec_quota, a.max_rank, a.pool, a.min_spares)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=0), encoding='utf-8', newline='\n')
    print(json.dumps(dict(family=a.family, entities=res['n_entities'], edits=res['edits'],
                          per_month=[(b['month'], b['n'], b['candidates']) for b in res['batches']],
                          stable=len(res['stable']), spares=len(res['spares']))))


if __name__ == '__main__':
    main()
