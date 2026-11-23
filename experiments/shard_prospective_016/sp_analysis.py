"""SP-16 analysis (PLAN.md section 4) from the stream records of one run.

Damage: a protected question answered correctly with routing off ('none') and wrongly by native SHARD. A damaged
question is a hit at budget k when one of the k highest-ranked slots, replayed alone with its native threshold, makes
the answer wrong (a sufficient slot). Hit rates pool numerators and denominators over streams. Random inspection of k
of N slots hits a damaged question with m sufficient slots with probability 1 - C(N-m,k)/C(N,k), zero when m = 0. The
frequency baseline ranks slots by how often they won a probe's prefill and passed tau. Intervals resample streams, then
protected questions within each drawn stream (10,000 draws, seed 0). Event gate: at least 50 damaged questions in at
least 4 streams; otherwise only counts are interpreted. Decision: risk ranking at least 10 points above random at the
primary budget, with the interval's lower bound above zero; extra value over the frequency baseline only if the paired
difference's lower bound is above zero.

Arms (all read the same table): damage, coverage of the edits' original prompts and paraphrases, and, against native,
repaired questions, new damage and coverage lost or gained. Matched thresholds against their within-block shuffles
(and against shuffles that keep disabled slots in place) are summarized as in the earlier studies.

Every rate is computed with the primary matcher and again with the author's matcher (sensitivity).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

import sp_core as C

DRAWS, SEED = 10_000, 0
MATCHERS = {'main': C.match_main, 'author': C.match_author}


def load_run(run: Path) -> dict:
    recs = {}
    for p in sorted(Path(run).glob('stream_*.json.gz')):
        r = json.loads(gzip.open(p).read().decode('utf-8'))
        recs[r['stream']] = r
    return recs


def oks(rec, arm, kind, match):
    return np.array([match(t, it['target']) for t, it in zip(rec['arms'][arm][kind]['text'], rec['items'][kind])],
                    dtype=bool)


def sufficient(rec, damaged, match) -> dict:
    out = {int(q): set() for q in np.flatnonzero(damaged)}
    for r in rec['replays']:
        if r['item'] in out and not match(r['text'], rec['items']['protect'][r['item']]['target']):
            out[r['item']].add(r['slot'])
    return out


def stream_block(rec, cfg, match) -> tuple[dict, dict]:
    n = rec['n_slots']
    base = oks(rec, 'none', 'protect', match)
    nat = oks(rec, 'native', 'protect', match)
    damaged = base & ~nat
    suff = sufficient(rec, damaged, match)
    rank, freq = rec['seal']['ranking'], rec['seal']['freq_ranking']
    items = len(base)
    per_item = {}
    budgets = {}
    for k in cfg['budgets']:
        top, ftop = set(rank[:k]), set(freq[:k])
        hit = np.zeros(items)
        fhit = np.zeros(items)
        rnd = np.zeros(items)
        for q, s in suff.items():
            hit[q] = bool(s & top)
            fhit[q] = bool(s & ftop)
            rnd[q] = C.random_hit(n, k, len(s))
        per_item[k] = dict(hit=hit, fhit=fhit, rnd=rnd)
        budgets[k] = dict(hits=int(hit.sum()), freq_hits=int(fhit.sum()), random_expected=float(rnd.sum()))
    witnessed = sum(1 for s in suff.values() if s)
    block = dict(protect=items, baseline_correct=int(base.sum()), damaged=int(damaged.sum()), witnessed=witnessed,
                 sufficient_per_damaged=[len(s) for s in suff.values()], budgets=budgets)
    return block, dict(damaged=damaged.astype(float), base=base.astype(float), **{
        f'{key}{k}': v for k, d in per_item.items() for key, v in d.items()})


def pooled(arrays: list[dict], k: int) -> dict:
    s = {key: sum(float(a[key].sum()) for a in arrays) for key in ('damaged', f'hit{k}', f'fhit{k}', f'rnd{k}')}
    d = s['damaged']
    return dict(damaged=int(d), hits=int(s[f'hit{k}']), freq_hits=int(s[f'fhit{k}']), random_expected=s[f'rnd{k}'],
                rate_risk=s[f'hit{k}'] / d if d else None, rate_freq=s[f'fhit{k}'] / d if d else None,
                rate_random=s[f'rnd{k}'] / d if d else None,
                diff_random=(s[f'hit{k}'] - s[f'rnd{k}']) / d if d else None,
                diff_freq=(s[f'hit{k}'] - s[f'fhit{k}']) / d if d else None)


def bootstrap(arrays: list[dict], k: int) -> dict:
    rng = np.random.default_rng(SEED)
    S = len(arrays)
    keys = ('damaged', f'hit{k}', f'fhit{k}', f'rnd{k}')
    stats = {'rate_risk': [], 'diff_random': [], 'diff_freq': []}
    for _ in range(DRAWS):
        acc = dict.fromkeys(keys, 0.0)
        for s in rng.integers(0, S, S):
            a = arrays[s]
            idx = rng.integers(0, len(a['damaged']), len(a['damaged']))
            for key in keys:
                acc[key] += float(a[key][idx].sum())
        d = acc['damaged']
        if d:
            stats['rate_risk'].append(acc[f'hit{k}'] / d)
            stats['diff_random'].append((acc[f'hit{k}'] - acc[f'rnd{k}']) / d)
            stats['diff_freq'].append((acc[f'hit{k}'] - acc[f'fhit{k}']) / d)
    return {key: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
            for key, v in stats.items()} | {'draws_with_damage': len(stats['rate_risk'])}


def arm_block(recs: dict, arm: str, match) -> dict:
    agg = dict(base=0, damaged=0, repaired=0, new_damage=0, edit_ok=0, edit_n=0, para_ok=0, para_n=0,
               edit_lost=0, edit_gained=0, para_lost=0, para_gained=0)
    per_stream = {}
    for s, rec in recs.items():
        base = oks(rec, 'none', 'protect', match)
        nat = oks(rec, 'native', 'protect', match)
        a = oks(rec, arm, 'protect', match)
        row = dict(base=int(base.sum()), damaged=int((base & ~a).sum()),
                   repaired=int((base & ~nat & a).sum()), new_damage=int((base & nat & ~a).sum()))
        for kind in ('edit', 'para'):
            ka, kn = oks(rec, arm, kind, match), oks(rec, 'native', kind, match)
            row.update({f'{kind}_ok': int(ka.sum()), f'{kind}_n': len(ka), f'{kind}_lost': int((kn & ~ka).sum()),
                        f'{kind}_gained': int((~kn & ka).sum())})
        row['damage'] = row['damaged'] / row['base'] if row['base'] else None
        per_stream[s] = row
        for key in agg:
            agg[key] += row[key]
    agg['damage'] = agg['damaged'] / agg['base'] if agg['base'] else None
    rates = [r['damage'] for r in per_stream.values() if r['damage'] is not None]
    agg['damage_equal_weight'] = float(np.mean(rates)) if rates else None
    agg['edit_coverage'] = agg['edit_ok'] / agg['edit_n'] if agg['edit_n'] else None
    agg['para_coverage'] = agg['para_ok'] / agg['para_n'] if agg['para_n'] else None
    return dict(pooled=agg, per_stream=per_stream)


def select_uniform(recs: dict, cfg: dict, match=C.match_main) -> dict:
    """Dev rule (PLAN 4.5): among grid levels with mean damage at most the cap, the highest mean paraphrase coverage;
    with no such level, the lowest damage. Ties: higher coverage, then the smaller threshold. Means weight streams
    equally."""
    grid = []
    arms = next(iter(recs.values()))['arms']
    for x in cfg['uniform_grid']:
        arm = f'uniform:{x:.2f}'
        if arm not in arms:
            if abs(x - cfg['editor']['tau']) > 1e-12:
                raise KeyError(f'{arm} was not evaluated on the dev streams')
            arm = 'native'
        dmg, cov = [], []
        for rec in recs.values():
            base = oks(rec, 'none', 'protect', match)
            a = oks(rec, arm, 'protect', match)
            dmg.append(float((base & ~a).sum() / base.sum()) if base.sum() else 0.0)
            cov.append(float(oks(rec, arm, 'para', match).mean()))
        grid.append(dict(level=x, damage=float(np.mean(dmg)), para_coverage=float(np.mean(cov))))
    cap = cfg['selection']['damage_cap']
    ok = [g for g in grid if g['damage'] <= cap + 1e-12]
    if ok:
        best, flag = min(ok, key=lambda g: (-g['para_coverage'], g['level'])), 'met'
    else:
        best, flag = min(grid, key=lambda g: (g['damage'], -g['para_coverage'], g['level'])), 'no level met the cap'
    return dict(level=best['level'], flag=flag, grid=grid)


def analyze(run: Path, dev: Path | None = None) -> dict:
    run = Path(run)
    cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    recs = load_run(run)
    k0 = cfg['primary_budget']
    out = dict(run=run.name, partition=cfg['partition'], streams=sorted(recs), config_sha256=next(iter(recs.values()))[
        'config_sha256'] if recs else None)
    for name, match in MATCHERS.items():
        blocks, arrays = {}, []
        for s in sorted(recs):
            b, arr = stream_block(recs[s], cfg, match)
            blocks[s] = b
            arrays.append(arr)
        res = dict(per_stream=blocks)
        dam = sum(b['damaged'] for b in blocks.values())
        with_dmg = sum(1 for b in blocks.values() if b['damaged'])
        res['events'] = dict(damaged=dam, streams_with_damage=with_dmg,
                             sufficient=dam >= cfg['gates']['min_damaged'] and with_dmg >= cfg['gates']['min_streams'])
        res['budgets'] = {}
        for k in cfg['budgets']:
            p = pooled(arrays, k)
            rates = [b['budgets'][k]['hits'] / b['damaged'] for b in blocks.values() if b['damaged']]
            p['rate_risk_equal_weight'] = float(np.mean(rates)) if rates else None
            p['ci'] = bootstrap(arrays, k) if dam else None
            p['leave_one_out_diff_random'] = {
                s: pooled([a for t, a in zip(sorted(recs), arrays) if t != s], k)['diff_random'] for s in sorted(recs)}
            res['budgets'][k] = p
        prim = res['budgets'][k0]
        g = cfg['gates']
        res['decision'] = dict(
            events_sufficient=res['events']['sufficient'],
            screening_passed=bool(res['events']['sufficient'] and prim['diff_random'] is not None
                                  and prim['diff_random'] >= g['min_gain'] and prim['ci']['diff_random'][0] > 0),
            beats_frequency=bool(res['events']['sufficient'] and prim['ci'] and prim['ci']['diff_freq']
                                 and prim['ci']['diff_freq'][0] > 0))
        res['arms'] = {arm: arm_block(recs, arm, match) for arm in cfg['arms'] if arm != 'none'}
        shuf = [a for a in cfg['arms'] if a.startswith('shuffled:')]
        act = [a for a in cfg['arms'] if a.startswith('active_shuffled:')]
        if 'matched' in cfg['arms'] and shuf:
            m = res['arms']['matched']['pooled']['damage']
            res['matched_vs_shuffled'] = dict(
                shuffled_mean=float(np.mean([res['arms'][a]['pooled']['damage'] for a in shuf])),
                active_mean=float(np.mean([res['arms'][a]['pooled']['damage'] for a in act])) if act else None,
                matched=m)
        out[name] = res
    out['checks'] = {s: r['checks'] for s, r in recs.items()}
    out['seconds'] = {s: r['seconds'] for s, r in recs.items()}
    if dev:
        out['dev_selection'] = select_uniform(load_run(dev), json.loads((Path(dev) / 'config.json').read_text()))
    return out


def write_analysis(run: Path, dev: Path | None = None):
    res = analyze(run, dev)
    Path(run, 'analysis.json').write_text(json.dumps(res, indent=1, default=float) + '\n', encoding='utf-8')
    for name in MATCHERS:
        r = res[name]
        p = r['budgets'][json.loads(Path(run, 'config.json').read_text())['primary_budget']]
        print(name, r['events'], {k: p[k] for k in ('hits', 'freq_hits', 'random_expected', 'rate_risk',
                                                       'rate_random', 'diff_random', 'diff_freq')}, r['decision'])
