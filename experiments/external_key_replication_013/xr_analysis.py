"""XR-13 analysis: FP-12's measures and rules for a single cell ('xr'), plus the concentration of residual damage
(guide v2 section 4.5) and leave-one-family-out sensitivity. Each test stream is one family, the statistical unit.

  tune     the uniform-threshold rule on the dev records (FP-12 PLAN section 4) -> config_test.json
  analyze  Delta (mechanism gate), Delta_U (practical gate), Q/R differences, per-family values, concentration
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'field_prefix_ablation_012'))
import fp_analysis as fa  # noqa: E402
import fp_run  # noqa: E402

CELL = 'xr'
MIN_KNOWN = 30


def tune(recs: dict, cfg: dict) -> dict:
    grid = [f'uniform:{x:.2f}' for x in cfg['arms']['uniform']]
    M = {k: fa.mean([fa.arm_metrics(r, f'{CELL}:matched')[k] for r in recs.values()]) for k in ('Q', 'R', 'damage')}
    pts = []
    for g in grid:
        m = {k: fa.mean([fa.arm_metrics(r, f'{CELL}:{g}')[k] for r in recs.values()]) for k in ('Q', 'R', 'damage')}
        pts.append((g, m, max(M['Q'] - fa.TOL - m['Q'], M['R'] - fa.TOL - m['R'])))
    ok = [p for p in pts if p[2] <= 1e-12]
    if ok:
        best, flag = min(ok, key=lambda p: (p[1]['damage'], -p[1]['Q'], float(p[0].split(':')[1]))), 'met'
    else:
        best, flag = min(pts, key=lambda p: (p[2], -p[1]['Q'], float(p[0].split(':')[1]))), 'dev gate not met'
    return dict(arm=best[0], flag=flag, matched=M, grid={p[0]: dict(p[1], shortfall=p[2]) for p in pts})


def concentration(rec, arm: str, top_frac: float = 0.01, min_items: int = 5) -> dict:
    """Under `arm` at T: culprits are entries that break a fact on their own (the 'alone' run). Share of the distinct
    damaged facts broken by the top ceil(top_frac x entries) culprits; streams with fewer than min_items damaged facts
    report counts only."""
    ivs = rec['interventions'].get(arm, [])
    answers = rec['loc_answers']
    culprits = {}
    for r in ivs:
        if 'alone' in r:
            culprits[r['i']] = [e for e, ans in r['alone'] if ans != answers[str(r['i'])]]
    counts = Counter(e for cs in culprits.values() for e in cs)
    k = max(1, math.ceil(top_frac * rec['tables']['n']))
    top = [e for e, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:k]]
    hit = sum(1 for cs in culprits.values() if set(cs) & set(top))
    damaged = len(ivs)
    return dict(damaged=damaged, with_culprit=sum(1 for cs in culprits.values() if cs), top_k=k, top_share_of_damaged=
                (hit / damaged) if damaged >= min_items else None, numerator=hit, judged=damaged >= min_items,
                top=[dict(entry=e, facts=counts[e], entity=rec['tables']['entity'][e]) for e in top])


def analyze(recs: dict, cfg: dict, dev: dict | None = None) -> dict:
    reps, seed = int(cfg['rule']['bootstrap']), int(cfg['rule']['bootstrap_seed'])
    names = sorted(recs)
    arms = sorted(next(iter(recs.values()))['arms'])
    table = {a: {n: fa.arm_metrics(recs[n], a) for n in names} for a in arms}
    means = {a: {k: fa.mean([v[k] for v in table[a].values()]) for k in next(iter(table[a].values()))} for a in arms}
    q = fa.summary({n: fa.quality_items(recs[n], CELL, 'cur') for n in names}, reps, seed)
    r = fa.summary({n: fa.quality_items(recs[n], CELL, 'ret') for n in names}, reps, seed)
    sel = (cfg.get('selected_uniform') or {}).get('arm')
    # families with fewer than MIN_KNOWN base-known old facts have no damage measurement (PLAN section 5, fixed before
    # the test run): they are listed with their counts and left out of Delta, Delta_U and the gates' unit counts
    no_damage = {n: recs[n]['screening']['known'] for n in names if recs[n]['screening']['known'] < MIN_KNOWN}
    keep = [n for n in names if n not in no_damage]
    d = fa.summary({n: fa.delta_items(recs[n], CELL) for n in keep}, reps, seed)
    du = fa.summary({n: fa.delta_items(recs[n], CELL, f'{CELL}:{sel}') for n in keep}, reps, seed) if sel else None
    mech = fa.gate(d, q, r)
    # practical gate: Q and R against the selected uniform arm, not against the shuffles (FP-12 analysis fix)
    qu = fa.summary({n: fa.quality_items(recs[n], CELL, 'cur', f'{CELL}:{sel}') for n in names}, reps, seed) if sel else None
    ru = fa.summary({n: fa.quality_items(recs[n], CELL, 'ret', f'{CELL}:{sel}') for n in names}, reps, seed) if sel else None
    prac = fa.gate(du, qu, ru) if du else None
    loo = {n: fa.mean([d['per_stream'].get(m) for m in keep if m != n]) for n in keep}
    conc = {n: concentration(recs[n], f'{CELL}:uniform:0.90') for n in names}
    ivs = {a: {n: fa.categorise(recs[n], a) for n in names} for a in recs[names[0]]['interventions']}
    low = {n: dict(shuffled_damage=fa.mean([table[a][n]['damage'] for a in arms if a.startswith(f'{CELL}:shuffled:')]),
                   matched_damage=table[f'{CELL}:matched'][n]['damage']) for n in names}
    checks = {n: dict(none_known_all_right=recs[n]['checks']['none_known_all_right'],
                      unfired_max=max(recs[n]['checks']['unfired_differs_from_none'].values()),
                      reversed_same=recs[n]['checks']['determinism']['loc_reversed_same'],
                      blocked_never_fire_in_context=recs[n]['checks']['blocked_never_fire_in_context'],
                      shuffle_multisets_equal=recs[n]['checks']['shuffle_multisets_equal'],
                      refused_never_active=recs[n]['checks']['refused_never_active'],
                      extraction=recs[n]['extraction'], known=recs[n]['screening']['known'],
                      candidates=recs[n]['screening']['candidates']) for n in names}
    return dict(streams=names, no_damage_measurement=no_damage, delta=d, delta_uniform=du, quality_q=q, quality_r=r,
                practical_quality_q=qu if sel else None, practical_quality_r=ru if sel else None,
                mechanism_gate=mech,
                practical_gate=prac, selected_uniform=cfg.get('selected_uniform'), leave_one_out=loo,
                low_damage=low, concentration=conc, interventions=ivs, arms=table, arm_means=means, checks=checks,
                dev_tuning=tune(dev, cfg) if dev else None)


def cmd_tune(dev_run: Path, cfg_dev_path, out_path: Path):
    cfg_dev_path, out_path = Path(cfg_dev_path), Path(out_path)
    if out_path.exists():
        raise SystemExit(f'{out_path} exists')
    cfg = json.loads(cfg_dev_path.read_text(encoding='utf-8'))
    recs = fp_run.load_records(Path(dev_run))
    if set(recs) != set(cfg['phases']['dev']['streams']):
        raise SystemExit(f'dev records {sorted(recs)} != {cfg["phases"]["dev"]["streams"]}')
    sel = tune(recs, cfg)
    test = dict(cfg)
    test['status'] = (f'TEST CONFIG written by xr_run.py tune from {Path(dev_run).name} on '
                      f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; config_dev.json sha256 '
                      f'{fp_run.sha(cfg_dev_path)}.')
    test['selected_uniform'] = sel
    out_path.write_text(json.dumps(test, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(arm=sel['arm'], flag=sel['flag']), indent=1))


def cmd_analyze(test_run: Path, dev_run: Path | None = None):
    cfg = json.loads((test_run / 'config.json').read_text(encoding='utf-8'))
    recs = fp_run.load_records(test_run)
    dev = fp_run.load_records(dev_run) if dev_run else None
    res = analyze(recs, cfg, dev)
    fp_run.write_json(test_run / 'analysis.json', res)
    fp_run.finish(test_run)
    print(json.dumps(dict(delta=res['delta'], mechanism=res['mechanism_gate']['passed'],
                          practical=(res['practical_gate'] or {}).get('passed')), indent=1, default=str))
