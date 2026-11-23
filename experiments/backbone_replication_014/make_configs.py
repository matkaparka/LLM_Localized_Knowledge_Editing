"""Write BR-14's dev configs (PLAN sections 1-4): FP-12's config_dev.json (2a5f828e...) with the second backbone and the
listed changes, in four design variants that differ only in how much is run (PLAN section 4: the host picks the first
variant whose projected cost fits, from the smoke run's measured speed, before any dev stream). Refuses to overwrite.

  python -I make_configs.py
"""
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FP = HERE.parent / 'field_prefix_ablation_012'
PARENT_SHA = '2a5f828efe5bbb06caf109df468bc6a026c11e453d0d96713962e2e6259b3d53'
GRID = [0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
SH = [f'shuffled:{p}' for p in range(8)]
AC = [f'active_shuffled:{p}' for p in range(8)]
UNI_GRID = [f'uniform:{x:.2f}' for x in GRID]
BAN_GRID = [f'ban_uniform:{x:.2f}' for x in GRID]
RULE = dict(delta=0.02, tol_q=0.02, tol_r=0.02, min_positive_share=0.75, bootstrap=10000, bootstrap_seed=0)
VARIANTS = {   # name: (dev streams, test cells, active shuffles)
    'full': (['D0', 'D1', 'D2', 'D3'], ['orig/on', 'corr/off'], AC),
    'r1': (['D0', 'D1', 'D2', 'D3'], ['corr/off'], AC),
    'r2': (['D0', 'D1'], ['corr/off'], AC),
    'r3': (['D0', 'D1'], ['corr/off'], AC[:4]),
}


def config(when: str, variant: str) -> dict:
    p = FP / 'config_dev.json'
    if hashlib.sha256(p.read_bytes()).hexdigest() != PARENT_SHA:
        raise SystemExit(f'{p} changed')
    c = json.loads(p.read_text(encoding='utf-8'))
    for k in ('arms', 'interventions', 'determinism_arm', 'phases', 'status', 'spans', 'spans_sha256', 'budget'):
        c.pop(k, None)
    dev_streams, test_cells, ac = VARIANTS[variant]
    c.update(
        status=(f'FROZEN {when} before any BR-14 model run, by Claude on the author\'s 2026-11-22 request to run the '
                f'review\'s supplementary experiments on vast.ai. Design in PLAN.md; parent '
                f'field_prefix_ablation_012/config_dev.json 2a5f828e; design variant {variant}.'),
        kind='fp', variant=variant,
        model_path='E:/research/pause2/models/olmo2_7b_base',
        spans='../field_prefix_ablation_012/annotation/spans_final.json',
        spans_sha256='53b16cd9178452da3b1d1925da1b7ca298820e8afdeff99b405d32750b6da8d6',
        screen_cap=3000,
        massive_rule=dict(base=200.0, ref_median_file='../backbone_replication_014/calibration/qwen_norms.json',
                          ref_layer_of_ref_model=8),
        norm_report=True,
        determinism_arm='corr/off:matched', interventions=dict(max_single=8), rule=RULE,
        acceptance=dict(phase='smoke', cell='corr/off', arm='matched', measure='update', min=0.80),
        budget=dict(gpu='RTX 4090', max_usd_per_hour=0.5, session_cap_usd=8.0, variant_limit_usd=7.5))
    c['local']['layers'] = [9]
    c['local']['chunk'] = 16
    c['eval']['batch'] = 32
    c['panels']['cov'] = False
    smoke_arms = ['matched', 'shuffled:0', 'shuffled:1', 'active_shuffled:0', 'noban_matched', 'uniform:0.90',
                  'ban_uniform:0.90']
    test_arms = {'corr/off': ['matched'] + SH + ac + ['noban_matched', 'uniform:0.90', 'uniform:SEL',
                                                      'ban_uniform:0.90', 'ban_uniform:SEL'],
                 'orig/on': ['matched'] + SH + ['noban_matched', 'uniform:0.90', 'ban_uniform:0.90']}
    c['phases'] = dict(
        smoke=dict(streams=['P0'], cap=12, order_check=True, intervention_check=True, cells=['orig/on', 'corr/off'],
                   arms={x: smoke_arms for x in ('orig/on', 'corr/off')},
                   interventions=['matched', 'shuffled:0', 'uniform:0.90']),
        dev=dict(streams=dev_streams, cap=150, cells=['corr/off'], arms={'corr/off': ['matched'] + UNI_GRID + BAN_GRID},
                 interventions=[]),
        test=dict(streams=['D4', 'D5', 'D6', 'D7'], cap=150, cells=test_cells,
                  arms={x: test_arms[x] for x in test_cells},
                  interventions=['uniform:0.90', 'matched', 'shuffled:0', 'active_shuffled:0', 'ban_uniform:0.90']))
    c['tune'] = dict(cells=['corr/off'], kinds=['uniform', 'ban_uniform'], levels=GRID)
    c['practical'] = {'corr/off': ['uniform:SEL', 'ban_uniform:SEL']}
    return c


def main():
    when = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    names = {v: HERE / f'config_dev_{v}.json' for v in VARIANTS}
    for p in names.values():
        if p.exists():
            raise SystemExit(f'{p} exists')
    for v, p in names.items():
        p.write_text(json.dumps(config(when, v), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(p.name, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
