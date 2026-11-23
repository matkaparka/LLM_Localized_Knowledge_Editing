"""Write RC-15's configs from the frozen parents (PLAN section 3). Every setting that is not listed here is the parent's:
FP-12's config_dev.json (2a5f828e...) for the WikiBigEdit runs, XR-13's config_dev.json (81f941ec...) for the
registry runs. Refuses to overwrite an existing file.

  python -I make_configs.py
"""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FP = HERE.parent / 'field_prefix_ablation_012'
XR = HERE.parent / 'external_key_replication_013'
PARENT_SHA = {FP / 'config_dev.json': '2a5f828efe5bbb06caf109df468bc6a026c11e453d0d96713962e2e6259b3d53',
              XR / 'config_dev.json': '81f941ec0a202dbfc5a3bcafe1d6cdd4d33bdb3dfdeb31983cb6582628beff6e'}
GRID = [0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
SH = [f'shuffled:{p}' for p in range(8)]
AC = [f'active_shuffled:{p}' for p in range(8)]
BAN_GRID = [f'ban_uniform:{x:.2f}' for x in GRID]
UNI_GRID = [f'uniform:{x:.2f}' for x in GRID]
FP_SEL = {'orig/on': 'uniform:0.94', 'corr/off': 'uniform:0.92'}   # FP-12's dev selections (its config_test.json)
XR_SEL = 'uniform:0.90'                                               # XR-13's dev selection
RULE = dict(delta=0.02, tol_q=0.02, tol_r=0.02, min_positive_share=0.75, bootstrap=10000, bootstrap_seed=0)
STATUS = ('FROZEN {when} before any RC-15 model run, by Claude on the author\'s 2026-11-22 request to run the review\'s '
          'supplementary experiments on vast.ai (pause2/Applied Intelligence/review_and_advise.md). Design in PLAN.md; '
          'parent config {parent}.')


def parent(p: Path) -> dict:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != PARENT_SHA[p]:
        raise SystemExit(f'{p} changed: {h}')
    return json.loads(p.read_text(encoding='utf-8'))


def fp_base(when: str) -> dict:
    c = parent(FP / 'config_dev.json')
    for k in ('arms', 'interventions', 'determinism_arm', 'phases', 'status', 'spans', 'spans_sha256', 'budget'):
        c.pop(k, None)
    c.update(status=STATUS.format(when=when, parent='field_prefix_ablation_012/config_dev.json 2a5f828e'),
             kind='fp', spans='../field_prefix_ablation_012/annotation/spans_final.json',
             spans_sha256='53b16cd9178452da3b1d1925da1b7ca298820e8afdeff99b405d32750b6da8d6',
             determinism_arm='corr/off:matched', interventions=dict(max_single=8), rule=RULE,
             budget=dict(gpu='RTX 4090', max_usd_per_hour=0.5, session_cap_usd=6.0))
    return c


def fp_dev(when):
    c = fp_base(when)
    cells = ['orig/on', 'corr/off']
    smoke_arms = ['matched', 'shuffled:0', 'shuffled:1', 'active_shuffled:0', 'active_shuffled:1', 'noban_matched',
                  'uniform:0.90', 'ban_uniform:0.90']
    c['phases'] = dict(
        smoke=dict(streams=['P0'], cap=12, order_check=True, intervention_check=True, cells=cells,
                   arms={x: smoke_arms for x in cells},
                   interventions=['matched', 'shuffled:0', 'active_shuffled:0', 'ban_uniform:0.90']),
        dev=dict(streams=['D0', 'D1', 'D2', 'D3'], cap=150, cells=cells, arms={x: ['matched'] + BAN_GRID for x in cells},
                 interventions=[]),
        test=dict(streams=['D4', 'D5', 'D6', 'D7'], cap=150, cells=cells,
                  arms={x: ['matched'] + SH + AC + ['noban_matched', 'uniform:0.90', FP_SEL[x], 'ban_uniform:0.90',
                                                    'ban_uniform:SEL'] for x in cells},
                  interventions=['uniform:0.90', 'matched', 'shuffled:0', 'active_shuffled:0', 'ban_uniform:0.90']))
    c['tune'] = dict(cells=cells, kinds=['ban_uniform'], levels=GRID)
    c['practical'] = {x: [FP_SEL[x], 'ban_uniform:SEL'] for x in cells}
    return c


def fp_variant(when, name, **opts):
    c = fp_base(when)
    arms = ['matched'] + SH
    if name.startswith('seed'):
        arms += AC + ['noban_matched', 'uniform:0.90', 'uniform:0.92', 'ban_uniform:0.90']
    else:
        arms += ['noban_matched'] + UNI_GRID + ['ban_uniform:0.90']
    c['phases'] = dict(test=dict(streams=['D4', 'D5', 'D6', 'D7'], cap=150, cells=['corr/off'], arms={'corr/off': arms},
                                 interventions=['uniform:0.90', 'matched', 'shuffled:0']))
    c['practical'] = {'corr/off': ['uniform:0.92']}
    for k, v in opts.items():
        if k == 'layer':
            c['local']['layers'] = [v]
        else:
            c[k] = v
    c['variant'] = name
    return c


def xr_dev(when):
    c = parent(XR / 'config_dev.json')
    for k in ('arms', 'interventions', 'phases', 'status', 'budget'):
        c.pop(k, None)
    c.update(status=STATUS.format(when=when, parent='external_key_replication_013/config_dev.json 81f941ec'),
             kind='xr', interventions=dict(max_single=8), rule=RULE,
             budget=dict(gpu='RTX 4090', max_usd_per_hour=0.5, session_cap_usd=6.0))
    test_streams = ['pypi', 'npm', 'crates', 'nuget', 'packagist', 'hackage', 'homebrew']   # CRAN: no known facts
    c['phases'] = dict(
        debug=dict(streams=['crates_debug'], cap=150, order_check=True, cells=['xr'],
                   arms={'xr': ['matched', 'shuffled:0', 'active_shuffled:0', 'noban_matched', 'uniform:0.90',
                                'ban_uniform:0.90']}, interventions=['matched']),
        dev=dict(streams=['pypi_dev', 'npm_dev', 'crates_dev', 'cran_dev'], cap=150, cells=['xr'],
                 arms={'xr': ['matched'] + BAN_GRID}, interventions=[]),
        test=dict(streams=test_streams, cap=150, cells=['xr'],
                  arms={'xr': ['matched'] + SH + AC + ['noban_matched', XR_SEL, 'ban_uniform:0.90', 'ban_uniform:SEL']},
                  interventions=['uniform:0.90', 'matched', 'shuffled:0', 'active_shuffled:0', 'ban_uniform:0.90']))
    c['tune'] = dict(cells=['xr'], kinds=['ban_uniform'], levels=GRID)
    c['practical'] = {'xr': [XR_SEL, 'ban_uniform:SEL']}
    return c


def main():
    import time
    when = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    out = {
        'config_fp_dev.json': fp_dev(when),
        'config_xr_dev.json': xr_dev(when),
        'config_seed1.json': fp_variant(when, 'seed1', write_seed=1),
        'config_seed2.json': fp_variant(when, 'seed2', write_seed=2),
        'config_layer6.json': fp_variant(when, 'layer6', layer=6, massive_rule=dict(base=200.0, ref_layer=8),
                                         norm_report=True),
        'config_layer12.json': fp_variant(when, 'layer12', layer=12, massive_rule=dict(base=200.0, ref_layer=8),
                                          norm_report=True),
        'config_raw.json': fp_variant(when, 'raw', centring='raw'),
    }
    for name, cfg in out.items():
        p = HERE / name
        if p.exists():
            raise SystemExit(f'{p} exists')
    for name, cfg in out.items():
        (HERE / name).write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(name, hashlib.sha256((HERE / name).read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
