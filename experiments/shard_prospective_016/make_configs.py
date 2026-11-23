"""Write SP-16's frozen dev configuration (PLAN.md of 2026-11-23). The test configuration is derived from it by
`sp_run.py select` after the dev streams, which fills in the development-selected uniform threshold.

  python make_configs.py            writes config_dev.json (refuses to overwrite)
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import sp_core as C

HERE = Path(__file__).resolve().parent
PERMS = 8
SHUFFLES = [f'shuffled:{p}' for p in range(PERMS)] + [f'active_shuffled:{p}' for p in range(PERMS)]
GRID = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def main():
    out = HERE / 'config_dev.json'
    if out.exists():
        raise SystemExit('config_dev.json exists')
    when = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    cfg = dict(
        status=(f'FROZEN {when} before any SP-16 model run, on the author\'s 2026-11-23 request to run GPT-6.2\'s '
                'plan (PLAN.md) on vast.ai. Editor: the author\'s single-slot SHARD at upstream commit '
                '96ede1c0bf23195492fe97c7b0ecc5339184512d (preset shard, tau 0.70, 200 steps, lr 1.0, no weight decay, '
                'norm cap 20) on Qwen2.5-1.5B-Instruct block 20; model loaded in float32; deterministic kernels.'),
        experiment='SP-16',
        model=dict(repo='Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
                   path='/workspace/models/qwen25_1.5b_instruct', dtype='float32'),
        data=dict(counterfact='/workspace/data/counterfact/counterfact.json', counterfact_sha256=C.SOURCE_SHA,
                  manifest='counterfact_roles.csv', manifest_sha256=C.MANIFEST_SHA),
        editor=dict(layer=20, tau=0.70, v_steps=200, v_lr=1.0, norm_cap=20.0, preset='shard'),
        eval=dict(max_new_tokens=12, batch_size=32),
        margin=0.02, n_perms=PERMS, budgets=[3, 13, 25], primary_budget=13,
        gates=dict(min_damaged=50, min_streams=4, min_gain=0.10),
        uniform_grid=GRID, selection=dict(damage_cap=0.02),
        partition='dev', streams=['D01', 'D02'],
        arms=['none', 'native'] + [f'uniform:{x:.2f}' for x in GRID] + ['matched'] + SHUFFLES + ['suppress_top13'],
        test_streams=[f'T{i:02d}' for i in range(1, 9)],
        test_arms=['none', 'native', 'uniform:SEL', 'matched'] + SHUFFLES + ['suppress_top13'],
        checks=dict(reversed_native=True),
        smoke=dict(stream='D01', writes=25, probes=100))
    out.write_text(json.dumps(cfg, indent=1) + '\n', encoding='utf-8')
    print('config_dev.json written')


if __name__ == '__main__':
    main()
