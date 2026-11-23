"""BR-14 tests (CPU only): the configs' differences from FP-12 and the design-variant rule on synthetic timings."""
import gzip
import json
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import br_design  # noqa: E402

VARIANTS = ('full', 'r1', 'r2', 'r3')


def cfgs():
    out = {}
    for v in VARIANTS:
        p = HERE / f'config_dev_{v}.json'
        if not p.exists():
            pytest.skip('configs not written yet')
        out[v] = json.loads(p.read_text(encoding='utf-8'))
    return out


def test_configs_differ_from_fp12_only_where_the_plan_says():
    fp = json.loads((HERE.parent / 'field_prefix_ablation_012' / 'config_dev.json').read_text(encoding='utf-8'))
    for v, c in cfgs().items():
        assert c['local']['layers'] == [9] and c['local']['chunk'] == 16 and c['eval']['batch'] == 32
        for k in ('tau', 'probe_margin', 'v_steps', 'v_lr', 'v_wd', 'clamp', 'kl', 'stop_loss', 'n_rand', 'rand_len',
                  'use_context', 'mode'):
            assert c['local'][k] == fp['local'][k], k
        assert c['streams'] == fp['streams'] and c['write_seed'] == 0 and c['rank'] == fp['rank']
        assert c['panels']['cov'] is False and c['screen_cap'] == 3000
        assert c['massive_rule']['base'] == 200.0 and c['phases']['test']['streams'] == ['D4', 'D5', 'D6', 'D7']
        assert set(c['phases']['dev']['streams']) <= {'D0', 'D1', 'D2', 'D3'}
        assert not set(c['phases']['dev']['streams']) & set(c['phases']['test']['streams'])


def fake_smoke(tmp_path, write_s=1.0, row_s=0.02):
    rec = dict(clock=dict(write=96 * write_s, screen=60.0, probes=30.0, eval_orig=row_s * 2000, eval_corr=row_s * 2000,
                          rows_orig=2000, rows_corr=2000),
               tables=dict(n=96), screening=dict(known=1000))
    p = tmp_path / 'stream_P0.json.gz'
    with gzip.open(p, 'wb') as g:
        g.write(json.dumps(rec).encode('utf-8'))
    return p


def test_design_rule_picks_the_first_variant_that_fits(tmp_path):
    cfgs()
    smoke = fake_smoke(tmp_path)
    inst = tmp_path / 'instance.json'
    inst.write_text(json.dumps(dict(deadline_epoch=time.time() + 100 * 3600)), encoding='utf-8')
    br_design.main(str(smoke), str(inst), str(tmp_path / 'd1.json'))
    d = json.loads((tmp_path / 'd1.json').read_text(encoding='utf-8'))
    assert d['variant'] == 'full'
    totals = [d['variants'][v]['total_h'] for v in VARIANTS]
    assert totals == sorted(totals, reverse=True)
    between = (totals[0] + totals[1]) / 2 * 3600 + br_design.MARGIN
    inst.write_text(json.dumps(dict(deadline_epoch=time.time() + between)), encoding='utf-8')
    br_design.main(str(smoke), str(inst), str(tmp_path / 'd2.json'))
    assert json.loads((tmp_path / 'd2.json').read_text(encoding='utf-8'))['variant'] == 'r1'
    inst.write_text(json.dumps(dict(deadline_epoch=time.time() + 600)), encoding='utf-8')
    with pytest.raises(SystemExit):
        br_design.main(str(smoke), str(inst), str(tmp_path / 'd3.json'))
