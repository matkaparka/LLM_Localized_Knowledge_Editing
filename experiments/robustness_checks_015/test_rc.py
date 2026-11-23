"""RC-15 tests: CPU only, SO-08's tiny random Qwen3 and character tokenizer, EC-11's synthetic WikiBigEdit-style data
and XR-13's synthetic streams (the fixtures of test_fp.py and test_xr.py)."""
import dataclasses
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rc_core  # noqa: E402
import rc_run  # noqa: E402
import rc_analysis  # noqa: E402
from rc_core import OFF, fp_core, fp_run  # noqa: E402
from so_data import shots_text  # noqa: E402
from so_model import fmt_prompt  # noqa: E402
from test_so import CharTok, tiny_model  # noqa: E402
from test_ec import corpus_texts, make_wm, write_data  # noqa: E402
import test_fp  # noqa: E402
import test_xr  # noqa: E402


# ---- arms -----------------------------------------------------------------------------------------------------------
def vectors():
    m = torch.tensor([0.9, 1.2, 0.95, 0.9, 1.05, 0.97, 0.9, 0.93, 1.3, 0.9, 0.99, 0.9])
    ends = [5, 12]
    active = torch.ones(12, dtype=torch.bool)
    active[3] = False
    active[8] = False          # a refused entry that would also be banned
    return m, ends, active


def test_active_shuffles_keep_bans_refusals_and_batch_multisets():
    m, ends, active = vectors()
    ban = rc_core.banned(m) & active
    assert ban.tolist() == [False, True, False, False, True, False, False, False, False, False, False, False]
    ref = torch.where(active, m, torch.full_like(m, OFF))
    moved = False
    for p in range(6):
        v = rc_core.arm_vector(f'active_shuffled:{p}', m, ends, active, 'D9', 0.9)
        assert fp_core.multisets_equal(v, ref, ends)
        assert torch.equal(v[ban], m[ban]) and bool((v[~active] >= OFF).all())
        moved |= not torch.equal(v, ref)
        s = rc_core.arm_vector(f'shuffled:{p}', m, ends, active, 'D9', 0.9)
        assert torch.equal(s, fp_core.arm_vector(f'shuffled:{p}', m, ends, active, 'D9'))
    assert moved
    a = rc_core.arm_vector('active_shuffled:1', m, ends, active, 'D9', 0.9)
    assert torch.equal(a, rc_core.arm_vector('active_shuffled:1', m, ends, active, 'D9', 0.9))   # fixed permutation


def test_ban_and_noban_arms():
    m, ends, active = vectors()
    ban = rc_core.banned(m) & active
    b = rc_core.arm_vector('ban_uniform:0.92', m, ends, active, 'D9', 0.9)
    assert bool((b[ban] >= OFF).all()) and bool((b[~active] >= OFF).all())
    assert torch.allclose(b[active & ~ban], torch.full_like(b[active & ~ban], 0.92))
    n = rc_core.arm_vector('noban_matched', m, ends, active, 'D9', 0.9)
    assert torch.allclose(n[ban], torch.full_like(n[ban], 0.9)) and torch.equal(n[active & ~ban], m[active & ~ban])
    assert bool((n[~active] >= OFF).all())
    with pytest.raises(ValueError):
        rc_core.arm_vector('bogus:1', m, ends, active, 'D9', 0.9)
    checks = rc_core.vector_checks({'c:ban_uniform:0.92': (b, 'r'), 'c:noban_matched': (n, 'r'),
                                    'c:active_shuffled:0': (rc_core.arm_vector('active_shuffled:0', m, ends, active,
                                                                               'D9', 0.9), 'r')},
                                   {'r': m}, {'r': ends}, {'r': rc_core.banned(m)}, ~active)
    assert all(checks.values())


def test_resolve_placeholders():
    sel = {'ban_uniform': dict(level=0.9), 'uniform': dict(level=0.94)}
    assert rc_run.resolve(['matched', 'ban_uniform:0.90', 'ban_uniform:SEL', 'uniform:SEL'], sel, 'x') == \
        ['matched', 'ban_uniform:0.90', 'uniform:0.94']
    with pytest.raises(SystemExit):
        rc_run.resolve(['uniform:SEL'], {}, 'x')


# ---- norms and spans ---------------------------------------------------------------------------------------------------
def test_norm_stats_match_hidden_states():
    wm = make_wm()
    texts = corpus_texts()[:13]
    st = rc_core.norm_stats(wm, wm.model.get_submodule('model.layers.0'), texts, batch_size=5)
    vals = []
    with torch.no_grad():
        for t in texts:
            ids = torch.tensor([wm.tok.encode(t)])
            h = wm.model(input_ids=ids, output_hidden_states=True).hidden_states[1][0]
            vals.append(h.float().norm(dim=-1)[1:])
    v = torch.cat(vals).double()
    assert st['positions'] == v.numel()
    assert abs(st['median'] - float(torch.quantile(v, 0.5))) < 1e-4


def test_retokenise_reproduces_the_char_tokenizer_flags(tmp_path):
    d = write_data(tmp_path / 'wbe')
    spans = tmp_path / 'spans.json'
    cfg = test_fp.tiny_cfg(d, str(spans))
    test_fp.synthetic_spans(d, spans, cfg)
    from so_data import load_edits
    edits, _, _ = load_edits(d, expect_steps=4)
    final = json.loads(spans.read_text(encoding='utf-8'))['final']
    out, rep = rc_core.retokenise(final, CharTok(), {e.uid: e for e in edits})
    assert rep['changed_differs_from_fp12'] == 0 and rep['changed'] == sum(bool(v['changed']) for v in final.values())


# ---- end to end: WikiBigEdit kind ----------------------------------------------------------------------------------------
CELLS = ['orig/on', 'corr/off']
TEST_ARMS = ['matched', 'shuffled:0', 'shuffled:1', 'active_shuffled:0', 'active_shuffled:1', 'noban_matched',
             'uniform:0.50', 'uniform:0.70', 'ban_uniform:0.50', 'ban_uniform:SEL']


def fp_cfg(data_dir: Path, spans: Path, **kw) -> dict:
    cfg = test_fp.tiny_cfg(data_dir, str(spans))
    for k in ('arms', 'determinism_arm'):
        cfg.pop(k)
    cfg['local']['probe_margin'] = 0.3          # some entries end above 1 (banned)
    cfg.update(kind='fp', spans=str(spans), spans_sha256=None, determinism_arm='corr/off:matched',
               interventions=dict(max_single=8),
               phases=dict(
                   smoke=dict(streams=['P0'], cap=2, order_check=True, intervention_check=True, cells=CELLS,
                              arms={c: [a for a in TEST_ARMS if 'SEL' not in a] for c in CELLS},
                              interventions=['matched', 'shuffled:0']),
                   dev=dict(streams=['D0'], cap=150, cells=CELLS,
                            arms={c: ['matched', 'ban_uniform:0.50', 'ban_uniform:0.70', 'ban_uniform:0.90']
                                  for c in CELLS}, interventions=[]),
                   test=dict(streams=['D1'], cap=150, cells=CELLS, arms={c: TEST_ARMS for c in CELLS},
                             interventions=['matched', 'shuffled:0', 'ban_uniform:0.50'])),
               tune=dict(cells=CELLS, kinds=['ban_uniform'], levels=[0.5, 0.7, 0.9]),
               practical={c: ['uniform:0.70', 'ban_uniform:SEL'] for c in CELLS})
    cfg.update(kw)
    return cfg


def fp_factory(cfg, phase):
    tok = CharTok()
    env = rc_core.RCEnv(cfg, model=tiny_model(len(tok), 0), tok=tok)
    for name, s in list(env.streams.items()):      # every candidate 'known': its answer is the base model's own
        ctx = shots_text(s.shots)
        ids = [tok.encode(fmt_prompt(q, ctx)) for q, _ in s.loc_hidden]
        outs = env.gen.generate(ids, [len(tok.encode(ctx))] * len(ids))
        env.streams[name] = dataclasses.replace(s, loc_hidden=tuple((q, o.text) for (q, _), o in zip(s.loc_hidden, outs)))
    return env


def assert_checks(c, fp=True):
    assert c['none_known_all_right'] and c['determinism']['loc_reversed_same']
    assert all(v == 0 for v in c['unfired_differs_from_none'].values())
    assert c['shuffle_multisets_equal'] and c['refused_never_active']
    assert c['active_shuffle_bans_fixed'] and c['ban_uniform_bans_off']
    if fp:
        assert c['blocked_cells_never_fire_in_context'] and c['probes_same_as_set_probes']
        assert c.get('corr_unchanged_bitwise', True)


def test_end_to_end_fp_smoke_dev_tune_test_analyze(tmp_path, monkeypatch):
    for k in ('RV_DATA_DIR', 'RV_MODEL_PATH', 'XR_STREAMS_DIR'):
        monkeypatch.delenv(k, raising=False)
    d = write_data(tmp_path / 'wbe')
    spans = tmp_path / 'spans_final.json'
    cfg = fp_cfg(d, spans)
    test_fp.synthetic_spans(d, spans, cfg)
    cpath = tmp_path / 'config_dev.json'
    cpath.write_text(json.dumps(cfg), encoding='utf-8')
    rc_run.cmd_run(cpath, tmp_path / 'smoke', 'smoke', env_factory=fp_factory)
    sm = fp_run.read_gz(tmp_path / 'smoke' / 'stream_P0.json.gz')
    assert_checks(sm['checks'])
    assert sm['checks']['screen_reversed_identical']
    assert sm['checks'].get('interventions_batched', dict(equal_single=True))['equal_single']
    with pytest.raises(SystemExit):                 # placeholders must be resolved first
        rc_run.cmd_run(cpath, tmp_path / 'too_early', 'test', env_factory=fp_factory)
    rc_run.cmd_run(cpath, tmp_path / 'dev', 'dev', env_factory=fp_factory)
    tpath = tmp_path / 'config_test.json'
    rc_run.cmd_tune(tmp_path / 'dev', cpath, tpath)
    tc = json.loads(tpath.read_text(encoding='utf-8'))
    assert set(tc['selected']) == set(CELLS) and not any('SEL' in a for a in tc['phases']['test']['arms']['corr/off'])
    assert not any('SEL' in a for v in tc['practical'].values() for a in v)
    rc_run.cmd_run(tpath, tmp_path / 'test', 'test', env_factory=fp_factory)
    rec = fp_run.read_gz(tmp_path / 'test' / 'stream_D1.json.gz')
    assert_checks(rec['checks'])
    n_cell = len(dict.fromkeys(tc['phases']['test']['arms']['corr/off']))
    assert len(rec['arms']) == 1 + 2 * n_cell
    assert set(rec['bans']) == {'orig', 'corr'} and rec['tables']['changed'] and rec['tables']['refused']
    rc_run.cmd_analyze(tmp_path / 'test', tmp_path / 'dev', tmp_path / 'test')
    res = json.loads((tmp_path / 'test' / 'analysis.json').read_text(encoding='utf-8'))
    for c in CELLS:
        r = res['cells_result'][c]
        assert {'delta', 'delta_active', 'delta_bans', 'delta_gate', 'delta_active_gate'} <= set(r)
        assert {'total', 'ban_flat', 'ban_cont', 'cont_ban', 'cont_noban', 'interaction'} <= set(r['factorial'])
        assert set(r['practical']) == set(tc['practical'][c])
        f = r['factorial']                          # the factorial identity holds per item, so for the means too
        assert abs(f['total']['mean'] - (f['ban_flat']['mean'] + f['cont_ban']['mean'])) < 1e-12
        assert abs(f['interaction']['mean'] - (f['ban_flat']['mean'] - f['ban_cont']['mean'])) < 1e-12
    assert all(v['identical_arms'] == v['shared_arms'] for v in res['reproduction'].values())
    import importlib.util
    spec = importlib.util.spec_from_file_location('recount', HERE / 'checks' / 'recount.py')
    rcnt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rcnt)
    rcnt.main(str(tmp_path / 'test'), str(tmp_path / 'dev'), str(tmp_path / 'recount.json'))
    out = json.loads((tmp_path / 'recount.json').read_text(encoding='utf-8'))
    assert out['max_abs_diff'] < 1e-9 and out['tuning_matches_config']


def test_options_raw_centring_layer_and_massive_rule(tmp_path, monkeypatch):
    for k in ('RV_DATA_DIR', 'RV_MODEL_PATH', 'XR_STREAMS_DIR'):
        monkeypatch.delenv(k, raising=False)
    d = write_data(tmp_path / 'wbe')
    spans = tmp_path / 'spans_final.json'
    cfg = fp_cfg(d, spans, centring='raw', write_seed=1, screen_cap=4, massive_rule=dict(base=200.0, ref_layer=0))
    cfg['local']['layers'] = [1]
    cfg['panels']['cov'] = False
    cfg['phases']['smoke']['cells'] = ['corr/off']
    cfg['determinism_arm'] = 'corr/off:matched'
    test_fp.synthetic_spans(d, spans, cfg)
    cpath = tmp_path / 'config.json'
    cpath.write_text(json.dumps(cfg), encoding='utf-8')
    env = fp_factory(cfg, 'smoke')
    assert float(env.editor.mu.abs().max()) == 0.0
    ni = env.norm_info
    assert ni['layer'] == 1 and ni['ref_layer'] == 0
    assert abs(env.editor.cfg.massive - 200.0 * ni['here']['median'] / ni['ref']['median']) < 1e-9
    rc_run.cmd_run(cpath, tmp_path / 'smoke', 'smoke', env_factory=lambda c, p: env)
    rec = fp_run.read_gz(tmp_path / 'smoke' / 'stream_P0.json.gz')
    assert_checks(rec['checks'])
    o = rec['options']
    assert o['centring'] == 'raw' and o['layer'] == 1 and o['write_seed'] == 1 and o['cov'] is False
    assert rec['screening']['candidates'] <= 4 and not any(k == 'cov' for k, _, _ in rec['items'])
    assert set(rec['tables']) >= {'orig', 'corr'} and all(a.startswith(('corr/off:', 'none')) for a in rec['arms'])


def test_massive_rule_from_a_reference_file(tmp_path, monkeypatch):
    for k in ('RV_DATA_DIR', 'RV_MODEL_PATH', 'XR_STREAMS_DIR'):
        monkeypatch.delenv(k, raising=False)
    d = write_data(tmp_path / 'wbe')
    spans = tmp_path / 'spans_final.json'
    ref = tmp_path / 'ref_norms.json'
    ref.write_text(json.dumps(dict(stats=dict(median=2.5))), encoding='utf-8')
    cfg = fp_cfg(d, spans, massive_rule=dict(base=200.0, ref_median_file=str(ref)), norm_report=True)
    test_fp.synthetic_spans(d, spans, cfg)
    env = fp_factory(cfg, 'smoke')
    ni = env.norm_info
    assert ni['ref_median'] == 2.5 and ni['ref_file_sha256'] == fp_run.sha(ref)
    assert abs(env.editor.cfg.massive - 200.0 * ni['here']['median'] / 2.5) < 1e-9


# ---- end to end: XR-13 kind --------------------------------------------------------------------------------------------
def xr_cfg(sdir: Path) -> dict:
    cfg = test_xr.tiny_cfg(sdir)
    cfg.pop('arms')
    cfg['local']['probe_margin'] = 0.3
    arms = ['matched', 'shuffled:0', 'shuffled:1', 'active_shuffled:0', 'active_shuffled:1', 'noban_matched',
            'uniform:0.50', 'ban_uniform:0.50', 'ban_uniform:SEL']
    cfg.update(kind='xr', interventions=dict(max_single=8),
               phases=dict(debug=dict(streams=['s_debug'], cap=150, order_check=True, cells=['xr'],
                                      arms={'xr': [a for a in arms if 'SEL' not in a]}, interventions=['matched']),
                           dev=dict(streams=['s_dev'], cap=150, cells=['xr'],
                                    arms={'xr': ['matched', 'ban_uniform:0.50', 'ban_uniform:0.70', 'ban_uniform:0.90']},
                                    interventions=[]),
                           test=dict(streams=['s_a', 's_b'], cap=150, cells=['xr'], arms={'xr': arms},
                                     interventions=['matched', 'shuffled:0'])),
               tune=dict(cells=['xr'], kinds=['ban_uniform'], levels=[0.5, 0.7, 0.9]),
               practical={'xr': ['uniform:0.50', 'ban_uniform:SEL']})
    return cfg


def test_end_to_end_xr_dev_tune_test_analyze(tmp_path, monkeypatch):
    for k in ('RV_DATA_DIR', 'RV_MODEL_PATH', 'XR_STREAMS_DIR'):
        monkeypatch.delenv(k, raising=False)
    sdir = tmp_path / 'streams'
    sdir.mkdir()
    for name, role in (('s_debug', 'debug'), ('s_dev', 'dev'), ('s_a', 'test'), ('s_b', 'test')):
        (sdir / f'{name}.json').write_text(json.dumps(test_xr.synth_stream(name, role)), encoding='utf-8')
    (sdir / 'mu_corpus.json').write_text(json.dumps(dict(texts=[f'Q: What is thing {i}?\nA:' for i in range(20)])),
                                         encoding='utf-8')
    cpath = tmp_path / 'config_dev.json'
    cpath.write_text(json.dumps(xr_cfg(sdir)), encoding='utf-8')
    rc_run.cmd_run(cpath, tmp_path / 'debug', 'debug', env_factory=test_xr.factory)
    rec = fp_run.read_gz(tmp_path / 'debug' / 'stream_s_debug.json.gz')
    assert_checks(rec['checks'], fp=False)
    assert rec['checks']['blocked_never_fire_in_context'] and rec['tables']['refused']
    rc_run.cmd_run(cpath, tmp_path / 'dev', 'dev', env_factory=test_xr.factory)
    tpath = tmp_path / 'config_test.json'
    rc_run.cmd_tune(tmp_path / 'dev', cpath, tpath)
    rc_run.cmd_run(tpath, tmp_path / 'test', 'test', env_factory=test_xr.factory)
    rc_run.cmd_analyze(tmp_path / 'test', tmp_path / 'dev')
    res = json.loads((tmp_path / 'test' / 'analysis.json').read_text(encoding='utf-8'))
    r = res['cells_result']['xr']
    assert set(res['streams']) == {'s_a', 's_b'} and 'delta_active' in r and 'ban_flat' in r['factorial']
    assert r['culprits']['matched']['s_a'] is not None
