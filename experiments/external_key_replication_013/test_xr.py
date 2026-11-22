"""XR-13 tests: CPU only, SO-08's tiny random Qwen3 and character tokenizer, synthetic streams in XR-13's format."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import xr_run  # noqa: E402
import xr_events  # noqa: E402
import xr_questions  # noqa: E402
import xr_extract  # noqa: E402
from test_so import CharTok, tiny_model  # noqa: E402


def test_versions_order_and_states():
    vs = [dict(v='1.9.0', t='2026-01-05T00:00:00Z'), dict(v='1.10.0', t='2026-02-01T00:00:00Z'),
          dict(v='2.0.0rc1', t='2026-02-10T00:00:00Z', pre=True), dict(v='1.10.1', t='2026-03-02T00:00:00Z', yanked=True),
          dict(v='1.4.9', t='2026-03-20T00:00:00Z')]   # a backport on an old line must not count as the latest
    assert xr_events.state(vs, '2026-02-01T00:00:00Z')['v'] == '1.9.0'
    assert xr_events.state(vs, '2026-03-01T00:00:00Z')['v'] == '1.10.0'
    assert xr_events.state(vs, '2026-04-01T00:00:00Z')['v'] == '1.10.0'
    assert xr_events.canon_licence('MIT + file LICENSE') == 'MIT' and xr_events.canon_licence('MIT License') == 'MIT'
    assert xr_events.canon_licence('GPL (>= 2)') == 'GPL (>= 2)'


def test_pools_split_entities_and_events_cap_per_month():
    ents = []
    for i in range(60):
        ents.append(dict(entity=f'pkg{i}', rank=i, license_latest='MIT', versions=[
            dict(v='1.0', t='2026-01-10T00:00:00Z', spec='>=3.9'),
            dict(v='1.1', t=f'2026-03-{1 + i % 27:02d}T00:00:00Z', spec='>=3.10' if i % 5 == 0 else '>=3.9')]))
    t = xr_events.build(ents, 'pypi', '2026-03', 2, 10, 2, 100, pool='test', min_spares=5)
    d = xr_events.build(ents, 'pypi', '2026-03', 2, 10, 2, 100, pool='dev', min_spares=5)
    te = {e['entity'] for b in t['batches'] for e in b['edits']} | {f['entity'] for f in t['stable']}
    de = {e['entity'] for b in d['batches'] for e in b['edits']} | {f['entity'] for f in d['stable']}
    assert te and de and not te & de
    b0 = t['batches'][0]
    assert b0['n'] <= 10 and sum(e['attribute'] == 'runtime requirement' for e in b0['edits']) <= 2
    assert len(t['spares']) >= 5
    assert all(f['attribute'] == 'licence' for f in t['stable'])


def test_question_cleaning_and_extractor_spans():
    q, why = xr_questions.clean_question('Q: What is the latest version of the PyPI package requests?', 'requests',
                                         '2.32.5')
    assert q == 'What is the latest version of the PyPI package requests?' and why == 'ok'
    assert xr_questions.clean_question('Is requests at 2.32.5 now?', 'requests', '2.32.5')[1] == 'states the answer'
    assert xr_questions.clean_question('What about it?', 'requests', None)[1] == 'entity name missing'
    q = 'Which licence does the npm package chalk use?'
    span, kind = xr_extract.to_span(q, '"chalk"')
    assert kind == 'ok' and span == [q.find('chalk'), q.find('chalk') + 5]
    assert xr_extract.to_span('Which licence?', 'NONE')[1] == 'none'


def synth_stream(name, role, steps=3, n=8):
    edits = []
    for t in range(steps):
        for i in range(n):
            ent = f'lib{t}x{i}'
            q = f'What is the latest version of the package {ent}?'
            k = q.find(ent)
            edits.append(dict(uid=f'{name}:{t}:{i}', step=t, entity=ent, attribute='latest version',
                              value=f'{t}.{i}.0', update=q, rephrase=f'Which release of {ent} is newest?',
                              retention=f'Newest version number of {ent}, please?',
                              span=None if i == 3 else [k, k + len(ent)], span_text=None if i == 3 else ent))
    hidden = [[f'Which licence does the package old{i} use?', 'x'] for i in range(10)]
    probes = [f'What is the licence of the package probe{i}?' for i in range(6)]
    return dict(stream=name, family='pypi', role=role, window=[], steps=steps, shots=[['What is 1+1?', '2']],
                edits=edits, hidden=hidden, probes=probes)


def factory(cfg, phase):
    import dataclasses
    from so_data import shots_text
    from so_model import fmt_prompt
    tok = CharTok()
    env = xr_run.XREnv(cfg, phase, model=tiny_model(len(tok), 0), tok=tok)
    for name, s in list(env.streams.items()):    # every hidden candidate 'known': its answer is the base model's own
        ctx = shots_text(s.shots)
        ids = [tok.encode(fmt_prompt(q, ctx)) for q, _ in s.loc_hidden]
        outs = env.gen.generate(ids, [len(tok.encode(ctx))] * len(ids))
        env.streams[name] = dataclasses.replace(s, loc_hidden=tuple((q, o.text) for (q, _), o in zip(s.loc_hidden, outs)))
    return env


def tiny_cfg(sdir: Path) -> dict:
    return dict(
        status='test', model_path='unused', dtype='float32', torch_seed=0, sites=['model.layers.{i}.mlp.down_proj'],
        layers='all', rank=2, scale=1.0, clip=1.0, streams_dir=str(sdir),
        phases=dict(debug=dict(streams=['s_debug'], cap=150, order_check=True), dev=dict(streams=['s_dev'], cap=150),
                    test=dict(streams=['s_a', 's_b'], cap=150)),
        panels=dict(max_cur=4, ret_per_step=2, max_loc=10, max_new=5, gen_batch=4),
        eval=dict(widths=[128, 256, 384], batch=4, max_new=5),
        local=dict(mode='route', layers=[0], tau=0.5, probe_margin=0.01, massive=200.0, v_steps=2, v_lr=0.5, v_wd=0.5,
                   clamp=0.75, kl=0.0625, stop_loss=0.05, n_rand=1, rand_len=3, use_context=True, chunk=4, key_batch=8),
        write_seed=0, arms=dict(n_shuffles=2, uniform=[0.5, 0.7, 0.9]),
        interventions=dict(arms=['uniform:0.90', 'matched'], max_single=8), rule=dict(bootstrap=50, bootstrap_seed=0))


def test_end_to_end_debug_dev_tune_test_analyze(tmp_path, monkeypatch):
    for k in ('RV_DATA_DIR', 'RV_MODEL_PATH', 'XR_STREAMS_DIR'):
        monkeypatch.delenv(k, raising=False)
    sdir = tmp_path / 'streams'
    sdir.mkdir()
    for name, role in (('s_debug', 'debug'), ('s_dev', 'dev'), ('s_a', 'test'), ('s_b', 'test')):
        (sdir / f'{name}.json').write_text(json.dumps(synth_stream(name, role)), encoding='utf-8')
    (sdir / 'mu_corpus.json').write_text(json.dumps(dict(texts=[f'Q: What is thing {i}?\nA:' for i in range(20)])),
                                         encoding='utf-8')
    cfg = tiny_cfg(sdir)
    cpath = tmp_path / 'config_dev.json'
    cpath.write_text(json.dumps(cfg), encoding='utf-8')
    xr_run.cmd_run(cpath, tmp_path / 'debug', 'debug', env_factory=factory)
    rec = xr_run.fp_run.read_gz(tmp_path / 'debug' / 'stream_s_debug.json.gz')
    c = rec['checks']
    assert c['screen_reversed_identical'] and c['none_known_all_right'] and c['determinism']['loc_reversed_same']
    assert all(v == 0 for v in c['unfired_differs_from_none'].values())
    assert c['blocked_never_fire_in_context'] and c['refused_never_active'] and c['shuffle_multisets_equal']
    assert rec['extraction']['refused'] == 3 and rec['tables']['refused']
    xr_run.cmd_run(cpath, tmp_path / 'dev', 'dev', env_factory=factory)
    import xr_analysis
    tpath = tmp_path / 'config_test.json'
    xr_analysis.cmd_tune(tmp_path / 'dev', cpath, tpath)
    xr_run.cmd_run(tpath, tmp_path / 'test', 'test', env_factory=factory)
    xr_analysis.cmd_analyze(tmp_path / 'test', tmp_path / 'dev')
    res = json.loads((tmp_path / 'test' / 'analysis.json').read_text(encoding='utf-8'))
    assert set(res['streams']) == {'s_a', 's_b'} and 'mechanism_gate' in res and res['concentration']
