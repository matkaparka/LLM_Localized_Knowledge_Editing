"""EC-11 tests: CPU only, SO-08's tiny random Qwen3 and character tokenizer, a tiny random BERT encoder, and a
synthetic WikiBigEdit-style dataset with four relation families (two per half)."""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ec_data  # noqa: E402
import ec_run  # noqa: E402
from ec_analysis import arrays, boot_means, frontier_value, select, table, verdict  # noqa: E402
from ec_arms import OFF, family, kt_thresholds, rad_thresholds, subfamily, with_hygiene  # noqa: E402
from ec_eval import FixedShapeGen, edit_items, loc_items, plain_prompts, relevant, score  # noqa: E402
import ec_grace  # noqa: E402
import ec_rag  # noqa: E402
from kt_core import LoggedEditor, build_entries, install, probe_max, probe_positions  # noqa: E402
from rv_editor import EditWritable, LocalCfg  # noqa: E402
from rv_policy import corpus, spare_rows  # noqa: E402
from so_data import build_streams, load_edits, norm, shots_text, split_halves  # noqa: E402
from so_model import fmt_prompt, fmt_target  # noqa: E402
from test_so import CharTok, tiny_model  # noqa: E402

SITES = ['model.layers.0.mlp.down_proj', 'model.layers.1.mlp.down_proj']
DATES = ['20240101', '20240201', '20240301', '20240401', '20240501']
WIDTHS = (128, 256, 384)


def make_wm(seed=0, dtype=torch.float32):
    tok = CharTok()
    return EditWritable(tiny_model(len(tok), seed, dtype), tok, SITES, rank=2, scale=1.0, clip=1.0)


def route_cfg(**kw):
    return LocalCfg(**dict(dict(mode='route', layers=(0,), tau=0.5, v_steps=3, n_rand=1, rand_len=3, chunk=4,
                                key_batch=8, probe_margin=0.01), **kw))


def corpus_texts():
    return [f'Q: What is the code of Thing{i}?\nA:' for i in range(30)]


# ---- synthetic data with four relations --------------------------------------------------------------------------
def rows_for(step, n=20):
    rows = []
    for r, rel in enumerate(('code', 'colour', 'size', 'shape')):
        for i in range(n):
            subj = f'S{r}x{step}y{i}'
            loc = f'Where is Shared{step}?' if i % 10 == 0 else f'Where is L{r}x{step}y{i}?'
            rows.append(dict(tag='new', subject=subj, subject_id=f'Q{r}_{step}_{i}', relation=rel,
                             relation_id=f'P{r + 1}', object=f'a{r}{step}{i}', object_id=None, ans=f'a{r}{step}{i}',
                             update=f'What is the {rel} of {subj}?', rephrase=f'Which {rel} does {subj} have?',
                             personas=f'Hey, {rel} of {subj}?', loc=loc, loc_ans=f'l{r}{step}{i}', mhop=None,
                             mhop_ans=None))
    return rows


def write_data(d: Path, steps=4):
    d.mkdir(parents=True)
    for s in range(steps):
        (d / f'wiki_big_edit_{DATES[s]}_{DATES[s + 1]}.json').write_text(json.dumps(rows_for(s)), encoding='utf-8')
    return d


SARGS = dict(n_pilot=1, n_dev=2, edits_per_stream=12, loc_probe=4, loc_hidden=6, n_shots=2)


# ---- data -------------------------------------------------------------------------------------------------------
def test_hygiene_rule_examples():
    m = ec_data.malformed
    for bad in ('What', 'In', 'of solar eclipse occurred on May 8 92 BC', 'School of X an example of',
                'Hilton Farmhouse located in Whitsome as part of the Outbuilding', 'LU'):
        assert m(bad, f'What about {bad}?'), bad
    for good in ('Roll On', 'Time Is a Machine', 'solar eclipse of July 11, 1536 BC', 'The Hague',
                 'freemasonry in Indonesia', 'Nicholson Catholic College'):
        assert not m(good, f'What about {good}?'), good
    assert ec_data.hygiene_flags('Foo', 'What is bar?') == ['not_in_question']
    assert not m('Foo', 'What is bar?')     # not_in_question is reported only


def test_prepare_half_matches_rv09_corpus_extends_pools_and_filters(tmp_path):
    d = write_data(tmp_path / 'wbe')
    edits, windows, _ = load_edits(d, expect_steps=4)
    halves = split_halves(edits)
    assert sorted(set(halves.values())) == ['confirm', 'dev']
    base_dev = build_streams(edits, 4, half='dev', **SARGS)
    st, mu, info = ec_data.prepare_half(edits, 4, 'dev', 9, [0, 1, 2], 20, stream_args=SARGS)
    assert mu == corpus(spare_rows(edits, base_dev, halves, 'dev'), {0, 1, 2}, 20)
    for b in base_dev:
        s = st[b.name]
        assert s.edits == b.edits and s.loc_probe == b.loc_probe and s.shots == b.shots
        assert s.loc_hidden[:len(b.loc_hidden)] == b.loc_hidden    # the base pool is a prefix (no corpus overlap)
        assert info[b.name]['edits_other_half'] == 0
    cf, _, cinfo = ec_data.prepare_half(edits, 4, 'confirm', 9, [0, 1, 2], 20, stream_args=SARGS)
    assert sorted(cf) == ['C0', 'C1', 'CP0']
    dev_texts = ec_data.pool_texts(build_streams(edits, 4, half='dev', **dict(SARGS, loc_hidden=9)))
    for name, s in cf.items():
        texts = {norm(q) for q in s.loc_probe} | {norm(q) for q, _ in s.loc_hidden}
        texts |= {t for e in s.edits for t in ec_data._edit_texts(e)}
        assert not texts & dev_texts
    assert any(v['probes_other_half'] + v['hidden_other_half'] for v in cinfo.values())   # the shared questions


# ---- fixed-shape generation ----------------------------------------------------------------------------------------
def prompts_of(tok, qs, ctx='Q: Who is Loc1?\nA: c1\n'):
    return [tok.encode(fmt_prompt(q, ctx)) for q in qs]


QS = ['What is the code of Sub1?', 'Which code does Sub2 have?', 'What colour is Loc9y2?', 'Hi?',
      'What is the size of a rather long subject name here?', 'Yo?', 'What shape is S2x1y3?']


def test_fixed_shape_rows_do_not_depend_on_batch_neighbours():
    wm = make_wm()
    gen = FixedShapeGen(wm, WIDTHS, batch=3, max_new=6)
    ids = prompts_of(wm.tok, QS)
    a = gen.generate(ids)
    order = [6, 3, 0, 5, 1, 4, 2]
    b = gen.generate([ids[i] for i in order])
    for k, i in enumerate(order):
        assert (a[i].text, a[i].n_gen) == (b[k].text, b[k].n_gen)
    single = [gen.generate([x])[0] for x in ids]
    assert [(x.text, x.n_gen) for x in a] == [(x.text, x.n_gen) for x in single]


def test_fixed_shape_greedy_matches_the_no_cache_reference():
    wm = make_wm(dtype=torch.float64)
    gen = FixedShapeGen(wm, WIDTHS, batch=4, max_new=6)
    ids = prompts_of(wm.tok, QS[:4])
    got = gen.generate(ids)
    ref = wm._greedy_nocache(ids, 6)
    assert [x.text for x in got] == [t for t, _, _ in ref]


def test_relevant_drops_padding_dummies_and_late_decode_positions():
    width, lens, nctx, gen = 8, [5, 3], [2, 2], [3, 1]
    # row 0 starts at 3 (context 3-4, question 5-7), row 1 at 5 (context 5-6, question 7); row 2 is a dummy
    log = [(0, 8, [0, 0, 1, 1, 1, 2], [2, 4, 4, 6, 7, 7], [7, 8, 9, 9, 9, 1]),
           (1, 1, [0, 1], [0, 0], [5, 6]),                              # row 1 generated one token: too late
           (3, 1, [0], [0], [3])]                                       # row 0 generated 3 tokens: k=3 too late
    got = sorted(relevant(log, width, lens, nctx, gen))
    assert got == [(0, (5, 2)), (0, (8, 0)), (1, (9, 0)), (1, (9, 1))]
    with pytest.raises(RuntimeError):
        relevant([(0, 7, [0], [3], [1])], width, lens, nctx, gen)


def facts(n=4):
    from rv_editor import Fact
    return [Fact(f'What is the code of Sub{i}?', f'z{i}', f'Sub{i}') for i in range(n)]


def test_kt_routing_through_fixed_shapes_leaves_unfired_rows_untouched():
    wm = make_wm()
    ed = LoggedEditor(route_cfg(tau=0.3, probe_margin=None))
    ed.prepare(wm, corpus_texts())
    ed.edit(wm, facts(), '', seed=1)
    gen = FixedShapeGen(wm, WIDTHS, batch=3, max_new=5)
    gen.loggers = [ed]
    ids = prompts_of(wm.tok, QS)
    nctx = [len(wm.tok.encode('Q: Who is Loc1?\nA: c1\n'))] * len(ids)
    routed = gen.generate(ids, nctx)
    saved = dict(wm.extra)
    wm.extra = {}
    plain = gen.generate(ids, nctx)
    wm.extra = saved
    assert any(o.fired for o in routed)
    for r, p in zip(routed, plain):
        if not r.fired:
            assert r.text == p.text
    ed.detach()


# ---- GRACE ----------------------------------------------------------------------------------------------------------
def test_grace_codebook_add_split_expand():
    keys = torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.5, 0.0], [0.4, 0.0], [20.0, 0.0]])
    vals = torch.arange(10.0).reshape(5, 2)
    cb = ec_grace.build_codebook(keys, vals, ['a', 'b', 'c', 'a', 'd'], eps_init=1.0)
    # 0 add; 1 far -> add; 2 near 1 (0.5 <= 1 + 1), other answer -> split; 3 near 0, same answer -> expand; 4 add
    assert cb.owner == [0, 1, 2, 4]
    assert cb.splits == 1 and cb.expansions == 1 and cb.added == 4
    assert torch.allclose(cb.eps, torch.tensor([1.0, 0.25 - 1e-5, 0.25, 1.0]))
    assert torch.equal(cb.values, vals[[0, 1, 2, 4]])
    two = torch.tensor([[0.0, 0.0], [0.4, 0.0]])
    far = ec_grace.build_codebook(two, vals[:2], ['a', 'a'], eps_init=0.1)       # 0.4 > 0.1 + 0.1: a new key
    assert far.added == 2 and far.expansions == 0
    near = ec_grace.build_codebook(two, vals[:2], ['a', 'a'], eps_init=0.3)      # 0.4 <= 0.6, same answer
    assert near.added == 1 and near.expansions == 1 and float(near.eps[0]) == pytest.approx(0.4)   # coverage


def test_grace_keys_equal_eval_queries_and_hook_replaces_within_radius():
    wm = make_wm()
    gen = FixedShapeGen(wm, WIDTHS, batch=3, max_new=4)
    hook = ec_grace.GraceHook()
    hook.attach(wm, 1)
    gen.loggers = [hook]
    ids = prompts_of(wm.tok, QS)
    keys = ec_grace.fixed_keys(gen, 1, ids)
    seen = []
    probe = ec_grace.site(wm, 1).register_forward_pre_hook(lambda m, inp: seen.append(inp[0][:, -1].float().clone()))
    gen.generate([ids[2], ids[5], ids[0]])
    probe.remove()
    assert torch.equal(seen[0][0], keys[2]) and torch.equal(seen[0][2], keys[0])
    plain = gen.generate(ids)
    big = torch.full((1, wm.model.config.hidden_size), 3.0)
    hook.cb = ec_grace.Codebook(keys[2:3].clone(), big, torch.tensor([1e-4]), owner=[2])
    out = gen.generate(ids)
    assert (2, 1) in out[2].fired           # owner 2, at the question's last token in the prompt forward
    for i, (o, p) in enumerate(zip(out, plain)):
        if i != 2 and not o.fired:
            assert o.text == p.text
    hook.detach()


def test_grace_value_training_lowers_loss_and_batching_does_not_change_values():
    wm = make_wm(dtype=torch.float64)
    tok = wm.tok
    ctx = 'Q: Who is Loc1?\nA: c1\n'
    ps = [tok.encode(fmt_prompt(f.question, ctx)) for f in facts(3)]
    ts = [tok.encode(fmt_target(f.answer)) for f in facts(3)]
    a = ec_grace.train_values(wm, 1, ps, ts, n_iter=25, lr=1.0, seed=0, chunk=3)
    assert all(l1 < l0 for l0, l1 in zip(a.loss_first, a.loss_last))
    b = ec_grace.train_values(wm, 1, ps[:1], ts[:1], n_iter=25, lr=1.0, seed=0, chunk=1)
    assert torch.allclose(a.values[0], b.values[0], atol=1e-3)


def test_probe_scale_is_median_nearest_neighbour_distance():
    k = torch.tensor([[0.0], [1.0], [3.0], [7.0]])
    assert ec_grace.probe_scale(k) == pytest.approx(1.0)     # nearest: 1, 1, 2, 4; torch's median is the lower middle


# ---- retrieval ------------------------------------------------------------------------------------------------------
def test_retrieval_arms_and_context_order():
    sim = torch.tensor([[0.9, 0.2, 0.7], [0.1, 0.55, 0.3], [0.2, 0.1, 0.0]])
    assert ec_rag.retrieve(sim, 'rag:always:2') == [[0, 2], [1, 2], [0, 1]]
    assert ec_rag.retrieve(sim, 'rag:2:0.6') == [[0, 2], [], []]
    assert ec_rag.retrieve(sim, 'rag:1:0.5') == [[0], [1], []]
    thr = torch.tensor([0.95, 0.5, 0.6])
    assert ec_rag.retrieve(sim, 'ragp:0.00', thr) == [[2], [1], []]
    recs = [('Q a?', 'x'), ('Q b?', 'y'), ('Q c?', 'z')]
    assert ec_rag.context_with('S\n', recs, [0, 2]) == 'S\nQ: Q c?\nA: z\nQ: Q a?\nA: x\n'
    rt = ec_rag.record_thresholds(torch.eye(3), torch.tensor([[1.0, 0, 0], [0, 0.3, 0.954]]), 0.05, 0.5)
    assert torch.allclose(rt, torch.tensor([1.05, 0.5, 1.004]))


def tiny_encoder():
    from transformers import BertConfig, BertModel
    torch.manual_seed(0)
    tok = CharTok()
    m = BertModel(BertConfig(vocab_size=len(tok), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                             intermediate_size=32, max_position_embeddings=512)).eval()
    return ec_rag.MeanPoolEncoder(m, tok, 'cpu', max_len=300, batch=2)


def test_mean_pool_encoder_ignores_padding():
    enc = tiny_encoder()
    a = enc.encode(['short one', 'a much longer sentence that forces padding'])
    b = enc.encode(['short one'])
    assert torch.allclose(a[0], b[0], atol=1e-5)
    assert torch.allclose(a.norm(dim=1), torch.ones(2), atol=1e-5)


# ---- thresholds on KT entries --------------------------------------------------------------------------------------
def test_kt_thresholds_reproduce_the_editors_own_and_rad_rules():
    wm = make_wm()
    ed = LoggedEditor(route_cfg())
    ed.prepare(wm, corpus_texts())
    ctx = 'Q: Who is Loc1?\nA: c1\n'
    probes = ['What colour is Loc3?', 'Where is Loc4?', 'Who made Loc5?']
    states, labels = probe_positions(wm, ed, probes, ctx)
    ed.set_probes(wm, probes, ctx)
    ed.edit(wm, facts(), ctx, seed=1)
    keys, matched = wm.extra['route_k'], wm.extra['route_t']
    pm = probe_max(states, labels, keys, len(probes) + 1)
    thr = kt_thresholds(pm, len(probes), len(probes), 0.01, 0.5)
    assert torch.allclose(thr, matched, atol=1e-5)
    flagged = torch.tensor([True, False, False, True])
    h = with_hygiene(thr, flagged)
    assert float(h[0]) == OFF and float(h[3]) == OFF and torch.equal(h[1:3], thr[1:3])
    ed.detach()
    k = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.05], [0.0, 1.0], [1.0, -0.05]]), dim=-1)
    t, st = rad_thresholds(k, ['a', 'b', 'c', 'a'], c=0.9)
    assert st['splits'] == 1 and st['expansions'] == 1 and st['active'] == 3
    assert float(t[3]) == OFF and float(t[2]) == pytest.approx(0.9)
    assert float(t[0]) > 0.9 and float(t[1]) > 0.9       # split: radii shrank, thresholds rose


# ---- analysis -------------------------------------------------------------------------------------------------------
OBJ = dict(cap=0.02, cap_tol=0.005, guard_tol=0.01, delta_q=0.03, min_positive_streams=6)


def test_verdict_rule():
    assert verdict(0.05, 0.01, 0.09, 7, 1, 0.015, 0.02, OBJ) == 'reference ahead'
    assert verdict(0.05, 0.01, 0.09, 5, 3, 0.015, 0.02, OBJ).startswith('undetermined: interval')
    assert verdict(0.05, 0.01, 0.09, 7, 1, 0.03, 0.02, OBJ).startswith('undetermined: reference ahead')
    assert verdict(-0.05, -0.09, -0.01, 0, 8, 0.015, 0.02, OBJ) == 'opponent ahead'
    assert verdict(0.0, -0.02, 0.025, 4, 4, 0.015, 0.02, OBJ) == 'equivalent'
    assert verdict(0.0, -0.02, 0.025, 4, 4, 0.015, 0.03, OBJ).startswith('undetermined: equivalent')
    assert verdict(0.01, -0.04, 0.06, 5, 3, 0.015, 0.02, OBJ) == 'undetermined: interval too wide'


def test_select_rule():
    tab = {a: dict(mean=dict(coverage=c, damage=d)) for a, c, d in
           (('x', 0.70, 0.01), ('y', 0.75, 0.019), ('z', 0.80, 0.03), ('w', 0.75, 0.005))}
    assert select(['x', 'y', 'z', 'w'], tab, 0.02) == ('w', True)       # tie on coverage: lower damage
    assert select(['z'], tab, 0.02) == ('z', False)


def test_boot_means_paired_and_reproducible():
    rng = np.random.default_rng(1)
    mats = [rng.integers(0, 2, size=(2, 50)).astype(np.float32) for _ in range(3)]
    mats = [np.stack([m[0], m[0]]) for m in mats]
    choice = np.random.default_rng(0).integers(0, 3, size=(100, 3))
    a = boot_means(mats, choice, np.random.default_rng(5))
    b = boot_means(mats, choice, np.random.default_rng(5))
    assert np.array_equal(a, b) and np.array_equal(a[0], a[1])


def test_frontier_value_interpolates_on_the_pareto_frontier():
    pts = [(0.01, 0.5), (0.03, 0.7), (0.02, 0.4), (0.05, 0.9)]     # (0.02, 0.4) is dominated
    assert frontier_value(pts, 0.005) is None
    assert frontier_value(pts, 0.02) == pytest.approx(0.6)
    assert frontier_value(pts, 0.10) == pytest.approx(0.9)


# ---- end to end -------------------------------------------------------------------------------------------------------
def tiny_cfg(data_dir: Path) -> dict:
    arms = ['none', 'kt:4:0.01:0.5', 'kt:2:0.01:0.5', 'kthyg:4:0.01:0.5', 'hyg:0.5', 'hyg:0.9', 'rad:0.5',
            'grace:0:s0.5', 'grace:1:s0.5', 'grace:1:a1.0', 'rag:always:2', 'rag:1:0.5', 'ragp:0.00']
    return dict(
        status='test', data_dir=str(data_dir), expect_steps=4, model_path='unused', dtype='float32', torch_seed=0,
        sites=['model.layers.{i}.mlp.down_proj'], layers='all', rank=2, scale=1.0, clip=1.0,
        streams=dict(args=SARGS, loc_hidden_eval=9, cap=150,
                     dev=dict(half='dev', names=['D0', 'D1']), test=dict(half='confirm', names=['C0', 'C1']),
                     smoke=dict(half='confirm', names=['CP0'], cap=2, loc_candidates=4, order_check=True),
                     mu_corpus=dict(buckets=[0, 1, 2], n=20)),
        eval=dict(widths=list(WIDTHS), batch=4, max_new=5),
        local=dict(mode='route', layers=[0], tau=0.5, probe_margin=0.01, massive=200.0, v_steps=2, v_lr=0.5,
                   v_wd=0.5, clamp=0.75, kl=0.0625, stop_loss=0.05, n_rand=1, rand_len=3, use_context=True,
                   chunk=4, key_batch=8),
        write_seed=0, kt_default='kt:4:0.01:0.5',
        arms=dict(smoke=arms, dev=arms, test=None),
        grace=dict(n_iter=3, edit_lr=1.0, chunk=4, seed=0),
        rag=dict(model_path='unused', max_len=300, batch=4, probe_floor=0.0),
        objective=dict(primary='final_rephrase_coverage', cap=0.02, delta_q=0.03, cap_tol=0.005, guard_tol=0.01,
                       min_positive_streams=1, bootstrap=50, bootstrap_seed=0, frontier_caps=[0.01, 0.02]),
        timing=dict(stream='C0', n64=4, n1=2, repeats=1))


def factory(tok_model):
    def make(cfg, phase):
        tok = CharTok()
        env = ec_run.ECEnv(cfg, phase, model=tiny_model(len(tok), 0), tok=tok, encoder=tiny_encoder())
        # make every locality candidate 'known': its answer is the base model's own answer in this layout
        for name, s in list(env.streams.items()):
            ctx = shots_text(s.shots)
            items = loc_items(list(s.loc_hidden))
            ids, nctx = plain_prompts(env.wm.tok, items, ctx)
            outs = env.gen.generate(ids, nctx)
            env.streams[name] = dataclasses.replace(s, loc_hidden=tuple((q, o.text) for (q, _), o in
                                                                        zip(s.loc_hidden, outs)))
        return env
    return make


def test_end_to_end_dev_tune_test_analyze(tmp_path, monkeypatch):
    for k in ('RV_DATA_DIR', 'RV_MODEL_PATH', 'EC_RETRIEVER_PATH'):   # the host run script exports these
        monkeypatch.delenv(k, raising=False)
    d = write_data(tmp_path / 'wbe')
    cfg = tiny_cfg(d)
    cpath = tmp_path / 'config_dev.json'
    cpath.write_text(json.dumps(cfg), encoding='utf-8')
    ec_run.cmd_run(cpath, tmp_path / 'smoke', 'smoke', env_factory=factory(None))
    sm = ec_run.read_gz(tmp_path / 'smoke' / 'stream_CP0.json.gz')
    assert sm['checks']['screen_reversed_identical'] and sm['checks']['none_known_all_right']
    assert all(v == 0 for v in sm['checks']['unfired_differs_from_none'].values())
    ec_run.cmd_run(cpath, tmp_path / 'dev', 'dev', env_factory=factory(None))
    test_cfg = tmp_path / 'config_test.json'
    ec_run.cmd_tune(tmp_path / 'dev', cpath, test_cfg)
    tc = json.loads(test_cfg.read_text(encoding='utf-8'))
    assert set(tc['selected']) >= {'kt', 'hyg', 'rad', 'grace', 'rag', 'rag_gated', 'ragp'}
    layers = {a.split(':')[1] for a in tc['arms']['test'] if a.startswith('grace:')}
    assert len(layers) == 1
    ec_run.cmd_run(test_cfg, tmp_path / 'test', 'test', env_factory=factory(None))
    res = ec_run.cmd_analyze(tmp_path / 'test', tmp_path / 'dev')
    assert set(res['primary']) == {'kt_vs_rag', 'kt_vs_grace', 'kt_vs_hyg'}
    assert all('verdict' in v for v in res['primary'].values())
    rec = ec_run.read_gz(tmp_path / 'test' / 'stream_C0.json.gz')
    assert rec['timing'] and set(rec['timing']) == set(tc['timing']['arms'])
    assert rec['checks']['determinism']['loc_reversed_same']
    assert (tmp_path / 'test' / 'manifest.json').exists()
