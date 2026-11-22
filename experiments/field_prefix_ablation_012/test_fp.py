"""FP-12 tests: CPU only, SO-08's tiny random Qwen3 and character tokenizer, EC-11's synthetic WikiBigEdit-style data.
The real Qwen3 tokenizer is used where it is available locally (skipped otherwise)."""
import json
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fp_data  # noqa: E402
import fp_core  # noqa: E402
import fp_run  # noqa: E402
import fp_analysis  # noqa: E402
from fp_core import OFF, FPEditor, FPGen, Row, arm_vector, at_step  # noqa: E402
from ec_eval import FixedShapeGen  # noqa: E402
from kt_core import LoggedEditor  # noqa: E402
from rv_editor import Fact, lookup  # noqa: E402
from so_data import shots_text  # noqa: E402
from so_model import fmt_prompt  # noqa: E402
from test_ec import SARGS, WIDTHS, corpus_texts, make_wm, route_cfg, write_data  # noqa: E402
from test_so import CharTok, tiny_model  # noqa: E402

CTX = 'Q: Who is Loc1?\nA: c1\nQ: Who is Loc2?\nA: c2\n'


def real_tok():
    p = Path(os.environ.get('RV_MODEL_PATH') or fp_data.MODEL)
    if not (p / 'tokenizer.json').exists():
        pytest.skip('real tokenizer not available')
    return fp_data.load_tokenizer(p)


# ---- anchors and positions ---------------------------------------------------------------------------------------
def test_orig_anchor_is_rv09_lookup_and_spans_align_to_tokens():
    tok = CharTok()
    for q, subj in (('What is the code of Sub1?', 'Sub1'), ('In which county is X located in Y?', 'In'),
                    ('What is carried by A, B?', 'A B'), ('Who is YG\'s partner?', 'YG')):
        text = fp_data.bare_prompt(q)
        ids, idx, found = lookup(tok, text, subj, 0)
        a = fp_data.orig_anchor(tok, q, subj)
        assert (a.token, a.found) == (idx, found)
        if found:
            k = q.strip().rfind(subj)
            s = fp_data.span_anchor(tok, q, (k, k + len(subj)))
            assert s.token == a.token and text[s.char] == subj[-1]
    with pytest.raises(ValueError):
        fp_data.span_anchor(tok, 'Short?', (3, 99))


def test_real_tokenizer_anchors_agree_with_lookup():
    tok = real_tok()
    for q, subj in (('Which territorial jurisdiction does Sozialgericht Freiburg have power over?', 'Sozialgericht Freiburg'),
                    ('What is the orthologous gene of wbp-2 in another species?', 'What'),
                    ('In which historic county is 40 Charlotte Square located in Edinburgh?',
                     '40 Charlotte Square located in Edinburgh')):
        text = fp_data.bare_prompt(q)
        _, idx, found = lookup(tok, text, subj, 0)
        a = fp_data.orig_anchor(tok, q, subj)
        assert (a.token, a.found) == (idx, found)
    q = 'In which historic county is 40 Charlotte Square located in Edinburgh?'
    k = q.find('40 Charlotte Square')
    s = fp_data.span_anchor(tok, q, (k, k + len('40 Charlotte Square')))
    text = fp_data.bare_prompt(q)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    a, b = enc['offset_mapping'][s.token]
    assert 'Square'.startswith(text[a:b].strip()) or text[a:b].strip().endswith('Square')


def test_marker_tokens_and_blocked_positions():
    tok = CharTok()
    text = fmt_prompt('What is it?', CTX)
    m = fp_core.marker_tokens(tok, text, len(CTX))
    q0 = len(CTX)
    assert m == [q0, q0 + 1, q0 + 2, len(text) - 3, len(text) - 2, len(text) - 1]
    nctx = len(tok.encode(CTX))
    assert fp_core.blocked_positions('on', nctx, m) == ()
    assert fp_core.blocked_positions('off', nctx, m) == tuple(range(nctx))
    assert set(fp_core.blocked_positions('strict', nctx, m)) == set(range(nctx)) | set(m)


def test_real_tokenizer_markers_are_only_the_fixed_marks():
    tok = real_tok()
    ctx = 'Q: Who wrote Hamlet?\nA: William Shakespeare\nQ: What is the capital of Peru?\nA: Lima\n'
    text = fmt_prompt('What is the code of Sub1?', ctx)
    m = fp_core.marker_tokens(tok, text, len(ctx))
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    pieces = [text[a:b] for a, b in (enc['offset_mapping'][i] for i in m)]
    # the space after 'Q:' joins the next word and the newline joins the question's last token: neither is a marker
    assert ''.join(p.strip() for p in pieces) == 'Q:A:'
    assert all(p.strip() in ('Q', ':', 'A', 'Q:', 'A:') for p in pieces)
    assert tok.encode(text, add_special_tokens=False)[:len(tok.encode(ctx, add_special_tokens=False))] == \
        tok.encode(ctx, add_special_tokens=False)


# ---- routing hook: per-row thresholds and blocking --------------------------------------------------------------------
def prompts(tok, qs, ctx=CTX):
    return [tok.encode(fmt_prompt(q, ctx)) for q in qs]


QS = ['What is the code of Sub1?', 'Which code does Sub2 have?', 'What colour is Loc9y2?', 'Hi?',
      'What is the size of a rather long subject name here?', 'Yo?', 'What shape is S2x1y3?']


def written(ed_cls=FPEditor, n=4, tau=0.3):
    wm = make_wm()
    ed = ed_cls(route_cfg(tau=tau, probe_margin=None))
    ed.prepare(wm, corpus_texts())
    ed.edit(wm, [Fact(f'What is the code of Sub{i}?', f'z{i}', f'Sub{i}') for i in range(n)], '', seed=1)
    return wm, ed


def test_per_row_thresholds_match_one_vector_and_mix_freely():
    wm, ed = written()
    gen = FPGen(wm, ed, WIDTHS, batch=3, max_new=5)
    ids = prompts(wm.tok, QS)
    nctx = len(wm.tok.encode(CTX))
    E = wm.extra['route_k'].shape[0]
    v1, v2 = torch.full((E,), 0.3), torch.full((E,), 0.9)
    plain = FixedShapeGen(wm, WIDTHS, batch=3, max_new=5)
    plain.loggers = [ed]
    ref = {}
    for k, v in ((0, v1), (1, v2)):
        wm.extra['route_t'] = v
        ref[k] = plain.generate(ids, [nctx] * len(ids))
    rows = [Row(x, nctx, (i % 2)) for i, x in enumerate(ids)]
    wm.extra['route_t'] = v1
    got = gen.run(rows, torch.stack([v1, v2]))
    for i, o in enumerate(got):
        r = ref[i % 2][i]
        assert (o.text, o.n_gen, o.fired) == (r.text, r.n_gen, r.fired)
    assert any(o.fired for o in got)
    ed.detach()


def block_state(wm, ed, ids, pos):
    """The layer's raw output at one prompt position (routing off), centred and normalised as a key."""
    from torch.nn import functional as F
    block = ed._block(wm, ed.cfg.layers[-1])
    got = {}

    def hook(mod, inp, out):
        got['h'] = (out[0] if isinstance(out, tuple) else out).float()[0, pos]
    h = block.register_forward_hook(hook)
    off, ed.route_off = ed.route_off, True
    try:
        with torch.no_grad():
            wm.model(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        h.remove()
        ed.route_off = off
    return F.normalize(got['h'] - ed.mu, dim=-1)


def test_a_key_that_fires_only_on_the_context_is_silenced_by_blocking():
    wm = make_wm()
    ed = FPEditor(route_cfg(tau=0.3, probe_margin=None))
    ed.prepare(wm, corpus_texts())
    tok = wm.tok
    ids = prompts(tok, ['What is the size of Sub3?'])[0]
    nctx = len(tok.encode(CTX))
    key = block_state(wm, ed, ids, 7)                    # a context position (inside 'Who is Loc1?')
    wm.extra = {'route_k': key[None], 'route_v': torch.full((1, key.shape[0]), 25.0), 'route_t': torch.tensor([0.999])}
    gen = FPGen(wm, ed, WIDTHS, batch=2, max_new=5)
    thr = torch.tensor([[0.999]])
    first = []
    head = wm.model.get_output_embeddings().register_forward_hook(
        lambda mod, inp, out: first.append(out[0].detach().clone()) if len(first) % 100 == 0 or True else None)

    def run(block, t):
        first.clear()
        o = gen.run([Row(ids, nctx, 0, block)], t)[0]
        return o, first[0]                                   # logits of the prompt forward
    on, l_on = run((), thr)
    off, l_off = run(fp_core.blocked_positions('off', nctx, []), thr)
    none, l_none = run((), torch.tensor([[OFF]]))
    head.remove()
    assert on.fired and all(g == 0 for _, g in on.fired)          # fires, and only in the context
    assert not off.fired and off.text == none.text and off.n_gen == none.n_gen
    assert torch.equal(l_off, l_none)                               # blocked: the residual is exactly zero
    assert not torch.allclose(l_on, l_none)                         # allowed: it reaches the answer position
    assert ids == prompts(tok, ['What is the size of Sub3?'])[0]    # same prompt bytes in both columns
    ed.detach()


def test_anchor_override_moves_the_key_and_leaves_other_facts_alone():
    wm, ed = written(LoggedEditor)
    base_k, base_v = wm.extra['route_k'].clone(), wm.extra['route_v'].clone()
    ed.detach()
    wm2 = make_wm()
    ed2 = FPEditor(route_cfg(tau=0.3, probe_margin=None))
    ed2.prepare(wm2, corpus_texts())
    q = 'What is the code of Sub0?'
    ed2.anchors[(q, 'code')] = (q.find('code'), q.find('code') + 4)
    fs = [Fact(f'What is the code of Sub{i}?', f'z{i}', f'Sub{i}') for i in range(4)]
    fs[0] = Fact(q, 'z0', 'code')
    ed2.edit(wm2, fs, '', seed=1)
    k2, v2 = wm2.extra['route_k'], wm2.extra['route_v']
    assert not torch.allclose(k2[0], base_k[0])
    assert torch.allclose(k2[1:], base_k[1:], atol=1e-6) and torch.allclose(v2[1:], base_v[1:], atol=1e-5)
    ed2.detach()


# ---- arms -------------------------------------------------------------------------------------------------------------
def test_shuffles_keep_batch_multisets_and_refused_slots():
    matched = torch.tensor([0.91, 0.95, 0.97, 0.9, 0.93, 0.99, 0.92])
    ends = [3, 7]
    active = torch.tensor([True, False, True, True, True, False, True])
    for p in range(8):
        v = arm_vector(f'shuffled:{p}', matched, ends, active, 'D9')
        m = arm_vector('matched', matched, ends, active, 'D9')
        assert fp_core.multisets_equal(v, m, ends)
        assert bool((v[~active] == OFF).all())
        assert torch.equal(v, arm_vector(f'shuffled:{p}', matched, ends, active, 'D9'))
    u = arm_vector('uniform:0.94', matched, ends, active, 'D9')
    assert bool((u[active] == 0.94).all()) and bool((u[~active] == OFF).all())
    assert bool((at_step(u, ends, 0)[3:] == OFF).all()) and torch.equal(at_step(u, ends, 1), u)


def test_batched_interventions_equal_one_item_at_a_time():
    wm, ed = written(n=4, tau=0.2)
    gen = FPGen(wm, ed, WIDTHS, batch=3, max_new=5)
    nctx = len(wm.tok.encode(CTX))
    ids = prompts(wm.tok, QS)
    E = wm.extra['route_k'].shape[0]
    thr = torch.full((E,), 0.2)
    rows = [Row(x, nctx, 0, ()) for x in ids]
    outs = gen.run(rows, thr[None])
    damaged = [i for i, o in enumerate(outs) if o.fired][:3]
    fired = {i: list(outs[i].fired) for i in damaged}
    got = fp_core.intervene(gen, thr, rows, damaged, fired, max_single=8)
    for rec in got:
        i = rec['i']
        solo = fp_core.intervene(gen, thr, rows, [i], fired, max_single=8)[0]
        assert rec == solo
        assert rec['rerun'] == fp_core.norm(outs[i].text)
    ed.detach()


# ---- analysis rules ------------------------------------------------------------------------------------------------------
def test_boot_is_paired_reproducible_and_brackets_the_mean():
    import numpy as np
    per = {'A': np.array([0.0, 1.0, 1.0, 0.0]), 'B': np.array([1.0, 1.0, 0.0, 1.0, 1.0])}
    a = fp_analysis.boot(per, 500, 0)
    assert a == fp_analysis.boot(per, 500, 0)
    m = (per['A'].mean() + per['B'].mean()) / 2
    assert a[0] <= m <= a[1]
    zero = {'A': np.zeros(5), 'B': np.zeros(3)}
    assert fp_analysis.boot(zero, 100, 0) == (0.0, 0.0)


def test_gate_rule():
    d = dict(mean=0.03, ci=(0.01, 0.05), positive=3, n=4)
    q, r = dict(mean=-0.01), dict(mean=0.0)
    assert fp_analysis.gate(d, q, r)['passed']
    assert not fp_analysis.gate(dict(d, positive=2), q, r)['passed']
    assert not fp_analysis.gate(dict(d, ci=(-0.01, 0.05)), q, r)['passed']
    assert not fp_analysis.gate(d, dict(mean=-0.03), r)['passed']
    assert not fp_analysis.gate(dict(d, mean=0.019), q, r)['passed']


# ---- end to end ----------------------------------------------------------------------------------------------------------
def tiny_cfg(data_dir: Path, spans: str) -> dict:
    return dict(
        status='test', data_dir=str(data_dir), expect_steps=4, model_path='unused', dtype='float32', torch_seed=0,
        sites=['model.layers.{i}.mlp.down_proj'], layers='all', rank=2, scale=1.0, clip=1.0,
        streams=dict(args=dict(SARGS, loc_hidden=6), mu_corpus=dict(buckets=[0, 1, 2], n=20)),
        phases=dict(smoke=dict(streams=['P0'], cap=2, order_check=True, intervention_check=True),
                    dev=dict(streams=['D0'], cap=150),
                    test=dict(streams=['D1'], cap=150)),
        panels=dict(max_cur=4, ret_per_step=2, max_loc=6, max_new=5, gen_batch=4),
        eval=dict(widths=list(WIDTHS), batch=4, max_new=5),
        local=dict(mode='route', layers=[0], tau=0.5, probe_margin=0.01, massive=200.0, v_steps=2, v_lr=0.5,
                   v_wd=0.5, clamp=0.75, kl=0.0625, stop_loss=0.05, n_rand=1, rand_len=3, use_context=True,
                   chunk=4, key_batch=8),
        write_seed=0, spans=spans,
        arms=dict(n_shuffles=2, uniform=[0.5, 0.7, 0.9]),
        determinism_arm='orig/on:matched', interventions=dict(arms=['uniform:0.50', 'matched'], max_single=8),
        rule=dict(bootstrap=50, bootstrap_seed=0))


def synthetic_spans(data_dir: Path, path: Path, cfg: dict):
    from so_data import load_edits
    edits, windows, _ = load_edits(data_dir, expect_steps=4)
    streams, _, _ = fp_data.kt10_half(edits, len(windows), cfg['streams']['args'], cfg['streams']['mu_corpus'])
    tok = CharTok()
    final = {}
    for s in streams.values():
        for j, e in enumerate(s.edits):
            q = e.update.strip()
            if j % 5 == 1:                       # re-anchor on the word 'the' (a deliberately different token)
                k = q.find('the')
                span, text = [k, k + 3], 'the'
            else:
                k = q.rfind(e.subject)
                span, text = [k, k + len(e.subject)], e.subject
            o = fp_data.orig_anchor(tok, e.update, e.subject)
            c = fp_data.span_anchor(tok, e.update, tuple(span))
            final[e.uid] = dict(span=span, text=text, unlocatable=(j % 7 == 3), changed=c.token != o.token)
    path.write_text(json.dumps(dict(final=final)), encoding='utf-8')


def factory(cfg):
    import dataclasses
    tok = CharTok()
    env = fp_run.FPEnv(cfg, model=tiny_model(len(tok), 0), tok=tok)
    for name, s in list(env.streams.items()):     # every candidate 'known': its answer is the base model's own
        ctx = shots_text(s.shots)
        nctx = len(tok.encode(ctx))
        ids = [tok.encode(fmt_prompt(q, ctx)) for q, _ in s.loc_hidden]
        outs = env.gen.generate(ids, [nctx] * len(ids))
        env.streams[name] = dataclasses.replace(s, loc_hidden=tuple((q, o.text) for (q, _), o in zip(s.loc_hidden, outs)))
    return env


def test_end_to_end_smoke_dev_tune_test_analyze(tmp_path, monkeypatch):
    for k in ('RV_DATA_DIR', 'RV_MODEL_PATH'):    # the host run script exports these
        monkeypatch.delenv(k, raising=False)
    d = write_data(tmp_path / 'wbe')
    spans = tmp_path / 'spans_final.json'
    cfg = tiny_cfg(d, str(spans))
    synthetic_spans(d, spans, cfg)
    cpath = tmp_path / 'config_dev.json'
    cpath.write_text(json.dumps(cfg), encoding='utf-8')
    fp_run.cmd_run(cpath, tmp_path / 'smoke', 'smoke', env_factory=factory)
    sm = fp_run.read_gz(tmp_path / 'smoke' / 'stream_P0.json.gz')
    c = sm['checks']
    assert c['screen_reversed_identical'] and c['none_known_all_right'] and c['determinism']['loc_reversed_same']
    assert all(v == 0 for v in c['unfired_differs_from_none'].values())
    assert c['shuffle_multisets_equal'] and c['refused_never_active'] and c['corr_unchanged_bitwise'] is not False
    assert c['blocked_cells_never_fire_in_context']
    assert c.get('interventions_batched', dict(equal_single=True))['equal_single']
    fp_run.cmd_run(cpath, tmp_path / 'dev', 'dev', env_factory=factory)
    tpath = tmp_path / 'config_test.json'
    fp_run.cmd_tune(tmp_path / 'dev', cpath, tpath)
    tc = json.loads(tpath.read_text(encoding='utf-8'))
    assert set(tc['selected_uniform']) == set(fp_run.CELLS)
    fp_run.cmd_run(tpath, tmp_path / 'test', 'test', env_factory=factory)
    fp_run.cmd_analyze(tmp_path / 'test', tmp_path / 'dev')
    res = json.loads((tmp_path / 'test' / 'analysis.json').read_text(encoding='utf-8'))
    assert set(res['decomposition']) >= {'corr_off', 'orig_on_minus_corr_off', 'interaction', 'field_on', 'mask_orig'}
    rec = fp_run.read_gz(tmp_path / 'test' / 'stream_D1.json.gz')
    assert rec['tables']['changed'] and rec['tables']['refused']
    n_arms = 1 + 4 * (1 + 2 + 3) + (1 + 2 + 1) + (1 + 2)
    assert len(rec['arms']) == n_arms
    assert (tmp_path / 'test' / 'manifest.json').exists()
    import fp_report
    fp_report.main(str(tmp_path / 'test'), str(tmp_path / 'TABLES.md'))
    assert 'Δ(正/关)' in (tmp_path / 'TABLES.md').read_text(encoding='utf-8')
    import importlib.util
    spec = importlib.util.spec_from_file_location('recount', HERE / 'checks' / 'recount.py')
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    rc.main(str(tmp_path / 'test'), str(tmp_path / 'dev'), str(tmp_path / 'recount.json'))
    out = json.loads((tmp_path / 'recount.json').read_text(encoding='utf-8'))
    assert out['max_abs_diff'] < 1e-9 and out['tuning_matches_config']
