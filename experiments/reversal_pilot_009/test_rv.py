"""RV-09 tests: CPU only, SO-08's tiny random Qwen3 and synthetic WikiBigEdit-style files."""
import dataclasses
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rv_editor import EditWritable, Fact, LocalCfg, LocalEditor, lookup  # noqa: E402
from rv_analysis import a1_gate, a2_verdict, family_estimates  # noqa: E402
from rv_policy import (RunCfg, SubjView, Writer, branch_pair, cap_stream, futures, parse_writer,  # noqa: E402
                       trajectory, value, write)
from so_data import FactIndex, build_streams, load_edits, norm, shots_text  # noqa: E402
from so_model import fmt_prompt, fmt_target  # noqa: E402
from so_policy import EvalCfg, WriteCfg, base_locality  # noqa: E402
from test_so import CharTok, tiny_model, write_dataset  # noqa: E402

SITES = ['model.layers.0.mlp.down_proj', 'model.layers.1.mlp.down_proj']


def make_wm(seed=0):
    tok = CharTok()
    return EditWritable(tiny_model(len(tok), seed), tok, SITES, rank=2, scale=1.0, clip=1.0)


def corpus_texts():
    return [f'Q: What is the code of Thing{i}?\nA:' for i in range(40)]


def route_editor(wm, **kw):
    ed = LocalEditor(LocalCfg(**dict(dict(mode='route', layers=(0,), tau=0.5, v_steps=4, n_rand=1, rand_len=3,
                                          chunk=4, key_batch=8), **kw)))
    ed.prepare(wm, corpus_texts())
    return ed


@pytest.fixture
def small(tmp_path):
    d = write_dataset(tmp_path / 'wbe4', 4)
    edits, windows, _ = load_edits(d, expect_steps=4)
    index = FactIndex(edits)
    streams = build_streams(edits, len(windows), n_pilot=1, n_dev=2, edits_per_stream=12, loc_probe=4, loc_hidden=4,
                            n_shots=1)
    return streams, index, edits


def small_rc():
    return RunCfg(write=WriteCfg(epochs=1, lr=1e-2, batch_size=8, accept_k=4, distill_k=4, max_new=6, gen_batch=16),
                  ev=EvalCfg(max_cur=3, ret_per_step=2, max_loc=4, max_new=6, gen_batch=16))


def logits_of(wm, text):
    ids = torch.tensor([wm.tok.encode(text)])
    with torch.no_grad():
        return wm.model(input_ids=ids, use_cache=False).logits


# ---- model layer ---------------------------------------------------------------------------------------------------
def test_extra_state_follows_snapshots():
    wm = make_wm()
    wm.extra['x'] = torch.ones(3)
    snap = wm.snapshot()
    wm.extra['x'] += 1
    wm.extra['y'] = torch.zeros(2)
    wm.restore(snap)
    assert set(wm.extra) == {'x'} and torch.equal(wm.extra['x'], torch.ones(3))
    wm.restore(wm.empty_state())
    assert wm.extra == {}


def test_commit_factors_is_an_exact_weight_update():
    wm = make_wm()
    text = 'Q: hi?\nA:'
    base = logits_of(wm, text)
    m = wm.mods[SITES[1]]
    a = torch.randn(3, m.base.in_features) * 0.1
    b = torch.randn(m.base.out_features, 3) * 0.1
    wm.commit_factors(SITES[1], a, b)
    stacked = logits_of(wm, text)
    snap = wm.snapshot()
    wm.restore(wm.empty_state())
    w0 = m.base.weight.detach().clone()
    with torch.no_grad():
        m.base.weight += b @ a
    direct = logits_of(wm, text)
    with torch.no_grad():
        m.base.weight.copy_(w0)
    wm.restore(snap)
    assert torch.allclose(stacked, direct, atol=1e-5)
    assert not torch.allclose(stacked, base)
    with pytest.raises(ValueError):
        wm.commit_factors(SITES[1], a[:, :5], b)


def test_lookup_subject_last_token_and_fallback():
    tok = CharTok()
    text = 'Q: What is the code of Sub3?\nA:'
    ids, idx, found = lookup(tok, text, 'Sub3', 0)
    assert found and tok.decode([ids[idx]]) == '3'
    ids, idx, found = lookup(tok, text, 'sub3', 0)          # case-insensitive match
    assert found and tok.decode([ids[idx]]) == '3'
    ids, idx, found = lookup(tok, text, 'Nobody', 0)        # fallback: last character of the question
    assert not found and tok.decode([ids[idx]]) == '?'
    ctx = 'Q: Sub3 here?\nA: x\n'
    ids, idx, found = lookup(tok, ctx + 'Q: code of Sub3?\nA:', 'Sub3', len(ctx))
    assert found and idx > len(tok.encode(ctx))             # never matches inside the context


# ---- editor ------------------------------------------------------------------------------------------------------
def nll(wm, q, a):
    per, _ = wm._per_example_ce([(fmt_prompt(q), fmt_target(a), 1.0)])
    return float(per[0])


def test_memit_solve_realises_its_targets_and_keeps_editor_state():
    """With negligible regularisation, one edit layer and only the bare template, the committed update maps each
    key exactly to its residual, so the edited layer's output reaches the optimised target z."""
    torch.manual_seed(0)
    wm = make_wm()
    ed = LocalEditor(LocalCfg(mode='memit', layers=(0,), lam=1e-9, v_steps=6, v_lr=0.5, n_rand=0, chunk=4,
                              key_batch=8))
    ed.prepare(wm, corpus_texts())
    facts = [Fact('What is the code of Sub1?', 'zz9', 'Sub1'), Fact('What is the code of Sub2?', 'qq7', 'Sub2')]
    before = [nll(wm, f.question, f.answer) for f in facts]
    rec = ed.edit(wm, facts, '', seed=3)
    assert rec['z_gap_after'] < 1e-3, rec
    assert rec['z_nll_mean'] < sum(before) / len(before)   # the target vectors lower the answer loss
    assert set(wm.extra) == {'Cp0'} and wm.committed_rank()[SITES[0]] == 2
    snap = wm.snapshot()
    ed.edit(wm, [Fact('What is the code of Sub3?', 'aa1', 'Sub3')], '', seed=4)
    assert wm.committed_rank()[SITES[0]] == 3
    wm.restore(snap)
    assert wm.committed_rank()[SITES[0]] == 2


def test_route_mode_is_exact_without_entries_and_entries_follow_snapshots():
    wm = make_wm()
    base = logits_of(wm, 'Q: What is the code of Sub1?\nA:')
    ed = route_editor(wm)
    assert torch.equal(logits_of(wm, 'Q: What is the code of Sub1?\nA:'), base)   # hook, no entries
    snap0 = wm.snapshot()
    facts = [Fact('What is the code of Sub1?', 'zz9', 'Sub1')]
    rec = ed.edit(wm, facts, '', seed=1)
    assert rec['entries'] == 1 and wm.extra['route_k'].shape[0] == 1
    edited = logits_of(wm, 'Q: What is the code of Sub1?\nA:')
    assert not torch.allclose(edited, base)
    ed.route_off = True
    assert torch.equal(logits_of(wm, 'Q: What is the code of Sub1?\nA:'), base)
    ed.route_off = False
    snap1 = wm.snapshot()
    wm.restore(snap0)
    assert torch.equal(logits_of(wm, 'Q: What is the code of Sub1?\nA:'), base)
    wm.restore(snap1)
    assert torch.equal(logits_of(wm, 'Q: What is the code of Sub1?\nA:'), edited)
    ed.detach()


def test_route_probe_thresholds_block_matches_on_probe_positions():
    wm = make_wm()
    ed = route_editor(wm, tau=0.0, probe_margin=0.01)
    q = 'What is the code of Sub1?'
    ed.set_probes(wm, [q], '')               # the edited question itself is a public probe
    base = logits_of(wm, fmt_prompt(q))
    rec = ed.edit(wm, [Fact(q, 'zz9', 'Sub1')], '', seed=1)
    assert rec['raised_thresholds'] == 1
    assert torch.equal(logits_of(wm, fmt_prompt(q)), base)   # cannot fire where the probes are
    ed.detach()


# ---- policy ------------------------------------------------------------------------------------------------------
def test_writers_record_and_replay_has_subjects(small):
    streams, index, _ = small
    s = streams[0]
    wm = make_wm()
    ed = route_editor(wm)
    t = next(t for t in range(1, s.steps) if SubjView(s, t).batch() and SubjView(s, t).latest_history())
    view = SubjView(s, t)
    assert all(f.subject for f in view.batch_facts())
    rep = view.replay_facts(3, 0)
    assert [(f.question, f.answer) for f in rep] == view.replay(3, 0)
    rc = small_rc()
    for spec in ('lora_plain', 'lora_replay:2', 'local', 'local_replay:2'):
        wm.restore(wm.empty_state())
        rec = write(wm, ed if spec.startswith('local') else None, view, parse_writer(spec), rc.write, 5)
        assert rec['action'] == spec
    ed.detach()


def test_futures_and_paired_branches(small):
    streams, index, edits = small
    s = streams[0]
    t = 1
    used = {e.uid for x in streams for e in x.edits}
    spare = [dataclasses.replace(e, uid=e.uid + 'x', subject_id=e.subject_id + 'x') for e in edits
             if e.step > t and e.uid in used][:12]
    from rv_policy import bucket_of
    b = bucket_of(spare[0].subject_id)
    spare = [e for e in spare if bucket_of(e.subject_id) == b]
    futs = futures(s, spare, b, t, 2, 3)
    assert len(futs) == 2
    hist = [e.uid for e in s.edits if e.step <= t]
    subj = []
    for f in futs:
        assert [e.uid for e in f.edits if e.step <= t] == hist
        subj.append({e.subject_id for e in f.edits if e.step > t})
    assert not (subj[0] & subj[1])
    wm = make_wm()
    ed = route_editor(wm)
    rc = small_rc()
    base_loc = base_locality(wm, s, rc.ev)
    traj = trajectory(wm, ed, s, index, Writer('local'), rc, 0, base_loc, max_steps=t, evaluate_steps=False)
    assert len(traj['steps']) == t
    state = wm.snapshot()
    br = branch_pair(wm, ed, futs, index, state, t, (Writer('local'), Writer('lora_plain')), Writer('local'), rc,
                     base_loc)
    assert set(br) == {'local', 'lora_plain'}
    for a in br.values():
        assert len(a['futures']) == 2 and all(x['value'] is not None for x in a['futures'])
    # the same action twice gives identical values in every future (paired, deterministic)
    br2 = branch_pair(wm, ed, futs, index, state, t, (Writer('local'), Writer('local_replay', 0)), Writer('local'),
                      rc, base_loc)
    va = [x['value'] for x in br2['local']['futures']]
    vb = [x['value'] for x in br2['local_replay:0']['futures']]
    assert va == vb
    ed.detach()


def test_cap_stream_keeps_at_most_cap_per_step(small):
    streams, _, _ = small
    s = cap_stream(streams[0], 2)
    for t in range(s.steps):
        assert len([e for e in s.edits if e.step == t]) <= 2
    assert cap_stream(streams[0], None) is streams[0]


# ---- analysis ----------------------------------------------------------------------------------------------------
def test_family_estimates_detect_a_planted_reversal():
    dmat = {0: [0.05, 0.04, 0.06, 0.05], 1: [-0.05, -0.06, -0.04, -0.05], 2: [0.04, 0.05, 0.05, 0.04]}
    e = family_estimates(dmat, [0, 1], [2, 3])
    assert e['reversal_candidate'] and e['reversal_confirmed']
    assert e['cross_fitted_gain'] > 0.01
    flat = {0: [0.02, -0.01, 0.0, 0.01], 1: [0.02, 0.0, 0.03, -0.01], 2: [0.03, 0.01, 0.0, 0.02]}
    e = family_estimates(flat, [0, 1], [2, 3])
    assert not e['reversal_candidate'] and abs(e['cross_fitted_gain']) < 1e-12
    noisy = {0: [0.03, 0.01, -0.01, 0.0], 1: [-0.01, -0.03, 0.0, 0.01], 2: [0.0, 0.01, 0.0, -0.01]}
    e = family_estimates(noisy, [0, 1], [2, 3])
    assert e['cross_fitted_gain'] <= 0 < e['in_sample_bound']   # noise inflates only the in-sample bound


def test_a2_verdicts():
    def recs(dm):
        out = []
        for f, states in dm.items():
            for s, ds in states.items():
                out.append(dict(family=f, seed=s, branches={
                    'a': dict(futures=[dict(value=0.5 + x) for x in ds]),
                    'b': dict(futures=[dict(value=0.5) for _ in ds])}))
        return out
    rule = dict(discovery=[0, 1], verification=[2, 3], delta=0.01)
    flat = {f: {s: [0.001, -0.001, 0.0, 0.0005] for s in range(3)} for f in ('D0', 'D1', 'D2', 'D3')}
    assert a2_verdict(recs(flat), 'a', 'b', rule)['verdict'].startswith('fixed write')
    planted = {f: {0: [0.05, 0.05, 0.05, 0.05], 1: [-0.05, -0.05, -0.05, -0.05], 2: [0.05] * 4}
               for f in ('D0', 'D1', 'D2', 'D3')}
    assert a2_verdict(recs(planted), 'a', 'b', rule)['verdict'].startswith('repeatable reversal')


def test_a1_gate_rules():
    gate = dict(learn_min=0.8, damage_max=0.1, delta=0.01, default='local')

    def row(f, w, learn, dmg, val):
        return dict(family=f, writer=w, seed=0, learn=learn, damage=dmg, value=val, rephrase=0.5, forgetting=0.0)
    fams = ('D0', 'D1')
    rows = [row(f, 'lora_plain', 0.9, 0.7, 0.3) for f in fams] + [row(f, 'local', 0.95, 0.05, 0.6) for f in fams]
    g = a1_gate(rows, gate)
    assert not g['go'] and g['feasible'] == ['local'] and 'only one' in g['reason']
    rows += [row('D0', 'local_replay:128', 0.95, 0.06, 0.62), row('D1', 'local_replay:128', 0.95, 0.06, 0.59)]
    g = a1_gate(rows, gate)
    assert g['go'] and g['default'] == 'local' and g['second'] == 'local_replay:128'
    rows = [r for r in rows if r['writer'] != 'local_replay:128']
    rows += [row(f, 'local_replay:128', 0.95, 0.06, 0.7) for f in fams]
    g = a1_gate(rows, gate)
    assert not g['go'] and 'fixed write is best' in g['reason']
    rows = [row(f, 'local', 0.95, 0.2, 0.6) for f in fams]
    assert a1_gate(rows, gate)['reason'] == 'no feasible writer'
