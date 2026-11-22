"""SO-08 tests: CPU only, tiny random Qwen3 and synthetic WikiBigEdit-style files. No real model or data."""
import json
import string
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from so_analysis import p0_gate, p0_verdict, select, state_dependence, swap_test  # noqa: E402
from so_branch import RunCfg, branch, matched_stream, p0_stream, run_default  # noqa: E402
from so_data import (FactIndex, PublicView, audit_stream, build_streams, find_steps, load_edits,  # noqa: E402
                     norm, split_halves)
from so_model import Writable, fmt_prompt, fmt_target  # noqa: E402
from so_policy import Action, EvalCfg, ReadoutCfg, WriteCfg, execute, parse_action  # noqa: E402
import so_run  # noqa: E402

DATES = ['20240201', '20240220', '20240301', '20240320', '20240401', '20240501', '20240601', '20240620', '20240701']
SITES = ['model.layers.0.mlp.down_proj', 'model.layers.1.mlp.down_proj']


class CharTok:
    def __init__(self):
        self.itos = ['<pad>', '<eos>'] + sorted(set(string.printable))
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_token_id, self.eos_token_id = 0, 1

    def __len__(self):
        return len(self.itos)

    def encode(self, text, add_special_tokens=False):
        return [self.stoi[c] for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(self.itos[i] for i in ids if i < len(self.itos) and not (skip_special_tokens and i < 2))


def tiny_model(vocab, seed=0, dtype=torch.float32):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512,
                      tie_word_embeddings=True)
    return Qwen3ForCausalLM(cfg).to(dtype).eval()


def synth_rows(step, n_subj=24):
    rows = []
    for i in range(n_subj):
        if i % 2 == 0 or step == 0:  # even subjects change their code every step; odd ones only appear once
            rows.append(dict(tag='new' if step == 0 else 'update', subject=f'Sub{i}', subject_id=f'Q{i}',
                             relation='code', relation_id='P1', object=f'v{i}x{step}', object_id=f'Q9{i}{step}',
                             ans=f'v{i}x{step}', update=f'What is the code of Sub{i}?',
                             rephrase=f'Which code does Sub{i} have?', personas=f'Yo, code of Sub{i}?',
                             loc=f'What colour is Loc{i}y{step}?', loc_ans=f'c{i}y{step}', mhop=None, mhop_ans=None))
    if step == 1:
        for v in ('red', 'blue'):
            rows.append(dict(tag='new', subject='Multi', subject_id='Q999', relation='tag', relation_id='P2',
                             object=v, object_id=None, ans=v, update='Which tag does Multi carry?',
                             rephrase='Name a tag of Multi', personas='Tags of Multi, mate?',
                             loc=f'What colour is LocM{v}?', loc_ans=f'cm{v}', mhop=None, mhop_ans=None))
    return rows


def write_dataset(d: Path, steps=8, jsonl_steps=()):
    d.mkdir(parents=True)
    for s in range(steps):
        rows = synth_rows(s)
        p = d / f'wiki_big_edit_{DATES[s]}_{DATES[s + 1]}.json'
        text = '\n'.join(json.dumps(r) for r in rows) + '\n' if s in jsonl_steps else json.dumps(rows)
        p.write_text(text, encoding='utf-8')
    return d


def small_cfg(**kw):
    return RunCfg(lag=0, write=WriteCfg(epochs=1, lr=1e-2, batch_size=8, accept_k=6, distill_k=4, max_new=8,
                                        gen_batch=16),
                  ev=EvalCfg(max_cur=3, ret_per_step=3, max_loc=4, max_new=8, gen_batch=16),
                  ro=ReadoutCfg(probe_k=3, cache_k=3), **kw)


@pytest.fixture
def data8(tmp_path):
    return write_dataset(tmp_path / 'wbe8', 8, jsonl_steps={3})


@pytest.fixture
def small(tmp_path):
    d = write_dataset(tmp_path / 'wbe4', 4)
    edits, windows, _ = load_edits(d, expect_steps=4)
    index = FactIndex(edits)
    streams = build_streams(edits, len(windows), n_pilot=1, n_dev=2, edits_per_stream=12, loc_probe=4, loc_hidden=4,
                            n_shots=1)
    return streams, index


def _view_with_history(streams):
    for s in streams:
        for t in range(1, s.steps):
            v = PublicView(s, t)
            if v.batch() and v.latest_history():
                return s, v
    raise AssertionError('fixture has no state with both a batch and an older history')


def make_wm(seed=0, dtype=torch.float32, grad_ckpt=False):
    tok = CharTok()
    return Writable(tiny_model(len(tok), seed, dtype), tok, SITES, rank=2, scale=1.0, grad_ckpt=grad_ckpt)


# ---- data ------------------------------------------------------------------------------------------------

def test_load_array_and_jsonl(data8):
    edits, windows, skipped = load_edits(data8)
    assert len(windows) == 8 and windows[0] == ('20240201', '20240220') and not skipped
    assert sum(1 for e in edits if e.step == 3) == len(synth_rows(3))


def test_nan_and_blank_fields_count_as_missing(tmp_path):
    d = tmp_path / 'nan'
    d.mkdir()
    rows = synth_rows(0)[:3]
    rows[0].update(tag=float('nan'), rephrase=float('nan'), personas='  ')
    rows[1].update(subject_id=None)
    rows[2].update(tag='')
    text = json.dumps(rows)  # writes NaN like the released files
    assert 'NaN' in text
    (d / f'wiki_big_edit_{DATES[0]}_{DATES[1]}.json').write_text(text, encoding='utf-8')
    edits, _, skipped = load_edits(d, expect_steps=1)
    assert skipped == {0: 1} and len(edits) == 2
    first = next(e for e in edits if e.subject_id == 'Q0')
    assert first.tag is None and first.rephrase is None and first.personas is None
    assert FactIndex(edits).category(first) == 'untagged'
    assert all(e.tag is None for e in edits)


def test_windows_must_be_contiguous(tmp_path):
    d = tmp_path / 'gap'
    d.mkdir()
    for a, b in (('20240201', '20240220'), ('20240301', '20240320')):
        (d / f'wiki_big_edit_{a}_{b}.json').write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError):
        find_steps(d)


def test_fact_index_supersession_and_multi(data8):
    edits, _, _ = load_edits(data8)
    idx = FactIndex(edits)
    assert idx.valid(('Q0', 'P1'), 0) == {'v0x0'} and idx.valid(('Q0', 'P1'), 3) == {'v0x3'}
    assert idx.valid(('Q1', 'P1'), 5) == {'v1x0'}
    first = next(e for e in edits if e.key == ('Q0', 'P1') and e.step == 0)
    assert idx.superseded(first, 2) and not idx.superseded(first, 0)
    assert ('Q999', 'P2') in idx.multi and idx.valid(('Q999', 'P2'), 1) == {'red', 'blue'}
    later = next(e for e in edits if e.key == ('Q0', 'P1') and e.step == 2)
    assert idx.category(later) == 'repeat' and idx.category(first) == 'new'


def test_streams_group_entities_and_pools_are_disjoint(data8):
    edits, windows, _ = load_edits(data8)
    halves = split_halves(edits)
    assert halves['P1'] != halves['P2']
    streams = build_streams(edits, len(windows), n_pilot=1, n_dev=2, edits_per_stream=20, loc_probe=5, loc_hidden=7,
                            n_shots=2)
    assert [s.name for s in streams] == ['P0', 'D0', 'D1']
    owner = {}
    for s in streams:
        assert audit_stream(s) == []
        assert len(s.loc_probe) <= 5 and len(s.loc_hidden) <= 7
        for e in s.edits:
            assert halves[e.relation_id] == 'dev'
            assert owner.setdefault(e.subject_id, s.name) == s.name
    by_key = {}
    for s in streams:
        for e in s.edits:
            by_key.setdefault(e.key, set()).add(s.name)
    assert all(len(v) == 1 for v in by_key.values())
    full = {e.key: sum(1 for x in edits if x.key == e.key) for s in streams for e in s.edits}
    assert all(sum(1 for e in s.edits if e.key == k) == n for s in streams for k, n in full.items()
               if any(e.key == k for e in s.edits))


def test_public_view_never_carries_hidden_text(data8):
    edits, windows, _ = load_edits(data8)
    s = build_streams(edits, len(windows), n_pilot=1, n_dev=2, edits_per_stream=20, loc_probe=5, loc_hidden=7)[1]
    v = PublicView(s, 3)
    assert not hasattr(v, '__dict__')
    seen = set()
    for blob in (v.batch(), v.latest_history(), v.replay(50, 0), v.acceptance(50), v.cache(50)):
        for item in blob:
            seen.update(str(x) for x in item)
    seen.update(v.probe_questions(100) + v.history_questions(100))
    hidden = {a for _, a in s.loc_hidden} | {e.loc_ans for e in s.edits} | {q for q, _ in s.loc_hidden}
    hidden |= {e.rephrase for e in s.edits} | {e.personas for e in s.edits}
    assert not (seen & hidden)
    now = {k for _, _, k in v.batch()}
    assert all(item[2] not in now for item in v.latest_history())
    assert len(v.replay(50, 0)) == len(v.latest_history())
    assert v.acceptance(4) == v.acceptance(4)


# ---- model -----------------------------------------------------------------------------------------------

def test_no_write_is_bit_exact():
    tok = CharTok()
    m = tiny_model(len(tok), 1)
    x = torch.randint(2, len(tok), (3, 11))
    with torch.no_grad():
        before = m(input_ids=x).logits.clone()
    Writable(m, tok, SITES, rank=2)
    with torch.no_grad():
        after = m(input_ids=x).logits
    assert torch.equal(before, after)


def test_commit_merge_shrink_and_restore_are_exact():
    wm = make_wm(2)
    x = torch.randint(2, len(wm.tok), (2, 9))
    mod = wm.mods[SITES[0]]
    wm.start_write(1)
    with torch.no_grad():
        mod.B.normal_()
    d1 = mod.delta().clone()
    wm.commit(0.5)
    assert torch.allclose(mod.delta(), 0.5 * d1)
    snap = wm.snapshot()
    with torch.no_grad():
        ref = wm.model(input_ids=x).logits.clone()
    wm.start_write(2)
    with torch.no_grad():
        mod.B.normal_()
    d2 = mod.delta() - 0.5 * d1
    wm.commit(1.0)
    wm.shrink(0.5)
    assert torch.allclose(mod.delta(), 0.5 * (0.5 * d1 + d2), atol=1e-6)
    wm.restore(snap)
    with torch.no_grad():
        assert torch.equal(wm.model(input_ids=x).logits, ref)
    wm.restore(wm.empty_state())
    assert wm.delta_norm() == 0.0


def test_fit_trains_only_the_write_and_checkpointing_matches():
    items = [(fmt_prompt(f'What is the code of Sub{i}?'), fmt_target(f'v{i}x1'), 1.0) for i in range(6)]
    a, b = make_wm(3), make_wm(3, grad_ckpt=True)
    base = {k: v.clone() for k, v in a.model.state_dict().items() if not k.endswith(('.A', '.B'))}
    for wm in (a, b):
        wm.start_write(7)
        wm.fit(items, epochs=2, lr=1e-2, batch_size=3, seed=4)
    for k, v in a.model.state_dict().items():
        if k in base:
            assert torch.equal(v, base[k]), k
    for n in SITES:
        assert torch.allclose(a.mods[n].A, b.mods[n].A, atol=1e-5)
        assert torch.allclose(a.mods[n].B, b.mods[n].B, atol=1e-5)
    assert a.mods[SITES[0]].B.abs().sum() > 0 and a.costs['backward'] == 4


def test_cached_greedy_matches_full_recomputation():
    wm = make_wm(5, dtype=torch.float64)
    qs = ['What is the code of Sub1?', 'Hi?', 'A somewhat longer question here?']
    fast = wm.generate(qs, max_new=10)
    slow = wm.generate(qs, max_new=10, use_cache=False)
    assert [(t, n) for t, _, n in fast] == [(t, n) for t, _, n in slow]
    assert all(abs(x[1] - y[1]) < 1e-9 for x, y in zip(fast, slow))


# ---- actions and branches --------------------------------------------------------------------------------

def _same_state(s1, s2):
    return all(torch.equal(x, y) for n in s1 for x, y in zip(s1[n][:4], s2[n][:4]))


def test_actions(small):
    streams, index = small
    s, v = _view_with_history(streams)
    wm = make_wm(6)
    cfg = small_cfg().write
    before = wm.snapshot()
    rec = execute(wm, v, Action('skip'), cfg, 1)
    assert not rec['wrote'] and _same_state(before, wm.snapshot())
    rec = execute(wm, v, Action('rollback', -1e9), cfg, 1)
    assert rec['rolled_back'] and _same_state(before, wm.snapshot())
    probes = [q for q in v.probe_questions(cfg.distill_k) + v.history_questions(cfg.distill_k)
              if norm(q) not in {norm(q2) for q2, _, _ in v.batch()}]
    expect = sum(1 for t, _, _ in wm.generate(probes, max_new=cfg.max_new) if t)
    rec = execute(wm, v, Action('selfdistill', 1.0), cfg, 1)
    assert rec['n_distill'] == expect and rec['wrote']
    wm.restore(before)
    rec = execute(wm, v, Action('replay', 100), cfg, 1)
    assert rec['n_replay'] == len(v.latest_history())
    wm.restore(before)
    rec = execute(wm, v, Action('merge', 0.5), cfg, 1)
    assert rec['wrote'] and wm.delta_norm() > 0
    assert parse_action('merge:0.5') == Action('merge', 0.5) and Action('merge', 0.5).label == 'merge:0.5'


def test_default_branch_reproduces_trajectory_and_restores(small):
    streams, index = small
    s = streams[1]
    wm = make_wm(7)
    cfg = small_cfg()
    traj = run_default(wm, s, index, cfg, seed=5, capture={1})
    cap = traj['captures'][1]
    out = branch(wm, s, index, cfg, cap, [Action('plain'), Action('skip')], 5, traj['base_loc'])
    assert [c['composite'] for c in out[0]['comps']] == [traj['steps'][t]['eval']['composite'] for t in (1, 2, 3)]
    assert _same_state(wm.snapshot(), cap['snapshot'])
    assert out[1]['exec']['wrote'] is False and len(out[1]['post']) == len(cap['probe_q']) + len(cap['cache'])


def test_p0_and_matched_runs_are_serialisable(small):
    streams, index = small
    wm = make_wm(8)
    cfg = small_cfg()
    menu = [Action('plain'), Action('skip'), Action('selfdistill', 1.0)]
    rec = p0_stream(wm, streams[0], index, cfg, 0, menu, repair_steps=[1])
    json.dumps(rec, default=list)
    v = p0_verdict(json.loads(json.dumps(rec, default=list)),
                   dict(conflict_min_r2w_fraction=0.1, repair_max_cur_drop=0.02, repair_min_r2w_reduction=0.5))
    assert set(v) >= {'capability', 'conflict', 'repairs', 'repairable'}
    records = [json.loads(json.dumps(matched_stream(wm, s, index, cfg, seed, menu, [1, 2]), default=list))
               for s in streams[1:] for seed in (0, 1)]
    sel = select(records, lam=1.0)
    assert sel['result']['n_states'] == 8
    assert state_dependence(records)['n_pairs'] == 4
    assert swap_test(records, 'current')['n'] == 8


# ---- analysis logic on constructed records ---------------------------------------------------------------

def _state(stream, seed, step, drift_plain, drift_sd, v_plain, v_sd):
    probe = ['a', 'b', 'c', 'd']
    changed = lambda k: [x + '!' if i < k else x for i, x in enumerate(probe)]  # noqa: E731
    br = [dict(action='plain', exec={}, post=changed(drift_plain), value=v_plain),
          dict(action='selfdistill:1', exec={}, post=changed(drift_sd), value=v_sd)]
    return dict(step=step, probe_q=['q1', 'q2', 'q3', 'q4'], cache=[], current=probe, lagged=probe, teacher=probe,
                branches=br)


def test_readout_selection_and_swap_on_constructed_records():
    # seed 0 states: plain drifts a lot and is worse; seed 1 states: the reverse. Public features cannot tell.
    records = []
    for k, stream in enumerate(['A', 'B', 'C', 'D']):
        for seed in (0, 1):
            st = [_state(stream, seed, t, 3, 0, 0.4, 0.6) if seed == 0 else _state(stream, seed, t, 0, 3, 0.6, 0.4)
                  for t in (1, 2)]
            records.append(dict(stream=stream, seed=seed, states=st))
    res = select(records, fsets=('public', 'public+current'), lam=0.1)['result']
    assert res['public+current']['mean_regret'] == 0.0
    assert res['public']['mean_regret'] > 0.05
    dep = state_dependence(records)
    assert dep['differ_rate'] == 1.0
    sw = swap_test(records, 'current', lam=0.1)
    assert sw['mean_regret_matched'] == 0.0 and sw['mean_regret_swapped'] > 0.15


def _ev(items):
    acc = {}
    for comp in ('cur', 'ret', 'loc', 'upd'):
        v = [ok for c, _, ok in items if c == comp]
        acc[comp] = sum(v) / len(v) if v else None
    return dict(items=[list(i) for i in items], acc=acc, n={}, composite=None)


def test_p0_verdict_logic():
    e0 = _ev([('cur', 'a|r', 1), ('cur', 'b|r', 1), ('upd', 'a|u', 1), ('upd', 'b|u', 1), ('loc', 'q1', 1),
              ('loc', 'q2', 1)])
    e1 = _ev([('cur', 'c|r', 1), ('upd', 'c|u', 1), ('ret', 'a|r', 1), ('ret', 'b|r', 0), ('loc', 'q1', 1),
              ('loc', 'q2', 0)])
    e2 = _ev([('cur', 'd|r', 1), ('upd', 'd|u', 1), ('ret', 'a|r', 0), ('ret', 'b|r', 0), ('ret', 'c|r', 1),
              ('loc', 'q1', 0), ('loc', 'q2', 0)])
    good2 = _ev([('cur', 'd|r', 1), ('upd', 'd|u', 1), ('ret', 'a|r', 1), ('ret', 'b|r', 1), ('ret', 'c|r', 1),
                 ('loc', 'q1', 1), ('loc', 'q2', 1)])
    lazy1 = _ev([('cur', 'c|r', 0), ('upd', 'c|u', 0), ('ret', 'a|r', 1), ('ret', 'b|r', 1), ('loc', 'q1', 1),
                 ('loc', 'q2', 1)])
    rec = dict(stream='P0', capability=dict(n=5, no_info=0.0, shots_only=0.0, fact_in_context=0.6),
               trajectory=dict(steps=[dict(eval=e) for e in (e0, e1, e2)],
                               base_loc={'q1': [True, 'x'], 'q2': [True, 'y']}),
               repairs={'1': [dict(action='plain', comps=[e1, e2], value=0.5),
                              dict(action='replay:64', comps=[e1, good2], value=0.9),
                              dict(action='skip', comps=[lazy1, good2], value=0.7)]})
    v = p0_verdict(rec, dict(conflict_min_r2w_fraction=0.25, repair_max_cur_drop=0.02, repair_min_r2w_reduction=0.5))
    assert v['conflict']['edits'] == dict(learned_tracked=3, right_to_wrong=2, fraction=2 / 3)
    assert v['conflict']['locality'] == dict(base_right=2, right_to_wrong=2, fraction=1.0)
    assert v['conflict']['pass']
    assert v['repairable'] and [p['action'] for p in v['repairs'][1]['passing']] == ['replay:64']
    assert v['capability']['upd_after_write'] == [1.0, 1.0, 1.0]
    gate = dict(learn_min_upd=0.8, repair_streams_min=1)
    assert p0_gate([v], gate)['passed']
    assert not p0_gate([v], dict(gate, learn_min_upd=1.01))['passed']
    worse = dict(v, conflict=dict(v['conflict'], **{'pass': False}))
    assert not p0_gate([v, worse], gate)['passed']


def test_screening_keeps_only_base_known_locality(small):
    from so_policy import screen_locality
    from so_data import shots_text
    streams, _ = small
    s = streams[1]
    wm = make_wm(10)
    cfg = EvalCfg(max_cur=3, ret_per_step=3, max_loc=2, max_new=6, gen_batch=16)
    screened, info = screen_locality(wm, s, cfg)
    assert info['candidates'] == len(s.loc_hidden) and info['kept'] == len(screened.loc_hidden) <= 2
    assert set(screened.loc_hidden) <= set(s.loc_hidden)
    ctx = shots_text(s.shots)
    again = wm.generate([q for q, _ in screened.loc_hidden], contexts=[ctx] * len(screened.loc_hidden), max_new=6)
    assert all(norm(t) == norm(a) for (t, _, _), (_, a) in zip(again, screened.loc_hidden))
    assert PublicView(s, 1).context() == ctx and (ctx == '' or ctx.startswith('Q: '))


def test_relation_targeted_probes_and_selfdistill(small):
    streams, _ = small
    s, v = _view_with_history(streams)
    assert len(s.probe_rel) == len(s.loc_probe) and set(s.probe_rel) <= {'P1', 'P2'}
    rels = v.batch_relations()
    assert rels and v.probe_questions_for({'P404'}, 10) == [] and v.history_questions_for({'P404'}, 10) == []
    picked = v.probe_questions_for(rels, 3)
    assert len(picked) <= 3 and all(r in rels for q, r in zip(s.loc_probe, s.probe_rel) if q in picked)
    assert all(item[2][1] in rels for item in v.latest_history() if item[0] in v.history_questions_for(rels, 50))
    wm = make_wm(11)
    cfg = small_cfg().write
    now = {norm(q) for q, _, _ in v.batch()}
    pool = [q for q in v.probe_questions_for(rels, cfg.distill_k) + v.history_questions_for(rels, cfg.distill_k)
            if norm(q) not in now]
    expect = sum(1 for t, _, _ in wm.generate(pool, contexts=[v.context()] * len(pool), max_new=cfg.max_new) if t)
    rec = execute(wm, v, parse_action('selfdistill_rel:1'), cfg, 2)
    assert rec['n_distill'] == expect and rec['action'] == 'selfdistill_rel:1'
    written = wm.snapshot()
    with wm.temporarily(wm.empty_state()):
        teacher = sum(1 for t, _, _ in wm.generate(pool, contexts=[v.context()] * len(pool), max_new=cfg.max_new) if t)
    assert _same_state(written, wm.snapshot())
    rec = execute(wm, v, parse_action('teacherdistill_rel:1'), cfg, 3)
    assert rec['n_distill'] == teacher and rec['wrote']


def _capture_fit(wm):
    """Record the training items of every write without changing what fit does."""
    seen, fit = [], wm.fit

    def wrapped(items, *a, **kw):
        seen.append(list(items))
        return fit(items, *a, **kw)
    wm.fit = wrapped
    return seen


def test_round2_actions_label_sources(small):
    streams, _ = small
    s, v = _view_with_history(streams)
    wm = make_wm(12)
    cfg = small_cfg().write
    execute(wm, v, Action('plain'), cfg, 4)  # leave the empty state, so self and base labels can differ
    written = wm.snapshot()
    ctx, now = v.context(), {norm(q) for q, _, _ in v.batch()}
    rels = v.batch_relations()
    probes = [q for q in v.probe_questions_for(rels, cfg.distill_k) if norm(q) not in now]
    gen = lambda: [t for t, _, _ in wm.generate(probes, contexts=[ctx] * len(probes), max_new=cfg.max_new)]  # noqa
    self_labels = gen()
    with wm.temporarily(wm.empty_state()):
        base_labels = gen()
    assert probes and any(self_labels) and any(base_labels)
    hist = [(q, a) for q, a in v.history_items_for(rels, cfg.distill_k) if norm(q) not in now]
    assert hist and all(item[2][1] in rels for item in v.latest_history() if (item[0], item[1]) in hist)
    seen = _capture_fit(wm)
    n_batch = len(v.batch())
    for spec, labels, with_hist, n_rep in (('selfprobe_rel:1', self_labels, True, 0),
                                           ('baseprobe_rel:1', base_labels, True, 0),
                                           ('replay_self:100', self_labels, False, len(v.latest_history())),
                                           ('replay_base:100', base_labels, False, len(v.latest_history()))):
        wm.restore(written)
        rec = execute(wm, v, parse_action(spec), cfg, 5)
        items = seen[-1]
        want = [(fmt_prompt(q), fmt_target(t), 1.0) for q, t in zip(probes, labels) if t]
        assert rec['action'] == spec and rec['wrote'] and rec['n_distill'] == len(want)
        assert rec['n_replay'] == n_rep and rec['n_history'] == (len(hist) if with_hist else 0)
        assert items[len(items) - len(want):] == want  # probe labels come last, from the stated source
        if with_hist:
            assert items[n_batch:n_batch + len(hist)] == [(fmt_prompt(q), fmt_target(a), 1.0) for q, a in hist]
    assert parse_action('replay_base:256').label == 'replay_base:256'


def test_persistent_continuation(small):
    streams, index = small
    s = streams[1]
    wm = make_wm(13)
    cfg = small_cfg(continuation='persistent')
    traj = run_default(wm, s, index, cfg, seed=6, capture={1})
    cap = traj['captures'][1]
    menu = [Action('plain'), Action('replay', 100)]
    out = branch(wm, s, index, cfg, cap, menu, 6, traj['base_loc'])
    assert [c['composite'] for c in out[0]['comps']] == [traj['steps'][t]['eval']['composite'] for t in (1, 2, 3)]
    assert [r['action'] for r in out[1]['later']] == ['replay:100', 'replay:100']
    assert all(r['n_replay'] > 0 for r in out[1]['later']) and out[1]['continuation'] == 'persistent'
    assert _same_state(wm.snapshot(), cap['snapshot'])
    once = branch(wm, s, index, small_cfg(), cap, menu, 6, traj['base_loc'])
    assert [r['action'] for r in once[1]['later']] == ['plain', 'plain']
    assert [c['composite'] for c in once[0]['comps']] == [c['composite'] for c in out[0]['comps']]
    with pytest.raises(ValueError):
        small_cfg(continuation='sometimes')
    rec = p0_stream(wm, s, index, cfg, 0, menu, repair_steps=[2])
    v = p0_verdict(json.loads(json.dumps(rec, default=list)),
                   dict(conflict_min_r2w_fraction=0.1, repair_max_cur_drop=0.02, repair_min_r2w_reduction=0.5),
                   rule='from_decision', cur_mode='mean')
    assert v['repair_rule'] == 'from_decision' and set(v['repairs'][2]) >= {'plain_cur', 'plain_damage', 'passing'}


def test_repair_rules_and_cur_modes():
    e0 = _ev([('cur', 'a|r', 1), ('cur', 'b|r', 1), ('upd', 'a|u', 1), ('upd', 'b|u', 1), ('loc', 'q1', 1),
              ('loc', 'q2', 0)])
    e1 = _ev([('cur', 'c|r', 1), ('upd', 'c|u', 1), ('ret', 'a|r', 1), ('ret', 'b|r', 0), ('loc', 'q1', 1),
              ('loc', 'q2', 0)])
    e2 = _ev([('cur', 'd|r', 1), ('upd', 'd|u', 1), ('ret', 'a|r', 0), ('ret', 'b|r', 0), ('ret', 'c|r', 1),
              ('loc', 'q1', 0), ('loc', 'q2', 0)])
    good2 = _ev([('cur', 'd|r', 1), ('upd', 'd|u', 1), ('ret', 'a|r', 1), ('ret', 'b|r', 1), ('ret', 'c|r', 1),
                 ('loc', 'q1', 1), ('loc', 'q2', 0)])
    lazy2 = _ev([('cur', 'd|r', 0), ('upd', 'd|u', 0), ('ret', 'a|r', 1), ('ret', 'b|r', 1), ('ret', 'c|r', 1),
                 ('loc', 'q1', 1), ('loc', 'q2', 0)])
    rec = dict(stream='P0', capability=dict(n=5, no_info=0.0, shots_only=0.0, fact_in_context=0.6),
               trajectory=dict(steps=[dict(eval=e) for e in (e0, e1, e2)],
                               base_loc={'q1': [True, 'x'], 'q2': [True, 'y']}),
               repairs={'1': [dict(action='plain', comps=[e1, e2], value=0.5),
                              dict(action='replay_base:256', comps=[e1, good2], value=0.9),
                              dict(action='selfprobe_rel:1', comps=[e1, lazy2], value=0.7)]})
    th = dict(conflict_min_r2w_fraction=0.25, repair_max_cur_drop=0.02, repair_min_r2w_reduction=0.5)
    v = p0_verdict(rec, th, 'plain', 'from_decision', 'mean')
    rep = v['repairs'][1]
    by = {c['action']: c for c in rep['candidates']}
    # after the decision: q1 was still right before step 1 (q2 broke at step 0); edits a, b right before, c learned
    assert by['plain']['locality'] == 1.0 and by['plain']['forgetting'] == pytest.approx(2 / 3)
    assert rep['plain_damage'] == pytest.approx(5 / 3) and by['replay_base:256']['reduction'] == 1.0
    assert by['selfprobe_rel:1']['cur'] == 0.5 and by['selfprobe_rel:1']['cur_decision'] == 1.0
    assert [p['action'] for p in rep['passing']] == ['replay_base:256'] and v['repairable']
    tot = {c['action']: c for c in v['repairs_other_rule']['total'][1]['candidates']}
    assert tot['replay_base:256']['reduction'] == pytest.approx(0.7)  # q2, broken before the decision, still counts
    v2 = p0_verdict(rec, th, 'plain', 'from_decision', 'decision')
    assert [p['action'] for p in v2['repairs'][1]['passing']] == ['replay_base:256', 'selfprobe_rel:1']
    r1 = p0_verdict(rec, th)  # round-1 defaults: damage from step 0, cur at the decision step
    assert r1['repair_rule'] == 'total' and r1['repairs'][1]['plain_damage'] == pytest.approx(5 / 3)
    assert [p['action'] for p in r1['repairs'][1]['passing']] == ['replay_base:256', 'selfprobe_rel:1']
    with pytest.raises(ValueError):
        p0_verdict(rec, th, 'plain', 'sometimes')


def test_run_default_respects_max_steps(small):
    streams, index = small
    wm = make_wm(9)
    traj = run_default(wm, streams[1], index, small_cfg(), seed=1, max_steps=2)
    assert [s['step'] for s in traj['steps']] == [0, 1]


# ---- command line ----------------------------------------------------------------------------------------

def test_streams_command_and_no_overwrite(data8, tmp_path):
    cfg = json.loads((HERE / 'config_draft.json').read_text(encoding='utf-8'))
    cfg.update(data_dir=str(data8))
    cfg['streams'].update(n_pilot=1, n_dev=2, edits_per_stream=20, loc_probe=5, loc_hidden=7)
    path = tmp_path / 'cfg.json'
    path.write_text(json.dumps(cfg), encoding='utf-8')
    out = tmp_path / 'run'
    so_run.main(['streams', str(path), str(out)])
    data = json.loads((out / 'streams.json').read_text(encoding='utf-8'))
    assert [s['name'] for s in data['streams']] == ['P0', 'D0', 'D1'] and all(not s['audit'] for s in data['streams'])
    assert (out / 'manifest.json').exists()
    with pytest.raises(FileExistsError):
        so_run.main(['streams', str(path), str(out)])
    bad = dict(cfg, default='replay:5')
    with pytest.raises(ValueError):
        so_run.menu(bad)
