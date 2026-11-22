"""KT-10 tests: CPU only, SO-08's tiny random Qwen3 and synthetic WikiBigEdit-style files."""
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kt_core import (OFF, Entries, LoggedEditor, build_entries, evaluate_logged, generate_logged, hidden_fire,  # noqa: E402
                     hidden_max, install, intervene, probe_max, probe_positions, relevant_hits, thresholds)
from kt_analysis import (arm_metrics, concentration, concentration_verdict, contrast, name_relation,  # noqa: E402
                         shuffled_arms, tune_uniform)
from rv_editor import EditWritable, Fact, LocalCfg, LocalEditor  # noqa: E402
from rv_policy import RunCfg, Writer, cap_stream, trajectory  # noqa: E402
from so_data import FactIndex, build_streams, load_edits, norm, shots_text  # noqa: E402
from so_model import fmt_prompt  # noqa: E402
from so_policy import EvalCfg, WriteCfg, base_locality, evaluate  # noqa: E402
from test_so import CharTok, tiny_model, write_dataset  # noqa: E402

SITES = ['model.layers.0.mlp.down_proj', 'model.layers.1.mlp.down_proj']
EV = EvalCfg(max_cur=3, ret_per_step=2, max_loc=4, max_new=6, gen_batch=4)


def make_wm(seed=0):
    tok = CharTok()
    return EditWritable(tiny_model(len(tok), seed), tok, SITES, rank=2, scale=1.0, clip=1.0)


def corpus_texts():
    return [f'Q: What is the code of Thing{i}?\nA:' for i in range(40)]


def cfg_route(**kw):
    return LocalCfg(**dict(dict(mode='route', layers=(0,), tau=0.5, v_steps=4, n_rand=1, rand_len=3, chunk=4,
                                key_batch=8, probe_margin=0.01), **kw))


def editor(wm, cls=LoggedEditor, **kw):
    ed = cls(cfg_route(**kw))
    ed.prepare(wm, corpus_texts())
    return ed


@pytest.fixture
def small(tmp_path):
    d = write_dataset(tmp_path / 'wbe4', 4)
    edits, windows, _ = load_edits(d, expect_steps=4)
    index = FactIndex(edits)
    streams = build_streams(edits, len(windows), n_pilot=1, n_dev=2, edits_per_stream=12, loc_probe=4, loc_hidden=4,
                            n_shots=1)
    return streams, index


def facts3():
    return [Fact(f'What is the code of Sub{i}?', f'z{i}', f'Sub{i}') for i in range(3)]


def prompts():
    return [fmt_prompt(q, 'Q: Who is Loc1?\nA: c1\n') for q in
            ('What is the code of Sub1?', 'Which code does Sub2 have?', 'What colour is Loc9y2?', 'Hi?')]


def logits_of(wm, text):
    with torch.no_grad():
        return wm.model(input_ids=torch.tensor([wm.tok.encode(text)]), use_cache=False).logits


# ---- the logging hook ---------------------------------------------------------------------------------------------
def test_logged_hook_routes_exactly_like_the_parent_and_logs_the_winner():
    wm = make_wm()
    parent = editor(wm, LocalEditor, tau=0.2, probe_margin=None)
    parent.edit(wm, facts3(), '', seed=1)
    wm.extra['route_t'] = torch.tensor([0.2, 0.35, 0.1])
    ref = [logits_of(wm, t) for t in prompts()]
    parent.detach()
    logged = editor(wm, tau=0.2, probe_margin=None)  # prepare() keeps the entries (snapshot / restore)
    assert torch.equal(logged.mu, parent.mu)
    logged.log, logged._fwd = [], 0
    got = [logits_of(wm, t) for t in prompts()]
    assert all(torch.equal(a, b) for a, b in zip(ref, got))
    # expected firings from the raw block outputs (routing off)
    block = wm.model.get_submodule('model.layers.0')
    cap = []
    hd = block.register_forward_hook(lambda m, i, o: cap.append((o[0] if isinstance(o, tuple) else o).detach().float()))
    logged.route_off = True
    for t in prompts():
        logits_of(wm, t)
    logged.route_off = False
    hd.remove()
    k, thr = wm.extra['route_k'], wm.extra['route_t']
    want = []
    for fwd, h in enumerate(cap):
        sims = F.normalize(h - logged.mu, dim=-1) @ k.T
        sims = torch.where(sims >= thr, sims, torch.full_like(sims, -2.0))
        for p in range(h.shape[1]):
            row = sims[0, p]
            if row.max() >= -1.0 and h[0, p].norm() < 200.0:
                best = float(row.max())
                e = max(j for j in range(k.shape[0]) if float(row[j]) == best)   # newest wins ties
                want.append((fwd, p, e))
    seen = [(fwd, p, e) for fwd, _, rows, pos, ents in logged.log for r, p, e in zip(rows, pos, ents)]
    assert seen == want and want
    logged.log = None
    wm.extra['route_t'] = torch.full((3,), OFF)
    assert torch.equal(logits_of(wm, prompts()[0]), logits_of(wm, prompts()[0]))
    logged.detach()


def test_relevant_hits_drops_padding_and_late_decode_positions():
    lens, nctx, gen = [5, 3], [2, 2], [3, 1]
    log = [(0, 5, [0, 0, 1, 1, 1], [0, 3, 1, 2, 4], [7, 8, 9, 9, 9]),   # row 1 starts at 2: position 1 is padding
           (1, 1, [0, 1], [0, 0], [5, 6]),                              # row 1 generated a single token: too late
           (2, 1, [0], [0], [4]),
           (3, 1, [0], [0], [3])]                                       # row 0 generated 3 tokens: k=3 too late
    assert relevant_hits(log, lens, nctx, gen) == [[(4, 2), (5, 2), (7, 0), (8, 1)], [(9, 0), (9, 1)]]


def test_generate_logged_matches_generate_and_no_firing_means_no_change():
    wm = make_wm()
    ed = editor(wm, tau=0.2, probe_margin=None)
    ed.edit(wm, facts3(), '', seed=1)
    qs = ['What is the code of Sub1?', 'Which code does Sub2 have?', 'What colour is Loc9y2?', 'Hi?',
          'What is the code of Sub0 now?']
    ctx = ['Q: Who is Loc1?\nA: c1\n'] * len(qs)
    for thr in (torch.tensor([0.2, 0.35, 0.1]), torch.tensor([0.6, 0.6, 0.6]), torch.full((3,), -1.0)):
        wm.extra['route_t'] = thr
        ref = wm.generate(qs, contexts=ctx, max_new=6, batch_size=2)
        got, hits = generate_logged(wm, ed, qs, ctx, 6, 2)
        assert got == ref
        wm.extra['route_t'] = torch.full((3,), OFF)
        base = wm.generate(qs, contexts=ctx, max_new=6, batch_size=2)
        for g, b, hh in zip(got, base, hits):
            if not hh:
                assert g == b
            assert all(reg in (0, 1, 2) and 0 <= e < 3 for e, reg in hh)
    assert any(hits)                                 # threshold -1 fires everywhere
    ed.detach()


def test_evaluate_logged_equals_so08_evaluate(small):
    streams, index = small
    s = streams[1]
    wm = make_wm()
    ed = editor(wm, tau=0.3)
    base_loc = base_locality(wm, s, EV)
    ed.set_probes(wm, list(s.loc_probe), shots_text(s.shots))
    ent = build_entries(wm, ed, s, 0)
    for thr in (ent.matched, torch.full((ent.n,), 0.3), torch.full((ent.n,), OFF)):
        for t in range(s.steps):
            install(wm, ent, t, thr)
            ref = evaluate(wm, s, index, t, 0, EV, base_loc)
            got, items = evaluate_logged(wm, ed, s, index, t, EV, base_loc)
            assert [[c, k, int(b)] for (c, k, _), b in zip(ref['items'], got['ok'])] == ref['items']
            for f in ('n', 'acc', 'composite', 'loc_right_to_wrong', 'loc_wrong_to_right', 'loc_agree_base'):
                assert got[f] == ref[f], f
            assert [k for _, _, _, k in items] == [k for _, k, _ in ref['items']]
    ed.detach()


# ---- entries, probes and thresholds --------------------------------------------------------------------------------
def test_entries_equal_the_rv09_local_trajectory(small):
    streams, index = small
    s = streams[0]
    wm = make_wm()
    ed = editor(wm)
    ctx = shots_text(s.shots)
    ed.set_probes(wm, list(s.loc_probe), ctx)
    ent = build_entries(wm, ed, s, 0)
    rc = RunCfg(write=WriteCfg(epochs=1, lr=1e-2, batch_size=8, accept_k=4, distill_k=4, max_new=6, gen_batch=4),
                ev=EV)
    trajectory(wm, ed, s, index, Writer('local'), rc, 0, {}, evaluate_steps=False)
    for k_ours, k_rv in (('keys', 'route_k'), ('offsets', 'route_v'), ('matched', 'route_t')):
        assert torch.equal(getattr(ent, k_ours), wm.extra[k_rv]), k_ours
    assert ent.ends[-1] == ent.n and len(ent.meta) == ent.n and ent.step == sorted(ent.step)
    assert {m['uid'] for m in ent.meta} <= {e.uid for e in s.edits}
    ed.detach()


def test_probe_positions_tables_and_threshold_arms(small):
    streams, index = small
    s = streams[0]
    wm = make_wm()
    ed = editor(wm)
    ctx = shots_text(s.shots)
    qs = list(s.loc_probe)
    states, labels = probe_positions(wm, ed, qs, ctx, batch_size=3)    # the context lands in a mixed last batch
    ed.set_probes(wm, qs, ctx, batch_size=3)
    assert torch.equal(states, ed.probe_states)
    assert labels.shape[0] == states.shape[0] and int(labels.max()) == len(qs)
    ent = build_entries(wm, ed, s, 0)
    nq = len(qs)
    pm = probe_max(states, labels, ent.keys, nq + 1, chunk=7)
    cfg = ed.cfg
    assert torch.allclose(torch.clamp(pm.max(0).values + cfg.probe_margin, min=cfg.tau), ent.matched, atol=1e-6)
    assert torch.allclose(thresholds(f'probes:{nq}', ent, pm, nq, cfg, s.name), ent.matched, atol=1e-6)
    ctx_only = thresholds('ctx_only', ent, pm, nq, cfg, s.name)
    assert bool((ctx_only <= ent.matched + 1e-6).all()) and bool((ctx_only >= cfg.tau).all())
    assert bool((thresholds('none', ent, pm, nq, cfg, s.name) == OFF).all())
    assert bool((thresholds('uniform:0.93', ent, pm, nq, cfg, s.name) == 0.93).all())
    a, b = (thresholds(f'shuffled:{p}', ent, pm, nq, cfg, s.name) for p in (0, 1))
    assert torch.equal(a, thresholds('shuffled:0', ent, pm, nq, cfg, s.name))      # fixed permutations
    start = 0
    for end in ent.ends:
        for x in (a, b):
            assert sorted(x[start:end].tolist()) == sorted(ent.matched[start:end].tolist())
        start = end
    # hidden_max on the probe questions themselves equals their probe rows (same masks)
    hm = hidden_max(wm, ed, qs, ctx, ent.keys, batch_size=3)
    assert torch.allclose(hm, pm[:nq], atol=1e-5)
    fire = hidden_fire(hm, pm[nq], ent.matched, torch.ones(nq, dtype=torch.bool))
    assert fire['all'] == 0.0                         # matched thresholds never fire on their own probes
    assert hidden_fire(hm, pm[nq], torch.full((ent.n,), -1.0), torch.ones(nq, dtype=torch.bool))['all'] == 1.0
    ed.detach()


def _iv_setup(dup=False):
    """Three entries; entry 0 gets a large offset (it changes answers). With dup, entry 1 is a copy of entry 0 (same
    key, large offset): being newer it wins every tie, so entry 0 fires only once entry 1 is switched off."""
    wm = make_wm()
    ed = editor(wm, tau=0.2, probe_margin=None)
    ed.edit(wm, facts3(), '', seed=1)
    x = wm.extra
    keys, big = x['route_k'].clone(), x['route_v'].clone()
    big[0] = torch.randn(big.shape[1], generator=torch.Generator().manual_seed(0)) * 20
    if dup:
        keys[1], big[1] = keys[0], torch.randn(big.shape[1], generator=torch.Generator().manual_seed(1)) * 20
    ent = Entries(keys, big, torch.full((3,), 0.2), [3], [0, 0, 0],
                  [dict(uid=str(i), step=0, subject=f'Sub{i}', answer=f'z{i}') for i in range(3)], [])
    ctx = 'Q: Who is Loc1?\nA: c1\n'
    qs = ['What is the code of Sub1?', 'Which code does Sub2 have?', 'What colour is Loc9y2?', 'Hi?']
    off = torch.full((3,), OFF)
    install(wm, ent, 0, off)
    base = [norm(t) for t, _, _ in wm.generate(qs, contexts=[ctx] * 4, max_new=6, batch_size=4)]
    items = [('loc', q, {b}, q) for q, b in zip(qs, base)]       # 'right' = the answer with nothing firing

    def ev(t):
        install(wm, ent, 0, t)
        preds, hits = generate_logged(wm, ed, qs, [ctx] * 4, 6, 4)
        return dict(ok=''.join('1' if norm(p[0]) in it[2] else '0' for p, it in zip(preds, items)),
                    hits={i: [list(h) for h in hh] for i, hh in enumerate(hits) if hh},
                    pred={i: norm(p[0]) for i, p in enumerate(preds)})
    return wm, ed, ent, ctx, items, base, off, ev


def test_intervene_finds_the_single_culprit():
    wm, ed, ent, ctx, items, base, off, ev = _iv_setup()
    thr = torch.tensor([-1.0, OFF, OFF])                          # entry 0 fires everywhere
    ev_none, ev_arm = ev(off), ev(thr)
    assert ev_none['ok'] == '1111' and '0' in ev_arm['ok']
    res = intervene(wm, ed, ent, 0, thr, items, ctx, ev_arm, ev_none, EvalCfg(max_new=6, gen_batch=4), 8)
    assert len(res) == ev_arm['ok'].count('0')
    for r in res:
        assert r['fired'] == [0] and r['rerun'] == [0, r['pred']]
        c = r['closure']
        assert c['entries'] == [0] and c['rounds'] == 1 and c['complete'] and c['answer'] == [1, base[r['i']]]
        assert r['alone'] == [[0, 0, r['pred']]] and r['single'] == [[0, 1, base[r['i']]]]
    assert torch.equal(wm.extra['route_t'], thr)                  # the arm's thresholds are back
    ed.detach()


def test_intervene_closure_switches_off_runner_ups():
    wm, ed, ent, ctx, items, base, off, ev = _iv_setup(dup=True)
    thr = torch.tensor([-1.0, -1.0, OFF])                         # entry 1 wins every tie with entry 0
    ev_none, ev_arm = ev(off), ev(thr)
    assert ev_none['ok'] == '1111' and '0' in ev_arm['ok']
    assert all({e for e, _ in h} == {1} for h in ev_arm['hits'].values())
    res = intervene(wm, ed, ent, 0, thr, items, ctx, ev_arm, ev_none, EvalCfg(max_new=6, gen_batch=4), 8)
    for r in res:
        assert r['fired'] == [1]
        c = r['closure']
        assert c['entries'] == [0, 1] and c['rounds'] == 2 and c['complete'] and c['answer'] == [1, base[r['i']]]
        assert r['alone'][0][:2] == [1, 0]                        # entry 1 alone breaks the item
        assert r['single'][0][0] == 1                             # switching entry 1 off lets entry 0 take over
    ed.detach()


# ---- analysis ---------------------------------------------------------------------------------------------------
def synth(name, wrong: dict, cov: dict, n_loc=100, n_cur=20, steps=2, conc=None):
    """A minimal stream record: arms with `wrong[arm]` damaged locality items at the last step and rephrase accuracy
    cov[arm] at every step."""
    items, arms = [], {}
    for t in range(steps):
        items.append([['cur', f'u{t}_{j}|r'] for j in range(n_cur)] + [['ret', f'u0_{j}|r'] for j in range(5) if t]
                     + [['loc', f'loc{j}'] for j in range(n_loc)])
    for a in wrong:
        st = []
        for t in range(steps):
            n_ok = round(cov[a] * n_cur)
            bits = ['1'] * n_ok + ['0'] * (n_cur - n_ok) + (['1'] * 5 if t else [])
            loc = ['0' if (t == steps - 1 and j < wrong[a]) else '1' for j in range(n_loc)]
            ok = ''.join(bits + loc)
            acc = dict(cur=n_ok / n_cur, ret=1.0 if t else None, loc=loc.count('1') / n_loc, upd=1.0)
            pres = [v for k, v in acc.items() if k != 'upd' and v is not None]
            st.append(dict(step=t, acc=acc, composite=sum(pres) / len(pres), ok=ok, hits={}, pred={},
                           loc_right_to_wrong=wrong[a] if t == steps - 1 else 0))
        arms[a] = st
    meta = [dict(uid=f'u{t}_{j}', step=t, subject=f'S{t}{j}', answer='x', subj_in_rephrase=j % 2 == 0)
            for t in range(steps) for j in range(n_cur)]
    ivs = {}
    if conc is not None:
        ivs['uniform:0.90'] = [dict(i=j, question=f'loc{j}', pred='x', fired=[conc[j % len(conc)]], regions=[1],
                                    hits=[[conc[j % len(conc)], 1]], rerun=[0, 'x'],
                                    closure=dict(entries=[conc[j % len(conc)]], rounds=1, answer=[1, 'y'], complete=True),
                                    alone=[[conc[j % len(conc)], 0, 'x']], single=[[conc[j % len(conc)], 1, 'y']])
                               for j in range(12)]
    return dict(stream=name, steps=steps, items=items, arms=arms, base_loc={f'loc{j}': [1, 'y'] for j in range(n_loc)},
                entries=dict(n=len(meta), meta=meta, matched=[0.95 if j == 3 else 0.9 for j in range(len(meta))]),
                interventions=ivs, hidden=dict(n=100, entry=dict(n_floor=[j % 7 for j in range(len(meta))])))


RULE = dict(delta_damage=0.02, tol_rephrase=0.02, tol_retention=0.02, min_positive_streams=3, bootstrap=2000,
            bootstrap_seed=0, concentration_top_frac=0.01, concentration_share=0.5, concentration_streams=3,
            concentration_min_items=5, concentration_arm='uniform:0.90')


def recs_with(wrong, cov):
    return {f'D{i}': synth(f'D{i}', wrong, cov) for i in range(4, 8)}


def test_metrics_and_h1_verdicts():
    base = dict(none=0, matched=2, **{'shuffled:0': 14, 'shuffled:1': 16})
    cov = dict(none=0.0, matched=0.8, **{'shuffled:0': 0.8, 'shuffled:1': 0.8})
    recs = recs_with(base, cov)
    m = arm_metrics(recs['D4'], 'matched')
    assert m['damage'] == 0.02 and m['n_loc'] == 100 and abs(m['coverage'] - 0.8) < 1e-12
    assert shuffled_arms(recs['D4']) == ['shuffled:0', 'shuffled:1']
    c = contrast(recs, list(recs), shuffled_arms, 'matched', RULE)
    assert abs(c['mean_delta'] - 0.13) < 1e-12 and c['ci95'][0] > 0 and c['verdict'] == 'supported'
    same = recs_with(dict(none=0, matched=5, **{'shuffled:0': 5, 'shuffled:1': 5}), cov)
    assert contrast(same, list(same), shuffled_arms, 'matched', RULE)['verdict'] == 'rejected'
    costly = recs_with(base, dict(cov, matched=0.7))
    assert contrast(costly, list(costly), shuffled_arms, 'matched', RULE)['verdict'].startswith('undetermined: rephrase')


def test_tuning_and_concentration():
    wrong = {'none': 0, 'uniform:0.90': 20, 'uniform:0.95': 4}
    cov = {'none': 0.0, 'uniform:0.90': 0.9, 'uniform:0.95': 0.5}
    recs = {f'D{i}': synth(f'D{i}', wrong, cov, conc=[3, 3, 3, 7]) for i in range(4)}
    best, vals = tune_uniform(recs, list(recs))
    assert best == 'uniform:0.90' and vals['uniform:0.90'] > vals['uniform:0.95']
    c = concentration(recs['D0'], 'uniform:0.90', RULE, 0.9)
    assert c['with_firing'] == 12 and c['alone_explained'] == 12 and c['distinct_culprits'] == 2 and c['top_k'] == 1
    assert c['closure_complete'] == 12 and c['culprit_regions'] == {'q': 12} and c['hijack'] == 12
    assert c['top_culprits'][0]['entry'] == 3 and abs(c['top_share'] - 0.75) < 1e-12 and c['culprits_raised'] == 0.5
    per = {s: concentration(r, 'uniform:0.90', RULE, 0.9) for s, r in recs.items()}
    assert concentration_verdict(per, RULE) == 'concentrated'
    spread = {f'D{i}': synth(f'D{i}', wrong, cov, conc=list(range(12))) for i in range(4)}
    per = {s: concentration(r, 'uniform:0.90', RULE, 0.9) for s, r in spread.items()}
    assert concentration_verdict(per, RULE) == 'not concentrated'


def test_name_relation():
    assert name_relation('What language is predominantly used at CILT-FM?', 'CHUO-FM') == 'other'
    assert name_relation('In what language is CJRT-FM primarily broadcasted?', 'CJRT-FM') == 'same name'
    assert name_relation('Who founded Acme Rockets Ltd?', 'Acme Rocket') == 'same name'
    assert name_relation('Who founded Acme Rocketry?', 'Acme Rockets') == 'near name'


# ---- end to end -------------------------------------------------------------------------------------------------
def test_run_stream_and_analysis_end_to_end(small, tmp_path):
    import json
    from types import SimpleNamespace

    import kt_run
    from kt_analysis import analyze
    import dataclasses
    streams, index = small
    wm = make_wm()
    ed = editor(wm, tau=0.3)
    by_name = {}
    for s in streams:   # make the locality pool base-known: its answers are the base model's own
        qs = [q for q, _ in s.loc_hidden]
        base = wm.generate(qs, contexts=[shots_text(s.shots)] * len(qs), max_new=EV.max_new, batch_size=EV.gen_batch)
        by_name[s.name] = dataclasses.replace(s, loc_hidden=tuple((q, t) for q, (t, _, _) in zip(qs, base)))
    env = SimpleNamespace(wm=wm, editor=ed, rc=SimpleNamespace(ev=EV), streams=by_name, index=index)
    cfg = dict(cap=None, write_seed=0, dev=['D0'], test=['D1'], local=dict(tau=0.3),
               arms=['none', 'matched', 'shuffled:0', 'shuffled:1', 'uniform:0.30', 'uniform:0.60', 'ctx_only',
                     'probes:2'],
               intervene=dict(arms=['uniform:0.30', 'matched'], max_single=8),
               rule=dict(RULE, tuning='highest mean dev value', concentration_arm='uniform:0.30'))
    recs = {}
    for name in ('D0', 'D1'):
        rec = kt_run.run_stream(env, name, cfg)
        kt_run.write_gz(tmp_path / f'stream_{name}.json.gz', rec)
        recs[name] = kt_run.read_gz(tmp_path / f'stream_{name}.json.gz')
        assert rec['determinism']['same_ok'] and rec['determinism']['same_hits']
        assert rec['entries']['threshold_recompute_max_abs_diff'] < 1e-6 and rec['probes']['same_as_set_probes']
        assert set(rec['arms']) == set(cfg['arms']) and len(rec['items']) == rec['steps']
    res = analyze(recs, cfg)
    assert res['tuning']['tuned'] in ('uniform:0.30', 'uniform:0.60')
    for role in ('dev', 'test'):
        assert {'h1_matched_vs_shuffled', 'h2_matched_vs_tuned_uniform', 'h3_concentration'} <= set(res[role])
    json.dumps(res)
    ed.detach()
