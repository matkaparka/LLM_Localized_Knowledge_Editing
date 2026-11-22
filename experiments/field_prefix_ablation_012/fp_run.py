"""FP-12 command line. Nothing is downloaded: WikiBigEdit, the base model and the final spans must be local.

  python fp_run.py run CONFIG OUT --phase smoke|dev|test [--streams a,b]
        every stream of the phase: screening, the orig and corr tables, every arm (PLAN section 4) on the per-step
        panels and at the last step, the per-key interventions and the checks; one gzipped JSON record per stream
  python fp_run.py tune DEV_RUN CONFIG_DEV CONFIG_TEST_OUT
        the uniform-threshold rule (PLAN section 4) over the dev records -> the test config
  python fp_run.py analyze TEST_RUN [--dev DEV_RUN]
        the four-cell tables, decompositions, intervals and checks -> TEST_RUN/analysis.json

OUT must not exist yet. A run writes its config copy, code hashes (FP-12, EC-11, KT-10, RV-09, SO-08), the environment,
one record per stream and a manifest of every output. RV_DATA_DIR and RV_MODEL_PATH may replace the data and model
paths (recorded in run.json); the config file itself stays byte-identical.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fp_data  # noqa: E402
import fp_core  # noqa: E402
from fp_core import FPEditor, FPGen, Row, Table, arm_vector, at_step  # noqa: E402
from kt_core import OFF, build_entries, probe_max, probe_positions  # noqa: E402
from kt_core import thresholds as kt_thresholds  # noqa: E402
from rv_editor import EditWritable, Fact, LocalCfg  # noqa: E402
from rv_policy import SubjView, cap_stream, write_seed  # noqa: E402
from so_data import FactIndex, load_edits, norm, shots_text  # noqa: E402
from so_model import fmt_prompt  # noqa: E402
from so_policy import EvalCfg, eval_items  # noqa: E402

EXP = HERE.parent
CODE = {'fp12': (HERE, ('fp_data.py', 'fp_core.py', 'fp_run.py', 'fp_analysis.py')),
        'ec11': (EXP / 'editor_compare_011', ('ec_eval.py', 'ec_data.py')),
        'kt10': (EXP / 'key_thresholds_010', ('kt_core.py',)),
        'rv09': (EXP / 'reversal_pilot_009', ('rv_editor.py', 'rv_policy.py')),
        'so08': (EXP / 'self_observation_008', ('so_data.py', 'so_model.py', 'so_policy.py'))}
CELLS = ('orig/on', 'orig/off', 'corr/on', 'corr/off')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def write_gz(path: Path, obj):
    data = json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    with gzip.GzipFile(filename='', mode='wb', fileobj=open(path, 'wb'), mtime=0) as g:
        g.write(data)


def read_gz(path: Path):
    with gzip.open(path, 'rb') as g:
        return json.loads(g.read().decode('utf-8'))


def code_hashes() -> dict:
    return {k: {f: sha(d / f) for f in files if (d / f).exists()} for k, (d, files) in CODE.items()}


def start(out: Path, cfg_path, command: str, phase: str) -> dict:
    cfg_path = Path(cfg_path)
    if out.exists():
        raise SystemExit(f'{out} exists')
    out.mkdir(parents=True)
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    (out / 'config.json').write_bytes(cfg_path.read_bytes())
    env = {k: os.environ.get(k) for k in ('RV_DATA_DIR', 'RV_MODEL_PATH') if os.environ.get(k)}
    write_json(out / 'run.json', dict(command=command, phase=phase, config=cfg_path.name, config_sha256=sha(cfg_path),
                                      started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), code=code_hashes(),
                                      env_overrides=env, spans_sha256=sha(spans_path(cfg))))
    return cfg


def finish(out: Path):
    man = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()
           and p.name != 'manifest.json'}
    write_json(out / 'manifest.json', man)


def spans_path(cfg) -> Path:
    return HERE / cfg['spans']


# ---- environment ----------------------------------------------------------------------------------------------
class FPEnv:
    """Data of the dev half (KT-10's streams and corpus), the base model with RV-09's EditWritable, the FP editor
    prepared on the corpus, the fixed-shape generator and the final spans."""

    def __init__(self, cfg: dict, model=None, tok=None):
        import torch
        self.cfg = cfg
        edits, windows, _ = load_edits(os.environ.get('RV_DATA_DIR') or cfg['data_dir'],
                                       expect_steps=cfg.get('expect_steps', 8))
        self.index = FactIndex(edits)
        self.by_uid = {e.uid: e for e in edits}
        sc = cfg['streams']
        self.streams, self.corpus, self.filters = fp_data.kt10_half(edits, len(windows), sc['args'], sc['mu_corpus'])
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if not torch.cuda.is_available():
                raise SystemExit('no CUDA device')
            torch.manual_seed(int(cfg.get('torch_seed', 0)))
            path = os.environ.get('RV_MODEL_PATH') or cfg['model_path']
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                                                         dtype=getattr(torch, cfg.get('dtype', 'bfloat16'))).to('cuda')
        layers = range(model.config.num_hidden_layers) if cfg.get('layers', 'all') == 'all' else cfg['layers']
        sites = [s.format(i=i) for i in layers for s in cfg['sites']]
        self.wm = EditWritable(model, tok, sites, rank=int(cfg['rank']), scale=float(cfg['scale']),
                               clip=cfg.get('clip'))
        local = dict(cfg['local'])
        local['layers'] = tuple(local['layers'])
        self.editor = FPEditor(LocalCfg(**local))
        self.prep = self.editor.prepare(self.wm, self.corpus)
        ev = cfg['eval']
        self.gen = FPGen(self.wm, self.editor, ev['widths'], ev['batch'], ev['max_new'])
        sp = json.loads(spans_path(cfg).read_text(encoding='utf-8'))
        self.spans = sp['final']

    def plain(self):
        self.wm.extra = {}


def _floats(t, nd=6) -> list:
    return [round(float(x), nd) for x in t.tolist()]


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---- the corrected table -------------------------------------------------------------------------------------------
def corr_table(env: FPEnv, s, orig: Table, seed: int):
    """orig with the entries whose final anchor differs rewritten at that anchor (same step, context and write seed;
    each step's changed facts written together as one small batch). Unchanged entries are orig's, bit for bit."""
    import torch
    wm, ed = env.wm, env.editor
    keys, offs, thr = orig.keys.clone(), orig.offsets.clone(), orig.matched.clone()
    changed, records = [], []
    for t in range(s.steps):
        view = SubjView(s, t)
        answer = {uid: a for _, a, _, uid in view._batch}
        idx = [j for j in range(len(orig.uids)) if orig.step[j] == t and env.spans[orig.uids[j]]['changed']
               and not env.spans[orig.uids[j]]['unlocatable']]
        if not idx:
            continue
        facts = []
        for j in idx:
            sp = env.spans[orig.uids[j]]
            e = env.by_uid[orig.uids[j]]
            ed.anchors[(e.update, sp['text'])] = tuple(sp['span'])
            facts.append(Fact(e.update, answer[orig.uids[j]], sp['text']))
        wm.restore(wm.empty_state())
        wm.extra = {}
        rec = ed.edit(wm, facts, view.context(), write_seed(seed, s, t))
        k, v, th = wm.extra['route_k'], wm.extra['route_v'], wm.extra['route_t']
        if k.shape[0] != len(idx):
            raise RuntimeError(f'step {t}: {k.shape[0]} entries for {len(idx)} changed facts')
        sel = torch.tensor(idx, device=keys.device)
        keys[sel], offs[sel], thr[sel] = k, v, th
        changed += idx
        records.append(dict(step=t, n=len(idx), found=rec.get('subject_found'), z_loss_mean=rec.get('z_loss_mean')))
    wm.extra = {}
    ed.anchors.clear()
    tab = Table('corr', keys, offs, thr, list(orig.ends), list(orig.step), list(orig.uids), changed=changed)
    return tab, records


# ---- items ---------------------------------------------------------------------------------------------------------
def stream_items(env: FPEnv, s, known) -> list[tuple]:
    """(comp, question, valid answers, key, step): cur/upd panels of every step, ret at T, the base-known locality
    facts at T and the whole stream's rephrasings at T ('cov')."""
    ev = EvalCfg(**env.cfg['panels'])
    T = s.steps - 1
    items = []
    for t in range(s.steps):
        for comp, q, valid, key in eval_items(s, env.index, t, 0, ev):
            if comp in ('cur', 'upd') or (t == T and comp == 'ret'):
                items.append((comp, q, frozenset(valid), key, t))
    for q, a in known:
        items.append(('loc', q, frozenset({norm(a)}), q, T))
    for e in s.edits:
        if e.rephrase:
            items.append(('cov', e.rephrase, frozenset(env.index.valid(e.key, T)), f'{e.uid}|c', T))
    return items


def own_entry(item_key: str, uid_index: dict):
    uid = item_key.split('|', 1)[0]
    return uid_index.get(uid)


# ---- one stream ----------------------------------------------------------------------------------------------------
def run_stream(env: FPEnv, name: str, phase: str, arms_of: dict, interventions: dict) -> dict:
    import torch
    cfg, wm, ed, gen = env.cfg, env.wm, env.editor, env.gen
    tok = wm.tok
    lc = ed.cfg
    clock, checks = {}, {}
    cap = cfg['phases'][phase].get('cap', fp_data.KT10_CAP)
    s = cap_stream(env.streams[name], cap)
    ed.anchors.clear()                    # explicit anchors only ever act inside corr_table
    T = s.steps - 1
    ctx = shots_text(s.shots)
    nctx = len(tok.encode(ctx, add_special_tokens=False))
    seed = int(cfg['write_seed'])

    # 1. base-known hidden facts in the evaluation's own fixed-shape layout
    t0 = time.time()
    env.plain()
    cand = list(s.loc_hidden)
    cids = [tok.encode(fmt_prompt(q, ctx), add_special_tokens=False) for q, _ in cand]
    couts = gen.generate(cids, [nctx] * len(cids))
    known = [(q, a) for (q, a), o in zip(cand, couts) if norm(o.text) == norm(a)]
    if cfg['phases'][phase].get('order_check'):
        rev = gen.generate(cids[::-1], [nctx] * len(cids))[::-1]
        checks['screen_reversed_identical'] = all(a.text == b.text for a, b in zip(couts, rev))
        checks['screen_reversed_mismatches'] = sum(a.text != b.text for a, b in zip(couts, rev))
    clock['screen'] = time.time() - t0

    # 2. probes, the orig table (KT-10's writer, written once), the corr table, strict thresholds
    t1 = time.time()
    probes = list(s.loc_probe)
    states, labels = probe_positions(wm, ed, probes, ctx)
    ed.set_probes(wm, probes, ctx)
    checks['probes_same_as_set_probes'] = bool(torch.equal(states, ed.probe_states))
    ed.probe_states = states
    nq = len(probes)
    ent = build_entries(wm, ed, s, seed)
    env.plain()
    uids = [m['uid'] for m in ent.meta]
    orig = Table('orig', ent.keys, ent.offsets, ent.matched, list(ent.ends), list(ent.step), uids)
    pm = probe_max(states, labels, ent.keys, nq + 1)
    recomputed = torch.clamp(pm.max(0).values + lc.probe_margin, min=lc.tau)
    checks['matched_recompute_max_abs_diff'] = float((recomputed - ent.matched).abs().max())
    clock['orig_table'] = time.time() - t1
    t2 = time.time()
    corr, corr_records = corr_table(env, s, orig, seed)
    unchanged = [j for j in range(orig.n) if j not in set(corr.changed)]
    if unchanged:
        u = torch.tensor(unchanged, device=orig.keys.device)
        checks['corr_unchanged_bitwise'] = bool(torch.equal(orig.keys[u], corr.keys[u])
                                                and torch.equal(orig.offsets[u], corr.offsets[u])
                                                and torch.equal(orig.matched[u], corr.matched[u]))
    strict_states = fp_core.strict_probe_states(wm, ed, probes, ctx)
    strict_matched = torch.clamp(fp_core.max_sim(strict_states, corr.keys) + lc.probe_margin, min=lc.tau)
    clock['corr_table'] = time.time() - t2
    dev = orig.keys.device
    refused = torch.tensor([bool(env.spans[u]['unlocatable']) for u in uids], device=dev)
    active_paired, active_all = ~refused, torch.ones_like(refused)

    # 3. every arm's vector at T
    vec, cell_of = {}, {}

    def put(arm, v, table, mask):
        vec[arm] = (v, table, mask)

    put('none', torch.full_like(orig.matched, OFF), 'orig', 'on')
    for cell in CELLS:
        row, mask = cell.split('/')
        tab = orig if row == 'orig' else corr
        for a in arms_of['cell']:
            put(f'{cell}:{a}', arm_vector(a, tab.matched, tab.ends, active_paired, s.name), row, mask)
    for a in arms_of['bridge']:
        v = kt_thresholds(a, ent, pm, nq, lc, s.name) if a.startswith('shuffled') else \
            arm_vector(a, orig.matched, orig.ends, active_all, s.name)
        put(f'bridge:{a}', v.to(dev).float(), 'orig', 'on')
    for a in arms_of['strict']:
        put(f'strict:{a}', arm_vector(a, strict_matched, corr.ends, active_paired, s.name), 'corr', 'strict')
    checks['shuffle_multisets_equal'] = all(
        fp_core.multisets_equal(vec[a][0], vec[a.rsplit(':', 2)[0] + ':matched'][0], orig.ends)
        for a in vec if ':shuffled:' in a)
    checks['refused_never_active'] = all(bool((vec[a][0][refused] >= OFF).all()) for a in vec
                                         if not a.startswith('bridge:'))

    # 4. items and rows; per table one generation pass over all its arms
    items = stream_items(env, s, known)
    ids = [tok.encode(fmt_prompt(q, ctx), add_special_tokens=False) for _, q, _, _, _ in items]
    markers = [fp_core.marker_tokens(tok, fmt_prompt(q, ctx), len(ctx)) for _, q, _, _, _ in items]
    for i in range(min(50, len(ids))):
        if ids[i][:nctx] != tok.encode(ctx, add_special_tokens=False):
            raise RuntimeError('the context does not tokenise as a prefix of the prompt')
    res = {}
    t3 = time.time()
    for tname, tab in (('orig', orig), ('corr', corr)):
        mine = [a for a in vec if vec[a][1] == tname]
        mats, where, rows, back = [], {}, [], []
        for a in mine:
            v, _, mask = vec[a]
            for t in range(s.steps):
                where[(a, t)] = len(mats)
                mats.append(at_step(v, tab.ends, t))
            seen = {}
            for i, (comp, q, valid, key, t) in enumerate(items):
                k = (q, t)
                if k not in seen:
                    seen[k] = len(rows)
                    rows.append(Row(ids[i], nctx, where[(a, t)], fp_core.blocked_positions(mask, nctx, markers[i])))
                back.append((a, i, seen[k]))
        tab.install(wm)
        outs = gen.run(rows, torch.stack(mats))
        for a, i, r in back:
            res.setdefault(a, [None] * len(items))[i] = outs[r]
        env.plain()
        clock[f'eval_{tname}'] = time.time() - t3
        t3 = time.time()

    # 5. per-arm records
    uid_index = {u: j for j, u in enumerate(uids)}
    none_out = res['none']
    arms, loc_idx = {}, [i for i, it in enumerate(items) if it[0] == 'loc']
    unfired_diff = {}
    for a, outs in res.items():
        ok, reg, own, wrong, hits = [], [], [], {}, {}
        diff = 0
        for i, (it, o) in enumerate(zip(items, outs)):
            good = norm(o.text) in it[2]
            ok.append('1' if good else '0')
            m = 0
            for e, g in o.fired:
                m |= 1 << g
            reg.append(str(m))
            if it[0] == 'loc':
                own.append('-')
                if not good:
                    wrong[i] = norm(o.text)
                if o.fired:
                    hits[i] = [list(x) for x in o.fired]
            else:
                oe = own_entry(it[3], uid_index)
                es = {e for e, _ in o.fired}
                own.append(str((1 if oe in es else 0) + (2 if es - {oe} else 0)))
            if not o.fired and o.text != none_out[i].text:
                diff += 1
        unfired_diff[a] = diff
        arms[a] = dict(ok=''.join(ok), reg=''.join(reg), own=''.join(own), wrong_loc=wrong, loc_hits=hits)
    checks['unfired_differs_from_none'] = unfired_diff
    checks['none_known_all_right'] = all(arms['none']['ok'][i] == '1' for i in loc_idx)
    checks['blocked_cells_never_fire_in_context'] = all(
        all(not (int(ch) & 1) for ch in arms[a]['reg']) for a in arms if '/off:' in a or a.startswith('strict:'))

    # 6. order check: one arm's locality rows in reversed order
    darm = cfg['determinism_arm']
    v, tname, mask = vec[darm]
    tab = orig if tname == 'orig' else corr
    tab.install(wm)
    rows = [Row(ids[i], nctx, 0, fp_core.blocked_positions(mask, nctx, markers[i])) for i in loc_idx][::-1]
    again = gen.run(rows, at_step(v, tab.ends, T).unsqueeze(0))[::-1]
    checks['determinism'] = dict(arm=darm, loc_reversed_same=all(o.text == res[darm][i].text
                                                                 for i, o in zip(loc_idx, again)))

    # 7. per-key interventions at T
    t4 = time.time()
    ivs = {}
    for cell in CELLS:
        tab = orig if cell.startswith('orig') else corr
        tab.install(wm)
        mask = cell.split('/')[1]
        loc_rows = {i: Row(ids[i], nctx, 0, fp_core.blocked_positions(mask, nctx, markers[i])) for i in loc_idx}
        rows_all = [loc_rows.get(i, Row(ids[i], nctx, 0, ())) for i in range(len(items))]
        for a in interventions['arms']:
            arm = f'{cell}:{a}'
            damaged = [i for i in loc_idx if arms['none']['ok'][i] == '1' and arms[arm]['ok'][i] == '0']
            fired = {i: [tuple(x) for x in arms[arm]['loc_hits'].get(i, [])] for i in damaged}
            ivs[arm] = fp_core.intervene(gen, at_step(vec[arm][0], tab.ends, T), rows_all, damaged, fired,
                                         int(interventions['max_single']))
            if cfg['phases'][phase].get('intervention_check') and damaged and 'interventions_batched' not in checks:
                solo = [fp_core.intervene(gen, at_step(vec[arm][0], tab.ends, T), rows_all, [i], fired,
                                          int(interventions['max_single']))[0] for i in damaged[:20]]
                checks['interventions_batched'] = dict(arm=arm, items=len(solo),
                                                       equal_single=solo == ivs[arm][:len(solo)])
    env.plain()
    clock['interventions'] = time.time() - t4

    # 8. the record
    spans = [env.spans[u] for u in uids]
    rec = dict(
        stream=name, phase=phase, steps=s.steps, cap=cap,
        screening=dict(candidates=len(cand), known=len(known)),
        probes=dict(questions=nq, positions=int(states.shape[0]), strict_positions=int(strict_states.shape[0])),
        tables=dict(orig=orig.digest(), corr=corr.digest(), n=orig.n, ends=orig.ends, step=orig.step, uids=uids,
                    changed=corr.changed, corr_records=corr_records, refused=[j for j in range(orig.n) if spans[j]['unlocatable']],
                    entity=[(sp.get('text') or env.by_uid[u].subject) for sp, u in zip(spans, uids)],
                    subject=[env.by_uid[u].subject for u in uids]),
        thresholds=dict(orig=_floats(orig.matched), corr=_floats(corr.matched), strict=_floats(strict_matched)),
        items=[[c, k, t] for c, _, _, k, t in items], loc_questions={i: items[i][1] for i in loc_idx},
        loc_answers={i: next(iter(items[i][2])) for i in loc_idx},
        arms=arms, interventions=ivs, checks=checks, clock=clock, costs=dict(wm.costs))
    return rec


# ---- commands ------------------------------------------------------------------------------------------------------
def phase_arms(cfg: dict, phase: str) -> dict:
    a = cfg['arms']
    cell = ['matched'] + [f'shuffled:{p}' for p in range(a['n_shuffles'])] + [f'uniform:{x:.2f}' for x in a['uniform']]
    bridge = ['matched'] + [f'shuffled:{p}' for p in range(a['n_shuffles'])] + ['uniform:0.90']
    strict = ['matched'] + [f'shuffled:{p}' for p in range(a['n_shuffles'])]
    return dict(cell=cell, bridge=bridge, strict=strict)


def measured(env: FPEnv, fn) -> dict:
    import torch
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rec = fn()
    rec['seconds'] = time.time() - t0
    if torch.cuda.is_available():
        rec['peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
    return rec


def cmd_run(cfg_path, out: Path, phase: str, streams: list[str] | None = None, env_factory=None):
    cfg = start(out, cfg_path, 'run', phase)
    names = list(cfg['phases'][phase]['streams'])
    if streams:
        unknown = [x for x in streams if x not in names]
        if unknown:
            raise SystemExit(f'not in the phase: {unknown}')
        names = [x for x in names if x in streams]
    env = env_factory(cfg) if env_factory else FPEnv(cfg)
    arms_of = phase_arms(cfg, phase)
    for name in names:
        print(f'stream {name}', flush=True)
        rec = measured(env, lambda: run_stream(env, name, phase, arms_of, cfg['interventions']))
        write_gz(out / f'stream_{name}.json.gz', rec)
        c = rec['checks']
        print(f'stream {name} done: {rec["seconds"]:.0f} s; known {rec["screening"]["known"]}; '
              f'unfired!=none max {max(c["unfired_differs_from_none"].values())}; '
              f'none right {c["none_known_all_right"]}; reversed same {c["determinism"]["loc_reversed_same"]}',
              flush=True)
    finish(out)


def load_records(run_dir: Path) -> dict:
    return {p.name[len('stream_'):-len('.json.gz')]: read_gz(p) for p in sorted(run_dir.glob('stream_*.json.gz'))}


def cmd_tune(dev_run: Path, cfg_dev_path, out_path: Path):
    import fp_analysis
    cfg_dev_path, out_path = Path(cfg_dev_path), Path(out_path)
    if out_path.exists():
        raise SystemExit(f'{out_path} exists')
    cfg = json.loads(cfg_dev_path.read_text(encoding='utf-8'))
    recs = load_records(Path(dev_run))
    if set(recs) != set(cfg['phases']['dev']['streams']):
        raise SystemExit(f'dev records {sorted(recs)} != {cfg["phases"]["dev"]["streams"]}')
    sel = fp_analysis.tune(recs, cfg)
    test = dict(cfg)
    test['status'] = (f'TEST CONFIG written by fp_run.py tune from {Path(dev_run).name} on '
                      f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; config_dev.json sha256 '
                      f'{sha(cfg_dev_path)}. Selected uniform thresholds per cell (PLAN section 4).')
    test['selected_uniform'] = sel
    out_path.write_text(json.dumps(test, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(sel, indent=1))


def cmd_analyze(test_run: Path, dev_run: Path | None = None):
    import fp_analysis
    test_run = Path(test_run)
    cfg = json.loads((test_run / 'config.json').read_text(encoding='utf-8'))
    recs = load_records(test_run)
    dev = load_records(Path(dev_run)) if dev_run else None
    res = fp_analysis.analyze(recs, cfg, dev)
    write_json(test_run / 'analysis.json', res)
    finish(test_run)
    print(json.dumps(res.get('headline', {}), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('run', 'tune', 'analyze'))
    ap.add_argument('args', nargs='+')
    ap.add_argument('--phase', default=None)
    ap.add_argument('--streams', default=None)
    ap.add_argument('--dev', default=None)
    a = ap.parse_args(argv)
    if a.command == 'run':
        cmd_run(a.args[0], Path(a.args[1]), a.phase, a.streams.split(',') if a.streams else None)
    elif a.command == 'tune':
        cmd_tune(Path(a.args[0]), a.args[1], Path(a.args[2]))
    else:
        cmd_analyze(Path(a.args[0]), Path(a.dev) if a.dev else None)


if __name__ == '__main__':
    main()
