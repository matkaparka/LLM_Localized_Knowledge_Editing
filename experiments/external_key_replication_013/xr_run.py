"""XR-13 command line (PLAN sections 1, 4, 5). The writer, the evaluator, the threshold arms, the blocking and the
interventions are FP-12's (fp_core), imported unchanged; only the data differ.

  python xr_run.py run CONFIG OUT --phase debug|dev|test [--streams a,b]
  python xr_run.py tune DEV_RUN CONFIG_DEV CONFIG_TEST_OUT
  python xr_run.py analyze TEST_RUN [--dev DEV_RUN]

Data: CONFIG['streams_dir'] holds one <family>.json per stream (edits with the three wordings, the extractor's span or
null, hidden candidates, public probes, format shots) and mu_corpus.json (the routing-centre corpus). Every key is
anchored at the extractor's span (fp_core.FPEditor explicit anchors); refused edits keep their entry and UID but can
never fire. Calibration and execution both block the format shots and the fixed 'Q: ' / '\\nA:' tokens (FP-12's
strict blocking); question and generated positions may fire.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FP = HERE.parent / 'field_prefix_ablation_012'
sys.path.insert(0, str(FP))
sys.path.insert(0, str(HERE))       # python -I does not put the script's folder on the path
import fp_data  # noqa: E402,F401  (import paths)
import fp_core  # noqa: E402
import fp_run  # noqa: E402
from fp_core import FPEditor, FPGen, Row, arm_vector, at_step  # noqa: E402
from kt_core import OFF, build_entries, probe_max, probe_positions  # noqa: E402
from rv_editor import EditWritable, LocalCfg  # noqa: E402
from rv_policy import cap_stream  # noqa: E402
from so_data import Edit, FactIndex, Stream, norm, shots_text  # noqa: E402
from so_model import fmt_prompt  # noqa: E402

fp_run.CODE['xr13'] = (HERE, ('xr_run.py', 'xr_analysis.py'))


def load_streams(sdir: Path, names: list[str]) -> tuple[dict, dict]:
    """Stream objects and, per stream, the extractor spans {uid: (start, end) or None} and the gold entity names."""
    streams, extra = {}, {}
    for name in names:
        d = json.loads((sdir / f'{name}.json').read_text(encoding='utf-8'))
        edits = []
        for e in d['edits']:
            edits.append(Edit(uid=e['uid'], step=int(e['step']), tag='update', subject=e.get('span_text') or '',
                              subject_id=f'{name}:{e["entity"]}', relation=e['attribute'], relation_id=e['attribute'],
                              answer=e['value'], object_id=None, update=e['update'], rephrase=e['rephrase'],
                              personas=e['retention'], loc=None, loc_ans=None, mhop=None, mhop_ans=None))
        streams[name] = Stream(name=name, half='xr13', steps=int(d['steps']), edits=tuple(edits),
                               loc_probe=tuple(d['probes']), loc_hidden=tuple((q, a) for q, a in d['hidden']),
                               shots=tuple((q, a) for q, a in d['shots']))
        extra[name] = dict(spans={e['uid']: e.get('span') for e in d['edits']},
                           entity={e['uid']: e['entity'] for e in d['edits']}, role=d.get('role'))
    return streams, extra


class XREnv:
    def __init__(self, cfg: dict, phase: str, model=None, tok=None):
        import torch
        self.cfg, self.phase = cfg, phase
        sdir = Path(os.environ.get('XR_STREAMS_DIR') or (HERE / cfg['streams_dir']))
        names = cfg['phases'][phase]['streams']
        self.streams, self.extra = load_streams(sdir, names)
        self.index = FactIndex([e for s in self.streams.values() for e in s.edits])
        self.corpus = json.loads((sdir / 'mu_corpus.json').read_text(encoding='utf-8'))['texts']
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
        self.wm = EditWritable(model, tok, sites, rank=int(cfg['rank']), scale=float(cfg['scale']), clip=cfg.get('clip'))
        local = dict(cfg['local'])
        local['layers'] = tuple(local['layers'])
        self.editor = FPEditor(LocalCfg(**local))
        self.prep = self.editor.prepare(self.wm, self.corpus)
        ev = cfg['eval']
        self.gen = FPGen(self.wm, self.editor, ev['widths'], ev['batch'], ev['max_new'])

    def plain(self):
        self.wm.extra = {}


def gold_span(question: str, entity: str):
    q = question.strip()
    k = q.rfind(entity)
    return [k, k + len(entity)] if k >= 0 else None


def run_stream(env: XREnv, name: str, phase: str) -> dict:
    import torch
    cfg, wm, ed, gen = env.cfg, env.wm, env.editor, env.gen
    tok, lc = wm.tok, ed.cfg
    clock, checks = {}, {}
    s = cap_stream(env.streams[name], cfg['phases'][phase].get('cap'))
    T = s.steps - 1
    ctx = shots_text(s.shots)
    nctx = len(tok.encode(ctx, add_special_tokens=False))
    seed = int(cfg['write_seed'])
    spans = env.extra[name]['spans']

    t0 = time.time()                                  # 1. base-known hidden facts, fixed-shape layout
    env.plain()
    cand = list(s.loc_hidden)
    cids = [tok.encode(fmt_prompt(q, ctx), add_special_tokens=False) for q, _ in cand]
    couts = gen.generate(cids, [nctx] * len(cids))
    known = [(q, a) for (q, a), o in zip(cand, couts) if norm(o.text) == norm(a)]
    if cfg['phases'][phase].get('order_check'):
        rev = gen.generate(cids[::-1], [nctx] * len(cids))[::-1]
        checks['screen_reversed_identical'] = all(a.text == b.text for a, b in zip(couts, rev))
    clock['screen'] = time.time() - t0

    t1 = time.time()                                  # 2. probes, the table at the extractor's anchors
    probes = list(s.loc_probe)
    states, labels = probe_positions(wm, ed, probes, ctx)
    ed.set_probes(wm, probes, ctx)
    ed.probe_states = states
    ed.anchors.clear()
    for e in s.edits:
        sp = spans.get(e.uid)
        if sp:
            ed.anchors[(e.update, e.subject)] = tuple(sp)
    ent = build_entries(wm, ed, s, seed)
    ed.anchors.clear()
    env.plain()
    uids = [m['uid'] for m in ent.meta]
    strict_states = fp_core.strict_probe_states(wm, ed, probes, ctx)
    matched = torch.clamp(fp_core.max_sim(strict_states, ent.keys) + lc.probe_margin, min=lc.tau)
    tab = fp_core.Table('xr', ent.keys, ent.offsets, matched, list(ent.ends), list(ent.step), uids)
    refused = torch.tensor([spans.get(u) is None for u in uids], device=ent.keys.device)
    active = ~refused
    clock['table'] = time.time() - t1

    arms_cfg = cfg['arms']
    names = (['matched'] + [f'shuffled:{p}' for p in range(arms_cfg['n_shuffles'])] +
             [f'uniform:{x:.2f}' for x in arms_cfg['uniform']])
    vec = {'none': torch.full_like(matched, OFF)}
    for a in names:
        vec[f'xr:{a}'] = arm_vector(a, matched, tab.ends, active, s.name)
    checks['shuffle_multisets_equal'] = all(fp_core.multisets_equal(vec[a], vec['xr:matched'], tab.ends)
                                            for a in vec if ':shuffled:' in a)
    checks['refused_never_active'] = all(bool((v[refused] >= OFF).all()) for v in vec.values())

    items = fp_run.stream_items(env, s, known)        # 3. every arm on the panels and at T, strict blocking
    ids = [tok.encode(fmt_prompt(q, ctx), add_special_tokens=False) for _, q, _, _, _ in items]
    markers = [fp_core.marker_tokens(tok, fmt_prompt(q, ctx), len(ctx)) for _, q, _, _, _ in items]
    block = [fp_core.blocked_positions('strict', nctx, m) for m in markers]
    mats, where, rows, back = [], {}, [], []
    for a, v in vec.items():
        for t in range(s.steps):
            where[(a, t)] = len(mats)
            mats.append(at_step(v, tab.ends, t))
        seen = {}
        for i, (comp, q, valid, key, t) in enumerate(items):
            if (q, t) not in seen:
                seen[(q, t)] = len(rows)
                rows.append(Row(ids[i], nctx, where[(a, t)], block[i]))
            back.append((a, i, seen[(q, t)]))
    t2 = time.time()
    tab.install(wm)
    outs = gen.run(rows, torch.stack(mats))
    res = {}
    for a, i, r in back:
        res.setdefault(a, [None] * len(items))[i] = outs[r]
    env.plain()
    clock['eval'] = time.time() - t2

    uid_index = {u: j for j, u in enumerate(uids)}
    arms, unfired = {}, {}
    loc_idx = [i for i, it in enumerate(items) if it[0] == 'loc']
    for a, o_list in res.items():
        ok, reg, own, wrong, hits, diff = [], [], [], {}, {}, 0
        for i, (it, o) in enumerate(zip(items, o_list)):
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
                oe = fp_run.own_entry(it[3], uid_index)
                es = {e for e, _ in o.fired}
                own.append(str((1 if oe in es else 0) + (2 if es - {oe} else 0)))
            if not o.fired and o.text != res['none'][i].text:
                diff += 1
        unfired[a] = diff
        arms[a] = dict(ok=''.join(ok), reg=''.join(reg), own=''.join(own), wrong_loc=wrong, loc_hits=hits)
    checks['unfired_differs_from_none'] = unfired
    checks['none_known_all_right'] = all(arms['none']['ok'][i] == '1' for i in loc_idx)
    checks['blocked_never_fire_in_context'] = all(all(not (int(ch) & 1) for ch in arms[a]['reg']) for a in arms)
    tab.install(wm)
    rrows = [Row(ids[i], nctx, 0, block[i]) for i in loc_idx][::-1]
    again = gen.run(rrows, at_step(vec['xr:matched'], tab.ends, T).unsqueeze(0))[::-1]
    checks['determinism'] = dict(arm='xr:matched', loc_reversed_same=all(o.text == res['xr:matched'][i].text
                                                                        for i, o in zip(loc_idx, again)))

    t3 = time.time()                                  # 4. per-key interventions at T
    ivs = {}
    rows_all = [Row(ids[i], nctx, 0, block[i]) for i in range(len(items))]
    for a in cfg['interventions']['arms']:
        arm = f'xr:{a}'
        damaged = [i for i in loc_idx if arms['none']['ok'][i] == '1' and arms[arm]['ok'][i] == '0']
        fired = {i: [tuple(x) for x in arms[arm]['loc_hits'].get(i, [])] for i in damaged}
        ivs[arm] = fp_core.intervene(gen, at_step(vec[arm], tab.ends, T), rows_all, damaged, fired,
                                     int(cfg['interventions']['max_single']))
    env.plain()
    clock['interventions'] = time.time() - t3

    gold = {u: gold_span(e.update, env.extra[name]['entity'][u]) for u, e in ((e.uid, e) for e in s.edits)}
    ext = dict(n=len(uids), refused=int(refused.sum()),
               exact=sum(1 for u in uids if spans.get(u) and gold.get(u) == spans[u]),
               same_anchor=sum(1 for u in uids if spans.get(u) and gold.get(u)
                               and fp_data.token_at(tok, fp_data.bare_prompt(env_q(s, u)), len(fp_data.Q_PREFIX) + spans[u][1] - 1)
                               == fp_data.token_at(tok, fp_data.bare_prompt(env_q(s, u)), len(fp_data.Q_PREFIX) + gold[u][1] - 1)))
    return dict(stream=name, phase=phase, role=env.extra[name]['role'], steps=s.steps,
                screening=dict(candidates=len(cand), known=len(known)),
                probes=dict(questions=len(probes), positions=int(states.shape[0]), strict_positions=int(strict_states.shape[0])),
                tables=dict(xr=tab.digest(), n=tab.n, ends=tab.ends, step=tab.step, uids=uids, changed=[],
                            refused=[j for j in range(tab.n) if bool(refused[j])],
                            entity=[env.extra[name]['entity'][u] for u in uids]),
                thresholds=dict(matched=fp_run._floats(matched), writer_own=fp_run._floats(ent.matched)),
                extraction=ext, items=[[c, k, t] for c, _, _, k, t in items],
                loc_questions={i: items[i][1] for i in loc_idx}, loc_answers={i: next(iter(items[i][2])) for i in loc_idx},
                arms=arms, interventions=ivs, checks=checks, clock=clock, costs=dict(wm.costs))


def env_q(s, uid):
    for e in s.edits:
        if e.uid == uid:
            return e.update
    raise KeyError(uid)


def cmd_run(cfg_path, out: Path, phase: str, streams=None, env_factory=None):
    cfg = json.loads(Path(cfg_path).read_text(encoding='utf-8'))
    if out.exists():
        raise SystemExit(f'{out} exists')
    out.mkdir(parents=True)
    (out / 'config.json').write_bytes(Path(cfg_path).read_bytes())
    fp_run.write_json(out / 'run.json', dict(command='run', phase=phase, config_sha256=fp_run.sha(Path(cfg_path)),
                                             started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                             code=fp_run.code_hashes()))
    names = [x for x in cfg['phases'][phase]['streams'] if not streams or x in streams]
    env = env_factory(cfg, phase) if env_factory else XREnv(cfg, phase)
    for name in names:
        print(f'stream {name}', flush=True)
        rec = fp_run.measured(env, lambda: run_stream(env, name, phase))
        fp_run.write_gz(out / f'stream_{name}.json.gz', rec)
        c = rec['checks']
        print(f'stream {name} done: {rec["seconds"]:.0f} s; known {rec["screening"]["known"]}; refused '
              f'{rec["extraction"]["refused"]}; unfired!=none max {max(c["unfired_differs_from_none"].values())}',
              flush=True)
    fp_run.finish(out)


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
    else:
        import xr_analysis
        if a.command == 'tune':
            xr_analysis.cmd_tune(Path(a.args[0]), a.args[1], Path(a.args[2]))
        else:
            xr_analysis.cmd_analyze(Path(a.args[0]), Path(a.dev) if a.dev else None)


if __name__ == '__main__':
    main()
