"""EC-11 command line. Nothing is downloaded: the WikiBigEdit files, the base model and the retriever must be local.

  python ec_run.py run CONFIG OUT --phase smoke|dev|test [--streams a,b]
        every stream of the phase: screening, KT entries, GRACE values, retrieval memory, then every arm of the
        phase evaluated once after the last write; one gzipped JSON record per stream
  python ec_run.py tune DEV_RUN CONFIG_DEV CONFIG_TEST_OUT
        the tuning rule (PLAN section 5) over the dev records -> the test config (selected arms filled in)
  python ec_run.py analyze TEST_RUN [--dev DEV_RUN]
        verdicts, frontiers and checks -> TEST_RUN/analysis.json

OUT must not exist yet. A run writes its config copy, code hashes (EC-11, KT-10, RV-09, SO-08), the environment, one
record per stream and a manifest of every output. Data, model and retriever paths may be replaced by RV_DATA_DIR,
RV_MODEL_PATH and EC_RETRIEVER_PATH (recorded in run.json); the config file itself stays byte-identical.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KT10 = HERE.parent / 'key_thresholds_010'
for d in (HERE, KT10):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import ec_data  # noqa: E402
from ec_arms import (OFF, family, kt_thresholds, parse_grace, parse_kt, rad_thresholds,  # noqa: E402
                     with_hygiene)
from ec_eval import FixedShapeGen, edit_items, loc_items, plain_prompts, score  # noqa: E402
import ec_grace  # noqa: E402
import ec_rag  # noqa: E402
from kt_core import LoggedEditor, build_entries, install, probe_max, probe_positions  # noqa: E402
from rv_editor import SO08, EditWritable, LocalCfg  # noqa: E402
from rv_policy import cap_stream  # noqa: E402
import rv_run  # noqa: E402
from so_data import FactIndex, load_edits, norm, shots_text  # noqa: E402
from so_model import fmt_prompt, fmt_target  # noqa: E402

CODE = ('ec_data.py', 'ec_eval.py', 'ec_grace.py', 'ec_rag.py', 'ec_arms.py', 'ec_run.py', 'ec_analysis.py')
KT10_CODE = ('kt_core.py',)
RV9_CODE = ('rv_editor.py', 'rv_policy.py', 'rv_analysis.py', 'rv_run.py')
SO08_CODE = ('so_data.py', 'so_model.py', 'so_policy.py', 'so_analysis.py')
PATH_ENV = {'data_dir': 'RV_DATA_DIR', 'model_path': 'RV_MODEL_PATH'}
RETRIEVER_ENV = 'EC_RETRIEVER_PATH'


# ---- files --------------------------------------------------------------------------------------------------------
def sha(path: Path) -> str:
    return rv_run.sha(path)


def write_json(path: Path, obj):
    rv_run.write_json(path, obj)


def write_gz(path: Path, obj):
    data = json.dumps(obj, ensure_ascii=False, separators=(',', ':'), default=rv_run._default).encode('utf-8')
    with open(path, 'wb') as f, gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as g:
        g.write(data)


def read_gz(path: Path):
    with gzip.open(path, 'rb') as g:
        return json.loads(g.read().decode('utf-8'))


def load_config(cfg_path) -> tuple[dict, dict]:
    cfg = json.loads(Path(cfg_path).read_text(encoding='utf-8'))
    over = {k: os.environ[v] for k, v in PATH_ENV.items() if os.environ.get(v)}
    cfg.update(over)
    if os.environ.get(RETRIEVER_ENV):
        over['rag.model_path'] = os.environ[RETRIEVER_ENV]
        cfg['rag'] = dict(cfg['rag'], model_path=os.environ[RETRIEVER_ENV])
    return cfg, over


def code_hashes() -> dict:
    return dict(code={f: sha(HERE / f) for f in CODE}, kt10_code={f: sha(KT10 / f) for f in KT10_CODE},
                rv09_code={f: sha(rv_run.HERE / f) for f in RV9_CODE},
                so08_code={f: sha(SO08 / f) for f in SO08_CODE})


def start(out: Path, cfg_path, command: str, phase: str) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    cfg, over = load_config(cfg_path)
    write_json(out / 'config.json', cfg)
    write_json(out / 'run.json', dict(command=command, phase=phase, config_sha256=sha(Path(cfg_path)),
                                      path_overrides=over, started=time.time(), **code_hashes(),
                                      environment=rv_run.environment()))
    return cfg


def finish(out: Path):
    rv_run.finish(out)


# ---- environment ---------------------------------------------------------------------------------------------------
class ECEnv:
    """Data of one phase's half, the base model with RV-09's EditWritable, the KT editor prepared on the half's own
    routing-centre corpus, the fixed-shape generator and (lazily) the retriever."""

    def __init__(self, cfg: dict, phase: str, model=None, tok=None, encoder=None):
        import torch
        self.cfg, self.phase = cfg, phase
        sc = cfg['streams']
        edits, windows, _ = load_edits(cfg['data_dir'], expect_steps=cfg.get('expect_steps', 8))
        self.index = FactIndex(edits)
        self.by_uid = {e.uid: e for e in edits}
        self.streams, self.corpus, self.filters = ec_data.prepare_half(
            edits, len(windows), sc[phase]['half'], sc['loc_hidden_eval'], sc['mu_corpus']['buckets'],
            sc['mu_corpus']['n'], stream_args=sc['args'], filter_other_half=sc[phase].get('filter_other_half'))
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if not torch.cuda.is_available():
                raise SystemExit('no CUDA device')
            torch.manual_seed(int(cfg.get('torch_seed', 0)))
            tok = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(cfg['model_path'], local_files_only=True,
                                                         dtype=getattr(torch, cfg.get('dtype', 'bfloat16'))).to('cuda')
        layers = range(model.config.num_hidden_layers) if cfg.get('layers', 'all') == 'all' else cfg['layers']
        sites = [s.format(i=i) for i in layers for s in cfg['sites']]
        self.wm = EditWritable(model, tok, sites, rank=int(cfg['rank']), scale=float(cfg['scale']),
                               clip=cfg.get('clip'))
        local = dict(cfg['local'])
        local['layers'] = tuple(local['layers'])
        self.editor = LoggedEditor(LocalCfg(**local))
        self.prep = self.editor.prepare(self.wm, self.corpus)
        ev = cfg['eval']
        self.gen = FixedShapeGen(self.wm, ev['widths'], ev['batch'], ev['max_new'])
        self.grace = ec_grace.GraceHook()
        self.gen.loggers = [self.editor, self.grace]
        self._encoder = encoder

    def encoder(self):
        if self._encoder is None:
            from transformers import AutoModel, AutoTokenizer
            rc = self.cfg['rag']
            tok = AutoTokenizer.from_pretrained(rc['model_path'], local_files_only=True)
            model = AutoModel.from_pretrained(rc['model_path'], local_files_only=True).to(self.wm.device).float()
            self._encoder = ec_rag.MeanPoolEncoder(model, tok, self.wm.device, rc['max_len'], rc['batch'])
        return self._encoder

    def plain(self):
        """No entries, no codebook: the base model (the KT hook returns at once without entries)."""
        self.wm.extra = {}
        self.grace.cb = None


def _floats(t, nd=6) -> list:
    return [round(float(x), nd) for x in t.tolist()]


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---- one stream ----------------------------------------------------------------------------------------------------
def run_stream(env: ECEnv, name: str, arms: list[str], timing: dict | None = None) -> dict:
    import torch
    cfg, wm, ed, gen = env.cfg, env.wm, env.editor, env.gen
    sc = cfg['streams']
    phase_cfg = sc[env.phase]
    clock, info = {}, {}
    s = cap_stream(env.streams[name], phase_cfg.get('cap', sc['cap']))
    T = s.steps - 1
    ctx = shots_text(s.shots)
    tok = wm.tok

    # 1. base-known locality facts, in the evaluation's own fixed-shape layout
    t0 = time.time()
    env.plain()
    cand = list(s.loc_hidden)[:phase_cfg.get('loc_candidates') or len(s.loc_hidden)]
    citems = loc_items(cand)
    cids, cnctx = plain_prompts(tok, citems, ctx)
    couts = gen.generate(cids, cnctx)
    known = [(q, a) for (q, a), o in zip(cand, couts) if norm(o.text) == norm(a)]
    screen_ans = {q: norm(o.text) for (q, _), o in zip(cand, couts)}
    checks = {}
    if phase_cfg.get('order_check'):
        rev = gen.generate(cids[::-1], cnctx[::-1])[::-1]
        checks['screen_reversed_identical'] = all(a.text == b.text for a, b in zip(couts, rev))
        checks['screen_reversed_mismatches'] = sum(a.text != b.text for a, b in zip(couts, rev))
    clock['screen'] = time.time() - t0
    info['screening'] = dict(candidates=len(cand), known=len(known), filters=env.filters.get(name))

    # 2. KT entries (RV-09's local writer, written once) and the probe table
    t1 = time.time()
    probes = list(s.loc_probe)
    states, labels = probe_positions(wm, ed, probes, ctx)
    ed.set_probes(wm, probes, ctx)
    same_probes = bool(torch.equal(states, ed.probe_states))
    ed.probe_states = states
    nq = len(probes)
    ent = build_entries(wm, ed, s, int(cfg['write_seed']))
    env.plain()
    _sync()
    clock['kt_entries'] = time.time() - t1
    t2 = time.time()
    pm = probe_max(states, labels, ent.keys, nq + 1)
    lc = ed.cfg
    recomputed = torch.clamp(pm.max(0).values + lc.probe_margin, min=lc.tau)
    _sync()
    clock['kt_table'] = time.time() - t2
    flags = [ec_data.hygiene_flags(m['subject'], env.by_uid[m['uid']].update) for m in ent.meta]
    flagged = torch.tensor([any(f in ec_data.HYGIENE_RULE for f in fl) for fl in flags], device=ent.keys.device)
    labels_kt = [m['answer'] for m in ent.meta]

    # 3. items: every edit's rephrase, persona and original wording; the base-known locality facts
    items = edit_items(s, env.index, T) + loc_items(known)
    pids, pnctx = plain_prompts(tok, items, ctx)
    kinds = [it.kind for it in items]

    # lazily built per-method state
    grace_store, rag_state = {}, {}

    def grace_layer(layer):
        if layer in grace_store:
            return grace_store[layer]
        g = cfg['grace']
        tg = time.time()
        eprompts = [tok.encode(fmt_prompt(e.update, ctx), add_special_tokens=False) for e in s.edits]
        etargets = [tok.encode(fmt_target(e.answer), add_special_tokens=False) for e in s.edits]
        keys = ec_grace.fixed_keys(gen, layer, eprompts)
        tv = time.time()
        vals = ec_grace.train_values(wm, layer, eprompts, etargets, int(g['n_iter']), float(g['edit_lr']),
                                     int(g['seed']), int(g['chunk']))
        _sync()
        train_s = time.time() - tv
        pprompts = [tok.encode(fmt_prompt(q, ctx), add_special_tokens=False) for q in probes]
        pkeys = ec_grace.fixed_keys(gen, layer, pprompts)
        scale = ec_grace.probe_scale(pkeys)
        _sync()
        st = dict(keys=keys, values=vals.values, labels=[norm(e.answer) for e in s.edits], scale=scale,
                  info=dict(scale=scale, seconds=time.time() - tg, train_seconds=train_s,
                            loss_first_mean=statistics.fmean(vals.loss_first) if vals.loss_first else None,
                            loss_last_mean=statistics.fmean(vals.loss_last) if vals.loss_last else None,
                            loss_last_max=max(vals.loss_last) if vals.loss_last else None,
                            loss_last=_floats(torch.tensor(vals.loss_last), 4),
                            key_norm_mean=float(keys.norm(dim=1).mean()) if len(keys) else None))
        grace_store[layer] = st
        return st

    def rag():
        if rag_state:
            return rag_state
        tr = time.time()
        enc = env.encoder()
        records = [(e.update, e.answer) for e in s.edits]
        rec_emb = enc.encode([q for q, _ in records])
        probe_emb = enc.encode(probes)
        _sync()
        write_s = time.time() - tr
        tq = time.time()
        q_emb = enc.encode([it.question for it in items])
        sim = q_emb @ rec_emb.T
        _sync()
        own = {e.uid: i for i, e in enumerate(s.edits)}
        rag_state.update(records=records, rec_emb=rec_emb, probe_emb=probe_emb, sim=sim, own=own,
                         info=dict(records=len(records), write_seconds=write_s, query_seconds=time.time() - tq,
                                   dim=int(rec_emb.shape[1]) if rec_emb.numel() else 0))
        return rag_state

    # 4. every arm, evaluated once after the last write
    out_arms, arm_info, arm_clock = {}, {}, {}
    thr_cache = {}
    for arm in arms:
        ta = time.time()
        fam = family(arm)
        prompts, nctx, extra = pids, pnctx, {}
        env.plain()
        if fam == 'none':
            pass
        elif fam in ('kt', 'hyg', 'rad'):
            if fam == 'kt':
                hyg, n, m, f = parse_kt(arm)
                thr = kt_thresholds(pm, nq, n, m, f)
                if hyg:
                    thr = with_hygiene(thr, flagged)
                extra['raised'] = int((thr > f + 1e-9).logical_and(thr < OFF).sum())
            elif fam == 'hyg':
                thr = with_hygiene(torch.full((ent.n,), float(arm.split(':')[1]), device=ent.keys.device), flagged)
            else:
                thr, st = rad_thresholds(ent.keys, labels_kt, float(arm.split(':')[1]))
                extra.update(st)
            thr = thr.to(wm.device).float()
            thr_cache[arm] = thr
            extra['off'] = int((thr >= OFF).sum())
            extra['threshold_mean_active'] = float(thr[thr < OFF].mean()) if bool((thr < OFF).any()) else None
            install(wm, ent, T, thr)
        elif fam == 'grace':
            layer, mode, x = parse_grace(arm)
            st = grace_layer(layer)
            eps_init = x * st['scale'] if mode == 's' else x
            cb = ec_grace.build_codebook(st['keys'], st['values'], st['labels'], eps_init)
            if env.grace.layer != layer:
                env.grace.attach(wm, layer)
            env.grace.cb = cb
            env.grace.hits = 0
            extra.update(layer=layer, eps_init=eps_init, keys=cb.added, splits=cb.splits, expansions=cb.expansions,
                         eps_mean=float(cb.eps.mean()) if cb.eps.numel() else None)
        elif fam == 'rag':
            rs = rag()
            p = ec_rag.parse(arm)
            thr_r = None
            if p['sub'] == 'probe':
                thr_r = ec_rag.record_thresholds(rs['rec_emb'], rs['probe_emb'], p['margin'],
                                                 float(cfg['rag']['probe_floor']))
            picked = ec_rag.retrieve(rs['sim'], arm, thr_r)
            prompts, nctx, retrieved, own_hit = [], [], [], []
            for it, ids, pk in zip(items, pids, picked):
                if pk:
                    c = ec_rag.context_with(ctx, rs['records'], pk)
                    prompts.append(tok.encode(fmt_prompt(it.question, c), add_special_tokens=False))
                    nctx.append(len(tok.encode(c, add_special_tokens=False)))
                else:
                    prompts.append(ids)
                    nctx.append(pnctx[0] if pnctx else 0)
                retrieved.append(len(pk))
                own_hit.append(it.kind != 'loc' and rs['own'].get(it.key) in pk)
            extra.update(retrieved=''.join(str(min(r, 9)) for r in retrieved),
                         own_retrieved=''.join('1' if h else '0' for h in own_hit),
                         prompt_tokens_mean=statistics.fmean(map(len, prompts)) if prompts else None)
        else:
            raise ValueError(arm)
        outs = gen.generate(prompts, nctx)
        ok, fired, wrong = score(items, outs)
        if fam == 'rag':      # retrieval "fires" when a record is put in the prompt
            fired = ''.join('1' if r != '0' else '0' for r in extra['retrieved'])
        out_arms[arm] = dict(ok=ok, fired=fired, wrong_loc=wrong)
        arm_info[arm] = extra
        env.plain()
        arm_clock[arm] = time.time() - ta
        cov = _mean_bits(ok, kinds, 'reph')
        dmg = 1 - _mean_bits(ok, kinds, 'loc') if 'loc' in kinds else None
        print(f'  {name} {arm}: reph {cov:.3f} damage {"-" if dmg is None else f"{dmg:.4f}"} '
              f'({arm_clock[arm]:.0f} s)', flush=True)
    clock['arms'] = arm_clock

    # 5. checks
    if 'none' in out_arms:
        base = out_arms['none']['ok']
        checks['none_known_all_right'] = all(b == '1' for b, k in zip(base, kinds) if k == 'loc')
        checks['none_known_wrong'] = sum(1 for b, k in zip(base, kinds) if k == 'loc' and b != '1')
        checks['unfired_differs_from_none'] = {a: sum(1 for x, y, f in zip(v['ok'], base, v['fired'])
                                                      if f == '0' and x != y) for a, v in out_arms.items()}
    det_arm = cfg.get('kt_default')
    if det_arm in thr_cache and phase_cfg.get('determinism_check', True):
        install(wm, ent, T, thr_cache[det_arm])
        li = [i for i, k in enumerate(kinds) if k == 'loc']
        rev = gen.generate([pids[i] for i in li][::-1], [pnctx[i] for i in li][::-1])[::-1]
        bits = ''.join('1' if norm(o.text) in items[i].valid else '0' for i, o in zip(li, rev))
        checks['determinism'] = dict(arm=det_arm, loc_reversed_same=bits == ''.join(out_arms[det_arm]['ok'][i]
                                                                                   for i in li))
        env.plain()
    checks['probes_same_as_set_probes'] = same_probes
    checks['matched_recompute_max_abs_diff'] = float((recomputed - ent.matched).abs().max()) if ent.n else 0.0

    # 6. timing (one test stream): the variable-width SO-08 generator, as a server would batch
    tim = None
    if timing:
        tim = time_arms(env, s, ent, thr_cache, items, ctx, timing, grace_layer, rag)

    d_key, d_off = int(ent.keys.shape[1]), int(ent.offsets.shape[1])
    storage = dict(kt_bytes_per_entry=4 * (d_key + d_off + 1))
    if grace_store:
        st = next(iter(grace_store.values()))
        storage['grace_bytes_per_key'] = 4 * (int(st['keys'].shape[1]) + int(st['values'].shape[1]) + 1)
    if rag_state:
        text_bytes = sum(len(q.encode('utf-8')) + len(a.encode('utf-8')) for q, a in rag_state['records'])
        storage['rag_bytes_per_record'] = 4 * rag_state['info']['dim'] + text_bytes / max(1, len(rag_state['records']))
    env.plain()
    wm.restore(wm.empty_state())
    edits_meta = [dict(uid=e.uid, step=e.step, subject=e.subject, answer=norm(e.answer),
                       literal=bool(e.subject) and e.subject.casefold() in (e.rephrase or '').casefold())
                  for e in s.edits]
    return dict(
        stream=name, phase=env.phase, half=phase_cfg['half'], steps=s.steps, info=info,
        edits=edits_meta,
        kt=dict(n=ent.n, ends=ent.ends, flags=[fl for fl in flags], flagged=int(flagged.sum()),
                matched=_floats(ent.matched), records=ent.records, probe_positions=int(states.shape[0]),
                d_key=d_key, d_off=d_off),
        grace={str(k): v['info'] for k, v in grace_store.items()},
        rag=rag_state.get('info') if rag_state else None,
        items=[[it.kind, it.key] for it in items],
        screen_answers=None if env.phase != 'smoke' else screen_ans,
        arms=out_arms, arm_info=arm_info, checks=checks, timing=tim, storage=storage, clock=clock)


def _mean_bits(bits: str, kinds, kind: str) -> float:
    v = [b == '1' for b, k in zip(bits, kinds) if k == kind]
    return sum(v) / len(v) if v else float('nan')


def time_arms(env: ECEnv, s, ent, thr_cache, items, ctx, timing: dict, grace_layer, rag) -> dict:
    """Median seconds per question over `repeats` passes of the SO-08 generator (variable widths) on the first
    n64 rephrase questions at batch 64 and the first n1 at batch 1, per arm; retrieval includes encoding."""
    import torch
    wm, tok = env.wm, env.wm.tok
    qs = [it.question for it in items if it.kind == 'reph']
    out = {}
    for arm in timing['arms']:
        fam = family(arm)
        res = {}
        for bs, n in ((64, int(timing['n64'])), (1, int(timing['n1']))):
            sub = qs[:n]
            times, toks = [], []
            for _ in range(int(timing['repeats'])):
                env.plain()
                if fam in ('kt', 'hyg', 'rad'):
                    install(wm, ent, s.steps - 1, thr_cache[arm])
                elif fam == 'grace':
                    layer, mode, x = parse_grace(arm)
                    st = grace_layer(layer)
                    eps_init = x * st['scale'] if mode == 's' else x
                    if env.grace.layer != layer:
                        env.grace.attach(wm, layer)
                    env.grace.cb = ec_grace.build_codebook(st['keys'], st['values'], st['labels'], eps_init)
                _sync()
                t0 = time.time()
                if fam == 'rag':
                    rs = rag()
                    p = ec_rag.parse(arm)
                    thr_r = None
                    if p['sub'] == 'probe':
                        thr_r = ec_rag.record_thresholds(rs['rec_emb'], rs['probe_emb'], p['margin'],
                                                         float(env.cfg['rag']['probe_floor']))
                    ctxs = []
                    for b0 in range(0, len(sub), bs):
                        qe = env.encoder().encode(sub[b0:b0 + bs])
                        picked = ec_rag.retrieve(qe @ rs['rec_emb'].T, arm, thr_r)
                        ctxs += [ec_rag.context_with(ctx, rs['records'], pk) for pk in picked]
                else:
                    ctxs = [ctx] * len(sub)
                wm.generate(sub, contexts=ctxs, max_new=int(env.cfg['eval']['max_new']), batch_size=bs)
                _sync()
                times.append((time.time() - t0) / max(1, len(sub)))
                toks.append(statistics.fmean(len(tok.encode(fmt_prompt(q, c), add_special_tokens=False))
                                             for q, c in zip(sub, ctxs)))
            res[f'batch{bs}'] = dict(seconds_per_question=statistics.median(times), runs=times,
                                     prompt_tokens_mean=statistics.fmean(toks), n=len(sub))
        out[arm] = res
        env.plain()
    return out


# ---- commands --------------------------------------------------------------------------------------------------------
def phase_arms(cfg: dict, phase: str) -> list[str]:
    arms = cfg['arms'][phase]
    if not arms or 'none' not in arms:
        raise SystemExit(f'{phase}: the arm list must include none')
    return list(arms)


def cmd_run(cfg_path, out: Path, phase: str, streams: list[str] | None = None, env_factory=None):
    cfg = start(out, cfg_path, 'run', phase)
    names = list(cfg['streams'][phase]['names'])
    if streams:
        unknown = [x for x in streams if x not in names]
        if unknown:
            raise SystemExit(f'not in the {phase} streams: {unknown}')
        names = [x for x in names if x in streams]
        meta = json.loads((out / 'run.json').read_text(encoding='utf-8'))
        meta['streams_override'] = names
        write_json(out / 'run.json', meta)
    env = env_factory(cfg, phase) if env_factory else ECEnv(cfg, phase)
    meta = json.loads((out / 'run.json').read_text(encoding='utf-8'))
    meta['editor_prepare'] = env.prep
    write_json(out / 'run.json', meta)
    arms = phase_arms(cfg, phase)
    tcfg = cfg.get('timing') or {}
    for name in names:
        print(f'stream {name}', flush=True)
        timing = tcfg if (phase == 'test' and name == tcfg.get('stream')) else None
        rec = measured(env, lambda: run_stream(env, name, arms, timing))
        write_gz(out / f'stream_{name}.json.gz', rec)
        print(f'stream {name} done: {rec["seconds"]:.0f} s', flush=True)
    finish(out)


def measured(env: ECEnv, fn) -> dict:
    import torch
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    g0, t0, c0 = rv_run.gpu_snapshot(), time.time(), dict(env.wm.costs)
    rec = fn()
    rec.update(seconds=time.time() - t0, costs={k: env.wm.costs[k] - c0.get(k, 0.0) for k in env.wm.costs},
               gpu_before=g0, gpu_after=rv_run.gpu_snapshot(),
               peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None)
    return rec


def load_records(run_dir: Path) -> dict:
    return {p.name[len('stream_'):-len('.json.gz')]: read_gz(p) for p in sorted(run_dir.glob('stream_*.json.gz'))}


def cmd_tune(dev_run: Path, cfg_dev_path, out_path: Path):
    """The dev tuning rule -> the frozen test config: the dev config with arms.test (GRACE restricted to the chosen
    layer), the selected arm per family and sub-family, and the tuning table. The dev config's hash is recorded."""
    from ec_analysis import tune
    cfg_dev = json.loads(Path(cfg_dev_path).read_text(encoding='utf-8'))
    run = json.loads((dev_run / 'run.json').read_text(encoding='utf-8'))
    if run['config_sha256'] != sha(Path(cfg_dev_path)):
        raise SystemExit('the dev run did not use this config file')
    if run.get('phase') != 'dev':
        raise SystemExit('not a dev run')
    recs = load_records(dev_run)
    missing = [n for n in cfg_dev['streams']['dev']['names'] if n not in recs]
    if missing:
        raise SystemExit(f'dev streams missing: {missing}')
    result = tune(recs, cfg_dev)
    cfg = json.loads(json.dumps(cfg_dev))
    layer = parse_grace(result['selected']['grace'])[0]
    cfg['arms']['test'] = [a for a in cfg['arms']['dev'] if family(a) != 'grace' or parse_grace(a)[0] == layer]
    cfg['selected'] = result['selected']
    cfg['tuning'] = dict(table=result['table'], notes=result['notes'], dev_run=dev_run.name,
                         dev_config_sha256=run['config_sha256'], dev_manifest_sha256=sha(dev_run / 'manifest.json'))
    tcfg = dict(cfg.get('timing') or {})
    tcfg['arms'] = ['none'] + [result['selected'][f] for f in ('kt', 'grace', 'rag')]
    cfg['timing'] = tcfg
    cfg['status'] = (f'FROZEN {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())} by `ec_run.py tune` from dev run '
                     f'{dev_run.name}, before any test stream was run. Only arms.test, selected, tuning and '
                     f'timing.arms differ from the dev config.')
    write_json(out_path, cfg)
    return cfg


def cmd_analyze(test_run: Path, dev_run: Path | None = None):
    from ec_analysis import analyze
    cfg = json.loads((test_run / 'config.json').read_text(encoding='utf-8'))
    run = json.loads((test_run / 'run.json').read_text(encoding='utf-8'))
    recs = load_records(test_run)
    dev = load_records(dev_run) if dev_run else None
    result = analyze(recs, cfg, dev_recs=dev)
    result['sources'] = dict(test_run=test_run.name, phase=run.get('phase'), config_sha256=run['config_sha256'],
                             dev_run=dev_run.name if dev_run else None, code=run['code'])
    write_json(test_run / 'analysis.json', result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('run', 'tune', 'analyze'))
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--phase', choices=('smoke', 'dev', 'test'), default=None)
    ap.add_argument('--streams', default=None)
    ap.add_argument('--dev', type=Path, default=None)
    a = ap.parse_args(argv)
    if a.command == 'run':
        if len(a.paths) != 2 or not a.phase:
            ap.error('run CONFIG OUT --phase PHASE')
        cmd_run(a.paths[0], Path(a.paths[1]), a.phase, a.streams.split(',') if a.streams else None)
    elif a.command == 'tune':
        if len(a.paths) != 3:
            ap.error('tune DEV_RUN CONFIG_DEV CONFIG_TEST_OUT')
        cmd_tune(Path(a.paths[0]), a.paths[1], Path(a.paths[2]))
    else:
        cmd_analyze(Path(a.paths[0]), a.dev)


if __name__ == '__main__':
    main()
