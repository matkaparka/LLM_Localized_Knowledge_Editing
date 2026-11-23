"""SP-16 command line (PLAN.md). Nothing is downloaded here: the model, counterfact.json and the manifest must be local.

  python sp_run.py smoke  CONFIG OUT                 D01's first 25 writes: acceptance against the unmodified author code
  python sp_run.py run    CONFIG OUT [--streams a,b] write, score probes, seal the ranking, then evaluate every arm
  python sp_run.py select DEV_RUN CONFIG_DEV CONFIG_TEST_OUT   uniform comparator from the dev records; freezes test
  python sp_run.py analyze RUN [--dev DEV_RUN]       analysis.json (sp_analysis)

SP_MODEL_PATH and SP_COUNTERFACT may replace the model and data paths of a config (recorded in run.json); the config
file itself stays byte-identical. OUT of `run` may already exist only as an unfinished run of the same config, whose
missing streams are then added (a restart resumes).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sp_core as C  # noqa: E402

CODE = [HERE / n for n in ('sp_core.py', 'sp_run.py', 'sp_analysis.py')] + \
       [C.UPSTREAM / n for n in ('ablations.py', 'ablations_realdata_patches.py', 'counterfact_data.py', 'evaluate.py',
                                 'run_baselines.py')]


def now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_hashes() -> dict:
    return {str(p.relative_to(HERE)).replace('\\', '/'): sha_file(p) for p in CODE if p.exists()}


def environment() -> dict:
    import transformers
    return dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def resolve(cfg: dict) -> tuple[dict, dict]:
    paths = dict(model=cfg['model']['path'], counterfact=cfg['data']['counterfact'],
                 manifest=str(HERE / cfg['data']['manifest']))
    over = {}
    for env, key in (('SP_MODEL_PATH', 'model'), ('SP_COUNTERFACT', 'counterfact')):
        if os.environ.get(env):
            paths[key] = over[env] = os.environ[env]
    return paths, over


def load_model(path, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype)).to(device)
    model.eval()
    return model, tok


def deterministic():
    """Repeatable kernels: deterministic cuBLAS, no TF32, and the math attention kernel (the flash and
    memory-efficient kernels have non-deterministic backward passes, which the writes go through)."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def write_gz(path: Path, obj):
    data = json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    with open(path, 'wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as g:
        g.write(data)


def read_gz(path: Path):
    return json.loads(gzip.open(path).read().decode('utf-8'))


def new_editor(model, tok, cfg):
    e = cfg['editor']
    return C.Editor(model, tok, e['layer'], e['tau'], e['v_steps'], e['v_lr'], e['norm_cap'])


# ---- one stream -------------------------------------------------------------------------------------------------------
def run_stream(cfg, src, model, tok, partition, stream, out: Path, log) -> dict:
    ev, tau, margin = cfg['eval'], cfg['editor']['tau'], cfg['margin']
    gen = dict(max_new_tokens=ev['max_new_tokens'], batch_size=ev['batch_size'])
    secs = {}
    t0 = time.time()
    ed = new_editor(model, tok, cfg)
    edits = src.edits(partition, stream)
    write_s = []
    for e in edits:
        t = time.time()
        ed.insert(e)
        write_s.append(round(time.time() - t, 3))
    secs['write'] = round(time.time() - t0, 1)
    snap = ed.snapshot()
    digest = C.table_digest(snap)
    torch.save(snap, out / f'table_{stream}.pt')
    n = len(ed.case_ids)
    log(f'{stream}: {n} writes in {secs["write"]} s; table {digest[:12]}')

    # unlabelled probes: prefill states and the native winner at the prompt's last position (PLAN 4.2)
    t = time.time()
    probes = src.probes(partition)
    pr = C.generate(model, tok, ed.w, [p['prompt'] for p in probes], max_new_tokens=1,
                    batch_size=ev['batch_size'], trace=True, capture=True)
    K_n = torch.nn.functional.normalize(snap['keys'].float(), dim=-1)
    S = pr['capture'].float() @ K_n.t()                                  # probes x slots
    maxcos = [round(float(v), 6) for v in S.max(dim=0).values]
    counts = [0] * n
    for tr in pr['traces']:
        if tr and tr[0][2]:                                            # prefill winner passed the threshold
            counts[tr[0][0]] += 1
    best_sim, best_idx = S.max(dim=1)
    recomputed = torch.bincount(best_idx[best_sim > tau], minlength=n).tolist()
    ranking = C.rank(maxcos, ed.case_ids)
    freq_ranking = C.rank(counts, ed.case_ids)
    matched = C.matched_thresholds(maxcos, tau, margin)
    disabled = frozenset(i for i in range(n) if matched[i] > 1.0)
    perms = {tag: [C.block_permutation(n, stream, tag, p, disabled if tag == 'active_shuffled' else frozenset())
                   for p in range(cfg['n_perms'])] for tag in ('shuffled', 'active_shuffled')}
    seal = dict(stream=stream, partition=partition, table_sha256=digest, case_ids=ed.case_ids,
                probe_case_ids=[p['case_id'] for p in probes], maxcos=maxcos, trigger_counts=counts,
                ranking=ranking, freq_ranking=freq_ranking, matched=matched, perms=perms, sealed_utc=now())
    seal_bytes = json.dumps(seal, sort_keys=True).encode()
    seal_sha = C.sha256(seal_bytes)
    (out / f'seal_{stream}.json').write_bytes(seal_bytes)
    src.seal(stream, seal_sha)
    secs['probes'] = round(time.time() - t, 1)
    log(f'{stream}: ranking sealed {seal_sha[:12]}; top-13 max cos {maxcos[ranking[0]]:.3f}..'
        f'{maxcos[ranking[12 if n > 12 else -1]]:.3f}; probes firing {sum(counts)}')

    # evaluation inputs are read only now
    protect = src.protect(partition, stream)
    items = dict(
        protect=[dict(case_id=q['case_id'], prompt=q['prompt'], target=q['target_true']) for q in protect],
        edit=[dict(case_id=e['case_id'], prompt=e['prompt'], target=e['target_new']) for e in edits],
        para=[dict(case_id=e['case_id'], k=k, prompt=p, target=e['target_new'])
              for e in edits for k, p in enumerate(e['paraphrases'])])
    arms = {}
    native_trace = None
    t = time.time()
    for arm in cfg['arms']:
        ro = C.arm_readout(arm, n, tau, matched, perms, ranking)
        res = {}
        for kind, rows in items.items():
            g = C.generate(model, tok, ed.w, [r['prompt'] for r in rows], readout=ro, trace=arm != 'none', **gen)
            res[kind] = dict(text=g['texts'])
            if g['traces'] is not None:
                res[kind]['fired'] = [sorted({s for s, _, _, a in tr if a}) for tr in g['traces']]
                if arm == 'native' and kind == 'protect':
                    native_trace = g['traces']
        arms[arm] = res
    secs['arms'] = round(time.time() - t, 1)
    log(f'{stream}: {len(cfg["arms"])} arms in {secs["arms"]} s')

    # single-slot replays of every slot that fired on a damaged protected question (PLAN 4.3); a question counts as
    # damaged under either matcher, so that the sensitivity analysis has its witnesses too
    t = time.time()
    none_t, nat_t, nat_f = arms['none']['protect']['text'], arms['native']['protect']['text'], arms['native']['protect']['fired']
    pairs = []
    for qi, q in enumerate(items['protect']):
        dm = C.match_main(none_t[qi], q['target']) and not C.match_main(nat_t[qi], q['target'])
        da = C.match_author(none_t[qi], q['target']) and not C.match_author(nat_t[qi], q['target'])
        if dm or da:
            pairs += [(qi, s) for s in nat_f[qi]]
    rp = C.generate(model, tok, ed.w, [items['protect'][qi]['prompt'] for qi, _ in pairs],
                    readout={'tau': tau}, single=[s for _, s in pairs], trace=True, **gen) if pairs else dict(texts=[], traces=[])
    replays = [dict(item=qi, slot=s, text=rp['texts'][j], fired=bool(any(a for _, _, _, a in rp['traces'][j])))
               for j, (qi, s) in enumerate(pairs)]
    secs['replays'] = round(time.time() - t, 1)

    # checks: native order invariance; the custom read-out at uniform tau equals the parent's forward
    checks = dict(trigger_counts_match_recomputed=counts == recomputed,
                  probe_trace_vs_recomputed_max_abs=round(float(max(
                      (abs(tr[0][1] - float(best_sim[i])) for i, tr in enumerate(pr['traces']) if tr), default=0.0)), 9))
    if cfg['checks'].get('reversed_native'):
        rev = C.generate(model, tok, ed.w, [r['prompt'] for r in items['protect']][::-1], **gen)['texts'][::-1]
        checks['reversed_native_differs'] = sum(a != b for a, b in zip(rev, nat_t))
    u = f'uniform:{tau:.2f}'
    if u in arms:
        checks['uniform_tau_differs_from_native'] = sum(
            a != b for kind in items for a, b in zip(arms[u][kind]['text'], arms['native'][kind]['text']))
    checks['seal_before_protect'] = True
    ed.detach()
    secs['total'] = round(time.time() - t0, 1)
    return dict(stream=stream, partition=partition, n_slots=n, case_ids=ed.case_ids, table_sha256=digest,
                write_seconds=write_s, seal_sha256=seal_sha, seal=dict(maxcos=maxcos, trigger_counts=counts,
                ranking=ranking, freq_ranking=freq_ranking, matched=matched, perms=perms),
                probes=dict(case_ids=[p['case_id'] for p in probes]),
                items={k: [{kk: vv for kk, vv in r.items() if kk != 'prompt'} for r in rows] for k, rows in items.items()},
                arms=arms, native_trace=native_trace, replays=replays, checks=checks, seconds=secs)


def cmd_run(cfg_path: Path, out: Path, streams=None):
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    paths, over = resolve(cfg)
    cfg_sha = sha_file(cfg_path)
    if out.exists():
        meta = json.loads((out / 'run.json').read_text(encoding='utf-8')) if (out / 'run.json').exists() else {}
        if meta.get('config_sha256') != cfg_sha or (out / 'manifest.json').exists():
            raise SystemExit(f'{out} exists and is not an unfinished run of this config')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_bytes(cfg_path.read_bytes())
    meta = dict(command='run', config=cfg_path.name, config_sha256=cfg_sha, partition=cfg['partition'],
                streams=streams or cfg['streams'], started=now(), code=code_hashes(), env_overrides=over,
                environment=environment(), upstream=json.loads((HERE / 'upstream' / 'UPSTREAM.json').read_text()))
    (out / 'run.json').write_text(json.dumps(meta, indent=1) + '\n', encoding='utf-8')
    deterministic()
    src = C.Source(paths['counterfact'], paths['manifest'], verify=cfg['data'].get('verify', True))
    model, tok = load_model(paths['model'], cfg['model']['dtype'])
    logf = (out / 'log.txt').open('a', encoding='utf-8')

    def log(msg):
        line = f'{now()} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
    for s in streams or cfg['streams']:
        if (out / f'stream_{s}.json.gz').exists():
            log(f'{s}: already written, skipped')
            continue
        rec = run_stream(cfg, src, model, tok, cfg['partition'], s, out, log)
        rec.update(config_sha256=cfg_sha, generation_config=model.generation_config.to_dict())
        write_gz(out / f'stream_{s}.json.gz', rec)
        log(f'{s}: done in {rec["seconds"]["total"]} s; checks {rec["checks"]}')
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name not in ('manifest.json', 'log.txt'))
    (out / 'manifest.json').write_text(json.dumps({p.name: sha_file(p) for p in files}, indent=1) + '\n',
                                       encoding='utf-8')
    log('manifest written')


# ---- acceptance on D01's first writes (PLAN 4.1) ------------------------------------------------------------------------
def cmd_smoke(cfg_path: Path, out: Path):
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    paths, over = resolve(cfg)
    if out.exists():
        raise SystemExit(f'{out} exists')
    out.mkdir(parents=True)
    deterministic()
    sm = cfg['smoke']
    src = C.Source(paths['counterfact'], paths['manifest'], verify=cfg['data'].get('verify', True))
    model, tok = load_model(paths['model'], cfg['model']['dtype'])
    edits = src.edits(cfg['partition'], sm['stream'])[:sm['writes']]
    prompts = [e['prompt'] for e in edits] + [p for e in edits for p in e['paraphrases']] + \
              [p['prompt'] for p in src.probes(cfg['partition'])[:sm['probes']]]
    e, ev = cfg['editor'], cfg['eval']
    res = dict(stream=sm['stream'], writes=len(edits), prompts=len(prompts), started=now(), code=code_hashes(),
               environment=environment(), env_overrides=over)

    # A: the author's method and wrapper, untouched; winners observed by a hook that only reads
    meth = C.ablations.AblatedSHARDMethod(layer_idx=e['layer'], sim_threshold=e['tau'], v_steps=e['v_steps'],
                                          v_lr=e['v_lr'], v_weight_decay=0.0, v_norm_constraint=e['norm_cap'],
                                          eps_init=1.0, **C.ablations.ABLATION_PRESETS['shard'])
    meth.setup(model, tok, kb=None)
    author = meth.wrapped_mlp
    t = time.time()
    for ed_ in edits:
        meth.insert(C.counterfact_data.CounterFactTriple(
            case_id=ed_['case_id'], subject=ed_['subject'], prompt_template=ed_['template'],
            target_new=ed_['target_new'], target_true='', relation_id=ed_['relation_id']))
    res['author_write_seconds'] = round(time.time() - t, 1)
    calls_a = []

    def observe(module, inputs, output):
        x_n = torch.nn.functional.normalize(inputs[0][:, -1, :].float(), dim=-1)
        K_n = torch.nn.functional.normalize(torch.stack(module.keys).float(), dim=-1)
        s, i = (x_n @ K_n.t()).max(dim=-1)
        calls_a.append((i.cpu(), (s > module.sim_threshold).cpu()))
    gens_a = []
    orig_generate = model.generate

    def recording_generate(*a, **k):
        g = orig_generate(*a, **k)
        gens_a.append((k['input_ids'].shape, g.cpu()))
        return g
    h = author.register_forward_hook(observe)
    model.generate = recording_generate
    t = time.time()
    try:
        texts_a = C.upstream_eval.generate_greedy(model, tok, prompts, max_new_tokens=ev['max_new_tokens'],
                                                  batch_size=ev['batch_size'])
    finally:
        h.remove()
        del model.generate
    res['author_generate_seconds'] = round(time.time() - t, 1)
    keys_a, vals_a = torch.stack(author.keys).cpu(), torch.stack(author.values).cpu()
    ids_a = [full[shape[1]:].tolist() for shape, g in gens_a for full in g]
    meth.wrapped_mlp = None
    model.model.layers[e['layer']].mlp = author.base_mlp

    # B: the same writes through Editor (author insert, SPWrapper read-out), traced
    ed = new_editor(model, tok, cfg)
    t = time.time()
    for ed_ in edits:
        ed.insert(ed_)
    res['write_seconds'] = round(time.time() - t, 1)
    t = time.time()
    b = C.generate(model, tok, ed.w, prompts, trace=True, max_new_tokens=ev['max_new_tokens'],
                   batch_size=ev['batch_size'])
    res['generate_seconds'] = round(time.time() - t, 1)
    snap = ed.snapshot()
    res['keys_equal'] = bool(torch.equal(snap['keys'], keys_a))
    res['values_equal'] = bool(torch.equal(snap['values'], vals_a))
    res['keys_max_abs_diff'] = float((snap['keys'] - keys_a).abs().max())
    res['values_max_abs_diff'] = float((snap['values'] - vals_a).abs().max())
    res['texts_differ'] = sum(x != y for x, y in zip(texts_a, b['texts']))
    res['ids_differ'] = sum(x != y for x, y in zip(ids_a, b['ids']))
    # winners and threshold decisions call by call: author's observed calls against SPWrapper's trace, rows aligned
    eos = C.eos_ids(model, tok)
    mism, k, rowbase = 0, 0, 0
    for shape, g in gens_a:
        bsz = shape[0]
        n_calls = g.shape[1] - shape[1]
        for j in range(bsz):
            used = C.steps_used(g[j, shape[1]:].tolist(), eos)
            got = [(c[0][j].item(), c[1][j].item()) for c in calls_a[k:k + n_calls][:used]]
            want = [(s, hit) for s, _, hit, _ in b['traces'][rowbase + j]]
            mism += got != want
        k += n_calls
        rowbase += bsz
    res['winner_or_decision_rows_differ'] = mism
    # one prompt at a time against left-padded batches (reported, not a gate)
    one = C.generate(model, tok, ed.w, prompts, max_new_tokens=ev['max_new_tokens'], batch_size=1)
    res['single_vs_batch_texts_differ'] = sum(x != y for x, y in zip(one['texts'], b['texts']))
    # save, reload into a fresh wrapper, decode again
    torch.save(snap, out / 'table_smoke.pt')
    ed.detach()
    ed2 = new_editor(model, tok, cfg)
    ed2.load(torch.load(out / 'table_smoke.pt'))
    c = C.generate(model, tok, ed2.w, prompts, trace=True, max_new_tokens=ev['max_new_tokens'],
                   batch_size=ev['batch_size'])
    res['reload_texts_differ'] = sum(x != y for x, y in zip(c['texts'], b['texts']))
    res['reload_traces_differ'] = sum(x != y for x, y in zip(c['traces'], b['traces']))
    ed2.detach()
    res['per_write_seconds'] = round(res['write_seconds'] / max(1, len(edits)), 3)
    res['gate'] = dict(texts=res['texts_differ'] == 0, ids=res['ids_differ'] == 0,
                       winners=res['winner_or_decision_rows_differ'] == 0, reload=res['reload_texts_differ'] == 0
                       and res['reload_traces_differ'] == 0)
    res['passed'] = all(res['gate'].values())
    res['finished'] = now()
    res['examples'] = [dict(prompt=p, author=x, ours=y) for p, x, y in list(zip(prompts, texts_a, b['texts']))[:6]]
    (out / 'smoke.json').write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in res.items() if k not in ('examples', 'code', 'environment')}, indent=1))
    if not res['passed']:
        raise SystemExit('acceptance failed')


# ---- uniform comparator from the dev records; freezes the test config ---------------------------------------------------
def cmd_select(dev: Path, cfg_dev_path: Path, out_cfg: Path):
    import sp_analysis as A
    cfg = json.loads(cfg_dev_path.read_text(encoding='utf-8'))
    if out_cfg.exists():
        raise SystemExit(f'{out_cfg} exists')
    recs = A.load_run(dev)
    sel = A.select_uniform(recs, cfg)
    test = json.loads(json.dumps(cfg))
    test['partition'], test['streams'] = 'test', cfg['test_streams']
    test['arms'] = [a.replace('uniform:SEL', f'uniform:{sel["level"]:.2f}') for a in cfg['test_arms']]
    test['selected'] = sel
    test['status'] = (f'TEST CONFIG written by sp_run.py select from {dev.name} on {now()}; '
                      f'{cfg_dev_path.name} sha256 {sha_file(cfg_dev_path)}; code {json.dumps(code_hashes())}')
    test.pop('smoke', None)
    out_cfg.write_text(json.dumps(test, indent=1) + '\n', encoding='utf-8')
    print(f'uniform {sel["level"]:.2f} ({sel["flag"]}); {out_cfg.name} sha256 {sha_file(out_cfg)}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('smoke'); s.add_argument('config', type=Path); s.add_argument('out', type=Path)
    r = sub.add_parser('run'); r.add_argument('config', type=Path); r.add_argument('out', type=Path)
    r.add_argument('--streams')
    t = sub.add_parser('select'); t.add_argument('dev', type=Path); t.add_argument('config_dev', type=Path)
    t.add_argument('config_test', type=Path)
    a = sub.add_parser('analyze'); a.add_argument('run', type=Path); a.add_argument('--dev', type=Path)
    args = ap.parse_args(argv)
    if args.cmd == 'smoke':
        cmd_smoke(args.config, args.out)
    elif args.cmd == 'run':
        cmd_run(args.config, args.out, args.streams.split(',') if args.streams else None)
    elif args.cmd == 'select':
        cmd_select(args.dev, args.config_dev, args.config_test)
    else:
        import sp_analysis as A
        A.write_analysis(args.run, args.dev)


if __name__ == '__main__':
    main()
