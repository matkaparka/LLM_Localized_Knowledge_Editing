"""SO-08 command line. Nothing is downloaded here: the dated WikiBigEdit files and the model must already be local.

  python so_run.py streams CONFIG OUT     build and audit the streams (CPU, no model)
  python so_run.py p0 CONFIG OUT          P0 on the pilot streams (loads the model)
  python so_run.py matched CONFIG OUT     matched-readout records on the chosen development streams (loads the model)
  python so_run.py analyze RUN_DIR        CPU analysis of the p0/matched files in RUN_DIR, written to RUN_DIR/analysis.json

OUT must not exist yet. A run writes its config copy, code hashes, environment and a manifest of every output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from so_analysis import p0_gate, p0_verdict, select, state_dependence, swap_test  # noqa: E402
from so_branch import RunCfg, matched_stream, p0_stream, run_default  # noqa: E402
from so_data import FactIndex, audit_stream, build_streams, dataset_report, load_edits, stream_report  # noqa: E402
from so_policy import EvalCfg, ReadoutCfg, WriteCfg, parse_action  # noqa: E402

CODE = ('so_data.py', 'so_model.py', 'so_policy.py', 'so_branch.py', 'so_analysis.py', 'so_run.py')


def sha(path: Path) -> str:
    d = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            d.update(block)
    return d.hexdigest()


def _default(o):
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(f'not JSON serialisable: {type(o).__name__}')


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_default) + '\n', encoding='utf-8')


def load_cfg(path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    for key in ('data_dir', 'streams', 'write', 'eval', 'readout', 'default', 'menu'):
        if key not in cfg:
            raise KeyError(f'config misses {key}')
    return cfg


def run_cfg(cfg) -> RunCfg:
    return RunCfg(lag=int(cfg.get('lag', 0)), write=WriteCfg(**cfg['write']), ev=EvalCfg(**cfg['eval']),
                  ro=ReadoutCfg(**cfg['readout']), default=parse_action(cfg['default']),
                  continuation=cfg.get('continuation', 'default'))


def menu(cfg):
    actions = [parse_action(s) for s in cfg['menu']]
    labels = [a.label for a in actions]
    if len(set(labels)) != len(labels):
        raise ValueError('duplicate menu entries')
    if parse_action(cfg['default']).label not in labels:
        raise ValueError('the default policy must be in the menu, so its branch can be checked against the trajectory')
    return actions


def build(cfg, with_edits=False):
    edits, windows, skipped = load_edits(cfg['data_dir'], expect_steps=cfg.get('expect_steps', 8))
    index = FactIndex(edits)
    streams = build_streams(edits, len(windows), **cfg['streams'])
    return (streams, index, windows, skipped, edits) if with_edits else (streams, index, windows, skipped)


def load_model(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from so_model import Writable
    if not torch.cuda.is_available() and not cfg.get('allow_cpu', False):
        raise SystemExit('no CUDA device and allow_cpu is false')
    print(f'keep-awake request: {keep_awake()}', flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(int(cfg.get('torch_seed', 0)))
    tok = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(cfg['model_path'], local_files_only=True,
                                                 dtype=getattr(torch, cfg.get('dtype', 'bfloat16'))).to(device)
    layers = cfg.get('layers', 'all')
    layers = range(model.config.num_hidden_layers) if layers == 'all' else layers
    sites = [s.format(i=i) for i in layers for s in cfg['sites']]
    return Writable(model, tok, sites, rank=int(cfg['rank']), scale=float(cfg['scale']),
                    grad_ckpt=bool(cfg.get('grad_ckpt', False)))


def environment() -> dict:
    env = dict(python=platform.python_version(), platform=platform.platform())
    try:
        import torch
        import transformers
        env.update(torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__,
                   device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu')
    except ImportError:
        pass
    return env


def start(out: Path, cfg_path, command: str) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_cfg(cfg_path)
    write_json(out / 'config.json', cfg)
    write_json(out / 'run.json', dict(command=command, config_sha256=sha(Path(cfg_path)), started=time.time(),
                                      code={f: sha(HERE / f) for f in CODE}, environment=environment()))
    return cfg


def finish(out: Path):
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    write_json(out / 'manifest.json', files)


def cmd_streams(cfg_path, out):
    cfg = start(out, cfg_path, 'streams')
    streams, index, windows, skipped, edits = build(cfg, with_edits=True)
    write_json(out / 'streams.json', dict(windows=windows, skipped_rows=skipped,
                                          dataset=dataset_report(edits, index, len(windows)),
                                          streams=[dict(stream_report(s, index), audit=audit_stream(s))
                                                   for s in streams]))
    finish(out)


def keep_awake() -> bool:
    """Ask Windows not to idle-sleep while this process runs (released when it exits). Changes no setting."""
    if sys.platform != 'win32':
        return False
    import ctypes
    es_continuous, es_system_required = 0x80000000, 0x00000001
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required))


def gpu_snapshot() -> dict:
    import subprocess
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,power.draw,memory.used',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # no driver tool: record why
        out = f'unavailable: {type(e).__name__}'
    return dict(time=time.time(), temp_c_sm_mhz_maxsm_mhz_power_w_mem_mib=out)


def measured(wm, fn) -> dict:
    """Run one stream job and attach time, cost deltas, peak CUDA memory and GPU state before and after."""
    import torch
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    g0, t0, c0 = gpu_snapshot(), time.time(), dict(wm.costs)
    rec = fn()
    rec.update(seconds=time.time() - t0, costs=_cost_delta(c0, wm.costs), gpu_before=g0, gpu_after=gpu_snapshot(),
               peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None)
    print(f'{rec.get("stream")} seed {rec.get("seed")}: {rec["seconds"]:.0f} s', flush=True)
    return rec


def pilot_streams(cfg, streams):
    names = cfg['p0'].get('streams')
    chosen = [s for s in streams if (s.name in names if names else s.name.startswith('P'))]
    if names and len(chosen) != len(names):
        raise ValueError('a requested pilot stream does not exist')
    return chosen


def cmd_p0(cfg_path, out):
    cfg = start(out, cfg_path, 'p0')
    streams, index, _, _ = build(cfg)
    rc, actions = run_cfg(cfg), menu(cfg)
    wm = load_model(cfg)
    for s in pilot_streams(cfg, streams):
        for seed in cfg['p0']['seeds']:
            rec = measured(wm, lambda: p0_stream(wm, s, index, rc, int(seed), actions, cfg['p0']['repair_steps']))
            write_json(out / f'p0_{s.name}_seed{seed}.json', rec)
    finish(out)


def cmd_calibrate(cfg_path, out):
    """Learning-rate check on pilot streams only: default-policy trajectories over the first few steps."""
    import dataclasses
    cfg = start(out, cfg_path, 'calibrate')
    streams, index, _, _ = build(cfg)
    base = run_cfg(cfg)
    wm = load_model(cfg)
    cal = cfg['calibrate']
    variants = cal.get('variants') or [dict(lr=float(lr)) for lr in cal['lrs']]
    for s in pilot_streams(dict(cfg, p0=dict(cfg['p0'], streams=cal['streams'])), streams):
        if cal.get('screen'):
            from so_policy import screen_locality
            s, screening = screen_locality(wm, s, base.ev)
            write_json(out / f'screening_{s.name}.json', screening)
        for i, var in enumerate(variants):
            var = dict(var)
            default = var.pop('default', None)
            rc = dataclasses.replace(base, write=dataclasses.replace(base.write, **var),
                                     default=parse_action(default) if default else base.default)
            var = dict(var, default=rc.default.label)
            rec = measured(wm, lambda: run_default(wm, s, index, rc, int(cal.get('seed', 0)), max_steps=cal['steps']))
            rec.pop('captures', None)
            rec.update(variant=var)
            write_json(out / f'calibrate_{s.name}_v{i}.json', rec)
    finish(out)


def _cost_delta(before, after) -> dict:
    return {k: after[k] - before.get(k, 0.0) for k in after}


def cmd_matched(cfg_path, out):
    cfg = start(out, cfg_path, 'matched')
    streams, index, _, _ = build(cfg)
    rc, actions = run_cfg(cfg), menu(cfg)
    chosen = [s for s in streams if s.name in set(cfg['matched']['streams'])]
    if len(chosen) != len(cfg['matched']['streams']):
        raise ValueError('a requested development stream does not exist')
    wm = load_model(cfg)
    for s in chosen:
        for seed in cfg['matched']['seeds']:
            rec = measured(wm, lambda: matched_stream(wm, s, index, rc, int(seed), actions, cfg['matched']['steps']))
            write_json(out / f'matched_{s.name}_seed{seed}.json', rec)
    finish(out)


def cmd_analyze(run_dir: Path):
    cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    default = parse_action(cfg['default']).label
    lam = float(cfg.get('analysis', {}).get('ridge_lambda', 1.0))
    result = {}
    p0 = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(run_dir.glob('p0_*.json'))]
    if p0:
        rule, cur_mode = cfg['p0'].get('repair_rule', 'total'), cfg['p0'].get('repair_cur', 'decision')
        result['p0'] = [p0_verdict(r, cfg['p0']['thresholds'], default, rule, cur_mode) for r in p0]
        if 'gate' in cfg['p0']:
            result['p0_gate'] = p0_gate(result['p0'], cfg['p0']['gate'])
    matched = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(run_dir.glob('matched_*.json'))]
    if matched:
        sel = select(matched, lam=lam, default_label=default)
        result['selection'] = sel['result']
        result['state_dependence'] = state_dependence(matched)
        result['swap'] = {k: swap_test(matched, k, lam) for k in ('current', 'lagged')}
    write_json(run_dir / 'analysis.json', result)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('streams', 'calibrate', 'p0', 'matched', 'analyze'))
    ap.add_argument('paths', nargs='+')
    a = ap.parse_args(argv)
    if a.command == 'analyze':
        cmd_analyze(Path(a.paths[0]))
        return
    if len(a.paths) != 2:
        ap.error('CONFIG and OUT are required')
    {'streams': cmd_streams, 'calibrate': cmd_calibrate, 'p0': cmd_p0,
     'matched': cmd_matched}[a.command](a.paths[0], Path(a.paths[1]))


if __name__ == '__main__':
    main()
