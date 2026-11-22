"""RV-09 command line. Nothing is downloaded: the WikiBigEdit files and the model must already be local.

  python rv_run.py calibrate CONFIG OUT     editor variants on the calibration (pilot) stream
  python rv_run.py a1 CONFIG OUT            A1: every writer as a fixed policy on every family
  python rv_run.py a2 CONFIG OUT A1_DIR     A2: two actions on reachable states x futures (only if A1 says go)
  python rv_run.py analyze RUN_DIR          CPU analysis of a1_*.json / a2_*.json in RUN_DIR -> analysis.json
  python rv_run.py auto CONFIG OUT          a1 -> analyze -> a2 (if go) -> analyze, under OUT/a1 and OUT/a2

OUT must not exist yet. A run writes its config copy, code hashes, environment and a manifest of every output.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rv_analysis import a1_gate, a1_table, a2_verdict  # noqa: E402
from rv_editor import SO08, EditWritable, LocalCfg, LocalEditor  # noqa: E402
from rv_policy import (NAMES, RunCfg, branch_pair, cap_stream, corpus, futures, parse_writer, screen,  # noqa: E402
                       spare_rows, trajectory)
from so_data import FactIndex, build_streams, load_edits, shots_text, split_halves  # noqa: E402
from so_policy import EvalCfg, WriteCfg  # noqa: E402

CODE = ('rv_editor.py', 'rv_policy.py', 'rv_analysis.py', 'rv_run.py')
SO08_CODE = ('so_data.py', 'so_model.py', 'so_policy.py', 'so_analysis.py')


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


PATH_ENV = {'data_dir': 'RV_DATA_DIR', 'model_path': 'RV_MODEL_PATH'}


def load_config(cfg_path) -> tuple[dict, dict]:
    """The frozen config, with data and model locations replaced by RV_DATA_DIR / RV_MODEL_PATH when set (another
    host keeps the same files elsewhere; the config file itself stays byte-identical). Returns (cfg, overrides)."""
    cfg = json.loads(Path(cfg_path).read_text(encoding='utf-8'))
    over = {k: os.environ[v] for k, v in PATH_ENV.items() if os.environ.get(v)}
    cfg.update(over)
    return cfg, over


def start(out: Path, cfg_path, command: str) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    cfg, over = load_config(cfg_path)
    write_json(out / 'config.json', cfg)
    write_json(out / 'run.json', dict(command=command, config_sha256=sha(Path(cfg_path)), path_overrides=over,
                                      started=time.time(), code={f: sha(HERE / f) for f in CODE},
                                      so08_code={f: sha(SO08 / f) for f in SO08_CODE}, environment=environment()))
    return cfg


def finish(out: Path):
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p.name != 'manifest.json'}
    write_json(out / 'manifest.json', files)


def keep_awake() -> bool:
    """Ask Windows not to idle-sleep while this process runs (released when it exits). Changes no setting."""
    if sys.platform != 'win32':
        return False
    import ctypes
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001))


def gpu_snapshot() -> dict:
    import subprocess
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,power.draw,memory.used',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        out = f'unavailable: {type(e).__name__}'
    return dict(time=time.time(), temp_c_sm_mhz_maxsm_mhz_power_w_mem_mib=out)


class Env:
    """Data, model and editor shared by the commands of one process."""

    def __init__(self, cfg: dict, load_model: bool = True):
        import torch
        self.cfg = cfg
        edits, windows, _ = load_edits(cfg['data_dir'], expect_steps=cfg.get('expect_steps', 8))
        self.edits, self.index = edits, FactIndex(edits)
        self.streams = {s.name: s for s in build_streams(edits, len(windows), **cfg['streams'])}
        self.spare = spare_rows(edits, list(self.streams.values()), split_halves(edits), cfg['streams']['half'])
        self.corpus = corpus(self.spare, set(cfg['corpus']['buckets']), int(cfg['corpus']['n']))
        self.rc = RunCfg(write=WriteCfg(**cfg['write']), ev=EvalCfg(**cfg['eval']))
        self.wm = self.editor = None
        if not load_model:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise SystemExit('no CUDA device')
        print(f'keep-awake request: {keep_awake()}', flush=True)
        torch.manual_seed(int(cfg.get('torch_seed', 0)))
        tok = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(cfg['model_path'], local_files_only=True,
                                                     dtype=getattr(torch, cfg.get('dtype', 'bfloat16'))).to('cuda')
        layers = range(model.config.num_hidden_layers) if cfg.get('layers', 'all') == 'all' else cfg['layers']
        sites = [s.format(i=i) for i in layers for s in cfg['sites']]
        self.wm = EditWritable(model, tok, sites, rank=int(cfg['rank']), scale=float(cfg['scale']),
                               clip=cfg.get('clip'))
        self.editor = self.make_editor(cfg['local'])

    def make_editor(self, local: dict) -> LocalEditor:
        local = dict(local)
        if 'layers' in local:
            local['layers'] = tuple(local['layers'])
        ed = LocalEditor(LocalCfg(**local))
        info = ed.prepare(self.wm, self.corpus)
        print(f'editor prepared: {json.dumps(info)[:300]}', flush=True)
        return ed

    def family(self, name: str, editor: LocalEditor | None = None):
        """Capped stream with base-known locality, the base answers, and the editor's public probe positions."""
        s = cap_stream(self.streams[name], self.cfg.get('cap'))
        s, info, base_loc = screen(self.wm, s, self.rc.ev)
        ed = editor or self.editor
        if ed is not None and ed.cfg.mode == 'route' and ed.cfg.probe_margin is not None:
            info['probes'] = ed.set_probes(self.wm, list(s.loc_probe), shots_text(s.shots))
        return s, info, base_loc


def measured(env: Env, fn) -> dict:
    import torch
    torch.cuda.reset_peak_memory_stats()
    g0, t0, c0 = gpu_snapshot(), time.time(), dict(env.wm.costs)
    rec = fn()
    rec.update(seconds=time.time() - t0, costs={k: env.wm.costs[k] - c0.get(k, 0.0) for k in env.wm.costs},
               gpu_before=g0, gpu_after=gpu_snapshot(), peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return rec


def _strip(traj: dict) -> dict:
    traj = dict(traj)
    traj.pop('captures', None)
    return traj


def fname(*parts) -> str:
    return '_'.join(str(p).replace(':', '-') for p in parts) + '.json'


# ---- commands -----------------------------------------------------------------------------------------------------
def cmd_calibrate(cfg_path, out: Path):
    cfg = start(out, cfg_path, 'calibrate')
    env = Env(cfg)
    cal = cfg['calibrate']
    for i, var in enumerate(cal['variants']):
        local = dict(cfg['local'], **var.get('local', {}))
        env.editor.detach()
        ed = env.make_editor(local)
        s, info, base_loc = env.family(cal['stream'], ed)
        writer = parse_writer(var.get('writer', 'local'))
        rec = measured(env, lambda: dict(traj=_strip(trajectory(env.wm, ed, s, env.index, writer, env.rc,
                                                                  int(cal.get('seed', 0)), base_loc,
                                                                  max_steps=cal['steps']))))
        rec.update(variant=var, local=local, family=cal['stream'], writer=writer.label, seed=int(cal.get('seed', 0)),
                   base_loc=base_loc, screening=info, trajectory=rec.pop('traj'))
        write_json(out / fname('calibrate', i), rec)
        print(f'variant {i} done: {rec["seconds"]:.0f} s', flush=True)
        env.editor = ed
    finish(out)


def cmd_a1(cfg_path, out: Path):
    """Writers outer, families inner (the first writers finish on every family first); each family is screened
    once and its public probe positions are kept for the local writers."""
    cfg = start(out, cfg_path, 'a1')
    env = Env(cfg)
    a1 = cfg['a1']
    writers = [parse_writer(w) for w in a1['writers']]
    fams = {}
    for fam in a1['families']:
        s, info, base_loc = env.family(fam)
        fams[fam] = (s, info, base_loc, env.editor.probe_states)
        print(f'family {fam}: {info}', flush=True)
    for w in writers:
        for fam in a1['families']:
            s, info, base_loc, probes = fams[fam]
            env.editor.probe_states = probes
            for seed in a1['seeds']:
                rec = measured(env, lambda: dict(trajectory=_strip(trajectory(
                    env.wm, env.editor if w.local else None, s, env.index, w, env.rc, int(seed), base_loc))))
                rec.update(family=fam, writer=w.label, seed=int(seed), base_loc=base_loc, screening=info)
                write_json(out / fname('a1', fam, w.label, f's{seed}'), rec)
                print(f'A1 {fam} {w.label} seed {seed}: {rec["seconds"]:.0f} s', flush=True)
    finish(out)


def cmd_a2(cfg_path, out: Path, a1_dir: Path, families=None):
    cfg = start(out, cfg_path, 'a2')
    if families is not None:
        cfg['a2']['families'] = list(families)
        write_json(out / 'families_override.json', dict(families=list(families),
                                                       reason='A1 exceeded its share of the budget (PLAN section 4)'))
    gate = json.loads((a1_dir / 'analysis.json').read_text(encoding='utf-8'))['a1_gate']
    if not gate['go']:
        write_json(out / 'a2_skipped.json', dict(reason=gate['reason'], a1=str(a1_dir)))
        finish(out)
        print(f'A2 skipped: {gate["reason"]}', flush=True)
        return
    env = Env(cfg)
    a2 = cfg['a2']
    d, b = parse_writer(gate['default']), parse_writer(gate['second'])
    t = int(a2['t'])
    for fam in a2.get('families', cfg['a1']['families']):
        s, info, base_loc = env.family(fam)
        futs = futures(s, env.spare, NAMES.index(fam), t, int(a2['futures']), int(a2['per_step']))
        sizes = [[sum(1 for e in f.edits if e.step == tau) for tau in range(f.steps)] for f in futs]
        for seed in a2['states']:
            def run():
                hist = trajectory(env.wm, env.editor if d.local else None, s, env.index, d, env.rc, int(seed),
                                  base_loc, max_steps=t, evaluate_steps=False)
                state = env.wm.snapshot()
                br = branch_pair(env.wm, env.editor, futs, env.index, state, t, (d, b), d, env.rc, base_loc)
                return dict(history=[dict(step=x['step'], exec=x['exec']) for x in hist['steps']], branches=br)
            rec = measured(env, run)
            rec.update(family=fam, seed=int(seed), t=t, actions=[d.label, b.label], default=d.label,
                       future_sizes=sizes, base_loc=base_loc, screening=info)
            write_json(out / fname('a2', fam, f's{seed}'), rec)
            print(f'A2 {fam} state {seed}: {rec["seconds"]:.0f} s', flush=True)
    finish(out)


def cmd_analyze(run_dir: Path):
    cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    result = {}
    a1 = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(run_dir.glob('a1_*.json'))]
    if a1:
        rows = a1_table(a1)
        result['a1_rows'] = rows
        result['a1_gate'] = a1_gate(rows, cfg['gate'])
    a2 = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(run_dir.glob('a2_*.json'))
          if p.name != 'a2_skipped.json']
    if a2:
        acts = a2[0]['actions']
        result['a2'] = a2_verdict(a2, acts[0], acts[1], dict(cfg['a2'], delta=cfg['gate']['delta']))
    write_json(run_dir / 'analysis.json', result)
    return result


def cmd_auto(cfg_path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    cmd_a1(cfg_path, out / 'a1')
    a1_hours = (time.time() - t0) / 3600
    res = cmd_analyze(out / 'a1')
    print(f'A1 gate: {res["a1_gate"]["reason"]} ({a1_hours:.2f} h)', flush=True)
    cfg, _ = load_config(cfg_path)
    limit = cfg.get('a1_hours_before_cut', 4.5)
    fams = cfg['a1']['families'][:3] if a1_hours > limit else None
    cmd_a2(cfg_path, out / 'a2', out / 'a1', families=fams)
    if any((out / 'a2').glob('a2_D*.json')):
        res = cmd_analyze(out / 'a2')
        print(f'A2 verdict: {res["a2"]["verdict"]}', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('calibrate', 'a1', 'a2', 'analyze', 'auto'))
    ap.add_argument('paths', nargs='+')
    a = ap.parse_args(argv)
    if a.command == 'analyze':
        cmd_analyze(Path(a.paths[0]))
    elif a.command == 'a2':
        if len(a.paths) != 3:
            ap.error('CONFIG OUT A1_DIR are required')
        cmd_a2(a.paths[0], Path(a.paths[1]), Path(a.paths[2]))
    else:
        if len(a.paths) != 2:
            ap.error('CONFIG and OUT are required')
        {'calibrate': cmd_calibrate, 'a1': cmd_a1, 'auto': cmd_auto}[a.command](a.paths[0], Path(a.paths[1]))


if __name__ == '__main__':
    main()
