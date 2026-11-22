"""KT-10 command line. Nothing is downloaded: the WikiBigEdit files and the model must already be local.

  python kt_run.py run CONFIG OUT        every stream of cfg['dev'] + cfg['test']: screen, probes, entries, all
                                         threshold arms at every step, interventions, prompt-level firing
  python kt_run.py analyze RUN_DIR       CPU analysis of stream_*.json.gz in RUN_DIR -> analysis.json

OUT must not exist yet. A run writes its config copy, code hashes (KT-10, RV-09, SO-08), environment, one gzipped
JSON record per stream and a manifest of every output. Data and model paths may be replaced by RV_DATA_DIR and
RV_MODEL_PATH (recorded in run.json); the config file itself stays byte-identical.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kt_core import (REGIONS, LoggedEditor, build_entries, evaluate_logged, hidden_fire, hidden_max, install,  # noqa: E402
                     intervene, probe_max, probe_positions, thresholds)
from rv_editor import SO08, LocalCfg  # noqa: E402
from rv_policy import cap_stream, screen  # noqa: E402
import rv_run  # noqa: E402
from so_data import shots_text  # noqa: E402

CODE = ('kt_core.py', 'kt_run.py', 'kt_analysis.py')
RV9_CODE = ('rv_editor.py', 'rv_policy.py', 'rv_analysis.py', 'rv_run.py')
SO08_CODE = ('so_data.py', 'so_model.py', 'so_policy.py', 'so_analysis.py')


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


def start(out: Path, cfg_path, command: str) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    cfg, over = rv_run.load_config(cfg_path)
    write_json(out / 'config.json', cfg)
    write_json(out / 'run.json', dict(command=command, config_sha256=sha(Path(cfg_path)), path_overrides=over,
                                      started=time.time(), code={f: sha(HERE / f) for f in CODE},
                                      rv09_code={f: sha(rv_run.HERE / f) for f in RV9_CODE},
                                      so08_code={f: sha(SO08 / f) for f in SO08_CODE},
                                      environment=rv_run.environment()))
    return cfg


def finish(out: Path):
    rv_run.finish(out)


class KTEnv(rv_run.Env):
    """RV-09's Env (data, streams, generic-key corpus, model, EditWritable) with the logging editor."""

    def make_editor(self, local: dict) -> LoggedEditor:
        local = dict(local)
        if 'layers' in local:
            local['layers'] = tuple(local['layers'])
        ed = LoggedEditor(LocalCfg(**local))
        info = ed.prepare(self.wm, self.corpus)
        print(f'editor prepared: {json.dumps(info)[:300]}', flush=True)
        return ed


def _floats(t) -> list:
    return [round(float(x), 6) for x in t.tolist()]


def run_stream(env: KTEnv, name: str, cfg: dict) -> dict:
    import torch
    wm, ed, ev = env.wm, env.editor, env.rc.ev
    lc = ed.cfg
    clock = {}
    t0 = time.time()
    s = cap_stream(env.streams[name], cfg.get('cap'))
    cand = [q for q, _ in s.loc_hidden]
    s, info, base_loc = screen(wm, s, ev)
    kept = {q for q, _ in s.loc_hidden}
    ctx = shots_text(s.shots)
    clock['screen'] = time.time() - t0

    t1 = time.time()
    states, labels = probe_positions(wm, ed, list(s.loc_probe), ctx)
    ed.set_probes(wm, list(s.loc_probe), ctx)
    same_probes = bool(torch.equal(states, ed.probe_states))
    ed.probe_states = states
    nq = len(s.loc_probe)
    ent = build_entries(wm, ed, s, int(cfg['write_seed']))
    clock['entries'] = time.time() - t1

    t2 = time.time()
    pm = probe_max(states, labels, ent.keys, nq + 1)
    recomputed = torch.clamp(pm.max(0).values + lc.probe_margin, min=lc.tau)
    thr_check = float((recomputed - ent.matched).abs().max())
    hm = hidden_max(wm, ed, cand, ctx, ent.keys)
    known = torch.tensor([q in kept for q in cand], device=hm.device)
    clock['tables'] = time.time() - t2

    arms, thr_of, items_at, arm_clock = {}, {}, {}, {}
    for arm in cfg['arms']:
        ta = time.time()
        thr = thresholds(arm, ent, pm, nq, lc, s.name).to(wm.device).float()
        thr_of[arm] = thr
        steps = []
        for t in range(s.steps):
            install(wm, ent, t, thr)
            out, items = evaluate_logged(wm, ed, s, env.index, t, ev, base_loc)
            steps.append(out)
            items_at[t] = items
        arms[arm] = steps
        arm_clock[arm] = time.time() - ta
        last = steps[-1]
        acc = {k: ('-' if v is None else f'{v:.3f}') for k, v in last['acc'].items()}
        print(f'  {name} {arm}: cur {acc["cur"]} loc {acc["loc"]} r2w {last.get("loc_right_to_wrong")} '
              f'({arm_clock[arm]:.0f} s)', flush=True)
    clock['arms'] = arm_clock

    T = s.steps - 1
    t3 = time.time()
    install(wm, ent, T, thr_of['matched'])
    again, _ = evaluate_logged(wm, ed, s, env.index, T, ev, base_loc)
    determinism = dict(arm='matched', step=T, same_ok=again['ok'] == arms['matched'][T]['ok'],
                       same_hits=again['hits'] == arms['matched'][T]['hits'])
    interventions = {}
    for arm in cfg['intervene']['arms']:
        if arm in arms:
            interventions[arm] = intervene(wm, ed, ent, T, thr_of[arm], items_at[T], ctx, arms[arm][T],
                                           arms['none'][T], ev, int(cfg['intervene']['max_single']))
    clock['interventions'] = time.time() - t3

    hidden = dict(n=len(cand), known=int(known.sum()), per_arm={a: hidden_fire(hm, pm[nq], thr_of[a], known)
                                                                for a in thr_of},
                  entry=dict(hmax=_floats(hm.max(0).values), n_floor=hm.ge(lc.tau).sum(0).tolist(),
                             n_matched=hm.ge(ent.matched.unsqueeze(0)).sum(0).tolist()))
    pmax_rows = {'ctx': _floats(pm[nq]), 'all': _floats(pm.max(0).values)}
    for n in sorted({int(a.split(':')[1]) for a in cfg['arms'] if a.startswith('probes:')}):
        pmax_rows[f'q{n}'] = _floats(pm[:n].max(0).values)
    rec = dict(
        stream=name, role='dev' if name in cfg['dev'] else 'test', steps=s.steps, screening=info,
        probes=dict(questions=nq, positions=int(states.shape[0]), same_as_set_probes=same_probes),
        base_loc={q: [int(v[0]), v[1]] for q, v in base_loc.items()},
        entries=dict(n=ent.n, ends=ent.ends, step=ent.step, meta=ent.meta, records=ent.records,
                     matched=_floats(ent.matched), threshold_recompute_max_abs_diff=thr_check),
        probe_max=pmax_rows,
        thresholds={a: _floats(t) for a, t in thr_of.items()},
        items=[[[c, k] for c, _, _, k in items_at[t]] for t in range(s.steps)],
        arms=arms, determinism=determinism, interventions=interventions, hidden=hidden, regions=REGIONS,
        clock=clock)
    install(wm, ent, T, thr_of['none'])
    wm.restore(wm.empty_state())
    return rec


def cmd_run(cfg_path, out: Path, streams: list[str] | None = None):
    """All streams of cfg['dev'] + cfg['test'], or only `streams` (recorded in run.json as streams_override; the
    config copy stays the frozen config)."""
    cfg = start(out, cfg_path, 'run')
    names = list(cfg['dev']) + list(cfg['test'])
    if streams:
        unknown = [s for s in streams if s not in names]
        if unknown:
            raise SystemExit(f'not in the config: {unknown}')
        names = [s for s in names if s in streams]
        meta = json.loads((out / 'run.json').read_text(encoding='utf-8'))
        meta['streams_override'] = names
        write_json(out / 'run.json', meta)
    env = KTEnv(cfg)
    for name in names:
        print(f'stream {name}', flush=True)
        rec = rv_run.measured(env, lambda: run_stream(env, name, cfg))
        write_gz(out / f'stream_{name}.json.gz', rec)
        print(f'stream {name} done: {rec["seconds"]:.0f} s', flush=True)
    finish(out)


def cmd_analyze(run_dir: Path, rv09_a1: Path | None = None, extra: list[Path] | None = None):
    """Analysis of the stream records in run_dir, plus those of `extra` run directories (a stream rerun elsewhere).
    Every extra directory must have run the same config file and the same KT-10, RV-09 and SO-08 code; a stream may
    come from only one directory. Sources are recorded in analysis.json."""
    from kt_analysis import analyze
    cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    main_run = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    dirs = [run_dir] + list(extra or [])
    recs, sources = {}, {}
    for d in dirs:
        meta = json.loads((d / 'run.json').read_text(encoding='utf-8'))
        for k in ('config_sha256', 'rv09_code', 'so08_code'):
            if meta[k] != main_run[k]:
                raise SystemExit(f'{d}: {k} differs from {run_dir}')
        diff = sorted(f for f in CODE if meta['code'].get(f) != main_run['code'].get(f))
        for p in sorted(d.glob('stream_*.json.gz')):
            name = p.name[len('stream_'):-len('.json.gz')]
            if name in recs:
                raise SystemExit(f'stream {name} in more than one run directory')
            recs[name] = read_gz(p)
            sources[name] = dict(run=d.name, kt10_code_differs_in=diff, host=meta['environment'].get('device'))
    rv = None
    if rv09_a1 is not None and rv09_a1.exists():
        rv = {p.name.split('_')[1]: json.loads(p.read_text(encoding='utf-8'))
              for p in sorted(rv09_a1.glob('a1_D*_local_s0.json'))}
    result = analyze(recs, cfg, rv09=rv)
    result['sources'] = sources
    write_json(run_dir / 'analysis.json', result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('run', 'analyze'))
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--rv09-a1', type=Path, default=None, help='RV-09 runs/run_002/a1 (reproduction check)')
    ap.add_argument('--streams', default=None, help='run: only these streams (comma separated)')
    ap.add_argument('--extra', type=Path, action='append', default=None,
                    help='analyze: another run directory holding further streams (repeatable)')
    a = ap.parse_args(argv)
    if a.command == 'analyze':
        cmd_analyze(Path(a.paths[0]), a.rv09_a1, a.extra)
    else:
        if len(a.paths) != 2:
            ap.error('CONFIG and OUT are required')
        cmd_run(a.paths[0], Path(a.paths[1]), a.streams.split(',') if a.streams else None)


if __name__ == '__main__':
    main()
