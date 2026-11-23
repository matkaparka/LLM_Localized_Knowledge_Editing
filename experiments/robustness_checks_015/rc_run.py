"""RC-15 / BR-14 command line. Nothing is downloaded: data, model and spans must be local.

  python rc_run.py run CONFIG OUT --phase P [--streams a,b]
  python rc_run.py tune DEV_RUN CONFIG_DEV CONFIG_TEST_OUT
  python rc_run.py analyze TEST_RUN [--dev DEV_RUN] [--ref REF_RUN]

CONFIG['kind'] is 'fp' (WikiBigEdit streams: rc_core.RCEnv, run_stream_fp) or 'xr' (XR-13 streams: xr_run.XREnv,
run_stream_xr). Every phase lists its streams, its cells, the arms of each cell and the intervention arms. In a dev
config, an arm or practical comparator written 'uniform:SEL' or 'ban_uniform:SEL' is a placeholder: `tune` replaces
it with the level that FP-12's selection rule picks on the dev records (CONFIG['tune']: cells, kinds, levels).

OUT must not exist. A run writes its config copy, code hashes (RC-15, XR-13, FP-12, EC-11, KT-10, RV-09, SO-08), the
environment overrides, one gzipped record per stream and a manifest of every output. RV_DATA_DIR, RV_MODEL_PATH and
XR_STREAMS_DIR may replace the data, model and stream paths (recorded in run.json); the config stays byte-identical.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rc_core  # noqa: E402
from rc_core import fp_run  # noqa: E402

CODE = dict(fp_run.CODE)
CODE['xr13'] = (rc_core.XR, ('xr_run.py', 'xr_analysis.py'))
CODE['rc15'] = (HERE, ('rc_core.py', 'rc_run.py', 'rc_analysis.py'))


def code_hashes() -> dict:
    return {k: {f: fp_run.sha(d / f) for f in files if (d / f).exists()} for k, (d, files) in CODE.items()}


def make_env(cfg: dict, phase: str):
    if cfg['kind'] == 'fp':
        return rc_core.RCEnv(cfg)
    if cfg['kind'] == 'xr':
        import xr_run
        return xr_run.XREnv(cfg, phase)
    raise SystemExit(f'unknown kind {cfg["kind"]}')


def cmd_run(cfg_path, out: Path, phase: str, streams: list[str] | None = None, env_factory=None):
    cfg_path, out = Path(cfg_path), Path(out)
    if out.exists():
        raise SystemExit(f'{out} exists')
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    if any('SEL' in a for arms in cfg['phases'][phase].get('arms', {}).values() for a in arms):
        raise SystemExit(f'phase {phase} still has SEL placeholders: run `tune` first')
    names = list(cfg['phases'][phase]['streams'])
    if streams:
        unknown = [x for x in streams if x not in names]
        if unknown:
            raise SystemExit(f'not in the phase: {unknown}')
        names = [x for x in names if x in streams]
    out.mkdir(parents=True)
    (out / 'config.json').write_bytes(cfg_path.read_bytes())
    env_over = {k: os.environ.get(k) for k in ('RV_DATA_DIR', 'RV_MODEL_PATH', 'XR_STREAMS_DIR') if os.environ.get(k)}
    fp_run.write_json(out / 'run.json', dict(
        command='run', phase=phase, config=cfg_path.name, config_sha256=fp_run.sha(cfg_path), streams=names,
        started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), code=code_hashes(), env_overrides=env_over,
        spans_sha256=fp_run.sha(rc_core.spans_path(cfg)) if cfg.get('spans') else None))
    env = env_factory(cfg, phase) if env_factory else make_env(cfg, phase)
    fn = rc_core.run_stream_fp if cfg['kind'] == 'fp' else rc_core.run_stream_xr
    for name in names:
        print(f'stream {name}', flush=True)
        rec = fp_run.measured(env, lambda: fn(env, name, phase))
        fp_run.write_gz(out / f'stream_{name}.json.gz', rec)
        c = rec['checks']
        print(f'stream {name} done: {rec["seconds"]:.0f} s; known {rec["screening"]["known"]}; '
              f'unfired!=none max {max(c["unfired_differs_from_none"].values())}; '
              f'none right {c["none_known_all_right"]}; reversed same {c["determinism"]["loc_reversed_same"]}; '
              f'clock {({k: round(v) for k, v in rec["clock"].items()})}', flush=True)
    fp_run.finish(out)


def resolve(arms: list[str], sel: dict, where: str) -> list[str]:
    out = []
    for a in arms:
        kind, _, val = a.partition(':')
        if val == 'SEL':
            if kind not in sel:
                raise SystemExit(f'{where}: no selection for {kind}')
            a = f'{kind}:{sel[kind]["level"]:.2f}'
        if a not in out:
            out.append(a)
    return out


def cmd_tune(dev_run: Path, cfg_dev_path, out_path: Path):
    import rc_analysis
    cfg_dev_path, out_path = Path(cfg_dev_path), Path(out_path)
    if out_path.exists():
        raise SystemExit(f'{out_path} exists')
    cfg = json.loads(cfg_dev_path.read_text(encoding='utf-8'))
    recs = fp_run.load_records(Path(dev_run))
    if set(recs) != set(cfg['phases']['dev']['streams']):
        raise SystemExit(f'dev records {sorted(recs)} != {cfg["phases"]["dev"]["streams"]}')
    sel = rc_analysis.tune(recs, cfg)
    test = copy.deepcopy(cfg)
    test['status'] = (f'TEST CONFIG written by rc_run.py tune from {Path(dev_run).name} on '
                      f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; {cfg_dev_path.name} sha256 '
                      f'{fp_run.sha(cfg_dev_path)}. Selected levels per cell (FP-12 rule).')
    test['selected'] = sel
    for pname, ph in test['phases'].items():
        for cell, lst in (ph.get('arms') or {}).items():
            if any(a.endswith(':SEL') for a in lst):
                ph['arms'][cell] = resolve(lst, sel.get(cell, {}), f'{pname}/{cell}')
    for cell, lst in (test.get('practical') or {}).items():
        test['practical'][cell] = resolve(lst, sel.get(cell, {}), f'practical/{cell}')
    out_path.write_text(json.dumps(test, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({c: {k: (v['arm'], v['flag']) for k, v in d.items()} for c, d in sel.items()}, indent=1))


def cmd_analyze(test_run: Path, dev_run: Path | None = None, ref_run: Path | None = None):
    import rc_analysis
    test_run = Path(test_run)
    cfg = json.loads((test_run / 'config.json').read_text(encoding='utf-8'))
    recs = fp_run.load_records(test_run)
    dev = fp_run.load_records(Path(dev_run)) if dev_run else None
    ref = fp_run.load_records(Path(ref_run)) if ref_run else None
    res = rc_analysis.analyze(recs, cfg, dev, ref)
    fp_run.write_json(test_run / 'analysis.json', res)
    fp_run.finish(test_run)
    print(json.dumps(res.get('headline', {}), indent=1)[:3000])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('run', 'tune', 'analyze'))
    ap.add_argument('args', nargs='+')
    ap.add_argument('--phase', default=None)
    ap.add_argument('--streams', default=None)
    ap.add_argument('--dev', default=None)
    ap.add_argument('--ref', default=None)
    a = ap.parse_args(argv)
    if a.command == 'run':
        cmd_run(a.args[0], Path(a.args[1]), a.phase, a.streams.split(',') if a.streams else None)
    elif a.command == 'tune':
        cmd_tune(Path(a.args[0]), a.args[1], Path(a.args[2]))
    else:
        cmd_analyze(Path(a.args[0]), Path(a.dev) if a.dev else None, Path(a.ref) if a.ref else None)


if __name__ == '__main__':
    main()
