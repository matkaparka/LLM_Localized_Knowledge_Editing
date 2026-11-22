"""Unattended stage-1 sequence: P0 -> analyze -> P0 gate -> matched readout -> analyze.

  python -I so_auto.py CONFIG RUN_ROOT

Each step is a separate so_run.py process (so a crash leaves the earlier outputs intact). The matched-readout run
starts only if the P0 gate written by `analyze` passed; otherwise the sequence stops and says why.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def step(*args):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'start', ' '.join(args), flush=True)
    r = subprocess.run([sys.executable, '-I', '-X', 'utf8', '-B', str(HERE / 'so_run.py'), *args], cwd=HERE)
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'end', args[0], 'exit', r.returncode, flush=True)
    if r.returncode:
        raise SystemExit(f'{args[0]} failed with exit code {r.returncode}')


def main(cfg: str, root: str):
    root_dir = Path(root)
    p0, matched = root_dir / 'p0', root_dir / 'matched'
    step('p0', cfg, str(p0))
    step('analyze', str(p0))
    gate = json.loads((p0 / 'analysis.json').read_text(encoding='utf-8')).get('p0_gate') or {}
    print('P0 gate:', json.dumps(gate, ensure_ascii=False), flush=True)
    if not gate.get('passed'):
        print('P0 gate not passed: the matched-readout run is skipped, as the plan requires.', flush=True)
        return
    step('matched', cfg, str(matched))
    step('analyze', str(matched))
    print('stage 1 finished', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
