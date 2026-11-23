"""BR-14 exploratory condition (PLAN section 9): the main test config with the floor lowered to 0.80, test streams
D4-D7, cell corr/off only. Written on the host after the main run's tune, so the practical comparator can be the
main run's dev-selected uniform level; everything else is fixed here. Refuses to overwrite.

  python -I make_explore_config.py CONFIG_TEST OUT
"""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

FLOOR = 0.80
SH = [f'shuffled:{p}' for p in range(8)]
AC = [f'active_shuffled:{p}' for p in range(8)]


def main(test_path, out_path):
    out = Path(out_path)
    if out.exists():
        raise SystemExit(f'{out} exists')
    src = Path(test_path)
    c = json.loads(src.read_text(encoding='utf-8'))
    sel = c['selected']['corr/off']['uniform']['level']
    e = copy.deepcopy(c)
    e['local']['tau'] = FLOOR
    arms = ['matched'] + SH + AC + ['noban_matched', 'uniform:0.80', 'uniform:0.84', 'uniform:0.88', 'uniform:0.90',
                                    'ban_uniform:0.80', f'uniform:{sel:.2f}']
    e['phases'] = dict(test=dict(streams=['D4', 'D5', 'D6', 'D7'], cap=150, cells=['corr/off'],
                                 arms={'corr/off': list(dict.fromkeys(arms))},
                                 interventions=['uniform:0.80', 'matched', 'shuffled:0']))
    e['practical'] = {'corr/off': list(dict.fromkeys(['uniform:0.80', f'uniform:{sel:.2f}']))}
    e.pop('tune', None)
    e['variant'] = 'explore_f080'
    e['status'] = (f'EXPLORATORY config written by make_explore_config.py on '
                   f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} from {src.name} '
                   f'(sha256 {hashlib.sha256(src.read_bytes()).hexdigest()}): floor 0.80 (PLAN section 9).')
    out.write_text(json.dumps(e, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(out.name, hashlib.sha256(out.read_bytes()).hexdigest(), 'selected uniform', sel)


if __name__ == '__main__':
    main(*sys.argv[1:3])
