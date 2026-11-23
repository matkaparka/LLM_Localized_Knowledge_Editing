"""BR-14 design variant from the smoke run's measured speed (PLAN section 4). Reads only timing and counts from the
smoke record (no outcome), projects every variant's remaining run time, and picks the first variant (full, r1, r2,
r3) that ends at least 45 minutes before the session deadline in /workspace/state/instance.json (the $8 cap).

  python -I br_design.py SMOKE_RECORD INSTANCE_JSON OUT_JSON

Per stream: screening + probes + model load and corpus statistics (per run, spread over its streams) + the measured
write time per entry x 1,060 entries (+ 45 rewritten corr entries) + arms x rows per arm x measured time per row
(+ the order check and 10 minutes of interventions on test streams). Rows per arm: 1,600 panel rows (cur and upd),
350 retention rows and the smoke stream's base-known count. Exit 3 when no variant fits.
"""
import gzip
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = ('full', 'r1', 'r2', 'r3')
ENTRIES, CORR, PANEL_ROWS = 1060, 45, 1600 + 350
LOAD = 420.0                 # model load, routing-centre statistics and norm report per run (seconds)
MARGIN = 2700.0


def main(smoke, inst, out):
    rec = json.loads(gzip.open(smoke).read().decode('utf-8'))
    c = rec['clock']
    t_write = c['write'] / max(1, rec['tables']['n'])
    rows = sum(v for k, v in c.items() if k.startswith('rows_'))
    t_row = sum(v for k, v in c.items() if k.startswith('eval_')) / max(1, rows)
    known = rec['screening']['known']
    fixed = c['screen'] + c['probes']
    per_arm = (PANEL_ROWS + known) * t_row
    deadline = json.loads(Path(inst).read_text(encoding='utf-8'))['deadline_epoch']
    left = deadline - time.time() - MARGIN
    table = {}
    pick = None
    for v in VARIANTS:
        cfg = json.loads((HERE / f'config_dev_{v}.json').read_text(encoding='utf-8'))
        dev, test = cfg['phases']['dev'], cfg['phases']['test']
        a_dev = sum(len(dict.fromkeys(x)) for x in dev['arms'].values())
        a_test = sum(len(dict.fromkeys(x)) for x in test['arms'].values())
        corr = CORR * t_write
        s_dev = fixed + ENTRIES * t_write + corr + a_dev * per_arm + known * t_row
        s_test = fixed + ENTRIES * t_write + corr + a_test * per_arm + known * t_row + 600
        total = 2 * LOAD + len(dev['streams']) * s_dev + len(test['streams']) * s_test + 300
        table[v] = dict(dev_streams=len(dev['streams']), dev_arms=a_dev, test_cells=test['cells'], test_arms=a_test,
                        stream_dev_s=round(s_dev), stream_test_s=round(s_test), total_h=round(total / 3600, 2),
                        fits=total <= left)
        if pick is None and total <= left:
            pick = v
    res = dict(variant=pick, left_h=round(left / 3600, 2), t_write_per_entry_s=t_write, t_row_s=t_row,
               smoke_entries=rec['tables']['n'], smoke_rows=rows, known=known, fixed_s=fixed, variants=table,
               written=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    Path(out).write_text(json.dumps(res, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(res, indent=1))
    if pick is None:
        sys.exit(3)
    print(f'VARIANT {pick}')


if __name__ == '__main__':
    main(*sys.argv[1:4])
