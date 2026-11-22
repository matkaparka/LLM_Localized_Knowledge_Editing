"""XR-13 dev acceptance (PLAN section 4; the lines were fixed before any dev run). The test streams run only if all pass.

  extraction   the extractor's anchor equals the gold anchor on >= 90% of the dev edits, and <= 10% are refused
  old facts    on average >= 300 base-known hidden facts per dev stream
  learning     the original question is right just after its write on >= 80% (xr:matched, mean over dev streams)

Also records the precision planning numbers: the dev streams' mean and standard deviation of Delta, and the 95%
half-width that 8 test families would give at that spread.

  python xr_accept.py DEV_RUN CONFIG OUT.json
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'field_prefix_ablation_012'))
import fp_analysis as fa  # noqa: E402
import fp_run  # noqa: E402

LINES = dict(extraction_anchor=0.90, max_refused=0.10, min_known=300, min_update=0.80)


def main(dev_run: str, cfg_path: str, out_path: str):
    recs = fp_run.load_records(Path(dev_run))
    ext = [r['extraction'] for r in recs.values()]
    n = sum(e['n'] for e in ext)
    refused = sum(e['refused'] for e in ext)
    anchor = sum(e['same_anchor'] for e in ext) / max(1, n - refused)
    known = statistics.mean(r['screening']['known'] for r in recs.values())
    upd = statistics.mean(fa.arm_metrics(r, 'xr:matched')['update'] for r in recs.values())
    deltas = [float(fa.delta_items(r, 'xr').mean()) for r in recs.values()]
    sd = statistics.stdev(deltas) if len(deltas) > 1 else None
    checks = dict(extraction_anchor=anchor >= LINES['extraction_anchor'],
                  refused=refused / max(1, n) <= LINES['max_refused'],
                  old_facts=known >= LINES['min_known'], learning=upd >= LINES['min_update'])
    out = dict(lines=LINES, values=dict(extraction_anchor=anchor, refused_share=refused / max(1, n), mean_known=known,
                                        update_after_write=upd), checks=checks, passed=all(checks.values()),
               precision=dict(dev_delta_per_stream=dict(zip(sorted(recs), deltas)),
                              dev_delta_mean=statistics.mean(deltas), dev_delta_sd=sd,
                              half_width_8=(2.365 * sd / math.sqrt(8)) if sd is not None else None))
    Path(out_path).write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=1))
    if not out['passed']:
        raise SystemExit('acceptance not passed')


if __name__ == '__main__':
    main(*sys.argv[1:4])
