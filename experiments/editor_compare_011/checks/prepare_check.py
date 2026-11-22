"""EC-11: what ec_data.prepare_half gives on the real files with the frozen dev config (CPU, no model).

  python checks/prepare_check.py CONFIG OUT.json

Per half: stream sizes after the per-step cap, probe and hidden pool sizes after the filters, what the filters removed,
and whether the dev-half streams keep KT-10's edits, probes and first 6,000 hidden candidates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import ec_data  # noqa: E402
from rv_policy import cap_stream  # noqa: E402
from so_data import build_streams, load_edits  # noqa: E402


def main(cfg_path: str, out_path: str):
    cfg = json.loads(Path(cfg_path).read_text(encoding='utf-8'))
    sc = cfg['streams']
    edits, windows, _ = load_edits(cfg['data_dir'], expect_steps=cfg['expect_steps'])
    report = {}
    for phase in ('dev', 'test', 'smoke'):
        half = sc[phase]['half']
        streams, corpus, info = ec_data.prepare_half(edits, len(windows), half, sc['loc_hidden_eval'],
                                                     sc['mu_corpus']['buckets'], sc['mu_corpus']['n'],
                                                     stream_args=sc['args'],
                                                     filter_other_half=sc[phase].get('filter_other_half'))
        rows = {}
        base = {s.name: s for s in build_streams(edits, len(windows), half=half, **sc['args'])} if half == 'dev' \
            else {}
        for name in sc[phase]['names']:
            s = cap_stream(streams[name], sc[phase].get('cap', sc['cap']))
            row = dict(info[name], edits_capped=len(s.edits), probes=len(s.loc_probe), hidden=len(s.loc_hidden))
            if name in base:
                b = cap_stream(base[name], sc['cap'])
                row['same_edits_as_kt10'] = [e.uid for e in s.edits] == [e.uid for e in b.edits]
                row['same_probes_as_kt10'] = s.loc_probe == b.loc_probe
                row['first_6000_hidden_as_kt10'] = s.loc_hidden[:len(b.loc_hidden)] == b.loc_hidden
            rows[name] = row
        report[phase] = dict(half=half, corpus=len(corpus), streams=rows)
    Path(out_path).write_text(json.dumps(report, indent=1), encoding='utf-8')
    for phase, r in report.items():
        print(phase, r['half'], 'corpus', r['corpus'])
        for name, row in r['streams'].items():
            print(' ', name, row)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
