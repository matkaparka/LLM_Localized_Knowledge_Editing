"""Readable tables from a finished SO-08 run directory (CPU only).

  python -I so_report.py RUN_DIR          # RUN_DIR holds p0_*.json and/or matched_*.json plus analysis.json
"""
import json
import sys
from pathlib import Path


def f(x, nd=3):
    return '—' if x is None else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def p0_tables(run: Path):
    a = json.loads((run / 'analysis.json').read_text(encoding='utf-8'))
    for v in a.get('p0', []):
        raw = json.loads((run / f'p0_{v["stream"]}_seed0.json').read_text(encoding='utf-8'))
        sc = v.get('locality_screening') or dict(known='?', candidates='?', kept='?')
        cap, con = v['capability'], v['conflict']
        print(f'\n## P0 {v["stream"]}  ({raw["seconds"] / 60:.1f} min, peak {raw["peak_allocated_bytes"] / 2**30:.2f} GiB)')
        print(f'locality screening: {sc["known"]}/{sc["candidates"]} base-known, kept {sc["kept"]}')
        print(f'capability: no info {f(cap["no_info"])}, shots only {f(cap["shots_only"])}, '
              f'fact in context {f(cap["fact_in_context"])}')
        print('| step | upd after write | cur (rephrase) | ret | loc |')
        print('|---:|---:|---:|---:|---:|')
        for t, (u, c, r, l) in enumerate(zip(cap['upd_after_write'], cap['cur_after_write'], con['retention_acc'],
                                             con['locality_acc'])):
            print(f'| {t} | {f(u)} | {f(c)} | {f(r)} | {f(l)} |')
        e, lo = con['edits'], con['locality']
        print(f'conflict: forgotten {e["right_to_wrong"]}/{e["learned_tracked"]} learned edits ({f(e["fraction"])}); '
              f'locality broken {lo["right_to_wrong"]}/{lo["base_right"]} ({f(lo["fraction"])}); pass={con["pass"]}')
        print(f'repair rule: {v.get("repair_rule", "total")}, cur: {v.get("repair_cur", "decision")}; '
              f'repairable under the other rule: {(v.get("repairs_other_rule") or {}).get("repairable", "—")}')
        for t, rep in v['repairs'].items():
            print(f'\nrepair at step {t}: default cur {f(rep["plain_cur"])}, default damage {f(rep["plain_damage"])}')
            print('| candidate | cur | forgetting | locality | damage | reduction | value |')
            print('|---|---:|---:|---:|---:|---:|---:|')
            for c in rep['candidates']:
                print(f'| {c["action"]} | {f(c["cur"])} | {f(c["forgetting"])} | {f(c["locality"])} | {f(c["damage"])} '
                      f'| {f(c["reduction"])} | {f(c["value"], 4)} |')
            print('passing:', [p['action'] for p in rep['passing']])
    if 'p0_gate' in a:
        g = a['p0_gate']
        print(f'\nP0 gate passed={g["passed"]}')
        for r in g['rows']:
            print(f'  {r["stream"]}: learn {f(r["learn_mean_upd"])} learnable={r["learnable"]} '
                  f'conflict={r["conflict"]} repairable={r["repairable"]}')


def matched_tables(run: Path):
    a = json.loads((run / 'analysis.json').read_text(encoding='utf-8'))
    if 'selection' not in a:
        return
    sel = a['selection']
    print(f'\n## Matched readout: {sel["n_states"]} decision states, streams {sel["streams"]}')
    print('| selector | mean regret | oracle hit rate |')
    print('|---|---:|---:|')
    for k, v in sel.items():
        if isinstance(v, dict) and 'mean_regret' in v:
            print(f'| {k} | {f(v["mean_regret"], 4)} | {f(v.get("oracle_hit_rate"))} |')
    dep = a['state_dependence']
    print(f'\nstate dependence: {dep["n_pairs"]} seed pairs, best candidate differs in {f(dep["differ_rate"])}')
    for k, s in a['swap'].items():
        if s.get('n'):
            print(f'swap {k}: n={s["n"]}, regret matched {f(s["mean_regret_matched"], 4)}, swapped '
                  f'{f(s["mean_regret_swapped"], 4)}, swapped worse {s["swapped_worse"]}, better {s["swapped_better"]}')
    best = {}
    for p in sorted(run.glob('matched_*.json')):
        raw = json.loads(p.read_text(encoding='utf-8'))
        for st in raw['states']:
            vals = {b['action']: b['value'] for b in st['branches']}
            top = max(vals, key=vals.get)
            best[top] = best.get(top, 0) + 1
    print('\nbest candidate counts over decision states:', dict(sorted(best.items(), key=lambda x: -x[1])))


if __name__ == '__main__':
    run_dir = Path(sys.argv[1])
    if list(run_dir.glob('p0_*.json')):
        p0_tables(run_dir)
    if list(run_dir.glob('matched_*.json')):
        matched_tables(run_dir)
