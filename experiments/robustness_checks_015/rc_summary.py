"""One comparison table across RC-15 / BR-14 runs for one cell (default corr/off): Δ, Δ_active, the factorial's ban
and continuous parts, ΔU against a fixed comparator, damage under the flat floor and under matched, matched Q / R,
and the share of banned and raised entries. Reads each run's analysis.json; writes JSON and prints a Markdown table.

  python -I rc_summary.py OUT_JSON CELL LABEL=RUN_DIR [LABEL=RUN_DIR ...]
"""
import json
import sys
from pathlib import Path


def pick(r, key):
    s = r.get(key)
    return None if not s else dict(mean=s['mean'], ci=s['ci'], positive=s['positive'], n=s['n'])


def main(out, cell, *pairs):
    rows = {}
    for p in pairs:
        label, _, run = p.partition('=')
        a = json.loads((Path(run) / 'analysis.json').read_text(encoding='utf-8'))
        r = a['cells_result'][cell]
        o = a['options'] or {}
        tau = f'{(o.get("tau") or 0.9):.2f}'
        means = a['arm_means']
        f = r.get('factorial') or {}
        prac = {k: dict(mean=v['delta']['mean'], ci=v['delta']['ci'], positive=v['delta']['positive'], n=v['delta']['n'],
                        q=v['q']['mean'], r=v['r']['mean'], passed=v['gate']['passed']) for k, v in (r.get('practical') or {}).items()}
        bans = r['bans']
        ent = sum(b['entries'] for b in bans.values())
        rows[label] = dict(
            run=str(run), streams=a['streams'], options=o, delta=pick(r, 'delta'), delta_active=pick(r, 'delta_active'),
            delta_bans=pick(r, 'delta_bans'), delta_gate=(r.get('delta_gate') or {}).get('passed'),
            delta_active_gate=(r.get('delta_active_gate') or {}).get('passed'),
            ban_flat=pick(f, 'ban_flat'), cont_ban=pick(f, 'cont_ban'), ban_cont=pick(f, 'ban_cont'),
            total=pick(f, 'total'), practical=prac,
            d_matched=means[f'{cell}:matched']['damage'], d_floor=(means.get(f'{cell}:uniform:{tau}') or {}).get('damage'),
            q_matched=means[f'{cell}:matched']['Q'], r_matched=means[f'{cell}:matched']['R'],
            update_matched=means[f'{cell}:matched']['update'],
            banned_share=sum(b['banned'] for b in bans.values()) / ent if ent else None,
            raised_share=sum(b['mid'] + b['banned'] for b in bans.values()) / ent if ent else None,
            known=sum(c['known'] for c in a['checks'].values()))
    Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    def pp(s):
        return '—' if not s or s.get('mean') is None else f'{100 * s["mean"]:+.2f} [{100 * s["ci"][0]:+.2f}, {100 * s["ci"][1]:+.2f}] {s["positive"]}/{s["n"]}'
    print('| 运行 | Δ | Δ_active | 禁用效应（统一时） | 连续效应（保留禁用） | 原配 D | 底线统一 D | 原配 Q / R | 抬高 / 禁用占比 |')
    print('|---|---|---|---|---|---:|---:|---|---|')
    for k, v in rows.items():
        floor = '—' if v['d_floor'] is None else f'{100 * v["d_floor"]:.2f}%'
        print(f'| {k} | {pp(v["delta"])} | {pp(v["delta_active"])} | {pp(v["ban_flat"])} | {pp(v["cont_ban"])} | '
              f'{100 * v["d_matched"]:.2f}% | {floor} | '
              f'{v["q_matched"]:.3f} / {v["r_matched"]:.3f} | {100 * v["raised_share"]:.1f}% / {100 * v["banned_share"]:.1f}% |')


if __name__ == '__main__':
    main(*sys.argv[1:])
