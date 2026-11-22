"""Print the RV-09 tables (markdown) from a run directory: python rv_report.py RUN_DIR (with a1/ and a2/)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def pct(x, d=1):
    return '—' if x is None else f'{100 * x:.{d}f}%'


def num(x, d=3):
    return '—' if x is None else f'{x:.{d}f}'


def a1_tables(an: dict) -> str:
    rows = an['a1_rows']
    gate = an['a1_gate']
    by = defaultdict(list)
    for r in rows:
        by[r['writer']].append(r)
    out = ['| 写法 | 家族 | 学会（原题） | 改写 | 流末保持 | 流末写坏 | 遗忘 | 价值 | 用时 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for w in sorted(by, key=lambda w: -gate['summary'][w]['value']):
        for r in sorted(by[w], key=lambda r: r['family']):
            out.append(f"| `{w}` | {r['family']} | {num(r['learn'])} | {num(r['rephrase'])} | {num(r['retention_final'])} | "
                       f"{r['damage_n']}/{r['base_right']} ({pct(r['damage'])}) | {pct(r['forgetting'])} | "
                       f"{num(r['value'])} | {r['seconds'] / 60:.1f} min |")
    out.append('')
    out.append('| 写法 | 学会均值 | 写坏均值 | 价值均值 | 可行 |')
    out.append('|---|---:|---:|---:|---|')
    for w, s in sorted(gate['summary'].items(), key=lambda kv: -kv[1]['value']):
        out.append(f"| `{w}` | {num(s['learn'])} | {pct(s['damage'])} | {num(s['value'])} | {'是' if s['feasible'] else '否'} |")
    out.append('')
    out.append(f"门：{gate['reason']}（go = {gate['go']}）")
    if 'value_diff_default_minus_second' in gate:
        diffs = ', '.join(f'{f} {v:+.4f}' for f, v in gate['value_diff_default_minus_second'].items())
        out.append(f"价值差（{gate.get('default')} − {gate.get('second')}）：{diffs}")
    return '\n'.join(out)


def a2_tables(an: dict) -> str:
    a = an['a2']
    out = [f"动作：`{a['actions'][0]}` 对 `{a['actions'][1]}`；D = G(前者) − G(后者)。", '',
           '| 家族 | 各状态 μ（找） | 各状态 μ（复核） | 交叉拟合增益 | 样本内上界 | 反转（候选/复核） | 状态内 SD | 状态间 SD |',
           '|---|---|---|---:|---:|---|---:|---:|']
    for f, e in a['families'].items():
        md = ', '.join(f'{x:+.4f}' for x in e['mu_disc'])
        mv = ', '.join(f'{x:+.4f}' for x in e['mu_ver'])
        out.append(f"| {f} | {md} | {mv} | {e['cross_fitted_gain']:+.4f} | {e['in_sample_bound']:.4f} | "
                   f"{'是' if e['reversal_candidate'] else '否'}/{'是' if e['reversal_confirmed'] else '否'} | "
                   f"{e['sd_within_state']:.4f} | {e['sd_between_states']:.4f} |")
    lo, hi = a['cross_fitted_ci']
    blo, bhi = a['in_sample_ci']
    out += ['', f"交叉拟合增益均值 {a['cross_fitted_gain']:+.4f}（家族 bootstrap 95% [{lo:+.4f}, {hi:+.4f}]）；"
                f"样本内上界均值 {a['in_sample_bound']:.4f}（[{blo:.4f}, {bhi:.4f}]）；平均 D {a['mean_D']:+.4f}；"
                f"复核过的反转 {a['confirmed_reversals']}/{len(a['families'])} 个家族。", f"结论：{a['verdict']}"]
    return '\n'.join(out)


def main(run: Path):
    a1 = run / 'a1' / 'analysis.json'
    if a1.exists():
        print('## A1\n')
        print(a1_tables(json.loads(a1.read_text(encoding='utf-8'))))
    a2 = run / 'a2' / 'analysis.json'
    if a2.exists():
        an = json.loads(a2.read_text(encoding='utf-8'))
        if 'a2' in an:
            print('\n## A2\n')
            print(a2_tables(an))
    skipped = run / 'a2' / 'a2_skipped.json'
    if skipped.exists():
        print('\nA2 未运行：' + json.loads(skipped.read_text(encoding='utf-8'))['reason'])


if __name__ == '__main__':
    main(Path(sys.argv[1]))
