"""Markdown tables (Chinese headings, like the other studies' TABLES.md) from SP-16's analysis.json files.

  python sp_report.py OUT_MD RUN [RUN ...]
"""
import json
import sys
from pathlib import Path


def pct(x):
    return '—' if x is None else f'{100 * x:.1f}%'


def pts(x):
    return '—' if x is None else f'{100 * x:+.1f}'


def ci(c):
    return '—' if not c else f'[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]'


def tables(run: Path) -> list[str]:
    a = json.loads((run / 'analysis.json').read_text(encoding='utf-8'))
    cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    k0 = str(cfg['primary_budget'])
    L = [f'## {run.name}（{a["partition"]}，{", ".join(a["streams"])}）', '']
    for name, title in (('main', '主匹配器（开头整词匹配）'), ('author', '作者匹配器（子串，敏感性）')):
        r = a[name]
        L += [f'### {title}', '', '| 流 | 基线答对 | 受损 | 有单槽见证 | 风险前3 | 风险前13 | 风险前25 | 频率前13 | 随机前13期望 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for s, b in r['per_stream'].items():
            bu = b['budgets']
            L.append(f'| {s} | {b["baseline_correct"]} | {b["damaged"]} | {b["witnessed"]} | {bu["3"]["hits"]} | '
                     f'{bu["13"]["hits"]} | {bu["25"]["hits"]} | {bu["13"]["freq_hits"]} | '
                     f'{bu["13"]["random_expected"]:.1f} |')
        ev = r['events']
        L += ['', f'事件：受损 {ev["damaged"]} 道，{ev["streams_with_damage"]} 条流有损伤；事件{"充足" if ev["sufficient"] else "不足"}。', '',
              '| 预算 | 命中/受损 | 风险命中率 | 随机期望 | 频率命中率 | 风险−随机 | 95% 区间 | 风险−频率 | 95% 区间 |',
              '|---:|---:|---:|---:|---:|---:|---|---:|---|']
        for k, p in r['budgets'].items():
            c = p.get('ci') or {}
            L.append(f'| {k} | {p["hits"]}/{p["damaged"]} | {pct(p["rate_risk"])} | {pct(p["rate_random"])} | '
                     f'{pct(p["rate_freq"])} | {pts(p["diff_random"])} | {ci(c.get("diff_random"))} | '
                     f'{pts(p["diff_freq"])} | {ci(c.get("diff_freq"))} |')
        d = r['decision']
        L += ['', f'判定（前 {k0} 槽）：事件充足 {d["events_sufficient"]}；筛查过门（比随机高 ≥10 个点且区间下限 > 0）'
                  f' {d["screening_passed"]}；胜过频率基线 {d["beats_frequency"]}。', '',
              '| 臂 | 损伤率 | 原题覆盖 | 改写覆盖 | 相对原生修复 | 相对原生新增损伤 | 改写丢失 | 改写新增 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm, ab in r['arms'].items():
            p = ab['pooled']
            L.append(f'| {arm} | {pct(p["damage"])} | {pct(p["edit_coverage"])} | {pct(p["para_coverage"])} | '
                     f'{p["repaired"]} | {p["new_damage"]} | {p["para_lost"]} | {p["para_gained"]} |')
        m = r.get('matched_vs_shuffled')
        if m:
            L += ['', f'逐槽门槛 {pct(m["matched"])}；同块打乱平均 {pct(m["shuffled_mean"])}；禁用槽不动的打乱平均 '
                      f'{pct(m["active_mean"])}。', '']
    if 'dev_selection' in a:
        s = a['dev_selection']
        L += [f'开发选定统一阈值 {s["level"]:.2f}（{s["flag"]}）：' +
              '；'.join(f'{g["level"]:.2f} 损伤 {pct(g["damage"])} 改写 {pct(g["para_coverage"])}' for g in s['grid']), '']
    L += ['耗时（秒）：' + '；'.join(f'{s} 写 {t["write"]:.0f}、臂 {t["arms"]:.0f}、合计 {t["total"]:.0f}'
                                for s, t in a['seconds'].items()), '']
    return L


def main(out, *runs):
    L = ['# SP-16 表格', '', '由 `sp_report.py` 从各运行的 `analysis.json` 生成。比例为题数之比；点差为百分点。', '']
    for r in runs:
        L += tables(Path(r))
    Path(out).write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(f'{out}: {len(L)} lines')


if __name__ == '__main__':
    main(*sys.argv[1:])
