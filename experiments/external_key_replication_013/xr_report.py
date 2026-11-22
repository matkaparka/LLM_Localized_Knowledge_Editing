"""XR-13 tables: TABLES.md from TEST_RUN/analysis.json.

  python xr_report.py TEST_RUN [OUT]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pp(x, nd=2):
    return '–' if x is None else f'{100 * x:+.{nd}f}'


def pct(x, nd=2):
    return '–' if x is None else f'{100 * x:.{nd}f}%'


def ci(c):
    return '–' if not c else f'[{100 * c[0]:+.2f}, {100 * c[1]:+.2f}]'


def f3(x):
    return '–' if x is None else f'{x:.3f}'


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def per_family_intervals(run: Path, a: dict, keep: list, sel: str | None) -> list:
    """Per-family 95% intervals: the analysis's bootstrap with a single unit, i.e. resampling the family's own items
    (10,000 times, seed 0). Descriptive; the gates use the two-level intervals over families."""
    sys.path.insert(0, str(HERE.parent / 'field_prefix_ablation_012'))
    import fp_analysis as fa
    import fp_run
    recs = fp_run.load_records(run)
    reps, seed = 10000, 0
    L = ['', '## 逐族区间（族内题目重采样，描述性）', '',
         '| 族 | Δ | 区间 | ΔU | 区间 | Q 差（原配 − 打乱） | 区间 | R 差 | 区间 |', '|---|---:|---|---:|---|---:|---|---:|---|']
    for n in a['streams']:
        rec = recs[n]
        cells = []
        if n in keep:
            for ref in (None, f'xr:{sel}'):
                if ref == 'xr:None':
                    cells += ['–', '–']
                    continue
                v = fa.delta_items(rec, 'xr', ref)
                cells += [pp(float(v.mean())), ci(fa.boot({n: v}, reps, seed))]
        else:
            cells += ['–'] * 4
        for kind in ('cur', 'ret'):
            v = fa.quality_items(rec, 'xr', kind)
            cells += [pp(float(v.mean())), ci(fa.boot({n: v}, reps, seed))]
        L.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    return L


def main(run: str, out: str | None = None):
    run = Path(run)
    a = json.loads((run / 'analysis.json').read_text(encoding='utf-8'))
    names = a['streams']
    nod = a['no_damage_measurement']
    keep = [n for n in names if n not in nod]
    sel = (a['selected_uniform'] or {}).get('arm')
    yn = lambda x: '是' if x else ('–' if x is None else '否')  # noqa: E731
    L = ['# XR-13 表格（`' + run.name + '`）', '',
         '由 `xr_report.py` 从 `analysis.json` 生成。D = 第 8 次写入后底座答对的隐藏旧题（许可证）里答错的比例；'
         'Δ = 8 个打乱的平均 D − 原配 D；ΔU = 选定统一阈值的 D − 原配 D（正值表示原配少写坏）；Q、R 差 = 原配 − 对照；'
         '单位：百分点。每个软件包仓库是一个统计单位，族等权；区间是两层配对 bootstrap（族、族内题目），10,000 次，种子 0。',
         '',
         f'底座答对的旧题少于 30 个、不能量写坏的族（PLAN 第 5 节，测试前写定）：'
         + ('、'.join(f'{n}（{k} 个）' for n, k in nod.items()) if nod else '无') +
         '。它们不进 Δ、ΔU 和两道门的单位数，Q、R 照常计入。', '']
    L += ['## 首要的数', '',
          '| 量 | 平均 | 95% 区间 | 为正的族 | ' + ' | '.join(names) + ' |',
          '|---|---:|---|---:|' + '---:|' * len(names)]
    for key, label in (('delta', 'Δ（原配 vs 打乱）'), ('delta_uniform', f'ΔU（原配 vs `{sel}`）'),
                       ('quality_q', 'Q 差（原配 − 打乱）'), ('quality_r', 'R 差（原配 − 打乱）'),
                       ('practical_quality_q', f'Q 差（原配 − `{sel}`）'), ('practical_quality_r', f'R 差（原配 − `{sel}`）')):
        s = a.get(key)
        if not s:
            continue
        L.append(f'| {label} | {pp(s["mean"])} | {ci(s["ci"])} | {s["positive"]}/{s["n"]} | ' +
                 ' | '.join(pp(s['per_stream'].get(n)) for n in names) + ' |')
    L += ['', '## 两道门（描述性判据，PLAN 第 5 节）', '',
          '| 门 | 平均 | 95% 区间 | 为正 / 需要 | Q 差 | R 差 | 结果 |', '|---|---:|---|---:|---:|---:|:---:|']
    for key, label in (('mechanism_gate', '机制门：原配 vs 打乱'), ('practical_gate', f'实用门：原配 vs `{sel}`')):
        g = a.get(key)
        if g:
            L.append(f'| {label} | {pp(g["mean_delta"])} | {ci(g["ci"])} | {g["positive"]} / {g["needed"]} | '
                     f'{pp(g["q_diff"])} | {pp(g["r_diff"])} | {"过" if g["passed"] else "不过"} |')
    s = a['selected_uniform'] or {}
    L += ['', f'统一阈值由开发流选定：`{s.get("arm")}`（{s.get("flag")}）。']
    if a.get('leave_one_out'):
        L += ['', '## 去掉一族后的 Δ 均值（描述性）', '', '| 去掉 | Δ 均值 |', '|---|---:|']
        L += [f'| {n} | {pp(v)} |' for n, v in a['leave_one_out'].items()]
    L += ['', '## 逐族', '',
          '| 族 | 隐藏候选 | 底座原对 | 原配 D | 打乱 D（8 个平均） | ' + f'`{sel}` D | 统一 0.98 D | Δ | ΔU | 原配 Q | 原配 R | 拒写 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in names:
        c = a['checks'][n]
        low = a['low_damage'][n]
        dm = a['arms']
        L.append(f'| {n} | {c["candidates"]} | {c["known"]} | {pct(low["matched_damage"])} | {pct(low["shuffled_damage"])} | '
                 f'{pct(dm["xr:" + sel][n]["damage"]) if sel else "–"} | {pct(dm["xr:uniform:0.98"][n]["damage"])} | '
                 f'{pp(a["delta"]["per_stream"].get(n))} | '
                 f'{pp((a["delta_uniform"] or {}).get("per_stream", {}).get(n))} | {f3(dm["xr:matched"][n]["Q"])} | '
                 f'{f3(dm["xr:matched"][n]["R"])} | {c["extraction"]["refused"]} |')
    L += per_family_intervals(run, a, keep, sel)
    L += ['', '## 各臂', '',
          'D 只在能量写坏的族上平均；其余各列在全部族上平均。', '',
          '| 臂 | D | Q | R | 整流覆盖 | 原题 | 学会率 | 旧题触发 | 问题区触发 | 生成区触发 | 示例区触发 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in sorted(a['arms']):
        if ':shuffled:' in arm and not arm.endswith(':shuffled:0'):
            continue
        per = a['arms'][arm]
        m = {k: avg([per[n][k] for n in names]) for k in next(iter(per.values()))}
        d = avg([per[n]['damage'] for n in keep])
        L.append(f'| `{arm}` | {pct(d)} | {f3(m["Q"])} | {f3(m["R"])} | {f3(m["coverage"])} | {f3(m["update"])} | '
                 f'{f3(m["learned"])} | {pct(m["fired_loc"])} | {pct(m["fired_loc_q"])} | {pct(m["fired_loc_gen"])} | '
                 f'{pct(m["fired_loc_ctx"])} |')
    sh = [a['arms'][k] for k in a['arms'] if k.startswith('xr:shuffled:')]
    L += ['', '（打乱臂只列 `shuffled:0`；Δ 用全部 8 个。8 个打乱的 D 平均：' +
          pct(avg([avg([s[n]['damage'] for n in keep]) for s in sh])) + '。）']
    L += ['', '## 写坏的集中度（`xr:uniform:0.90`，PLAN 第 5 节）', '',
          '前 1% 的键 = 按单键致错的不同旧事实数排在前 ⌈1% × 条目数⌉ 的键；写坏不足 5 题的族只列计数。', '',
          '| 族 | 写坏题 | 有单键致错者 | 前 1% 键数 | 前 1% 键致错的题 | 占写坏题 | 致错最多的键（实体：题数） |',
          '|---|---:|---:|---:|---:|---:|---|']
    for n in names:
        c = a['concentration'][n]
        top = '；'.join(f'{t["entity"]}：{t["facts"]}' for t in c['top'][:3])
        L.append(f'| {n} | {c["damaged"]} | {c["with_culprit"]} | {c["top_k"]} | {c["numerator"]} | '
                 f'{pct(c["top_share_of_damaged"], 1) if c["judged"] else "（不判）"} | {top} |')
    cats = ('shared prefix', 'near name', 'other', 'several kinds', 'no single culprit', 'too many fired', 'no firing')
    cn = {'shared prefix': '共享前缀', 'near name': '近名', 'other': '其他', 'several kinds': '多类', 'no single culprit': '无法单键归因',
          'too many fired': '触发键 > 8', 'no firing': '无触发'}
    L += ['', '## 逐键干预：写坏题按致错键的类别（全部族合计，一题只计一次）', '',
          '| 臂 | 写坏题 | ' + ' | '.join(cn[c] for c in cats) + ' | 重跑不符 | 闭包复原 |',
          '|---|---:|' + '---:|' * (len(cats) + 2)]
    for arm, per in a['interventions'].items():
        av = [per[n] for n in names if per[n].get('available')]
        tot = {c: sum(v['per_item'].get(c, 0) for v in av) for c in cats}
        L.append(f'| `{arm}` | {sum(v["damaged"] for v in av)} | ' + ' | '.join(str(tot[c]) for c in cats) +
                 f' | {sum(v["rerun_mismatch"] for v in av)} | {sum(v["closure_right"] for v in av)} |')
    L += ['', '## 提取与核对', '',
          '| 族 | 条目 | 拒写 | 与金跨度逐字相同 | 锚点相同 | none 全对 | 未触发≠none 最多 | 倒序相同 | 示例区不触发 | 排列多重集 | 拒写槽不触发 |',
          '|---|---:|---:|---:|---:|:---:|---:|:---:|:---:|:---:|:---:|']
    for n in names:
        c = a['checks'][n]
        e = c['extraction']
        L.append(f'| {n} | {e["n"]} | {e["refused"]} | {e["exact"]} | {e["same_anchor"]} | {yn(c["none_known_all_right"])} | '
                 f'{c["unfired_max"]} | {yn(c["reversed_same"])} | {yn(c["blocked_never_fire_in_context"])} | '
                 f'{yn(c["shuffle_multisets_equal"])} | {yn(c["refused_never_active"])} |')
    dt = a.get('dev_tuning')
    if dt:
        L += ['', '## 开发流调参（4 条开发流平均）', '', '| 臂 | D | Q | R | 质量缺口 |', '|---|---:|---:|---:|---:|',
              f'| `xr:matched` | {pct(dt["matched"]["damage"])} | {f3(dt["matched"]["Q"])} | {f3(dt["matched"]["R"])} | – |']
        L += [f'| `xr:{g}` | {pct(v["damage"])} | {f3(v["Q"])} | {f3(v["R"])} | {pp(v["shortfall"])} |'
              for g, v in dt['grid'].items()]
        L += ['', f'选定：`{dt["arm"]}`（{dt["flag"]}）。缺口 = max(Q_M − 0.02 − Q_U, R_M − 0.02 − R_U)，≤ 0 为守住两道质量门。']
    dest = Path(out) if out else HERE / 'TABLES.md'
    dest.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(dest.name, len(L), 'lines')


if __name__ == '__main__':
    main(*sys.argv[1:3])
