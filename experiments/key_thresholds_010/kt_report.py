"""Render RUN_DIR/analysis.json as Markdown tables (TABLES.md next to this script).

  python kt_report.py RUN_DIR
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def f(x, d=3, pct=False):
    if x is None:
        return '–'
    if isinstance(x, bool):
        return '是' if x else '否'
    if isinstance(x, (int,)) and not pct:
        return str(x)
    return f'{100 * x:.{max(d - 2, 1)}f}%' if pct else f'{x:.{d}f}'


def arms_table(block, arms):
    rows = ['| 臂 | 写坏（相对 none） | 写坏（相对底座） | 改写 | 原题 | 保持 | 遗忘 | 价值 | locality 触发 | 改写被自己的键触发 | 被别的键触发 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in arms:
        v = block['arms'].get(a)
        if not v:
            continue
        rows.append(f"| `{a}` | {f(v['damage'], 4, True)} | {f(v['damage_base'], 4, True)} | {f(v['coverage'])} | "
                    f"{f(v['upd'])} | {f(v['retention'])} | {f(v['forgetting'], 4, True)} | {f(v['value'], 4)} | "
                    f"{f(v['loc_fire'], 4, True)} | {f(v['own_hit'])} | {f(v['other_hit'])} |")
    return rows


def contrast_table(c, label):
    rows = [f'**{label}**：平均 Δ = {f(c["mean_delta"], 4, True)}，两层 bootstrap 95% '
            f'[{f(c["ci95"][0], 4, True)}, {f(c["ci95"][1], 4, True)}]；Δ>0 的流 {c["positive_streams"]} 条；'
            f'改写差 {f(c["coverage_diff"], 4)}，保持差 {f(c["retention_diff"], 4)}；判定 **{c["verdict"]}**。', '',
            '| 流 | 事实数 | 对手写坏 | matched 写坏 | Δ | 改写（matched / 对手） | 保持（matched / 对手） | locality 触发（matched / 对手） |',
            '|---|---:|---:|---:|---:|---|---|---|']
    for r in c['rows']:
        rows.append(f"| {r['stream']} | {r['n_items']} | {f(r['damage_comp'], 4, True)} | {f(r['damage_ref'], 4, True)} | "
                    f"{f(r['delta'], 4, True)} | {f(r['coverage_ref'])} / {f(r['coverage_comp'])} | "
                    f"{f(r['retention_ref'])} / {f(r['retention_comp'])} | {f(r['loc_fire_ref'], 4, True)} / "
                    f"{f(r['loc_fire_comp'], 4, True)} |")
    return rows


def conc_table(per, label):
    rows = [f'**{label}**', '',
            '| 流 | 写坏（有触发） | 只开一个就能写坏 | 只关一个就恢复 | 闭包完成 / 恢复 | 闭包平均大小 | 致因键 / 条目 | 头部 k | 头部份额 | 致因键被抬高 | 全部条目被抬高 | 致因键在隐藏候选上过下限（中位） | 错答=致因键的答案 | 主语关系 | 致因键生效处 |',
            '|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|']
    for s, c in per.items():
        if not c.get('available'):
            continue
        rows.append(f"| {s} | {c['damaged']}（{c['with_firing']}） | {c['alone_explained']} | {c['single_recovers']} | "
                    f"{c['closure_complete']} / {c['closure_recovered']} | {f(c['closure_mean_size'], 1)} | "
                    f"{c['distinct_culprits']} / {c['entries']} | {c['top_k']} | {f(c['top_share'])} | "
                    f"{f(c['culprits_raised'])} | {f(c['entries_raised'])} | "
                    f"{f(c['culprit_hidden_floor_share_median'], 4, True)} | {c['hijack']} | "
                    f"{', '.join(f'{k} {v}' for k, v in sorted(c['name_relation'].items())) or '–'} | "
                    f"{', '.join(f'{k} {v}' for k, v in sorted(c['culprit_regions'].items())) or '–'} |")
    tops = [(s, t) for s, c in per.items() if c.get('available') for t in c['top_culprits'][:3]]
    if tops:
        rows += ['', '头部致因键（每流前 3 个）：', '', '| 流 | 条目 | 写坏事实数 | 主语 | 写入步 | 原配阈值 | 被抬高 | 在隐藏候选上过下限 |',
                 '|---|---:|---:|---|---:|---:|---|---:|']
        rows += [f"| {s} | {t['entry']} | {t['items']} | {t['subject']} | {t['step']} | {f(t['threshold'])} | "
                 f"{f(t['raised'])} | {f(t['hidden_floor_share'], 4, True)} |" for s, t in tops]
    return rows


def main(run: Path):
    a = json.loads((run / 'analysis.json').read_text(encoding='utf-8'))
    cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    arms = cfg['arms']
    out = [f'# KT-10 表格（`{run.name}`）', '',
           f'由 `kt_report.py` 从 `runs/{run.name}/analysis.json` 生成。写坏以第 7 步为准；「相对 none」= `none` 下答对、'
           '该臂下答错的底座已知事实所占比例。百分数均为底座已知事实的比例。', '']
    if 'tuning' in a:
        t = a['tuning']
        out += ['## 开发流上调统一阈值', '', '| 统一阈值 | 开发流平均价值 |', '|---|---:|']
        out += [f'| `{k}` | {f(v, 4)} |' for k, v in t['dev_values'].items()]
        out += ['', f'选定：`{t["tuned"]}`（{t["rule"]}）。', '']
    for role, title in (('test', '测试流 D4–D7（判定用）'), ('dev', '开发流 D0–D3（不作确认）')):
        if role not in a:
            continue
        b = a[role]
        out += [f'## {title}', '', '### 各臂（四流平均）', ''] + arms_table(b, arms) + ['']
        out += ['### H1：原配 vs 打乱', ''] + contrast_table(b['h1_matched_vs_shuffled'], 'H1') + ['']
        out += ['### H2：原配 vs 调好的统一阈值', ''] + contrast_table(b['h2_matched_vs_tuned_uniform'], 'H2') + ['']
        h3 = b['h3_concentration']
        out += ['### H3：损伤是否集中', ''] + conc_table(h3['per_stream'], f'`{h3["arm"]}`：判定 **{h3["verdict"]}**')
        for arm, per in b['interventions_other'].items():
            out += [''] + conc_table(per, f'`{arm}`（描述）')
        fr = b['frontier']
        out += ['', '### 统一阈值前沿', '', '| 臂 | 改写 | 写坏（相对 none） |', '|---|---:|---:|']
        out += [f"| `{p['arm']}` | {f(p['coverage'])} | {f(p['damage'], 4, True)} |" for p in fr['points']]
        out += ['', f"`matched`：改写 {f(fr['matched']['coverage'])}，写坏 {f(fr['matched']['damage'], 4, True)}；"
                f"同改写处统一阈值的插值写坏 {f(fr['uniform_damage_at_matched_coverage'], 4, True)}。", '']
        h = b['hidden']
        out += ['### 6,000 个隐藏候选上的提示位置触发', '', '| 臂 | 全部候选 | 底座已知 |', '|---|---:|---:|']
        out += [f"| `{k}` | {f(v['all'], 4, True)} | {f(v['known'], 4, True)} |" for k, v in h['per_arm'].items()]
        out += ['', '| 流 | Spearman(探针最大, 隐藏最大) | AUROC(探针最大 → 隐藏过下限) | 在隐藏候选上过下限的键 | 按原配阈值仍会触发的键 |',
                '|---|---:|---:|---:|---:|']
        out += [f"| {s} | {f(v['spearman_probe_hidden'])} | {f(v['auroc_probe_max_fires_on_hidden'])} | "
                f"{f(v['entries_firing_on_hidden_at_floor'], 4, True)} | {f(v['entries_firing_on_hidden_at_matched'], 4, True)} |"
                for s, v in h['per_stream'].items()]
        lit = b['literal']
        out += ['', '### 改写题：主语是否原样出现', '', '| 臂 | 原样出现 | 未出现 |', '|---|---|---|']
        out += [f"| `{k}` | {f(v['literal']['acc'])}（{v['literal']['n']}） | {f(v['not_literal']['acc'])}（{v['not_literal']['n']}） |"
                for k, v in lit.items()]
        nf = b['noise_floor']
        out += ['', '### 噪声底（`none` 相对底座）', '', '| 流 | 底座已知 | 各步原对→错 | 各步原错→对 |', '|---|---:|---|---|']
        out += [f"| {s} | {v['base_right']} | {', '.join(map(str, v['r2w_by_step']))} | {', '.join(map(str, v['w2r_by_step']))} |"
                for s, v in nf.items()]
        out += ['']
    out += ['## 检查', '', '| 流 | 确定性（ok / 触发） | 阈值重算最大差 | 探针位置与 set_probes 相同 | 候选 / 已知 | 条目 |',
            '|---|---|---:|---|---|---:|']
    for s, c in a['checks'].items():
        d = c['determinism']
        out.append(f"| {s} | {f(d['same_ok'])} / {f(d['same_hits'])} | {c['threshold_recompute_max_abs_diff']:.2g} | "
                   f"{f(c['probes_same_as_set_probes'])} | {c['screening']['candidates']} / {c['screening']['known']} | "
                   f"{c['entries']} |")
    if 'reproduction_vs_rv09' in a:
        out += ['', '## 与 RV-09 A1 `local` 的逐项一致率（改写、原题、保持题）', '', '| 流 | 各步一致率 |', '|---|---|']
        for s, rows in a['reproduction_vs_rv09'].items():
            out.append(f"| {s} | {', '.join(f(r['agree']) for r in rows)} |")
    (HERE / 'TABLES.md').write_text('\n'.join(out) + '\n', encoding='utf-8', newline='\n')
    print('TABLES.md written')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
