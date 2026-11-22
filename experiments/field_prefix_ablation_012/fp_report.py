"""FP-12 tables: TABLES.md from TEST_RUN/analysis.json (and the stream records for the per-stream counts).

  python fp_report.py TEST_RUN
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = ('orig/on', 'orig/off', 'corr/on', 'corr/off')
NAMES = {'orig/on': '原/开', 'orig/off': '原/关', 'corr/on': '正/开', 'corr/off': '正/关', 'bridge': '桥接（全量原/开）',
         'strict': '严口径（正/关）'}


def pp(x, nd=2):
    return '–' if x is None else f'{100 * x:+.{nd}f}'


def pct(x, nd=2):
    return '–' if x is None else f'{100 * x:.{nd}f}%'


def ci(c):
    return '–' if not c else f'[{100 * c[0]:+.2f}, {100 * c[1]:+.2f}]'


def f3(x):
    return '–' if x is None else f'{x:.3f}'


def per_stream_intervals(run: Path, names: list) -> list:
    """Per-stream 95% intervals of the two headline numbers: the analysis's bootstrap with a single unit, i.e.
    resampling the stream's own base-known items (10,000 times, seed 0). Descriptive."""
    sys.path.insert(0, str(HERE))
    import fp_analysis as fa
    import fp_run
    recs = fp_run.load_records(run)
    L = ['', '逐流区间（流内题目重采样，描述性）：', '', '| 流 | Δ(正/关) | 区间 | Δ(原/开) | 区间 | Δ(原/开) − Δ(正/关) | 区间 |',
         '|---|---:|---|---:|---|---:|---|']
    for n in names:
        a, b = fa.delta_items(recs[n], 'corr/off'), fa.delta_items(recs[n], 'orig/on')
        cells = []
        for v in (a, b, b - a):
            cells += [pp(float(v.mean())), ci(fa.boot({n: v}, 10000, 0))]
        L.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    return L


def main(run: str, out: str | None = None):
    run = Path(run)
    a = json.loads((run / 'analysis.json').read_text(encoding='utf-8'))
    names = a['streams']
    L = ['# FP-12 表格（`' + run.name + '`）', '',
         '由 `fp_report.py` 从 `analysis.json` 生成。D = 第 8 次写入后 `none` 答对的旧题里答错的比例；'
         'Δ = 8 个打乱的平均 D − 原配 D（正值表示原配少写坏）；单位：百分点。区间是两层配对 bootstrap，'
         '只有四条已见流，是描述性的。', '']
    h = a['headline']
    L += ['## 首要的数', '',
          '| 量 | 平均 | 95% 区间 | ' + ' | '.join(names) + ' |', '|---|---:|---|' + '---:|' * len(names)]
    for key, label in (('delta_corr_off', 'Δ(正/关)'), ('delta_orig_on', 'Δ(原/开)'),
                       ('orig_on_minus_corr_off', 'Δ(原/开) − Δ(正/关)'), ('interaction', '交互')):
        s = h[key]
        L.append(f'| {label} | {pp(s["mean"])} | {ci(s["ci"])} | ' +
                 ' | '.join(pp(s['per_stream'].get(n)) for n in names) + ' |')
    L += per_stream_intervals(run, names)
    L += ['', '## 四格、桥接与严口径', '',
          '| 格 | Δ | 区间 | 为正的流 | 原配 D | 打乱 D（平均） | Q 差（原配 − 打乱） | R 差 | 机制门（描述） |',
          '|---|---:|---|---:|---:|---:|---:|---:|:---:|']
    for p in CELLS + ('bridge', 'strict'):
        d, q, r, g = a['deltas'][p], a['quality_q'][p], a['quality_r'][p], a['gates'][p]
        m = a['arm_means'][f'{p}:matched']['damage']
        sh = [a['arm_means'][k]['damage'] for k in a['arm_means'] if k.startswith(p + ':shuffled:')]
        L.append(f'| {NAMES[p]} | {pp(d["mean"])} | {ci(d["ci"])} | {d["positive"]}/{d["n"]} | {pct(m)} | '
                 f'{pct(sum(sh) / len(sh))} | {pp(q["mean"])} {ci(q["ci"])} | {pp(r["mean"])} {ci(r["ci"])} | '
                 f'{"过" if g["passed"] else "未过"} |')
    dec = a['decomposition']
    L += ['', '## 分解', '', '| 量 | 平均 | 95% 区间 |', '|---|---:|---|']
    for key, label in (('field_on', '字段效应，开列：Δ(原/开) − Δ(正/开)'), ('field_off', '字段效应，关列：Δ(原/关) − Δ(正/关)'),
                       ('mask_orig', '屏蔽效应，原行：Δ(原/开) − Δ(原/关)'), ('mask_corr', '屏蔽效应，正行：Δ(正/开) − Δ(正/关)'),
                       ('interaction', '交互'), ('bridge_minus_orig_on', '桥接 − 配对基线'), ('strict', 'Δ(严口径)')):
        s = dec[key]
        L.append(f'| {label} | {pp(s["mean"])} | {ci(s["ci"])} |')
    if a.get('without_d7'):
        L += ['', '去掉 D7 的描述性均值：' + '；'.join(f'{NAMES[p]} {pp(v)}' for p, v in a['without_d7'].items()) + '。']
    L += ['', '## 实用对照（选定的统一阈值）', '',
          '| 格 | 选定 | 开发流是否达门 | ΔU | 区间 | Q 差（原配 − 统一） | R 差 | 实用门（描述） |',
          '|---|---|---|---:|---|---:|---:|:---:|']
    for c in CELLS:
        s = a['selected_uniform'].get(c, {})
        du = a['practical'].get(c)
        g = a['practical_gates'].get(c, {})
        pq, pr = a.get('practical_quality_q', {}).get(c), a.get('practical_quality_r', {}).get(c)
        L.append(f'| {NAMES[c]} | `{s.get("arm", "–")}` | {s.get("flag", "–")} | {pp(du["mean"]) if du else "–"} | '
                 f'{ci(du["ci"]) if du else "–"} | {pp(pq["mean"]) + " " + ci(pq["ci"]) if pq else "–"} | '
                 f'{pp(pr["mean"]) + " " + ci(pr["ci"]) if pr else "–"} | {"过" if g.get("passed") else "未过"} |')
    L += ['', '## 各臂（四流平均）', '',
          '| 臂 | D | Q | R | 整流覆盖 | 原题 | 学会率 | 旧题触发 | 示例区触发 | 拒写 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, m in a['arm_means'].items():
        if ':shuffled:' in arm and not arm.endswith(':shuffled:0'):
            continue
        L.append(f'| `{arm}` | {pct(m["damage"])} | {f3(m["Q"])} | {f3(m["R"])} | {f3(m["coverage"])} | '
                 f'{f3(m["update"])} | {f3(m["learned"])} | {pct(m["fired_loc"])} | {pct(m["fired_loc_ctx"])} | '
                 f'{pct(m["refusal"], 1)} |')
    L += ['', '（打乱臂只列 `shuffled:0`；Δ 用全部 8 个。）', '', '## 逐流：旧题原对数与写坏数', '',
          '| 流 | 旧题原对 | ' + ' | '.join(f'{NAMES[c]} 原配 / 打乱均值' for c in CELLS) + ' |',
          '|---|---:|' + '---|' * len(CELLS)]
    for n in names:
        row = [n, str(a['arms']['none'][n]['known'])]
        for c in CELLS:
            mm = a['arms'][f'{c}:matched'][n]
            shs = [a['arms'][k][n]['damaged'] for k in a['arms'] if k.startswith(c + ':shuffled:')]
            row.append(f'{mm["damaged"]} / {sum(shs) / len(shs):.1f}')
        L.append('| ' + ' | '.join(row) + ' |')
    L += ['', '## 逐键干预：按题分类（四流合计）', '']
    cats = ('bad anchor', 'shared prefix', 'near name', 'other', 'several kinds', 'no single culprit', 'too many fired',
            'no firing')
    cn = {'bad anchor': '坏锚点', 'shared prefix': '共享前缀', 'near name': '正常近名', 'other': '其他', 'several kinds': '多类',
          'no single culprit': '无法单键归因', 'too many fired': '触发键 > 8', 'no firing': '无触发'}
    L += ['| 格:臂 | 写坏题 | ' + ' | '.join(cn[c] for c in cats) + ' | 重跑不符 |', '|---|---:|' + '---:|' * (len(cats) + 1)]
    for arm, per in a['interventions'].items():
        tot = {c: sum(per[n]['per_item'].get(c, 0) for n in names) for c in cats}
        dam = sum(per[n]['damaged'] for n in names)
        mis = sum(per[n]['rerun_mismatch'] for n in names)
        L.append(f'| `{arm}` | {dam} | ' + ' | '.join(str(tot[c]) for c in cats) + f' | {mis} |')
    L += ['', '## 核对', '', '| 流 | none 全对 | 未触发≠none 最多 | 倒序相同 | 阈值重算差 | 正行未改条目逐位相同 | 排列多重集 | 拒写槽不触发 | 改锚 | 拒写 | 条目 |',
          '|---|:---:|---:|:---:|---:|:---:|:---:|:---:|---:|---:|---:|']
    for n in names:
        c = a['checks'][n]
        yn = lambda x: '是' if x else ('–' if x is None else '否')  # noqa: E731
        L.append(f'| {n} | {yn(c["none_known_all_right"])} | {c["unfired_max"]} | {yn(c["reversed_same"])} | '
                 f'{c["matched_recompute"]:.1e} | {yn(c["corr_unchanged_bitwise"])} | {yn(c["shuffle_multisets_equal"])} | '
                 f'{yn(c["refused_never_active"])} | {c["changed"]} | {c["refused"]} | {c["entries"]} |')
    dest = Path(out) if out else HERE / 'TABLES.md'
    dest.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(dest.name, len(L), 'lines')


if __name__ == '__main__':
    main(*sys.argv[1:3])
