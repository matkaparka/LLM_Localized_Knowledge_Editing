"""EC-11 tables: TABLES.md from a test run's analysis.json (and the test config's tuning table).

  python ec_report.py TEST_RUN
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ec_arms import family

FAM = {'kt': 'KT（逐键探针阈值）', 'hyg': 'HYG（主语清洗 + 统一阈值）', 'rad': 'RAD（GRACE 半径规则，KT 条目）',
       'grace': 'GRACE-IV', 'rag': 'RAG（同信息检索）', 'rag_gated': 'RAG 全局门槛', 'ragp': 'RAG 逐条探针门槛'}


def f(x, d=3, pct=False):
    if x is None:
        return '–'
    if isinstance(x, str):
        return x
    return f'{100 * x:.{max(0, d - 1)}f}%' if pct else f'{x:.{d}f}'


def ci(v, d=3, pct=False):
    return '–' if not v else f'[{f(v[0], d, pct)}, {f(v[1], d, pct)}]'


def tuning_table(cfg) -> list[str]:
    t = cfg['tuning']
    sel = set(cfg['selected'].values())
    out = ['| 族 | 臂 | 开发流覆盖 | 开发流写坏 | 选定 |', '|---|---|---:|---:|:---:|']
    for a, row in t['table'].items():
        if a == 'none':
            continue
        out.append(f'| {family(a)} | `{a}` | {f(row["coverage"])} | {f(row["damage"], 3, True)} | '
                   f'{"✔" if a in sel else ""} |')
    out += ['', '| 族 | 选定的臂 | 开发流上是否达上限 |', '|---|---|---|']
    for g, a in cfg['selected'].items():
        note = t['notes'][g]
        out.append(f'| {FAM.get(g, g)} | `{a}` | {"是" if note["met_cap"] else "否（取写坏最低者）"} |')
    return out


def arms_table(res, arms) -> list[str]:
    out = ['| 臂 | 流末改写覆盖 | 人设 | 原题 | 非字面改写 | 写坏 | 写坏 95% 区间 | locality 触发 |',
           '|---|---:|---:|---:|---:|---:|---|---:|']
    for a in arms:
        m, c = res['arms'][a], res['cap'][a]
        out.append(f'| `{a}` | {f(m["coverage"])} | {f(m["personas"])} | {f(m["update"])} | '
                   f'{f(m["coverage_nonliteral"])} | {f(m["damage"], 3, True)} | {ci(c["damage_ci"], 3, True)} | '
                   f'{f(m["fired_loc"], 3, True)} |')
    return out


def contrast_rows(cs: dict) -> list[str]:
    out = ['| 对比 | 参照 | 对手 | Δ 覆盖 | 95% 区间 | 为正的流 | 参照写坏 | 对手写坏 | 判定 |',
           '|---|---|---|---:|---|---:|---:|---:|---|']
    for name, c in cs.items():
        out.append(f'| {name} | `{c["reference"]}` | `{c["opponent"]}` | {f(c["delta"])} | {ci(c["ci"])} | '
                   f'{c["positive"]}/{len(c["per_stream"])} | {f(c["damage_reference"], 3, True)} | '
                   f'{f(c["damage_opponent"], 3, True)} | {c["verdict"]} |')
    return out


def per_stream_rows(res, arms) -> list[str]:
    streams = sorted(next(iter(res['per_stream'].values())))
    out = ['| 臂 | ' + ' | '.join(streams) + ' |', '|---|' + '---:|' * len(streams)]
    for a in arms:
        cells = [f'{f(res["per_stream"][a][s]["coverage"])} / {f(res["per_stream"][a][s]["damage"], 2, True)}'
                 for s in streams]
        out.append(f'| `{a}` | ' + ' | '.join(cells) + ' |')
    return out


def frontier_rows(res) -> list[str]:
    caps = list(res['frontier'])
    groups = list(next(iter(res['frontier'].values())))
    out = ['| 族 | ' + ' | '.join(f'写坏 = {100 * float(c):.0f}% 时的覆盖（Δ vs KT，95% 区间）' for c in caps) + ' |',
           '|---|' + '---|' * len(caps)]
    for g in groups:
        cells = []
        for c in caps:
            r = res['frontier'][c][g]
            s = f(r['point'])
            if 'delta_ci' in r:
                s += f'（{f(r.get("delta_vs_kt"))}，{ci(r["delta_ci"])}）'
            cells.append(s)
        out.append(f'| {FAM.get(g, g)} | ' + ' | '.join(cells) + ' |')
    return out


def cost_rows(res) -> list[str]:
    out = []
    t = res['costs'].get('timing')
    if t:
        base = t['none']['batch64']['seconds_per_question']
        out += ['| 臂 | 批 64：秒/题 | 额外开销 | 批 1：秒/题 | 提示词数 |', '|---|---:|---:|---:|---:|']
        for a, v in t.items():
            b64, b1 = v['batch64'], v['batch1']
            out.append(f'| `{a}` | {b64["seconds_per_question"]:.4f} | '
                       f'{(b64["seconds_per_question"] - base) / base:+.1%} | {b1["seconds_per_question"]:.4f} | '
                       f'{b64["prompt_tokens_mean"]:.1f} |')
        out.append('')
    rows = res['costs']['per_stream']
    keys = sorted({k for r in rows.values() for k in r if k.endswith('_seconds_per_edit')})
    out += ['| 流 | 编辑数 | ' + ' | '.join(keys) + ' |', '|---|---:|' + '---:|' * len(keys)]
    for s, r in rows.items():
        out.append(f'| {s} | {r["edits"]} | ' + ' | '.join(f(r.get(k), 4) for k in keys) + ' |')
    st = next(iter(rows.values()))['storage']
    out += ['', '每个编辑的存储：' + '；'.join(f'{k} = {v:.0f} 字节' for k, v in st.items())]
    if res.get('cost_claims'):
        out += ['', '成本主张（只在判「相当」的对比上）：`' + json.dumps(res['cost_claims'], ensure_ascii=False) + '`']
    return out


def check_rows(res) -> list[str]:
    out = ['| 流 | none 对已知事实全对 | 倒序重评相同 | 未触发却与 none 不同（最多的臂） |', '|---|:---:|:---:|---|']
    for s, c in res['checks'].items():
        worst = max(c.get('unfired_differs_from_none', {}).items(), key=lambda kv: kv[1], default=('–', 0))
        out.append(f'| {s} | {"是" if c.get("none_known_all_right") else "否"} | '
                   f'{"是" if (c.get("determinism") or {}).get("loc_reversed_same") else "否"} | '
                   f'{worst[1]}（`{worst[0]}`） |')
    return out


def main(run: Path):
    res = json.loads((run / 'analysis.json').read_text(encoding='utf-8'))
    cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    sel = res['selected']
    arms = list(cfg['arms']['test'])
    picked = ['none'] + [sel[g] for g in ('kt', 'rag', 'grace', 'hyg', 'rad')]
    ov = res['overall']
    out = [f'# EC-11 表格（`{run.name}`）', '',
           f'由 `ec_report.py` 从 `runs/{run.name}/analysis.json` 生成。全部在第 8 次写入之后评测。覆盖 = 流里全部编辑的改写问法'
           '答对的比例；写坏 = 底座在同一定形评测器下答对的已知事实里答错的比例。八条测试流等权平均。', '',
           '## 总判断', '',
           f'- P1 KT vs RAG：{ov["p1"]}', f'- P2 KT vs GRACE：{ov["p2"]}', f'- P3 KT vs HYG：{ov["p3"]}',
           f'- KT 领先强对手：{"是" if ov["kt_leads_strong_opponents"] else "否"}；胜过清洗：'
           f'{"是" if ov["kt_beats_cleaning"] else "否"}', '',
           '## 开发流调参（D0–D7）', ''] + tuning_table(cfg) + [
           '', '## 测试流 C0–C7', '', '### 各族选定的臂', ''] + arms_table(res, picked) + [
           '', '### 主对比', ''] + contrast_rows(res['primary']) + [
           '', '### 次对比', ''] + contrast_rows(res['secondary']) + [
           '', '### 前沿（各族全部格点，线性插值）', ''] + frontier_rows(res) + [
           '', '### 逐流（覆盖 / 写坏）', ''] + per_stream_rows(res, picked) + [
           '', '### 全部测试臂', ''] + arms_table(res, arms) + [
           '', '## 成本', ''] + cost_rows(res) + [
           '', '## 核对', ''] + check_rows(res) + ['']
    (run.parent.parent / 'TABLES.md').write_text('\n'.join(out), encoding='utf-8')
    print('wrote', run.parent.parent / 'TABLES.md')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
