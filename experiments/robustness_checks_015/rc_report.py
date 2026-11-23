"""TABLES.md for RC-15 / BR-14 from the analysis.json files of finished test runs (numbers only; the prose is in
REPORT.md). Units: percentage points of damage D (base-known facts answered wrongly at the end); positive = matched
breaks fewer facts.

  python -I rc_report.py OUT_MD RUN_DIR [RUN_DIR ...]
"""
import json
import sys
from pathlib import Path


def pp(x, nd=2):
    return '—' if x is None else f'{100 * x:+.{nd}f}'


def pct(x, nd=2):
    return '—' if x is None else f'{100 * x:.{nd}f}%'


def ci(s):
    c = s.get('ci') if s else None
    return '—' if not c else f'[{100 * c[0]:+.2f}, {100 * c[1]:+.2f}]'


def row_summary(name, s, gate=None):
    if not s:
        return None
    g = '' if gate is None else (' 过' if gate.get('passed') else ' 不过')
    return f'| {name} | {pp(s["mean"])} | {ci(s)} | {s["positive"]}/{s["n"]} |{g} |'


def run_tables(run: Path) -> list[str]:
    a = json.loads((run / 'analysis.json').read_text(encoding='utf-8'))
    cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    o = a.get('options') or {}
    L = [f'## {run.name}', '',
         f'流：{", ".join(a["streams"])}；写入种子 {o.get("write_seed")}，路由层 {o.get("layer")}，截断 {o.get("massive"):.1f}，'
         f'中心化 {o.get("centring")}，底线 {o.get("tau")}。', '']
    for P, r in a['cells_result'].items():
        L += [f'### 格 {P}', '', '| 对比 | 均值 | 95% 区间 | 为正的流 | 门 |', '|---|---:|---|---:|---|']
        for key, label in (('delta', 'Δ = 打乱 − 原配'), ('delta_active', 'Δ_active = 活跃置换 − 原配'),
                           ('delta_bans', 'Δ − Δ_active（挪动禁用）')):
            if key in r:
                line = row_summary(label, r[key], r.get(f'{key}_gate'))
                if line:
                    L.append(line)
        for key in ('delta', 'delta_active'):
            if f'{key}_q' in r:
                L.append(f'| {key} 的 Q 差 / R 差（原配 − 对照） | {pp(r[key + "_q"]["mean"])} / {pp(r[key + "_r"]["mean"])} | '
                         f'{ci(r[key + "_q"])} / {ci(r[key + "_r"])} | | |')
        f = r.get('factorial') or {}
        if f:
            L += ['', '| 二乘二拆解 | 均值 | 95% 区间 | 为正的流 |', '|---|---:|---|---:|']
            for key, label in (('total', '总差 D(统一) − D(原配)'), ('ban_flat', '禁用效应（统一时）'),
                               ('ban_cont', '禁用效应（逐键时）'), ('cont_ban', '连续阈值效应（保留禁用）'),
                               ('cont_noban', '连续阈值效应（不禁用）'), ('interaction', '交互')):
                if key in f:
                    L.append(f'| {label} | {pp(f[key]["mean"])} | {ci(f[key])} | {f[key]["positive"]}/{f[key]["n"]} |')
        pr = r.get('practical') or {}
        if pr:
            L += ['', '| 实用对照 | ΔU | 95% 区间 | 为正的流 | Q 差 | R 差 | 实用门 |', '|---|---:|---|---:|---:|---:|---|']
            for comp, v in pr.items():
                L.append(f'| {comp} | {pp(v["delta"]["mean"])} | {ci(v["delta"])} | {v["delta"]["positive"]}/{v["delta"]["n"]} | '
                         f'{pp(v["q"]["mean"])} | {pp(v["r"]["mean"])} | {"过" if v["gate"]["passed"] else "不过"} |')
        arms = a['arms']
        tau = f'{(o.get("tau") or 0.9):.2f}'
        L += ['', f'| 流 | 旧题原对 | 原配 D | 打乱 D | 活跃置换 D | 统一 {tau} D | 禁用+统一 {tau} D | noban D | '
                  '禁用键 / 中段 / 底线 |', '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for n in a['streams']:
            def D(arm):
                m = arms.get(f'{P}:{arm}', {}).get(n)
                return None if not m else m['damage']
            sh = [D(f'shuffled:{p}') for p in range(8) if f'{P}:shuffled:{p}' in arms]
            ac = [D(f'active_shuffled:{p}') for p in range(8) if f'{P}:active_shuffled:{p}' in arms]
            mean = lambda v: None if not v or any(x is None for x in v) else sum(v) / len(v)  # noqa: E731
            b = r['bans'][n]
            known = arms[f'{P}:matched'][n]['known']
            L.append(f'| {n} | {known} | {pct(D("matched"))} | {pct(mean(sh))} | {pct(mean(ac))} | '
                     f'{pct(D(f"uniform:{tau}"))} | {pct(D(f"ban_uniform:{tau}"))} | {pct(D("noban_matched"))} | '
                     f'{b["banned"]} / {b["mid"]} / {b["floor"]} |')
        front = r.get('frontier') or {}
        for kind in ('uniform', 'ban_uniform'):
            pts = front.get(kind) or {}
            if len(pts) > 2:
                L += ['', f'| {kind} 前沿 | D | Q | R | 原题 |', '|---|---:|---:|---:|---:|']
                for x, m in pts.items():
                    L.append(f'| {x} | {pct(m["damage"])} | {m["Q"]:.3f} | {m["R"]:.3f} | {m["update"]:.3f} |')
        mm = a['arm_means'].get(f'{P}:matched')
        if mm:
            L += ['', f'原配（四流平均）：D {pct(mm["damage"])}，Q {mm["Q"]:.3f}，R {mm["R"]:.3f}，原题 {mm["update"]:.3f}。']
        cul = r.get('culprits') or {}
        if cul:
            L += ['', '| 干预臂 | 写坏题 | 致错键都被原配禁用 | 都未禁用 | 两类都有 | 无单键 | 触发太多 |', '|---|---:|---:|---:|---:|---:|---:|']
            for arm, per in cul.items():
                tot = {}
                dmg = 0
                for v in per.values():
                    if not v:
                        continue
                    dmg += v['damaged']
                    for k, c in v['per_item'].items():
                        tot[k] = tot.get(k, 0) + c
                L.append(f'| {arm} | {dmg} | {tot.get("banned", 0)} | {tot.get("active", 0)} | {tot.get("both", 0)} | '
                         f'{tot.get("no single culprit", 0)} | {tot.get("too many fired", 0)} |')
        L.append('')
    sel = cfg.get('selected')
    if sel:
        L += ['开发流选定：' + '；'.join(f'{c} {k} {v["level"]:.2f}（{v["flag"]}）' for c, d in sel.items()
                                    for k, v in d.items()), '']
    chk = a.get('checks') or {}
    bad = {n: [k for k, v in c.items() if v is False] for n, c in chk.items()}
    L += [f'核对：未触发与 none 的最大差 {max(c["unfired_max"] for c in chk.values())}；失败项 '
          f'{ {n: v for n, v in bad.items() if v} or "无"}；每流秒数 '
          f'{ {n: round(c["seconds"]) for n, c in chk.items() if c.get("seconds")} }。', '']
    rc = run / 'recount.json'
    if rc.exists():
        r = json.loads(rc.read_text(encoding='utf-8'))
        L += [f'独立复算：{r["compared"]} 个均值，最大差 {r["max_abs_diff"]:.2e}；调参一致 {r["tuning_matches_config"]}。', '']
    return L


def main(out, *runs):
    L = ['# 表格（由 rc_report.py 从各运行的 analysis.json 生成）', '']
    for r in runs:
        p = Path(r)
        if (p / 'analysis.json').exists():
            L += run_tables(p)
    Path(out).write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(f'{out}: {len(L)} lines')


if __name__ == '__main__':
    main(*sys.argv[1:])
