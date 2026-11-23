"""RC-15 core: threshold arms that separate banning from continuous calibration, a block-output norm statistic, the
FP-style environment with RC-15's options, and one stream run with a configurable set of cells and arms.

Arms over one table's matched vector m (all entries), its active mask A (refused entries never fire) and its banned
set B = {i in A : m_i > 1}. A centred cosine never exceeds 1, so a banned entry never fires under `matched`.
  none                 nothing fires (fp_core)
  matched              m (fp_core)
  shuffled:p           m permuted within each write batch among A (fp_core, FP-12's permutations; banned values move)
  active_shuffled:p    m permuted within each write batch among A minus B; banned entries keep their value
  noban_matched        m with every banned entry at the floor tau
  uniform:x            x on A (fp_core)
  ban_uniform:x        x on A minus B; banned entries never fire
Options (config): write_seed; the routing layer (local.layers); massive_rule, which rescales the 200 norm cut-off by
the ratio of median block-output norms on the routing-centre corpus (ref_layer: another layer of the same model;
ref_median: a number measured on another model); centring ('mu' as in RV-09, or 'raw': mu = 0 for keys, probes and
routing); screen_cap (only the first N hidden candidates are screened); panels.cov (the whole stream's rephrasings at
T, default on).

The writer, the corrected table, the evaluator, the blocking and the interventions are FP-12's (fp_core, fp_run) and
XR-13's (xr_run), imported unchanged. Records keep FP-12's layout, so fp_analysis.arm_metrics reads them.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FP = EXP / 'field_prefix_ablation_012'
XR = EXP / 'external_key_replication_013'
for _d in (XR, FP, HERE):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import fp_data  # noqa: E402  (import paths for EC-11, KT-10, RV-09, SO-08)
import fp_core  # noqa: E402
import fp_run  # noqa: E402
from fp_core import FPEditor, FPGen, Row, Table, at_step  # noqa: E402
from kt_core import OFF, build_entries, probe_max, probe_positions  # noqa: E402
from rv_editor import EditWritable, LocalCfg, StopForward  # noqa: E402
from rv_policy import cap_stream  # noqa: E402
from so_data import FactIndex, load_edits, norm, shots_text  # noqa: E402
from so_model import fmt_prompt  # noqa: E402
from so_policy import EvalCfg, derive, eval_items  # noqa: E402

BAN = 1.0
KINDS = ('none', 'matched', 'shuffled', 'active_shuffled', 'noban_matched', 'uniform', 'ban_uniform')


# ---- arms --------------------------------------------------------------------------------------------------------
def banned(matched: torch.Tensor) -> torch.Tensor:
    """Entries whose matched threshold is above 1: they never fire under `matched`."""
    return matched > BAN


def active_permutation(stream_name: str, step: int, p: int, n: int) -> list[int]:
    order = list(range(n))
    random.Random(derive('rc15-active-perm', stream_name, step, p)).shuffle(order)
    return order


def active_shuffled(matched: torch.Tensor, ends: list, active: torch.Tensor, stream_name: str, p: int) -> torch.Tensor:
    """matched permuted within each write batch among its active, non-banned entries (fixed permutation p); banned
    and refused entries keep their values."""
    keep = active & ~banned(matched)
    out, start = matched.clone(), 0
    for t, end in enumerate(ends):
        idx = [j for j in range(start, end) if bool(keep[j])]
        if len(idx) > 1:
            perm = active_permutation(stream_name, t, p, len(idx))
            src = torch.tensor([idx[k] for k in perm], device=matched.device)
            out[torch.tensor(idx, device=matched.device)] = matched[src]
        start = end
    return out


def arm_kind(arm: str) -> str:
    return arm.partition(':')[0]


def arm_vector(arm: str, matched: torch.Tensor, ends: list, active: torch.Tensor, stream_name: str,
               tau: float) -> torch.Tensor:
    """Threshold of every entry under one arm at the last step; refused entries never fire."""
    kind, _, val = arm.partition(':')
    if kind in ('none', 'matched', 'shuffled', 'uniform'):
        return fp_core.arm_vector(arm, matched, ends, active, stream_name)
    ban = banned(matched)
    if kind == 'active_shuffled':
        v = active_shuffled(matched, ends, active, stream_name, int(val))
    elif kind == 'noban_matched':
        v = torch.where(ban, torch.full_like(matched, float(tau)), matched)
    elif kind == 'ban_uniform':
        v = torch.where(ban, torch.full_like(matched, OFF), torch.full_like(matched, float(val)))
    else:
        raise ValueError(f'unknown arm {arm}')
    return torch.where(active, v, torch.full_like(v, OFF))


def split_arm(name: str) -> tuple[str, str]:
    """'corr/off:shuffled:3' -> ('corr/off', 'shuffled:3'); 'none' -> ('', 'none')."""
    if ':' not in name:
        return '', name
    cell, _, arm = name.partition(':')
    return cell, arm


def vector_checks(vec: dict, matched_of: dict, ends_of: dict, banned_of: dict, refused: torch.Tensor) -> dict:
    """vec: name -> (vector, row); matched_of / ends_of / banned_of: row -> tensor / list / mask."""
    multisets, bans_fixed, ban_off = True, True, True
    for name, (v, row) in vec.items():
        kind = arm_kind(split_arm(name)[1])
        m = matched_of[row]
        live = ~refused
        if kind in ('shuffled', 'active_shuffled'):
            ref = torch.where(live, m, torch.full_like(m, OFF))
            multisets &= fp_core.multisets_equal(v, ref, ends_of[row])
        b = banned_of[row] & live
        if kind == 'active_shuffled':
            bans_fixed &= bool(torch.equal(v[b], m[b]))
        if kind == 'ban_uniform':
            ban_off &= bool((v[b] >= OFF).all())
    return dict(shuffle_multisets_equal=bool(multisets), active_shuffle_bans_fixed=bool(bans_fixed),
                ban_uniform_bans_off=bool(ban_off),
                refused_never_active=all(bool((v[refused] >= OFF).all()) for v, _ in vec.values()))


# ---- block-output norms ------------------------------------------------------------------------------------------
@torch.no_grad()
def norm_stats(wm: EditWritable, block, texts, batch_size: int = 64) -> dict:
    """Quantiles of the block-output norm over every position after the first of `texts` (base state; call before
    any routing hook or entry exists), and the median at the first position."""
    vals, first, cur = [], [], {}

    def hook(mod, inp, out):
        h = (out[0] if isinstance(out, tuple) else out).float()
        n = h.norm(dim=-1)
        ok = cur['m'].clone()
        ok[:, 0] = False
        vals.append(n[ok].cpu())
        first.append(n[:, 0].cpu())
        raise StopForward
    handle = block.register_forward_hook(hook)
    saved = wm.snapshot()
    wm.restore(wm.empty_state())
    try:
        for s in range(0, len(texts), batch_size):
            x, mask = FPEditor._pad([wm.tok.encode(t, add_special_tokens=False) for t in texts[s:s + batch_size]],
                                    wm.pad)
            cur['m'] = mask.bool().to(wm.device)
            try:
                wm._decoder()(input_ids=x.to(wm.device), attention_mask=mask.to(wm.device), use_cache=False)
            except StopForward:
                pass
    finally:
        handle.remove()
        wm.restore(saved)
    v = torch.cat(vals).double()
    f = torch.cat(first).double()
    qs = torch.quantile(v, torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float64)).tolist()
    return dict(positions=int(v.numel()), median=float(qs[0]), q90=float(qs[1]), q99=float(qs[2]), q999=float(qs[3]),
                max=float(v.max()), first_median=float(f.median()), above_200=float((v >= 200.0).double().mean()))


# ---- spans for another tokenizer -----------------------------------------------------------------------------------
def retokenise(final: dict, tok, by_uid: dict) -> tuple[dict, dict]:
    """FP-12's final spans with `changed` recomputed for this tokenizer: an entry is rewritten in the corr row when its
    span's anchor token differs from rv_editor.lookup's token for the subject field. Returns (spans, report)."""
    out, differs = {}, 0
    for uid, sp in final.items():
        d = dict(sp)
        if uid in by_uid and not sp.get('unlocatable') and sp.get('span'):
            e = by_uid[uid]
            o = fp_data.orig_anchor(tok, e.update, e.subject)
            c = fp_data.span_anchor(tok, e.update, tuple(sp['span']))
            d['changed'] = c.token != o.token
        differs += int(bool(d.get('changed')) != bool(sp.get('changed')))
        out[uid] = d
    return out, dict(entries=len(out), changed_differs_from_fp12=differs,
                     changed=sum(bool(d.get('changed')) for d in out.values()))


# ---- environment ---------------------------------------------------------------------------------------------------
class RCEnv:
    """FP-12's environment (fp_run.FPEnv) with RC-15's options; the attributes fp_run.corr_table reads are the same."""

    def __init__(self, cfg: dict, model=None, tok=None):
        self.cfg = cfg
        edits, windows, _ = load_edits(os.environ.get('RV_DATA_DIR') or cfg['data_dir'],
                                       expect_steps=cfg.get('expect_steps', 8))
        self.index = FactIndex(edits)
        self.by_uid = {e.uid: e for e in edits}
        sc = cfg['streams']
        self.streams, self.corpus, self.filters = fp_data.kt10_half(edits, len(windows), sc['args'], sc['mu_corpus'])
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if not torch.cuda.is_available():
                raise SystemExit('no CUDA device')
            torch.manual_seed(int(cfg.get('torch_seed', 0)))
            path = os.environ.get('RV_MODEL_PATH') or cfg['model_path']
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                                                         dtype=getattr(torch, cfg.get('dtype', 'bfloat16'))).to('cuda')
        layers = range(model.config.num_hidden_layers) if cfg.get('layers', 'all') == 'all' else cfg['layers']
        sites = [s.format(i=i) for i in layers for s in cfg['sites']]
        self.wm = EditWritable(model, tok, sites, rank=int(cfg['rank']), scale=float(cfg['scale']),
                               clip=cfg.get('clip'))
        local = dict(cfg['local'])
        local['layers'] = tuple(local['layers'])
        block_fmt = local.get('block', LocalCfg.block)
        layer = local['layers'][-1]
        self.norm_info = None
        mr = cfg.get('massive_rule')
        if mr or cfg.get('norm_report'):
            here = norm_stats(self.wm, self.wm.model.get_submodule(block_fmt.format(i=layer)), self.corpus)
            info = dict(layer=layer, here=here)
            if mr:
                if mr.get('ref_layer') is not None:
                    ref = norm_stats(self.wm, self.wm.model.get_submodule(block_fmt.format(i=int(mr['ref_layer']))),
                                     self.corpus)
                    info.update(ref_layer=int(mr['ref_layer']), ref=ref)
                    ref_med = ref['median']
                elif mr.get('ref_median_file'):
                    p = Path(mr['ref_median_file'])
                    p = p if p.is_absolute() else (Path(os.environ.get('RC_CONFIG_DIR') or HERE) / p)
                    ref_rec = json.loads(p.read_text(encoding='utf-8'))
                    ref_med = float(ref_rec['stats']['median'])
                    info.update(ref_file=str(p), ref_file_sha256=fp_run.sha(p), ref=ref_rec)
                else:
                    ref_med = float(mr['ref_median'])
                local['massive'] = float(mr['base']) * here['median'] / ref_med
                info.update(ref_median=ref_med, base=float(mr['base']), massive=local['massive'])
            self.norm_info = info
        self.editor = FPEditor(LocalCfg(**local))
        self.prep = self.editor.prepare(self.wm, self.corpus)
        self.centring = cfg.get('centring', 'mu')
        if self.centring == 'raw':
            self.editor.mu = torch.zeros_like(self.editor.mu)
        elif self.centring != 'mu':
            raise ValueError(f'unknown centring {self.centring}')
        ev = cfg['eval']
        self.gen = FPGen(self.wm, self.editor, ev['widths'], ev['batch'], ev['max_new'])
        sp_path = spans_path(cfg)
        if cfg.get('spans_sha256') and fp_run.sha(sp_path) != cfg['spans_sha256']:
            raise SystemExit(f'{sp_path} does not match spans_sha256')
        final = json.loads(sp_path.read_text(encoding='utf-8'))['final']
        self.spans, self.span_report = retokenise(final, tok, self.by_uid)

    def plain(self):
        self.wm.extra = {}


def spans_path(cfg: dict) -> Path:
    p = Path(cfg['spans'])
    return p if p.is_absolute() else (HERE / p).resolve()


# ---- items ---------------------------------------------------------------------------------------------------------
def stream_items(env, s, known) -> list[tuple]:
    """fp_run.stream_items; the whole-stream rephrasings at T ('cov') only when panels.cov is not false."""
    panels = dict(env.cfg['panels'])
    cov = panels.pop('cov', True)
    ev = EvalCfg(**panels)
    T = s.steps - 1
    items = []
    for t in range(s.steps):
        for comp, q, valid, key in eval_items(s, env.index, t, 0, ev):
            if comp in ('cur', 'upd') or (t == T and comp == 'ret'):
                items.append((comp, q, frozenset(valid), key, t))
    for q, a in known:
        items.append(('loc', q, frozenset({norm(a)}), q, T))
    if cov:
        for e in s.edits:
            if e.rephrase:
                items.append(('cov', e.rephrase, frozenset(env.index.valid(e.key, T)), f'{e.uid}|c', T))
    return items


def screen(env, s, ctx: str, nctx: int, order_check: bool, checks: dict):
    """Base-known hidden facts in the evaluation's own fixed-shape layout (optionally the first screen_cap only)."""
    tok, gen = env.wm.tok, env.gen
    env.plain()
    cand = list(s.loc_hidden)
    if env.cfg.get('screen_cap'):
        cand = cand[:int(env.cfg['screen_cap'])]
    cids = [tok.encode(fmt_prompt(q, ctx), add_special_tokens=False) for q, _ in cand]
    couts = gen.generate(cids, [nctx] * len(cids))
    known = [(q, a) for (q, a), o in zip(cand, couts) if norm(o.text) == norm(a)]
    if order_check:
        rev = gen.generate(cids[::-1], [nctx] * len(cids))[::-1]
        checks['screen_reversed_identical'] = all(a.text == b.text for a, b in zip(couts, rev))
        checks['screen_reversed_mismatches'] = sum(a.text != b.text for a, b in zip(couts, rev))
    return cand, known


def arm_records(items, res: dict, uid_index: dict) -> tuple[dict, dict]:
    """fp_run.run_stream's per-arm records (ok, firing regions, own-entry flags, wrong locality answers and hits) and
    the unfired-differs-from-none counts."""
    none_out = res['none']
    arms, unfired = {}, {}
    for a, outs in res.items():
        ok, reg, own, wrong, hits = [], [], [], {}, {}
        diff = 0
        for i, (it, o) in enumerate(zip(items, outs)):
            good = norm(o.text) in it[2]
            ok.append('1' if good else '0')
            m = 0
            for e, g in o.fired:
                m |= 1 << g
            reg.append(str(m))
            if it[0] == 'loc':
                own.append('-')
                if not good:
                    wrong[i] = norm(o.text)
                if o.fired:
                    hits[i] = [list(x) for x in o.fired]
            else:
                oe = fp_run.own_entry(it[3], uid_index)
                es = {e for e, _ in o.fired}
                own.append(str((1 if oe in es else 0) + (2 if es - {oe} else 0)))
            if not o.fired and o.text != none_out[i].text:
                diff += 1
        unfired[a] = diff
        arms[a] = dict(ok=''.join(ok), reg=''.join(reg), own=''.join(own), wrong_loc=wrong, loc_hits=hits)
    return arms, unfired


def evaluate(env, vec: dict, tables: dict, items, ids, nctx: int, blocks: dict, steps: int, clock: dict) -> dict:
    """Every arm on its items, one generation pass per table. vec: name -> (vector, row, mask); blocks: mask -> list of
    blocked positions per item."""
    wm, gen = env.wm, env.gen
    res = {}
    for tname, tab in tables.items():
        mine = [a for a in vec if vec[a][1] == tname]
        if not mine:
            continue
        t0 = time.time()
        mats, where, rows, back = [], {}, [], []
        for a in mine:
            v, _, mask = vec[a]
            for t in range(steps):
                where[(a, t)] = len(mats)
                mats.append(at_step(v, tab.ends, t))
            seen = {}
            for i, (comp, q, valid, key, t) in enumerate(items):
                k = (q, t)
                if k not in seen:
                    seen[k] = len(rows)
                    rows.append(Row(ids[i], nctx, where[(a, t)], blocks[mask][i]))
                back.append((a, i, seen[k]))
        tab.install(wm)
        outs = gen.run(rows, torch.stack(mats))
        for a, i, r in back:
            res.setdefault(a, [None] * len(items))[i] = outs[r]
        env.plain()
        clock[f'eval_{tname}'] = time.time() - t0
        clock[f'rows_{tname}'] = len(rows)
    return res


# ---- one WikiBigEdit stream (FP-12 style) ----------------------------------------------------------------------------
def run_stream_fp(env: RCEnv, name: str, phase: str) -> dict:
    cfg, wm, ed, gen = env.cfg, env.wm, env.editor, env.gen
    ph = cfg['phases'][phase]
    tok, lc = wm.tok, ed.cfg
    clock, checks = {}, {}
    cap = ph.get('cap', fp_data.KT10_CAP)
    s = cap_stream(env.streams[name], cap)
    ed.anchors.clear()
    T = s.steps - 1
    ctx = shots_text(s.shots)
    nctx = len(tok.encode(ctx, add_special_tokens=False))
    seed = int(cfg['write_seed'])
    cells = list(ph['cells'])
    arms_of = ph['arms']

    t0 = time.time()                                   # 1. base-known hidden facts
    cand, known = screen(env, s, ctx, nctx, bool(ph.get('order_check')), checks)
    clock['screen'] = time.time() - t0

    t1 = time.time()                                   # 2. probes and the orig table (written once)
    probes = list(s.loc_probe)
    states, labels = probe_positions(wm, ed, probes, ctx)
    ed.set_probes(wm, probes, ctx)
    checks['probes_same_as_set_probes'] = bool(torch.equal(states, ed.probe_states))
    ed.probe_states = states
    nq = len(probes)
    clock['probes'] = time.time() - t1
    tw = time.time()
    ent = build_entries(wm, ed, s, seed)
    clock['write'] = time.time() - tw
    env.plain()
    uids = [m['uid'] for m in ent.meta]
    orig = Table('orig', ent.keys, ent.offsets, ent.matched, list(ent.ends), list(ent.step), uids)
    pm = probe_max(states, labels, ent.keys, nq + 1)
    recomputed = torch.clamp(pm.max(0).values + lc.probe_margin, min=lc.tau)
    checks['matched_recompute_max_abs_diff'] = float((recomputed - ent.matched).abs().max())
    clock['orig_table'] = time.time() - t1
    tables, corr_records = {'orig': orig}, []
    if any(c.split('/')[0] == 'corr' for c in cells):
        t2 = time.time()                               # 3. the corr table (FP-12's code)
        corr, corr_records = fp_run.corr_table(env, s, orig, seed)
        unchanged = [j for j in range(orig.n) if j not in set(corr.changed)]
        if unchanged:
            u = torch.tensor(unchanged, device=orig.keys.device)
            checks['corr_unchanged_bitwise'] = bool(torch.equal(orig.keys[u], corr.keys[u])
                                                    and torch.equal(orig.offsets[u], corr.offsets[u])
                                                    and torch.equal(orig.matched[u], corr.matched[u]))
        tables['corr'] = corr
        clock['corr_table'] = time.time() - t2
    dev = orig.keys.device
    refused = torch.tensor([bool(env.spans[u]['unlocatable']) for u in uids], device=dev)
    active = ~refused

    vec = {'none': (torch.full_like(orig.matched, OFF), 'orig', 'on')}   # 4. arm vectors at T
    for cell in cells:
        row, mask = cell.split('/')
        tab = tables[row]
        for a in dict.fromkeys(arms_of[cell]):
            vec[f'{cell}:{a}'] = (arm_vector(a, tab.matched, tab.ends, active, s.name, lc.tau), row, mask)
    checks.update(vector_checks({k: (v[0], v[1]) for k, v in vec.items() if k != 'none'},
                                {r: t.matched for r, t in tables.items()}, {r: t.ends for r, t in tables.items()},
                                {r: banned(t.matched) for r, t in tables.items()}, refused))
    checks['refused_never_active'] = checks['refused_never_active'] and bool((vec['none'][0][refused] >= OFF).all())

    items = stream_items(env, s, known)                # 5. items and rows
    ids = [tok.encode(fmt_prompt(q, ctx), add_special_tokens=False) for _, q, _, _, _ in items]
    markers = [fp_core.marker_tokens(tok, fmt_prompt(q, ctx), len(ctx)) for _, q, _, _, _ in items]
    for i in range(min(50, len(ids))):
        if ids[i][:nctx] != tok.encode(ctx, add_special_tokens=False):
            raise RuntimeError('the context does not tokenise as a prefix of the prompt')
    masks = sorted({v[2] for v in vec.values()})
    blocks = {m: [fp_core.blocked_positions(m, nctx, mk) for mk in markers] for m in masks}
    res = evaluate(env, vec, tables, items, ids, nctx, blocks, s.steps, clock)

    uid_index = {u: j for j, u in enumerate(uids)}     # 6. per-arm records and checks
    arms, unfired = arm_records(items, res, uid_index)
    loc_idx = [i for i, it in enumerate(items) if it[0] == 'loc']
    checks['unfired_differs_from_none'] = unfired
    checks['none_known_all_right'] = all(arms['none']['ok'][i] == '1' for i in loc_idx)
    checks['blocked_cells_never_fire_in_context'] = all(
        all(not (int(ch) & 1) for ch in arms[a]['reg']) for a in arms if '/off:' in a or '/strict:' in a)

    darm = cfg['determinism_arm']                      # 7. order check: one arm's locality rows reversed
    v, tname, mask = vec[darm]
    tab = tables[tname]
    tab.install(wm)
    rows = [Row(ids[i], nctx, 0, blocks[mask][i]) for i in loc_idx][::-1]
    again = gen.run(rows, at_step(v, tab.ends, T).unsqueeze(0))[::-1]
    checks['determinism'] = dict(arm=darm, loc_reversed_same=all(o.text == res[darm][i].text
                                                                 for i, o in zip(loc_idx, again)))
    env.plain()

    t4 = time.time()                                   # 8. per-key interventions at T
    ivs = {}
    for cell in cells:
        row, mask = cell.split('/')
        tab = tables[row]
        tab.install(wm)
        loc_rows = {i: Row(ids[i], nctx, 0, blocks[mask][i]) for i in loc_idx}
        rows_all = [loc_rows.get(i, Row(ids[i], nctx, 0, ())) for i in range(len(items))]
        for a in ph.get('interventions', []):
            arm = f'{cell}:{a}'
            if arm not in vec:
                continue
            damaged = [i for i in loc_idx if arms['none']['ok'][i] == '1' and arms[arm]['ok'][i] == '0']
            fired = {i: [tuple(x) for x in arms[arm]['loc_hits'].get(i, [])] for i in damaged}
            ivs[arm] = fp_core.intervene(gen, at_step(vec[arm][0], tab.ends, T), rows_all, damaged, fired,
                                         int(cfg['interventions']['max_single']))
            if ph.get('intervention_check') and damaged and 'interventions_batched' not in checks:
                solo = [fp_core.intervene(gen, at_step(vec[arm][0], tab.ends, T), rows_all, [i], fired,
                                          int(cfg['interventions']['max_single']))[0] for i in damaged[:20]]
                checks['interventions_batched'] = dict(arm=arm, items=len(solo),
                                                       equal_single=solo == ivs[arm][:len(solo)])
    env.plain()
    clock['interventions'] = time.time() - t4

    spans = [env.spans[u] for u in uids]
    thr = {r: fp_run._floats(t.matched) for r, t in tables.items()}
    ban_lists = {}
    for r, t in tables.items():
        b = banned(t.matched).tolist()
        ban_lists[r] = [j for j in range(orig.n) if b[j] and not spans[j]['unlocatable']]
    return dict(
        stream=name, phase=phase, kind='fp', steps=s.steps, cap=cap, cells=cells,
        options=dict(write_seed=seed, layer=int(lc.layers[-1]), massive=float(lc.massive), tau=float(lc.tau),
                     centring=env.centring, screen_cap=cfg.get('screen_cap'), cov=cfg['panels'].get('cov', True)),
        norm_info=env.norm_info, span_report=env.span_report,
        screening=dict(candidates=len(cand), known=len(known)),
        probes=dict(questions=nq, positions=int(states.shape[0])),
        tables=dict({r: t.digest() for r, t in tables.items()}, n=orig.n, ends=orig.ends, step=orig.step, uids=uids,
                    changed=tables['corr'].changed if 'corr' in tables else [], corr_records=corr_records,
                    refused=[j for j in range(orig.n) if spans[j]['unlocatable']],
                    entity=[(sp.get('text') or env.by_uid[u].subject) for sp, u in zip(spans, uids)],
                    subject=[env.by_uid[u].subject for u in uids]),
        thresholds=thr, bans=ban_lists,
        items=[[c, k, t] for c, _, _, k, t in items], loc_questions={i: items[i][1] for i in loc_idx},
        loc_answers={i: next(iter(items[i][2])) for i in loc_idx},
        arms=arms, interventions=ivs, checks=checks, clock=clock, costs=dict(wm.costs))


# ---- one XR-13 stream --------------------------------------------------------------------------------------------------
def run_stream_xr(env, name: str, phase: str) -> dict:
    """xr_run.run_stream with RC-15's arms: the table at the extractor's anchors, thresholds calibrated and executed
    with the format shots and the fixed 'Q: ' / '\\nA:' tokens blocked (FP-12's strict blocking)."""
    import xr_run
    cfg, wm, ed, gen = env.cfg, env.wm, env.editor, env.gen
    ph = cfg['phases'][phase]
    tok, lc = wm.tok, ed.cfg
    clock, checks = {}, {}
    s = cap_stream(env.streams[name], ph.get('cap'))
    T = s.steps - 1
    ctx = shots_text(s.shots)
    nctx = len(tok.encode(ctx, add_special_tokens=False))
    seed = int(cfg['write_seed'])
    spans = env.extra[name]['spans']

    t0 = time.time()
    cand, known = screen(env, s, ctx, nctx, bool(ph.get('order_check')), checks)
    clock['screen'] = time.time() - t0

    t1 = time.time()
    probes = list(s.loc_probe)
    states, labels = probe_positions(wm, ed, probes, ctx)
    ed.set_probes(wm, probes, ctx)
    ed.probe_states = states
    ed.anchors.clear()
    for e in s.edits:
        sp = spans.get(e.uid)
        if sp:
            ed.anchors[(e.update, e.subject)] = tuple(sp)
    ent = build_entries(wm, ed, s, seed)
    ed.anchors.clear()
    env.plain()
    uids = [m['uid'] for m in ent.meta]
    strict_states = fp_core.strict_probe_states(wm, ed, probes, ctx)
    matched = torch.clamp(fp_core.max_sim(strict_states, ent.keys) + lc.probe_margin, min=lc.tau)
    tab = Table('xr', ent.keys, ent.offsets, matched, list(ent.ends), list(ent.step), uids)
    refused = torch.tensor([spans.get(u) is None for u in uids], device=ent.keys.device)
    active = ~refused
    clock['table'] = time.time() - t1

    vec = {'none': (torch.full_like(matched, OFF), 'xr', 'strict')}
    for a in dict.fromkeys(ph['arms']['xr']):
        vec[f'xr:{a}'] = (arm_vector(a, matched, tab.ends, active, s.name, lc.tau), 'xr', 'strict')
    checks.update(vector_checks({k: (v[0], v[1]) for k, v in vec.items() if k != 'none'}, {'xr': matched},
                                {'xr': tab.ends}, {'xr': banned(matched)}, refused))

    items = fp_run.stream_items(env, s, known)
    ids = [tok.encode(fmt_prompt(q, ctx), add_special_tokens=False) for _, q, _, _, _ in items]
    markers = [fp_core.marker_tokens(tok, fmt_prompt(q, ctx), len(ctx)) for _, q, _, _, _ in items]
    blocks = {'strict': [fp_core.blocked_positions('strict', nctx, m) for m in markers]}
    res = evaluate(env, vec, {'xr': tab}, items, ids, nctx, blocks, s.steps, clock)

    uid_index = {u: j for j, u in enumerate(uids)}
    arms, unfired = arm_records(items, res, uid_index)
    loc_idx = [i for i, it in enumerate(items) if it[0] == 'loc']
    checks['unfired_differs_from_none'] = unfired
    checks['none_known_all_right'] = all(arms['none']['ok'][i] == '1' for i in loc_idx)
    checks['blocked_never_fire_in_context'] = all(all(not (int(ch) & 1) for ch in arms[a]['reg']) for a in arms)
    darm = 'xr:matched'
    tab.install(wm)
    rrows = [Row(ids[i], nctx, 0, blocks['strict'][i]) for i in loc_idx][::-1]
    again = gen.run(rrows, at_step(vec[darm][0], tab.ends, T).unsqueeze(0))[::-1]
    checks['determinism'] = dict(arm=darm, loc_reversed_same=all(o.text == res[darm][i].text
                                                                 for i, o in zip(loc_idx, again)))

    t3 = time.time()
    ivs = {}
    rows_all = [Row(ids[i], nctx, 0, blocks['strict'][i]) for i in range(len(items))]
    for a in ph.get('interventions', []):
        arm = f'xr:{a}'
        if arm not in vec:
            continue
        damaged = [i for i in loc_idx if arms['none']['ok'][i] == '1' and arms[arm]['ok'][i] == '0']
        fired = {i: [tuple(x) for x in arms[arm]['loc_hits'].get(i, [])] for i in damaged}
        ivs[arm] = fp_core.intervene(gen, at_step(vec[arm][0], tab.ends, T), rows_all, damaged, fired,
                                     int(cfg['interventions']['max_single']))
    env.plain()
    clock['interventions'] = time.time() - t3

    return dict(stream=name, phase=phase, kind='xr', role=env.extra[name]['role'], steps=s.steps, cells=['xr'],
                options=dict(write_seed=seed, layer=int(lc.layers[-1]), massive=float(lc.massive), tau=float(lc.tau),
                             centring='mu'),
                screening=dict(candidates=len(cand), known=len(known)),
                probes=dict(questions=len(probes), positions=int(states.shape[0]),
                            strict_positions=int(strict_states.shape[0])),
                tables=dict(xr=tab.digest(), n=tab.n, ends=tab.ends, step=tab.step, uids=uids, changed=[],
                            refused=[j for j in range(tab.n) if bool(refused[j])],
                            entity=[env.extra[name]['entity'][u] for u in uids]),
                thresholds=dict(xr=fp_run._floats(matched), writer_own=fp_run._floats(ent.matched)),
                bans=dict(xr=[j for j, (b, r) in enumerate(zip(banned(matched).tolist(), refused.tolist()))
                              if b and not r]),
                items=[[c, k, t] for c, _, _, k, t in items], loc_questions={i: items[i][1] for i in loc_idx},
                loc_answers={i: next(iter(items[i][2])) for i in loc_idx},
                arms=arms, interventions=ivs, checks=checks, clock=clock, costs=dict(wm.costs),
                xr_code=fp_run.sha(Path(xr_run.__file__)))
