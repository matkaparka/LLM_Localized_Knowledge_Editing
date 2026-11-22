"""KT-10 analysis (CPU, standard library only) of the per-stream records written by kt_run.py.

Per stream and arm, with T the last step:
  damage       share of base-known locality facts that are right with nothing firing (arm 'none', same step, same
               batches) and wrong under the arm, at T                                              [primary]
  damage_base  RV-09's definition: base-known facts (base_locality) wrong at T
  coverage     mean over steps of the rephrase accuracy on the batch just written
  upd          mean over steps of the original-wording accuracy
  retention    retention accuracy at T (earlier panels, rephrase and persona wording)
  forgetting   SO-08 edit_forgetting at T
  value        mean over steps of the SO-08 composite (rephrase, retention, locality)
  loc_fire     share of the facts counted by `damage` on which some entry fired (relevant positions)
  own_hit      share of rephrase items (all steps) on which their own entry fired; other_hit: some other entry fired

Pre-registered rules (PLAN.md section 5): H1 matched vs the mean of the shuffled arms, H2 matched vs the uniform
threshold tuned on the development streams, both on the test streams; H3 concentration of the damage under the
floor threshold, from the single-entry interventions.
"""
from __future__ import annotations

import math
import random
from collections import Counter

# ---- small helpers ----------------------------------------------------------------------------------------------


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _hits(step_rec) -> dict:
    return {int(k): v for k, v in step_rec['hits'].items()}


def _pred(step_rec) -> dict:
    return {int(k): v for k, v in step_rec['pred'].items()}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def auroc(scores, labels):
    pos = sum(1 for y in labels if y)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    r = ranks(scores)
    return (sum(ri for ri, y in zip(r, labels) if y) - pos * (pos + 1) / 2) / (pos * neg)


def lcs_len(a: str, b: str) -> int:
    best, prev = 0, [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def name_relation(question: str, subject: str | None) -> str:
    """How a firing entry's subject relates to the question it fired on: 'same name' (the subject string occurs in
    the question), 'near name' (a common substring of >= 4 characters covering >= half the subject), else 'other'."""
    q, s = (question or '').casefold(), (subject or '').casefold()
    if not s:
        return 'no subject'
    if s in q:
        return 'same name'
    n = lcs_len(q, s)
    return 'near name' if n >= 4 and n >= 0.5 * len(s) else 'other'


# ---- per stream and arm -----------------------------------------------------------------------------------------
def loc_base_set(rec) -> list[int]:
    """Locality items at T that are right with nothing firing (the denominator of `damage`)."""
    T = rec['steps'] - 1
    none = rec['arms']['none'][T]['ok']
    return [i for i, (c, _) in enumerate(rec['items'][T]) if c == 'loc' and none[i] == '1']


def arm_metrics(rec, arm: str) -> dict:
    T = rec['steps'] - 1
    items, st = rec['items'], rec['arms'][arm]
    base_set = loc_base_set(rec)
    ok_t = st[T]['ok']
    wrong = [i for i in base_set if ok_t[i] == '0']
    right_base = [i for i, (c, k) in enumerate(items[T]) if c == 'loc' and rec['base_loc'].get(k, [0])[0]]
    first = {}
    for t in range(T + 1):
        for i, (c, k) in enumerate(items[t]):
            if c == 'cur':
                first.setdefault(k, st[t]['ok'][i] == '1')
    final = {k: ok_t[i] == '1' for i, (c, k) in enumerate(items[T]) if c == 'ret' and k.endswith('|r')}
    tracked = [k for k, ok in first.items() if ok and k in final]
    hits_t = _hits(st[T])
    uid_entry = {m['uid']: j for j, m in enumerate(rec['entries']['meta'])}
    n_cur = own = other = 0
    for t in range(T + 1):
        h = _hits(st[t])
        for i, (c, k) in enumerate(items[t]):
            if c != 'cur':
                continue
            n_cur += 1
            mine = uid_entry.get(k.rsplit('|', 1)[0])
            fired = {e for e, _ in h.get(i, [])}
            own += mine in fired
            other += bool(fired - {mine})
    return dict(
        damage=len(wrong) / len(base_set) if base_set else None, n_loc=len(base_set), n_damaged=len(wrong),
        damage_base=(sum(1 for i in right_base if ok_t[i] == '0') / len(right_base)) if right_base else None,
        coverage=mean(s['acc']['cur'] for s in st), upd=mean(s['acc']['upd'] for s in st),
        retention=st[T]['acc']['ret'], value=mean(s['composite'] for s in st),
        forgetting=(sum(1 for k in tracked if not final[k]) / len(tracked)) if tracked else None,
        loc_fire=(sum(1 for i in base_set if i in hits_t) / len(base_set)) if base_set else None,
        own_hit=own / n_cur if n_cur else None, other_hit=other / n_cur if n_cur else None,
        r2w_by_step=[s.get('loc_right_to_wrong') for s in st])


def shuffled_arms(rec) -> list[str]:
    return sorted((a for a in rec['arms'] if a.startswith('shuffled:')), key=lambda a: int(a.split(':')[1]))


def item_diffs(rec, comp_arms: list[str], ref: str) -> list[float]:
    """Per locality item of loc_base_set at T: mean over comp_arms of 'wrong' minus 'wrong' under ref."""
    T = rec['steps'] - 1
    oks = [rec['arms'][a][T]['ok'] for a in comp_arms]
    ok_r = rec['arms'][ref][T]['ok']
    return [sum(o[i] == '0' for o in oks) / len(oks) - (ok_r[i] == '0') for i in loc_base_set(rec)]


def boot2(per_stream: dict, reps: int, seed: int) -> tuple[float, float]:
    """Two-level percentile bootstrap of the mean over streams of the per-stream mean: streams with replacement,
    then items within each drawn stream with replacement."""
    rng = random.Random(seed)
    names = sorted(per_stream)
    vals = []
    for _ in range(reps):
        tot = 0.0
        for _ in names:
            d = per_stream[rng.choice(names)]
            tot += sum(rng.choices(d, k=len(d))) / len(d) if d else 0.0
        vals.append(tot / len(names))
    vals.sort()
    return vals[int(0.025 * reps)], vals[min(reps - 1, int(math.ceil(0.975 * reps)) - 1)]


def contrast(recs: dict, names: list[str], comp_arms_of, ref: str, rule: dict) -> dict:
    """Damage under the comparison arm(s) minus damage under `ref` (positive: ref damages less), with the rephrase
    and retention guards (ref minus comparison)."""
    rows, per = [], {}
    for s in names:
        rec = recs[s]
        comp = comp_arms_of(rec)
        d = item_diffs(rec, comp, ref)
        if d:
            per[s] = d
        mc = [arm_metrics(rec, a) for a in comp]
        mr = arm_metrics(rec, ref)
        rows.append(dict(stream=s, delta=mean(d), n_items=len(d), damage_ref=mr['damage'],
                         damage_comp=mean(m['damage'] for m in mc),
                         coverage_ref=mr['coverage'], coverage_comp=mean(m['coverage'] for m in mc),
                         coverage_diff=mr['coverage'] - mean(m['coverage'] for m in mc),
                         retention_ref=mr['retention'], retention_comp=mean(m['retention'] for m in mc),
                         retention_diff=mr['retention'] - mean(m['retention'] for m in mc),
                         loc_fire_ref=mr['loc_fire'], loc_fire_comp=mean(m['loc_fire'] for m in mc),
                         comp_arms=comp))
    md = mean(r['delta'] for r in rows)
    cov = mean(r['coverage_diff'] for r in rows)
    ret = mean(r['retention_diff'] for r in rows)
    if md is None:
        return dict(rows=rows, mean_delta=None, ci95=[None, None], positive_streams=0, coverage_diff=cov,
                    retention_diff=ret, guards=None, verdict='undetermined: no locality items')
    lo, hi = boot2(per, int(rule['bootstrap']), int(rule['bootstrap_seed']))
    pos = sum(1 for r in rows if r['delta'] is not None and r['delta'] > 0)
    guards = (cov is None or cov >= -rule['tol_rephrase']) and (ret is None or ret >= -rule['tol_retention'])
    effect = md >= rule['delta_damage'] and lo > 0 and pos >= rule['min_positive_streams']
    if effect and guards:
        verdict = 'supported'
    elif hi < rule['delta_damage']:
        verdict = 'rejected'
    elif effect:
        verdict = 'undetermined: rephrase or retention cost beyond tolerance'
    else:
        verdict = 'undetermined'
    return dict(rows=rows, mean_delta=md, ci95=[lo, hi], positive_streams=pos, coverage_diff=cov,
                retention_diff=ret, guards=guards, verdict=verdict)


def tune_uniform(recs: dict, dev: list[str]) -> tuple[str, dict]:
    """The uniform arm with the highest mean value over the development streams (ties: the lower threshold)."""
    uni = sorted({a for s in dev for a in recs[s]['arms'] if a.startswith('uniform:')})
    vals = {a: mean(arm_metrics(recs[s], a)['value'] for s in dev) for a in uni}
    best = max(uni, key=lambda a: (vals[a], -float(a.split(':')[1])))
    return best, vals


def _region_class(hits, e) -> str:
    regs = {g for x, g in hits if x == e}
    return {frozenset({0}): 'ctx', frozenset({1}): 'q', frozenset({2}): 'gen'}.get(frozenset(regs), 'mixed')


def concentration(rec, arm: str, rule: dict, tau: float) -> dict:
    """From the interventions of one arm at T (PLAN section 5, amended 2026-11-20 before the main run).
    A culprit of a damaged fact is an entry that fired on it under the arm and, with every other entry switched off,
    breaks it by itself ('alone'). The damaged facts with some firing are the base; the top ceil(top_frac x E)
    entries by number of facts they break alone carry `top_share` of them. Also: closure (switching off everything
    that fires, runner-ups included, until nothing fires), the switch-one-off test ('single'), where each culprit
    fired (format context, question, generation), how generic it is (share of the hidden candidates on which it
    reaches the floor at a question position), and its subject's relation to the question."""
    ivs = rec['interventions'].get(arm)
    if ivs is None:
        return dict(available=False)
    meta, matched = rec['entries']['meta'], rec['entries']['matched']
    n_entries = rec['entries']['n']
    n_floor, n_hidden = rec['hidden']['entry']['n_floor'], rec['hidden']['n']
    k = max(1, math.ceil(rule['concentration_top_frac'] * n_entries))
    fired = [r for r in ivs if r['fired']]
    culprits = {r['i']: [e for e, ok, _ in r.get('alone', []) if ok == 0] for r in fired}
    counts = Counter(e for cs in culprits.values() for e in cs)
    top = [e for e, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:k]]
    share = (sum(1 for r in fired if set(culprits[r['i']]) & set(top)) / len(fired)) if fired else None
    pairs, regions = Counter(), Counter()
    for r in fired:
        for e in culprits[r['i']]:
            pairs[name_relation(r['question'], meta[e]['subject'])] += 1
            regions[_region_class(r.get('hits', []), e)] += 1
    closure = [r['closure'] for r in fired if 'closure' in r]
    raised = [j for j, t in enumerate(matched) if t > tau + 1e-9]
    generic = [n_floor[e] / n_hidden for e in counts] if n_hidden else []
    return dict(
        available=True, damaged=len(ivs), no_firing=len(ivs) - len(fired), with_firing=len(fired),
        rerun_mismatch=sum(1 for r in fired if r.get('rerun') and r['rerun'][1] != r['pred']),
        alone_explained=sum(1 for r in fired if culprits[r['i']]),
        single_recovers=sum(1 for r in fired if any(ok == 1 for _, ok, _ in r.get('single', []))),
        many_fired=sum(1 for r in fired if 'alone' not in r),
        closure_complete=sum(1 for c in closure if c['complete']),
        closure_recovered=sum(1 for c in closure if c['answer'] and c['answer'][0] == 1),
        closure_mean_size=mean(len(c['entries']) for c in closure),
        fired_mean=mean(len(r['fired']) for r in fired),
        distinct_culprits=len(counts), entries=n_entries, top_k=k, top_share=share,
        top_culprits=[dict(entry=e, items=counts[e], subject=meta[e]['subject'], step=meta[e]['step'],
                           raised=matched[e] > tau + 1e-9, threshold=matched[e],
                           hidden_floor_share=(n_floor[e] / n_hidden) if n_hidden else None) for e in top],
        culprits_raised=(sum(1 for e in counts if matched[e] > tau + 1e-9) / len(counts)) if counts else None,
        entries_raised=len(raised) / n_entries if n_entries else None,
        culprit_hidden_floor_share_median=sorted(generic)[len(generic) // 2] if generic else None,
        hijack=sum(1 for r in fired if any(meta[e]['answer'] == r['pred'] for e in culprits[r['i']])),
        name_relation=dict(pairs), culprit_regions=dict(regions),
        firing_regions={str(g): n for g, n in sorted(Counter(g for r in ivs for g in r['regions']).items())})


def concentration_verdict(per_stream: dict, rule: dict) -> str:
    ev = {s: c for s, c in per_stream.items() if c.get('available') and c['with_firing'] >= rule['concentration_min_items']}
    if len(ev) < rule['concentration_streams']:
        return 'undetermined: too little attributable damage'
    conc = sum(1 for c in ev.values() if c['top_share'] >= rule['concentration_share'])
    if conc >= rule['concentration_streams']:
        return 'concentrated'
    if len(ev) - conc >= 2:
        return 'not concentrated'
    return 'undetermined'


# ---- descriptive summaries --------------------------------------------------------------------------------------
def arm_table(recs: dict, names: list[str], arms: list[str]) -> dict:
    keys = ('damage', 'damage_base', 'coverage', 'upd', 'retention', 'forgetting', 'value', 'loc_fire', 'own_hit',
            'other_hit')
    out = {}
    for a in arms:
        ms = [arm_metrics(recs[s], a) for s in names if a in recs[s]['arms']]
        out[a] = {k: mean(m[k] for m in ms) for k in keys}
        out[a]['per_stream'] = {s: {k: arm_metrics(recs[s], a)[k] for k in ('damage', 'coverage', 'retention')}
                                for s in names if a in recs[s]['arms']}
    return out


def frontier(table: dict) -> dict:
    """Uniform arms by coverage; the uniform damage linearly interpolated at the matched arm's coverage."""
    pts = sorted((v['coverage'], v['damage'], a) for a, v in table.items() if a.startswith('uniform:'))
    m = table.get('matched')
    interp = None
    if m and len(pts) >= 2:
        c = m['coverage']
        for (c0, d0, _), (c1, d1, _) in zip(pts, pts[1:]):
            if c0 <= c <= c1 and c1 > c0:
                interp = d0 + (d1 - d0) * (c - c0) / (c1 - c0)
    return dict(points=[dict(arm=a, coverage=c, damage=d) for c, d, a in pts],
                uniform_damage_at_matched_coverage=interp,
                matched=dict(coverage=m['coverage'], damage=m['damage']) if m else None)


def literal_split(recs: dict, names: list[str], arms: list[str]) -> dict:
    """Rephrase accuracy (all steps, pooled over streams) by whether the edit's subject occurs literally in the
    rephrase wording."""
    out = {}
    for a in arms:
        tot = {True: [0, 0], False: [0, 0]}
        for s in names:
            rec = recs[s]
            if a not in rec['arms']:
                continue
            flag = {m['uid']: m['subj_in_rephrase'] for m in rec['entries']['meta']}
            for t, st in enumerate(rec['arms'][a]):
                for i, (c, k) in enumerate(rec['items'][t]):
                    if c == 'cur':
                        f = bool(flag.get(k.rsplit('|', 1)[0], False))
                        tot[f][0] += st['ok'][i] == '1'
                        tot[f][1] += 1
        out[a] = {('literal' if f else 'not_literal'): dict(acc=v[0] / v[1] if v[1] else None, n=v[1])
                  for f, v in tot.items()}
    return out


def hidden_summary(recs: dict, names: list[str], arms: list[str], tau: float) -> dict:
    per_arm = {a: dict(all=mean(recs[s]['hidden']['per_arm'][a]['all'] for s in names if a in recs[s]['hidden']['per_arm']),
                       known=mean(recs[s]['hidden']['per_arm'][a]['known'] for s in names
                                  if a in recs[s]['hidden']['per_arm']))
               for a in arms}
    per_stream = {}
    for s in names:
        rec = recs[s]
        pm, hx = rec['probe_max']['all'], rec['hidden']['entry']['hmax']
        nf, nm = rec['hidden']['entry']['n_floor'], rec['hidden']['entry']['n_matched']
        per_stream[s] = dict(spearman_probe_hidden=spearman(pm, hx),
                             auroc_probe_max_fires_on_hidden=auroc(pm, [n > 0 for n in nf]),
                             entries_firing_on_hidden_at_floor=sum(1 for n in nf if n > 0) / len(nf),
                             entries_firing_on_hidden_at_matched=sum(1 for n in nm if n > 0) / len(nm))
    return dict(per_arm=per_arm, per_stream=per_stream)


def noise_floor(recs: dict, names: list[str]) -> dict:
    out = {}
    for s in names:
        rec = recs[s]
        st = rec['arms']['none']
        out[s] = dict(r2w_by_step=[x.get('loc_right_to_wrong') for x in st],
                      w2r_by_step=[x.get('loc_wrong_to_right') for x in st],
                      base_right=sum(1 for v in rec['base_loc'].values() if v[0]))
    return out


def reproduction(recs: dict, rv09: dict) -> dict:
    """Per development stream and step: agreement of the matched arm's rephrase, original and retention items with
    RV-09's A1 `local` run (same writer, seed and items; only the locality pool and its batches differ)."""
    out = {}
    for s, rv in rv09.items():
        if s not in recs:
            continue
        rec = recs[s]
        rows = []
        for t, st in enumerate(rec['arms']['matched']):
            mine = {(c, k): st['ok'][i] == '1' for i, (c, k) in enumerate(rec['items'][t]) if c in ('cur', 'upd', 'ret')}
            theirs = {(c, k): bool(ok) for c, k, ok in rv['trajectory']['steps'][t]['eval']['items']
                      if c in ('cur', 'upd', 'ret')}
            common = [x for x in mine if x in theirs]
            rows.append(dict(step=t, items=len(common), agree=(sum(mine[x] == theirs[x] for x in common) / len(common))
                             if common else None))
        out[s] = rows
    return out


# ---- the whole analysis -----------------------------------------------------------------------------------------
def analyze(recs: dict, cfg: dict, rv09: dict | None = None) -> dict:
    rule, tau = cfg['rule'], float(cfg['local']['tau'])
    dev = [s for s in cfg['dev'] if s in recs]
    test = [s for s in cfg['test'] if s in recs]
    arms = list(cfg['arms'])
    out = dict(streams=dict(dev=dev, test=test),
               checks={s: dict(determinism=recs[s]['determinism'],
                               threshold_recompute_max_abs_diff=recs[s]['entries']['threshold_recompute_max_abs_diff'],
                               probes_same_as_set_probes=recs[s]['probes']['same_as_set_probes'],
                               screening=recs[s]['screening'], entries=recs[s]['entries']['n'])
                       for s in recs})
    if dev:
        tuned, vals = tune_uniform(recs, dev)
        out['tuning'] = dict(rule=rule['tuning'], dev_values=vals, tuned=tuned)
    else:
        tuned = None
    for role, names in (('dev', dev), ('test', test)):
        if not names:
            continue
        table = arm_table(recs, names, arms)
        block = dict(arms=table, frontier=frontier(table), literal=literal_split(recs, names, arms),
                     hidden=hidden_summary(recs, names, arms, tau), noise_floor=noise_floor(recs, names))
        if any(shuffled_arms(recs[s]) for s in names):
            block['h1_matched_vs_shuffled'] = contrast(recs, names, shuffled_arms, 'matched', rule)
        if tuned:
            block['h2_matched_vs_tuned_uniform'] = contrast(recs, names, lambda rec: [tuned], 'matched', rule)
        conc = {s: concentration(recs[s], rule['concentration_arm'], rule, tau) for s in names}
        block['h3_concentration'] = dict(arm=rule['concentration_arm'], per_stream=conc,
                                         verdict=concentration_verdict(conc, rule))
        block['interventions_other'] = {a: {s: concentration(recs[s], a, rule, tau) for s in names}
                                        for a in cfg['intervene']['arms'] if a != rule['concentration_arm']}
        out[role] = block
    if rv09:
        out['reproduction_vs_rv09'] = reproduction(recs, rv09)
    return out
