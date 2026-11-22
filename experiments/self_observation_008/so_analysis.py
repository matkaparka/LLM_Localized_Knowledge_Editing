"""SO-08 offline analysis (CPU only, no model): P0 verdicts, matched-readout selection, state dependence and the
readout swap test.

Selection: for each decision state the candidates' remaining-episode values are centred within the state; a
ridge regression on per-candidate features predicts the centred value; it is cross-fitted by leaving one stream
out; the selector takes the argmax. Feature sets differ only in which readout is added to the public features.
Swap test: for two states of the same stream and step that differ only in the write seed (same public history),
the self-readout features of one state are replaced by those of the other; candidates stay on their real states.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from so_data import norm
from so_policy import KINDS

READOUTS = ('current', 'lagged', 'teacher', 'cache')


def _agree(xs, ys) -> float:
    n = min(len(xs), len(ys))
    return sum(1 for a, b in zip(xs[:n], ys[:n]) if a == b) / n if n else 0.0


def public_features(br: dict) -> list[float]:
    kind, _, v = br['action'].partition(':')
    ex = br['exec']
    da = 0.0
    if ex.get('accept_after') is not None and ex.get('accept_before') is not None:
        da = ex['accept_after'] - ex['accept_before']
    return [1.0 if kind == k else 0.0 for k in KINDS] + [
        float(v or 0.0), da, float(ex.get('train_loss_last') or 0.0), float(bool(ex.get('rolled_back')))]


def readout_features(state: dict, br: dict, kind: str) -> list[float]:
    """Agreement of the trial-written model's answers with one readout, plus the pre-write agreement."""
    p = len(state['probe_q'])
    post_probe, post_cache = br['post'][:p], br['post'][p:]
    cur_probe, cur_cache = state['current'][:p], state['current'][p:]
    if kind == 'current':
        return [_agree(post_probe, cur_probe)]
    if kind in ('lagged', 'teacher'):
        ref = state[kind]
        return [_agree(post_probe, ref), _agree(cur_probe, ref)]
    if kind == 'cache':
        labels = [norm(a) for _, a in state['cache']]
        return [_agree(post_cache, labels), _agree(cur_cache, labels)]
    raise ValueError(kind)


def features(state: dict, br: dict, fset: str, readout_state: dict | None = None,
             readout_br: dict | None = None) -> list[float]:
    x = public_features(br)
    if fset != 'public':
        kind = fset.split('+', 1)[1]
        x += readout_features(readout_state or state, readout_br or br, kind)
    return x


def _ridge_fit(x, y, lam):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    w = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ y)
    return mu, sd, w


def _ridge_predict(model, x):
    mu, sd, w = model
    return ((np.asarray(x, float) - mu) / sd) @ w


def _states(records):
    """Flatten matched records into decision states with stream, seed and step."""
    out = []
    for rec in records:
        for st in rec['states']:
            vals = [b['value'] for b in st['branches']]
            if any(v is None for v in vals):
                continue
            out.append(dict(st, stream=rec['stream'], seed=rec['seed']))
    return out


def _rows(states, fset):
    xs, ys, owner = [], [], []
    for i, st in enumerate(states):
        vals = np.array([b['value'] for b in st['branches']], float)
        for b, v in zip(st['branches'], vals - vals.mean()):
            xs.append(features(st, b, fset))
            ys.append(v)
            owner.append(i)
    return xs, ys, owner


def select(records, fsets=('public',) + tuple(f'public+{r}' for r in READOUTS), lam: float = 1.0,
           default_label: str = 'plain') -> dict:
    states = _states(records)
    streams = sorted({s['stream'] for s in states})
    if len(streams) < 2:
        raise ValueError('cross-fitting needs at least two streams')
    result, models = {}, {}
    for fset in fsets:
        xs, ys, owner = _rows(states, fset)
        pred = np.zeros(len(ys))
        for s in streams:
            train = [j for j, o in enumerate(owner) if states[o]['stream'] != s]
            test = [j for j, o in enumerate(owner) if states[o]['stream'] == s]
            model = _ridge_fit([xs[j] for j in train], [ys[j] for j in train], lam)
            models[(fset, s)] = model
            pred[test] = _ridge_predict(model, [xs[j] for j in test])
        regrets, hits, picks = [], 0, []
        for i, st in enumerate(states):
            idx = [j for j, o in enumerate(owner) if o == i]
            vals = [st['branches'][k]['value'] for k in range(len(idx))]
            choice = int(np.argmax(pred[idx]))
            best = int(np.argmax(vals))
            regrets.append(vals[best] - vals[choice])
            hits += choice == best
            picks.append(st['branches'][choice]['action'])
        result[fset] = dict(mean_regret=float(np.mean(regrets)), oracle_hit_rate=hits / len(states),
                            regrets=[float(r) for r in regrets], picks=picks)
    base = []
    for st in states:
        vals = [b['value'] for b in st['branches']]
        labels = [b['action'] for b in st['branches']]
        d = labels.index(default_label) if default_label in labels else None
        base.append(max(vals) - vals[d] if d is not None else None)
    result['default_policy'] = dict(mean_regret=float(np.mean([b for b in base if b is not None])) if any(
        b is not None for b in base) else None)
    result['n_states'] = len(states)
    result['streams'] = streams
    return dict(result=result, models=models, states=states)


def state_dependence(records) -> dict:
    """Pairs of states with the same stream and step but different seeds: does the best candidate change?"""
    states = _states(records)
    groups = defaultdict(list)
    for st in states:
        groups[(st['stream'], st['step'])].append(st)
    pairs = []
    for (_, _), sts in groups.items():
        for a, b in combinations(sts, 2):
            va = [x['value'] for x in a['branches']]
            vb = [x['value'] for x in b['branches']]
            ba, bb = int(np.argmax(va)), int(np.argmax(vb))
            pairs.append(dict(stream=a['stream'], step=a['step'], seeds=(a['seed'], b['seed']), differ=ba != bb,
                              loss_a_using_b_best=va[ba] - va[bb], loss_b_using_a_best=vb[bb] - vb[ba]))
    return dict(n_pairs=len(pairs), differ_rate=(sum(p['differ'] for p in pairs) / len(pairs)) if pairs else None,
                pairs=pairs)


def swap_test(records, kind: str = 'current', lam: float = 1.0) -> dict:
    """Matched versus swapped self-readout for paired states (same stream and step, different seed)."""
    fset = f'public+{kind}'
    sel = select(records, fsets=(fset,), lam=lam)
    states = sel['states']
    groups = defaultdict(list)
    for i, st in enumerate(states):
        groups[(st['stream'], st['step'])].append(i)
    rows = []
    for (stream, _), idx in groups.items():
        model = sel['models'][(fset, stream)]
        for i, j in ((a, b) for a in idx for b in idx if a != b):
            si, sj = states[i], states[j]
            vals = [b['value'] for b in si['branches']]
            labels_j = {b['action']: b for b in sj['branches']}
            matched = [features(si, b, fset) for b in si['branches']]
            swapped = [features(si, b, fset, readout_state=sj, readout_br=labels_j[b['action']])
                       for b in si['branches']]
            cm = int(np.argmax(_ridge_predict(model, matched)))
            cs = int(np.argmax(_ridge_predict(model, swapped)))
            rows.append(dict(stream=stream, step=si['step'], seed=si['seed'], other_seed=sj['seed'],
                             regret_matched=max(vals) - vals[cm], regret_swapped=max(vals) - vals[cs]))
    if not rows:
        return dict(kind=kind, n=0)
    rm = np.array([r['regret_matched'] for r in rows])
    rs = np.array([r['regret_swapped'] for r in rows])
    return dict(kind=kind, n=len(rows), mean_regret_matched=float(rm.mean()), mean_regret_swapped=float(rs.mean()),
                swapped_worse=int((rs > rm).sum()), swapped_better=int((rs < rm).sum()), rows=rows)


def p0_gate(verdicts, gate: dict) -> dict:
    """Stop/go for the matched-readout experiment, fixed before P0 is run.

    learnable: mean original-wording accuracy right after each write >= learn_min_upd on every pilot stream;
    conflict: the conflict test passes on every pilot stream; repairable: on at least repair_streams_min streams.
    """
    rows = []
    for v in verdicts:
        upd = [x for x in v['capability']['upd_after_write'] if x is not None]
        learn = sum(upd) / len(upd) if upd else None
        rows.append(dict(stream=v['stream'], learn_mean_upd=learn,
                         learnable=learn is not None and learn >= gate['learn_min_upd'],
                         conflict=v['conflict']['pass'], repairable=v['repairable']))
    passed = (bool(rows) and all(r['learnable'] for r in rows) and all(r['conflict'] for r in rows)
              and sum(r['repairable'] for r in rows) >= gate['repair_streams_min'])
    return dict(passed=passed, rows=rows, rule=gate)


def _items(ev: dict, comp: str, suffix: str | None = None) -> dict:
    return {k: ok for c, k, ok in ev['items'] if c == comp and (suffix is None or k.endswith(suffix))}


def edit_forgetting(evals) -> dict:
    """Edits answered correctly right after their write (rephrase wording) that are wrong at the last evaluation."""
    first = {}
    for ev in evals:
        for c, k, ok in ev['items']:
            if c == 'cur':
                first.setdefault(k, ok)
    final = _items(evals[-1], 'ret', '|r')
    tracked = [k for k, ok in first.items() if ok and k in final]
    r2w = sum(1 for k in tracked if not final[k])
    return dict(learned_tracked=len(tracked), right_to_wrong=r2w, fraction=r2w / len(tracked) if tracked else None)


def locality_damage(ev: dict, base_loc: dict) -> dict:
    """Locality facts the base model answered correctly that are wrong now."""
    final = _items(ev, 'loc')
    right = [q for q in final if base_loc.get(q, (False,))[0]]
    r2w = sum(1 for q in right if not final[q])
    return dict(base_right=len(right), right_to_wrong=r2w, fraction=r2w / len(right) if right else None)


def _damage(evals, base_loc) -> tuple[float, dict, dict]:
    ef, ld = edit_forgetting(evals), locality_damage(evals[-1], base_loc)
    return (ef['fraction'] or 0.0) + (ld['fraction'] or 0.0), ef, ld


REPAIR_RULES = ('total', 'from_decision')
CUR_MODES = ('decision', 'mean')


def _cur(b: dict, cur_mode: str):
    """Just-written rephrase accuracy of a branch: at the decision step, or averaged over the decision step and every
    later step (for persistent candidates, which also act at the later writes)."""
    if cur_mode == 'decision':
        return b['comps'][0]['acc']['cur']
    vals = [c['acc']['cur'] for c in b['comps'] if c['acc']['cur'] is not None]
    return sum(vals) / len(vals) if vals else None


def repair_from_decision(rec: dict, th: dict, default_label: str = 'plain', cur_mode: str = 'decision') -> dict:
    """The repair test counting only damage a candidate can affect. Added 2026-11-19 after run_001, where it was a
    post hoc diagnostic; frozen as the round-2 rule (config p0.repair_rule = 'from_decision') before round 2 ran.

    At decision step t the candidates share everything before t, so damage counted from step 0 mostly cannot be
    changed. Here: locality = facts correct after the write of step t-1 (every base-known fact when t = 0) that are
    wrong at the end; forgetting = edits correct after step t-1 (rephrase wording) or first answered correctly
    after a write from step t on, that are wrong at the end. damage = locality share + forgetting share. Same
    thresholds as the total rule; cur per cur_mode (see _cur).
    """
    evals = [s['eval'] for s in rec['trajectory']['steps']]
    base_loc = rec['trajectory']['base_loc']
    out = {}
    for t, branches in rec['repairs'].items():
        t = int(t)
        if t > 0:
            before = evals[t - 1]['items']
            loc_ok = {k for c, k, ok in before if c == 'loc' and ok}
            kept = {k for c, k, ok in before if c in ('cur', 'ret') and k.endswith('|r') and ok}
        else:
            loc_ok, kept = {q for q, v in base_loc.items() if v[0]}, set()
        rows = []
        for b in branches:
            final = b['comps'][-1]
            later = set()
            for ev in b['comps']:
                for c, k, ok in ev['items']:
                    if c == 'cur' and ok:
                        later.add(k)
            loc_final = _items(final, 'loc')
            ret_final = _items(final, 'ret', '|r')
            loc_set = [k for k in loc_ok if k in loc_final]
            edit_set = [k for k in kept | later if k in ret_final]
            loc_frac = sum(1 for k in loc_set if not loc_final[k]) / len(loc_set) if loc_set else 0.0
            edit_frac = sum(1 for k in edit_set if not ret_final[k]) / len(edit_set) if edit_set else 0.0
            rows.append(dict(action=b['action'], cur=_cur(b, cur_mode), cur_decision=b['comps'][0]['acc']['cur'],
                             locality=loc_frac, forgetting=edit_frac, damage=loc_frac + edit_frac, n_loc=len(loc_set),
                             n_edits=len(edit_set), value=b['value']))
        by = {r['action']: r for r in rows}
        plain = by.get(default_label)
        passing = []
        for r in rows:
            r['reduction'] = 1 - r['damage'] / plain['damage'] if plain and plain['damage'] else 0.0
            if plain and r['action'] != default_label and r['cur'] is not None and plain['cur'] is not None \
                    and r['cur'] >= plain['cur'] - th['repair_max_cur_drop'] \
                    and r['reduction'] >= th['repair_min_r2w_reduction']:
                passing.append(dict(action=r['action'], reduction=r['reduction']))
        out[t] = dict(plain_cur=plain['cur'] if plain else None, plain_damage=plain['damage'] if plain else None,
                      candidates=rows, passing=passing)
    return out


def repair_total(rec: dict, th: dict, default_label: str = 'plain', cur_mode: str = 'decision') -> dict:
    """The round-1 repair test: damage (forgetting share + locality share) counted from step 0 to the end."""
    evals = [s['eval'] for s in rec['trajectory']['steps']]
    base_loc = rec['trajectory']['base_loc']
    repairs = {}
    for t, branches in rec['repairs'].items():
        t = int(t)
        by = {b['action']: b for b in branches}
        if default_label not in by:
            continue
        prefix = evals[:t]
        plain = by[default_label]
        p_cur = _cur(plain, cur_mode)
        p_dmg, _, _ = _damage(prefix + plain['comps'], base_loc)
        rows, ok = [], []
        for lab, b in by.items():
            dmg, ef, ld = _damage(prefix + b['comps'], base_loc)
            cur = _cur(b, cur_mode)
            red = 1 - dmg / p_dmg if p_dmg else 0.0
            rows.append(dict(action=lab, cur=cur, damage=dmg, forgetting=ef['fraction'], locality=ld['fraction'],
                             reduction=red, value=b['value']))
            if lab != default_label and cur is not None and p_cur is not None \
                    and cur >= p_cur - th['repair_max_cur_drop'] and red >= th['repair_min_r2w_reduction']:
                ok.append(dict(action=lab, reduction=red))
        repairs[t] = dict(plain_cur=p_cur, plain_damage=p_dmg, candidates=rows, passing=ok)
    return repairs


def p0_verdict(rec: dict, th: dict, default_label: str = 'plain', rule: str = 'total',
               cur_mode: str = 'decision') -> dict:
    """Capability, conflict and repairability for one P0 stream, using thresholds fixed in the config.

    conflict: under the default policy, the share of learned edits forgotten by the end, or the share of
    base-known locality facts broken by the end, reaches conflict_min_r2w_fraction.
    repairable: at some repair step some candidate keeps the just-written edits (rephrase accuracy within
    repair_max_cur_drop of the default, per cur_mode) and cuts damage by at least repair_min_r2w_reduction relative
    to the default. rule 'total' (round 1) counts damage from step 0 (repair_total); rule 'from_decision' (round 2)
    counts only damage after the decision (repair_from_decision). The other rule is reported alongside.
    """
    if rule not in REPAIR_RULES or cur_mode not in CUR_MODES:
        raise ValueError(f'unknown repair rule {rule} or cur mode {cur_mode}')
    steps = rec['trajectory']['steps']
    evals = [s['eval'] for s in steps]
    base_loc = rec['trajectory']['base_loc']
    _, forget, loc = _damage(evals, base_loc)
    shares = [x for x in (forget['fraction'], loc['fraction']) if x is not None]
    conflict = dict(edits=forget, locality=loc, retention_acc=[e['acc']['ret'] for e in evals],
                    locality_acc=[e['acc']['loc'] for e in evals])
    conflict['pass'] = bool(shares) and max(shares) >= th['conflict_min_r2w_fraction']
    by_rule = dict(total=repair_total(rec, th, default_label, cur_mode),
                   from_decision=repair_from_decision(rec, th, default_label, cur_mode))
    repairs = by_rule[rule]
    other = 'total' if rule == 'from_decision' else 'from_decision'
    cap = rec['capability']
    return dict(stream=rec['stream'], locality_screening=rec.get('locality_screening'),
                capability=dict(cap, cur_after_write=[e['acc']['cur'] for e in evals],
                                upd_after_write=[e['acc']['upd'] for e in evals]),
                conflict=conflict, repair_rule=rule, repair_cur=cur_mode, repairs=repairs,
                repairable=any(v['passing'] for v in repairs.values()),
                repairs_other_rule={other: by_rule[other],
                                    'repairable': any(v['passing'] for v in by_rule[other].values())})
