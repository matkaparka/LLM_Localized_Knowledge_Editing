"""SP-16 post-hoc description (written 2026-11-23 after the test analysis; not part of the frozen protocol, no gate).
Standard library only; reuses the matchers of checks/recount.py.

  python checks/describe_post_hoc.py RUN OUT_JSON

Per stream, main matcher:
  - spread of the label-free risk score (max cosine of each slot to the 800 background probes);
  - background probes whose prefill winner passes 0.70 (= sum of the trigger counts) and how concentrated they are;
  - damaged protected facts per sufficient slot: distinct witness slots, and the damaged facts covered by the 13 slots
    that witness the most damage (an after-the-fact ceiling for any 13-slot list, using test labels);
  - Spearman correlation over the 250 slots between witnessed damage and the max-cosine score / the trigger count;
  - what the damaged native answers became: the written object of a slot that fired on that row (main matcher, i.e.
    the answer starts with it), the written object anywhere in the answer (author matcher), or something else.
"""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recount import author_match, main_match  # noqa: E402

BUDGET = 13


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


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else None


def quantiles(xs):
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]
    return dict(min=s[0], q25=q(0.25), median=q(0.5), q75=q(0.75), max=s[-1])


def stream(rec):
    n = rec['n_slots']
    prot, edit = rec['items']['protect'], rec['items']['edit']
    assert [e['case_id'] for e in edit] == rec['case_ids'], 'slot order differs from the edit items'
    seal = rec['seal']
    maxcos, trig = seal['maxcos'], seal['trigger_counts']
    none = [main_match(t, it['target']) for t, it in zip(rec['arms']['none']['protect']['text'], prot)]
    nat_text = rec['arms']['native']['protect']['text']
    nat_fired = rec['arms']['native']['protect']['fired']
    nat = [main_match(t, it['target']) for t, it in zip(nat_text, prot)]
    dam = [i for i in range(len(prot)) if none[i] and not nat[i]]
    suff = {i: set() for i in dam}
    for r in rec['replays']:
        if r['item'] in suff and not main_match(r['text'], prot[r['item']]['target']):
            suff[r['item']].add(r['slot'])
    w = [0] * n
    for s in suff.values():
        for slot in s:
            w[slot] += 1
    oracle = sorted(range(n), key=lambda s: (-w[s], s))[:BUDGET]
    covered = sum(1 for s in suff.values() if s & set(oracle))
    kinds = dict(starts_with_written_object=0, contains_written_object=0, other=0)
    for i in dam:
        objs = [edit[s]['target'] for s in nat_fired[i]]
        if any(main_match(nat_text[i], o) for o in objs):
            kinds['starts_with_written_object'] += 1
        elif any(author_match(nat_text[i], o) for o in objs):
            kinds['contains_written_object'] += 1
        else:
            kinds['other'] += 1
    risk_top, freq_top = seal['ranking'][:BUDGET], seal['freq_ranking'][:BUDGET]
    total_trig = sum(trig)
    return dict(
        maxcos=quantiles(maxcos), maxcos_top13=[min(maxcos[s] for s in risk_top), max(maxcos[s] for s in risk_top)],
        slots_maxcos_above_095=sum(1 for x in maxcos if x > 0.95),
        probes_firing=total_trig, slots_with_triggers=sum(1 for x in trig if x > 0),
        trigger_share_freq_top13=sum(trig[s] for s in freq_top) / total_trig if total_trig else None,
        trigger_share_risk_top13=sum(trig[s] for s in risk_top) / total_trig if total_trig else None,
        protect_rows_with_a_fired_slot=sum(1 for f in nat_fired if f),
        baseline_correct=sum(none), damaged=len(dam), witnessed=sum(1 for s in suff.values() if s),
        witness_slots=sum(1 for x in w if x > 0), max_damage_per_slot=max(w),
        oracle_top13_covered=covered,
        spearman_damage_maxcos=spearman(w, maxcos), spearman_damage_triggers=spearman(w, trig),
        spearman_maxcos_triggers=spearman(maxcos, trig),
        damaged_answer_kinds=kinds)


def main(run, out):
    run = Path(run)
    res, allcos = {}, []
    for p in sorted(run.glob('stream_*.json.gz')):
        rec = json.loads(gzip.open(p).read().decode('utf-8'))
        res[rec['stream']] = stream(rec)
        allcos += rec['seal']['maxcos']
    keys_sum = ('probes_firing', 'protect_rows_with_a_fired_slot', 'baseline_correct', 'damaged', 'witnessed',
                'witness_slots', 'oracle_top13_covered')
    pooled = {k: sum(r[k] for r in res.values()) for k in keys_sum}
    pooled['damaged_answer_kinds'] = {k: sum(r['damaged_answer_kinds'][k] for r in res.values())
                                      for k in ('starts_with_written_object', 'contains_written_object', 'other')}
    for k in ('spearman_damage_maxcos', 'spearman_damage_triggers', 'spearman_maxcos_triggers',
              'trigger_share_freq_top13', 'trigger_share_risk_top13'):
        v = [r[k] for r in res.values() if r[k] is not None]
        pooled[k + '_mean'] = sum(v) / len(v) if v else None
    pooled['maxcos_all'] = quantiles(allcos)
    pooled['slots_maxcos_above_095'] = sum(1 for x in allcos if x > 0.95)
    pooled['slots'] = len(allcos)
    doc = dict(run=run.name, note='post hoc, descriptive; not part of the frozen SP-16 protocol', budget=BUDGET,
               matcher='main', streams=res, pooled=pooled)
    Path(out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(pooled, ensure_ascii=False))


if __name__ == '__main__':
    main(*sys.argv[1:])
