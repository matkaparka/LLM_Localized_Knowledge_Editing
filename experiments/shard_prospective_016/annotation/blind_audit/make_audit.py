"""SP-16 blind human audit (PLAN section 4.3: "then blind-check samples on which the arms disagree, and the error types"), built 2026-11-23 after the test
analysis at the author's request; the size (140 rows) was his choice.

Rows come from the test streams T01-T08 (protected questions only), with the main matcher's verdicts as the
automatic labels, drawn with a fixed seed:
  S1  damage pairs     40 questions damaged by native SHARD (routing off correct, native wrong), 5 per stream; both
                       continuations are shown, as two separate rows (80 rows)
  S2  matched arm      the per-slot-threshold continuation of 10 questions it repaired and 10 it still damages
  S3  matcher split    20 continuations on which the main matcher and the author's substring matcher disagree
                       (arms none, native, matched, uniform:0.95)
  S4  random control   20 continuations drawn uniformly from the same four arms
A question appears in one stratum only (S1 questions twice). Rows are shuffled. The sheet shows the prompt the model
saw, the reference answer (CounterFact target_true) and the continuation (newlines shown as ⏎); it shows no arm,
stream, case id, slot or automatic verdict. Those are in audit_key.json, which the auditor does not open.

  python -I make_audit.py COUNTERFACT_JSON     writes audit_sheet.csv and audit_key.json (refuses to overwrite)
"""
import csv
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SP = HERE.parents[1]
sys.path.insert(0, str(SP / 'checks'))
from recount import author_match, main_match  # noqa: E402

SEED = 20261123
RUN = SP / 'runs' / 'test_001'
COUNTERFACT_SHA = 'd017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f'
ARMS = ('none', 'native', 'matched', 'uniform:0.95')
N_PAIRS_PER_STREAM, N_REPAIRED, N_REMAINING, N_SPLIT, N_RANDOM = 5, 10, 10, 20, 20


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def show(text: str) -> str:
    return text.rstrip().replace('\r', '').replace('\n', ' ⏎ ')


def main(cf_path):
    for name in ('audit_sheet.csv', 'audit_key.json'):
        if (HERE / name).exists():
            raise SystemExit(f'{name} exists')
    cf_path = Path(cf_path)
    if sha(cf_path) != COUNTERFACT_SHA:
        raise SystemExit('counterfact.json does not match the published SHA-256')
    cf = {r['case_id']: r['requested_rewrite'] for r in json.loads(cf_path.read_text(encoding='utf-8'))}
    recs = {}
    for p in sorted(RUN.glob('stream_*.json.gz')):
        rec = json.loads(gzip.open(p).read().decode('utf-8'))
        recs[rec['stream']] = rec
    assert list(recs) == [f'T0{i}' for i in range(1, 9)], list(recs)

    def verdicts(s, i, arm):
        rec = recs[s]
        t, ref = rec['arms'][arm]['protect']['text'][i], rec['items']['protect'][i]['target']
        return main_match(t, ref), author_match(t, ref)

    rng = random.Random(SEED)
    used, rows = set(), []

    def add(stratum, s, i, arm, **extra):
        m, a = verdicts(s, i, arm)
        rows.append(dict(stratum=stratum, stream=s, item=i, arm=arm, auto_main=m, auto_author=a, **extra))
        used.add((s, i))

    # S1: damage pairs, 5 per stream
    for s, rec in recs.items():
        n = len(rec['items']['protect'])
        dam = [i for i in range(n) if verdicts(s, i, 'none')[0] and not verdicts(s, i, 'native')[0]]
        for i in rng.sample(dam, N_PAIRS_PER_STREAM):
            add('S1_damage_pair', s, i, 'none')
            add('S1_damage_pair', s, i, 'native')
    # S2: matched arm, repaired and remaining damage
    rep, rem = [], []
    for s, rec in recs.items():
        for i in range(len(rec['items']['protect'])):
            if (s, i) in used or not verdicts(s, i, 'none')[0]:
                continue
            nat, mat = verdicts(s, i, 'native')[0], verdicts(s, i, 'matched')[0]
            if not nat and mat:
                rep.append((s, i))
            elif not mat:
                rem.append((s, i))
    for s, i in rng.sample(rep, N_REPAIRED):
        add('S2_matched_repaired', s, i, 'matched')
    for s, i in rng.sample(rem, N_REMAINING):
        add('S2_matched_still_damaged', s, i, 'matched')
    # S3: main and author matchers disagree, one row per question
    split = []
    for s, rec in recs.items():
        for i in range(len(rec['items']['protect'])):
            for arm in ARMS:
                m, a = verdicts(s, i, arm)
                if m != a:
                    split.append((s, i, arm))
    rng.shuffle(split)
    taken = 0
    for s, i, arm in split:
        if taken == N_SPLIT:
            break
        if (s, i) not in used:
            add('S3_matcher_split', s, i, arm)
            taken += 1
    assert taken == N_SPLIT
    # S4: uniform control, one row per question
    pool = [(s, i, arm) for s, rec in recs.items() for i in range(len(rec['items']['protect'])) for arm in ARMS]
    rng.shuffle(pool)
    taken = 0
    for s, i, arm in pool:
        if taken == N_RANDOM:
            break
        if (s, i) not in used:
            add('S4_random', s, i, arm)
            taken += 1
    assert taken == N_RANDOM

    rng.shuffle(rows)
    key, sheet = [], []
    for no, r in enumerate(rows, 1):
        rec = recs[r['stream']]
        it = rec['items']['protect'][r['item']]
        rr = cf[it['case_id']]
        assert rr['target_true']['str'] == it['target'], (r, rr['target_true'])
        prompt = rr['prompt'].format(rr['subject'])
        text = rec['arms'][r['arm']]['protect']['text'][r['item']]
        fired = rec['arms'][r['arm']]['protect'].get('fired')
        objs = sorted({rec['items']['edit'][slot]['target'] for slot in fired[r['item']]}) if fired else []
        sheet.append([no, prompt, it['target'], show(text), '', '', ''])
        key.append(dict(no=no, **r, case_id=it['case_id'], relation_id=rr['relation_id'], prompt=prompt,
                        reference=it['target'], continuation=text, fired_written_objects=objs))
    for k in key:
        if k['stratum'] == 'S1_damage_pair':
            k['pair_no'] = next(o['no'] for o in key if o is not k and o['stratum'] == 'S1_damage_pair'
                                and o['stream'] == k['stream'] and o['item'] == k['item'])
    with (HERE / 'audit_sheet.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['no', 'prompt', 'reference', 'continuation', 'verdict', 'error_type', 'note'])
        w.writerows(sheet)
    doc = dict(seed=SEED, built_utc='2026-11-23', run=str(RUN.relative_to(SP)), arms=list(ARMS),
               sizes=dict(pairs_per_stream=N_PAIRS_PER_STREAM, repaired=N_REPAIRED, remaining=N_REMAINING,
                          split=N_SPLIT, random=N_RANDOM),
               counterfact_sha256=COUNTERFACT_SHA, analysis_sha256=sha(RUN / 'analysis.json'),
               stream_sha256={p.name: sha(p) for p in sorted(RUN.glob('stream_*.json.gz'))},
               candidates=dict(matched_repaired=len(rep), matched_still_damaged=len(rem), matcher_split_rows=len(split)),
               rows=key)
    (HERE / 'audit_key.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    counts = {}
    for k in key:
        counts[k['stratum']] = counts.get(k['stratum'], 0) + 1
    print(len(key), 'rows', counts, 'candidates', doc['candidates'])


if __name__ == '__main__':
    main(*sys.argv[1:])
