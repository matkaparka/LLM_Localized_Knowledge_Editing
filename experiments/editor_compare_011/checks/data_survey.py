"""EC-11 design survey (CPU, no model): confirm-half test streams, and what the subject-hygiene rule would catch.

1. Rebuilds the dev-half streams exactly as RV-09/KT-10 did and checks them against KT-10's entry lists.
2. Applies the hygiene rule to every dev-half entry and splits KT-10's uniform:0.90 and matched damage by whether
   the entries that broke a fact on their own are flagged (KT-10 interventions; dev-half streams only).
3. Builds the ten confirm-half streams (CP0, CP1, C0-C7) with the same settings, caps them at 150 per step and
   reports sizes, flag counts and leakage checks. Confirm-half subjects and texts are not printed or stored.

  python checks/data_survey.py OUT.json
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EC = HERE.parent
sys.path.insert(0, str(EC))

from ec_data import HYGIENE_RULE, STREAM_ARGS, confirm_streams, hygiene_flags  # noqa: E402
from rv_policy import cap_stream, pool_texts  # noqa: E402
from so_data import audit_stream, build_streams, load_edits, norm  # noqa: E402

DATA = Path('E:/research/pause2/data/wikibigedit')
KT = EC.parent / 'key_thresholds_010' / 'runs'
CAP = 150


def kt_record(name: str) -> dict:
    p = KT / ('run_003_D7' if name == 'D7' else 'run_002') / f'stream_{name}.json.gz'
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        return json.load(f)


def attribution(rec: dict, arm: str, flagged: set) -> dict:
    """Damaged facts of an arm by whether their alone-sufficient culprits are flagged."""
    out = Counter()
    for it in rec['interventions'].get(arm, []):
        culprits = {e for e, ok, _ in it.get('alone', []) if not ok}
        if not it.get('fired'):
            out['no_firing'] += 1
        elif not culprits:
            out['no_alone_culprit'] += 1
        elif culprits <= flagged:
            out['all_culprits_flagged'] += 1
        elif culprits & flagged:
            out['some_culprits_flagged'] += 1
        else:
            out['no_culprit_flagged'] += 1
    return dict(out)


def main(out_path: str):
    edits, windows, _ = load_edits(DATA, expect_steps=8)
    by_uid = {e.uid: e for e in edits}
    report = dict(rule=list(HYGIENE_RULE), cap=CAP, stream_args=STREAM_ARGS)

    # 1-2: dev half, against KT-10
    dev = {s.name: cap_stream(s, CAP) for s in build_streams(edits, len(windows), half='dev', **STREAM_ARGS)}
    rows, flagged_subjects = {}, {}
    for name in [f'D{i}' for i in range(8)]:
        rec = kt_record(name)
        meta = rec['entries']['meta']
        same = [m['uid'] for m in meta] == [e.uid for e in dev[name].edits]
        flags = [hygiene_flags(m['subject'], by_uid[m['uid']].update) for m in meta]
        flagged = {i for i, f in enumerate(flags) if any(x in HYGIENE_RULE for x in f)}
        reasons = Counter(x for f in flags for x in f)
        rows[name] = dict(entries=len(meta), same_entries_as_kt10=same, flagged=len(flagged),
                          reasons=dict(reasons),
                          uniform090=attribution(rec, 'uniform:0.90', flagged),
                          matched=attribution(rec, 'matched', flagged))
        flagged_subjects[name] = sorted({meta[i]['subject'] for i in flagged})
    report['dev_half'] = rows
    report['dev_half_flagged_subjects'] = flagged_subjects

    # 3: confirm half
    conf = confirm_streams(edits, len(windows))
    dev_all = build_streams(edits, len(windows), half='dev', **STREAM_ARGS)
    dev_texts = pool_texts(dev_all)
    dev_subjects = {e.subject_id for s in dev_all for e in s.edits}
    crow = {}
    for name, s in conf.items():
        c = cap_stream(s, CAP)
        flags = [hygiene_flags(e.subject, e.update) for e in c.edits]
        parts = dict(edit_texts={norm(t) for e in c.edits for t in (e.update, e.rephrase, e.personas) if t},
                     loc_probe={norm(q) for q in c.loc_probe}, loc_hidden={norm(q) for q, _ in c.loc_hidden},
                     shots={norm(q) for q, _ in c.shots})
        texts = set().union(*parts.values())
        overlap_parts = {k: len(v & dev_texts) for k, v in parts.items()}
        lit = [bool(e.subject) and e.subject.casefold() in (e.rephrase or '').casefold() for e in c.edits if e.rephrase]
        crow[name] = dict(
            edits_uncapped=len(s.edits), edits=len(c.edits),
            per_step=[sum(1 for e in c.edits if e.step == t) for t in range(c.steps)],
            subjects=len({e.subject_id for e in c.edits}), relations=len({e.relation_id for e in c.edits}),
            with_rephrase=sum(1 for e in c.edits if e.rephrase), with_personas=sum(1 for e in c.edits if e.personas),
            subject_literal_in_rephrase=round(sum(lit) / len(lit), 4) if lit else None,
            flagged=sum(1 for f in flags if any(x in HYGIENE_RULE for x in f)),
            reasons=dict(Counter(x for f in flags for x in f)),
            loc_probe=len(c.loc_probe), loc_hidden=len(c.loc_hidden), shots=len(c.shots),
            audit=audit_stream(c), text_overlap_with_dev_half=len(texts & dev_texts), overlap_parts=overlap_parts,
            subjects_also_in_dev_streams=len({e.subject_id for e in c.edits} & dev_subjects))
    report['confirm_half'] = crow
    dev_rel = {e.relation_id for s in dev_all for e in s.edits}
    conf_rel = {e.relation_id for s in conf.values() for e in s.edits}
    report['relation_overlap_dev_confirm'] = len(dev_rel & conf_rel)
    Path(out_path).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')
    for k, v in rows.items():
        print(k, {x: v[x] for x in ('entries', 'same_entries_as_kt10', 'flagged')}, 'u0.90', v['uniform090'],
              'matched', v['matched'])
    for k, v in crow.items():
        print(k, {x: v[x] for x in ('edits', 'flagged', 'audit', 'overlap_parts', 'subjects_also_in_dev_streams')})
    print('relation overlap dev/confirm:', report['relation_overlap_dev_confirm'])


if __name__ == '__main__':
    main(sys.argv[1])
