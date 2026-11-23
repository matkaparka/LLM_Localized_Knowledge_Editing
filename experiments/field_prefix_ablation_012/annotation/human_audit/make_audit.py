"""FP-12 human audit of the final entity spans (PLAN section 3: "随机抽 100 条定稿，由作者核对"; review Exp-0A).

Draws, with a fixed seed, 100 rows uniformly from the 8,412 D0-D7 rows of annotation/manifest.json, and separately
50 of the 510 rows that went to adjudication (reported apart, never pooled). The sheets show only what the
annotators saw: the question and the entity's Wikidata labels, aliases and description. They do not show the
WikiBigEdit subject field, the final span, either annotator's span, the adjudication or any model output.

  python -I make_audit.py          writes audit_sheet.csv, audit_sheet_disagreements.csv and audit_sample.json
"""
import csv
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANN = HERE.parent
SEED = 20261122
LANGS = ('en', 'mul', 'de', 'fr', 'es', 'it', 'nl')


def evidence(ent: dict) -> tuple[str, str, str]:
    labels, aliases, desc = ent.get('labels') or {}, ent.get('aliases') or {}, ent.get('descriptions') or {}
    lab = []
    for lg in LANGS + tuple(sorted(labels)):
        v = labels.get(lg)
        if v and v not in lab:
            lab.append(v)
    al = []
    for lg in LANGS:
        for v in aliases.get(lg) or []:
            if v not in al and v not in lab:
                al.append(v)
    d = next((desc[lg] for lg in LANGS if desc.get(lg)), '')
    return ' | '.join(lab[:5]), ' | '.join(al[:6]), d


def main():
    for name in ('audit_sheet.csv', 'audit_sheet_disagreements.csv', 'audit_sample.json'):
        if (HERE / name).exists():
            raise SystemExit(f'{name} exists')
    man = json.loads((ANN / 'manifest.json').read_text(encoding='utf-8'))
    wd = json.loads((ANN / 'wikidata_subjects.json').read_text(encoding='utf-8'))['entities']
    dis = json.loads((ANN / 'disagreements.json').read_text(encoding='utf-8'))['items']
    rows = [r for r in man['rows'] if r['stream'] in {f'D{i}' for i in range(8)}]
    by_uid = {r['uid']: r for r in rows}
    rng = random.Random(SEED)
    main_uids = [r['uid'] for r in rng.sample(rows, 100)]
    dis_uids = [d['uid'] for d in rng.sample([d for d in dis if d['uid'] in by_uid], 50)]
    for fname, uids in (('audit_sheet.csv', main_uids), ('audit_sheet_disagreements.csv', dis_uids)):
        with (HERE / fname).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['no', 'uid', 'question', 'entity_labels', 'entity_aliases', 'entity_description',
                        'your_mention', 'status', 'note'])
            for k, u in enumerate(uids, 1):
                r = by_uid[u]
                lab, al, d = evidence(wd.get(r['subject_id']) or {})
                w.writerow([k, u, r['question'].strip(), lab, al, d, '', '', ''])
    rec = dict(seed=SEED, population=len(rows), main=main_uids, disagreements=dis_uids,
               manifest_sha256=hashlib.sha256((ANN / 'manifest.json').read_bytes()).hexdigest(),
               spans_final_sha256=hashlib.sha256((ANN / 'spans_final.json').read_bytes()).hexdigest())
    (HERE / 'audit_sample.json').write_text(json.dumps(rec, indent=1) + '\n', encoding='utf-8')
    print(f'{len(main_uids)} + {len(dis_uids)} rows written')


if __name__ == '__main__':
    main()
