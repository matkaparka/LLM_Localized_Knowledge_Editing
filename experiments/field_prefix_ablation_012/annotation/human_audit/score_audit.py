"""Score the author's filled audit sheets against annotation/spans_final.json (review Exp-0A).

Per row: the author's `status` is empty or 'ok' (then `your_mention` must be a substring of the question; its last
occurrence is taken, as annotator A did), 'none' (the question does not name the entity) or 'ambiguous'. Agreement:
  final_char   the mention ends on the same character as the final span (this fixes the key's anchor token)
  exact        same start and end
  flags        the author's none / ambiguous against the final unlocatable flag
Proportions with Wilson 95% intervals; every disagreement is listed for classification.

  python -I score_audit.py audit_sheet.csv [audit_sheet_disagreements.csv] OUT_JSON
"""
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def wilson(k, n, z=1.959964):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]


def score(sheet: Path, final: dict) -> dict:
    rows = list(csv.DictReader(sheet.open(encoding='utf-8-sig')))
    n = fc = ex = 0
    bad, skipped = [], []
    for r in rows:
        st = (r.get('status') or '').strip().lower() or 'ok'
        fin = final[r['uid']]
        q = r['question']
        if st in ('none', 'ambiguous'):
            n += 1
            ok = bool(fin.get('unlocatable'))
            fc += ok
            ex += ok
            if not ok:
                bad.append(dict(no=r['no'], uid=r['uid'], question=q, author=st, final=fin.get('text')))
            continue
        m = (r.get('your_mention') or '').strip()
        if not m:
            skipped.append(r['no'])
            continue
        k = q.rfind(m)
        if k < 0:
            skipped.append(r['no'])
            continue
        n += 1
        if fin.get('unlocatable') or not fin.get('span'):
            bad.append(dict(no=r['no'], uid=r['uid'], question=q, author=m, final='unlocatable'))
            continue
        a, b = fin['span']
        same_end = k + len(m) == b
        fc += same_end
        ex += same_end and k == a
        if not (same_end and k == a):
            bad.append(dict(no=r['no'], uid=r['uid'], question=q, author=m, final=fin.get('text'),
                            same_final_char=same_end))
    return dict(rows=len(rows), scored=n, skipped=skipped, final_char=fc, exact=ex,
                final_char_rate=fc / n if n else None, final_char_ci=wilson(fc, n),
                exact_rate=ex / n if n else None, exact_ci=wilson(ex, n), disagreements=bad)


def main(args):
    out = Path(args[-1])
    final = json.loads((HERE.parent / 'spans_final.json').read_text(encoding='utf-8'))['final']
    res = {Path(a).name: score(Path(a), final) for a in args[:-1]}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    for k, v in res.items():
        print(k, {x: v[x] for x in ('scored', 'final_char', 'final_char_ci', 'exact', 'exact_ci')})


if __name__ == '__main__':
    main(sys.argv[1:])
