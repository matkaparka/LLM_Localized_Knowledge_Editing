"""Score the author's filled SP-16 blind audit (audit_sheet.csv) against audit_key.json.

Per row the author's `verdict` is correct / wrong / ambiguous (initials and 对 / 错 / 不确定 also accepted; blank =
not scored) and, for wrong rows, `error_type` other (a different specific answer) or none (no answer). Rows are
joined on `no`; prompt, reference and continuation must still match the key. Reports, with Wilson 95% intervals:
  agreement     human correct/wrong against the main matcher and against the author's matcher, overall and per stratum
                (ambiguous rows counted apart)
  damage_pairs  S1 questions where the human reads the routing-off answer as correct and the native answer as wrong
  matched       S2 repairs the human confirms (correct) and remaining damage the human confirms (wrong)
  split         S3 rows (author's matcher correct, main matcher wrong) the human reads as correct
  error_types   among human-wrong rows; for S1 native rows read as wrong, whether the continuation starts with or
                contains the written object of a slot that fired on that row
Every disagreement is listed.

  python -I score_audit.py audit_sheet.csv OUT_JSON
"""
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'checks'))
from recount import author_match, main_match  # noqa: E402

VERDICT = {'correct': 'correct', 'c': 'correct', '对': 'correct', '正确': 'correct',
           'wrong': 'wrong', 'w': 'wrong', '错': 'wrong', '错误': 'wrong',
           'ambiguous': 'ambiguous', 'a': 'ambiguous', '不确定': 'ambiguous', '?': 'ambiguous'}
ERROR = {'other': 'other', 'o': 'other', '另一个答案': 'other', 'none': 'none', 'n': 'none', '没答': 'none'}


def wilson(k, n, z=1.959964):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]


def frac(k, n):
    return dict(k=k, n=n, rate=k / n if n else None, ci=wilson(k, n))


def agreement(rows, field):
    judged = [r for r in rows if r['human'] in ('correct', 'wrong')]
    k = sum(1 for r in judged if (r['human'] == 'correct') == r[field])
    return dict(frac(k, len(judged)), ambiguous=sum(1 for r in rows if r['human'] == 'ambiguous'),
                blank=sum(1 for r in rows if r['human'] is None))


def main(sheet, out):
    key = {k['no']: k for k in json.loads((HERE / 'audit_key.json').read_text(encoding='utf-8'))['rows']}
    filled = list(csv.DictReader(Path(sheet).open(encoding='utf-8-sig')))
    problems, rows = [], []
    for f in filled:
        no = int(f['no'])
        k = key[no]
        if f['prompt'].strip() != k['prompt'].strip() or f['reference'].strip() != k['reference'].strip():
            problems.append(dict(no=no, problem='prompt or reference changed'))
        v = (f.get('verdict') or '').strip().lower()
        e = (f.get('error_type') or '').strip().lower()
        if v and v not in VERDICT:
            problems.append(dict(no=no, problem=f'unknown verdict {v!r}'))
        if e and e not in ERROR:
            problems.append(dict(no=no, problem=f'unknown error_type {e!r}'))
        rows.append(dict(k, human=VERDICT.get(v), error=ERROR.get(e), note=(f.get('note') or '').strip()))
    missing = sorted(set(key) - {r['no'] for r in rows})
    if missing:
        problems.append(dict(problem=f'rows missing from the sheet: {missing}'))

    strata = sorted({r['stratum'] for r in rows})
    res = dict(rows=len(rows), problems=problems,
               agreement={'all': dict(main=agreement(rows, 'auto_main'), author=agreement(rows, 'auto_author'))})
    for s in strata:
        sub = [r for r in rows if r['stratum'] == s]
        res['agreement'][s] = dict(main=agreement(sub, 'auto_main'), author=agreement(sub, 'auto_author'))

    by_no = {r['no']: r for r in rows}
    pairs, unconfirmed = 0, []
    seen = set()
    for r in rows:
        if r['stratum'] != 'S1_damage_pair' or r['no'] in seen:
            continue
        o = by_no[r['pair_no']]
        seen |= {r['no'], o['no']}
        before, after = (r, o) if r['arm'] == 'none' else (o, r)
        ok = before['human'] == 'correct' and after['human'] == 'wrong'
        pairs += ok
        if not ok:
            unconfirmed.append(dict(before_no=before['no'], after_no=after['no'], prompt=before['prompt'],
                                    reference=before['reference'], before=before['continuation'],
                                    after=after['continuation'], human_before=before['human'],
                                    human_after=after['human'], notes=[before['note'], after['note']]))
    res['damage_pairs'] = dict(frac(pairs, len(seen) // 2), unconfirmed=unconfirmed)

    rep = [r for r in rows if r['stratum'] == 'S2_matched_repaired']
    rem = [r for r in rows if r['stratum'] == 'S2_matched_still_damaged']
    res['matched'] = dict(repaired_confirmed=frac(sum(r['human'] == 'correct' for r in rep), len(rep)),
                          still_damaged_confirmed=frac(sum(r['human'] == 'wrong' for r in rem), len(rem)))
    split = [r for r in rows if r['stratum'] == 'S3_matcher_split']
    res['split'] = dict(human_correct=frac(sum(r['human'] == 'correct' for r in split), len(split)),
                        human_wrong=frac(sum(r['human'] == 'wrong' for r in split), len(split)),
                        human_ambiguous=sum(r['human'] == 'ambiguous' for r in split))

    wrong = [r for r in rows if r['human'] == 'wrong']
    nat = [r for r in wrong if r['stratum'] == 'S1_damage_pair' and r['arm'] == 'native']
    res['error_types'] = dict(
        all_wrong={t: sum(1 for r in wrong if r['error'] == t) for t in ('other', 'none')}
        | {'unlabelled': sum(1 for r in wrong if r['error'] is None)},
        s1_native_wrong=dict(n=len(nat), other=sum(r['error'] == 'other' for r in nat),
                             none=sum(r['error'] == 'none' for r in nat),
                             starts_with_written_object=sum(any(main_match(r['continuation'], o)
                                                                for o in r['fired_written_objects']) for r in nat),
                             contains_written_object=sum(any(author_match(r['continuation'], o)
                                                             for o in r['fired_written_objects']) for r in nat)))

    res['disagreements_with_main'] = [
        dict(no=r['no'], stratum=r['stratum'], arm=r['arm'], prompt=r['prompt'], reference=r['reference'],
             continuation=r['continuation'], human=r['human'], auto_main=r['auto_main'],
             auto_author=r['auto_author'], error=r['error'], note=r['note'])
        for r in rows if r['human'] in ('correct', 'wrong') and (r['human'] == 'correct') != r['auto_main']]
    res['ambiguous_rows'] = [dict(no=r['no'], stratum=r['stratum'], arm=r['arm'], prompt=r['prompt'],
                                  reference=r['reference'], continuation=r['continuation'], note=r['note'])
                             for r in rows if r['human'] == 'ambiguous']
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    a = res['agreement']['all']
    print('problems:', len(problems))
    print('agreement with main matcher:', a['main']['k'], '/', a['main']['n'], 'ambiguous', a['main']['ambiguous'],
          'blank', a['main']['blank'])
    print('agreement with author matcher:', a['author']['k'], '/', a['author']['n'])
    print('damage pairs confirmed:', res['damage_pairs']['k'], '/', res['damage_pairs']['n'])
    print('matched repaired / still damaged confirmed:', res['matched']['repaired_confirmed']['k'], '/',
          res['matched']['repaired_confirmed']['n'], ',', res['matched']['still_damaged_confirmed']['k'], '/',
          res['matched']['still_damaged_confirmed']['n'])
    print('split rows read as correct:', res['split']['human_correct']['k'], '/', res['split']['human_correct']['n'])
    print('error types:', res['error_types'])


if __name__ == '__main__':
    main(*sys.argv[1:])
