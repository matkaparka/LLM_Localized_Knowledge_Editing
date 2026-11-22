"""FP-12 annotator A: a fixed rule that finds the subject entity's Wikidata label or alias in the question.

Reads only the question and the Wikidata evidence of subject_id (never the WikiBigEdit subject field, model outputs,
risk rankings, hidden questions or intervention results). PLAN section 3:

1. Candidates: every label and alias of subject_id in the fetched languages.
2. Matching is case- and punctuation-insensitive (, . ' " ? ! ; : curly quotes, guillemets, parentheses and brackets
   become spaces; runs of spaces collapse) and on word boundaries. The longest matching candidate wins; between equally long matches the last
   occurrence wins (the writer's rfind).
3. No match: retry with reduced candidates (a trailing parenthetical removed, everything after the first comma
   removed, a leading article removed).
4. Still none: the longest run of >= 2 consecutive candidate words that occurs in the question, if it holds a word
   with a digit or a capital letter in the candidate and is not made of stop words only.
5. Otherwise no span.

Spans are (start, end) offsets into question.strip(), end exclusive.

  python annotator_a.py MANIFEST.json LABELS.json OUT.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PUNCT = set(',.\'"“”‘’„«»()[]?!;:')
STOP = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'by', 'de', 'la', 'le', 'el', 'der',
        'die', 'das', 'von', 'van', 'du', 'des', 'di', 'da', 'is', 'was', 'with', 'from', 'as'}


def normalise(text: str) -> tuple[str, list[int]]:
    """Lower-cased text with punctuation as spaces and space runs collapsed, plus the original index of every kept
    character. Characters whose lower case is not one character are kept as they are (same length)."""
    out, idx = [], []
    for i, c in enumerate(text):
        if c in PUNCT or c.isspace():
            c = ' '
        else:
            lc = c.lower()
            c = lc if len(lc) == 1 else c
        if c == ' ' and (not out or out[-1] == ' '):
            continue
        out.append(c)
        idx.append(i)
    while out and out[-1] == ' ':
        out.pop()
        idx.pop()
    return ''.join(out), idx


def find_all(hay: str, needle: str) -> list[int]:
    """Word-boundary occurrences of needle in hay (both normalised)."""
    if not needle:
        return []
    pos, out = 0, []
    while True:
        k = hay.find(needle, pos)
        if k < 0:
            return out
        end = k + len(needle)
        if (k == 0 or hay[k - 1] == ' ') and (end == len(hay) or hay[end] == ' '):
            out.append(k)
        pos = k + 1


def candidates(ev: dict) -> list[str]:
    seen, out = set(), []
    for v in list((ev.get('labels') or {}).values()) + [a for vs in (ev.get('aliases') or {}).values() for a in vs]:
        v = (v or '').strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def reduced(c: str) -> list[str]:
    out = []
    r = re.sub(r'\s*\([^)]*\)\s*$', '', c).strip()
    if r and r != c:
        out.append(r)
    if ',' in c:
        out.append(c.split(',', 1)[0].strip())
    for art in ('the ', 'The ', 'a ', 'A ', 'an ', 'An '):
        if c.startswith(art):
            out.append(c[len(art):].strip())
    return [x for x in out if x]


def best_match(qn: str, cands: list[str]) -> tuple[int, int, str, int] | None:
    """(normalised start, normalised end, candidate, occurrences) of the longest candidate in qn."""
    best = None
    for c in cands:
        cn, _ = normalise(c)
        occ = find_all(qn, cn)
        if not occ:
            continue
        key = (len(cn), occ[-1])
        if best is None or key > best[0]:
            best = (key, (occ[-1], occ[-1] + len(cn), c, len(occ)))
    return best[1] if best else None


def ngram_match(qn: str, cands: list[str]) -> tuple[int, int, str, int] | None:
    best = None
    for c in cands:
        words = c.split()
        for n in range(len(words), 1, -1):
            for s in range(0, len(words) - n + 1):
                gram = words[s:s + n]
                if not any(re.search(r'[0-9A-Z]', w) for w in gram):
                    continue
                gn, _ = normalise(' '.join(gram))
                if all(w in STOP for w in gn.split()):
                    continue
                occ = find_all(qn, gn)
                if occ:
                    key = (len(gn), occ[-1])
                    if best is None or key > best[0]:
                        best = (key, (occ[-1], occ[-1] + len(gn), ' '.join(gram), len(occ)))
    return best[1] if best else None


def annotate(question: str, ev: dict | None) -> dict:
    q = question.strip()
    if not ev:
        return dict(span=None, method='no evidence')
    qn, idx = normalise(q)
    cands = candidates(ev)
    for method, pool in (('label', cands), ('reduced', [r for c in cands for r in reduced(c)])):
        m = best_match(qn, pool)
        if m:
            break
    else:
        m, method = ngram_match(qn, cands), 'ngram'
    if not m:
        return dict(span=None, method='none', n_candidates=len(cands))
    a, b, cand, occ = m
    start, end = idx[a], idx[b - 1] + 1
    # amendment 1 (2026-11-21, after the first comparison with B, before any adjudication): punctuation that ends the
    # matched name ('Jr.', 'E tu...', a closing quote) belongs to the mention when the question has it right there
    tail = len(cand) - len(cand.rstrip(''.join(PUNCT)))
    if tail:
        want = cand[len(cand) - tail:]
        if q[end:end + tail] == want:
            end += tail
    return dict(span=[start, end], text=q[start:end], method=method, candidate=cand, occurrences=occ)


def main(manifest_path: str, labels_path: str, out_path: str):
    rows = json.loads(Path(manifest_path).read_text(encoding='utf-8'))['rows']
    ents = json.loads(Path(labels_path).read_text(encoding='utf-8'))['entities']
    out = {r['uid']: annotate(r['question'], ents.get(r['subject_id'])) for r in rows}
    methods = {}
    for v in out.values():
        methods[v['method']] = methods.get(v['method'], 0) + 1
    Path(out_path).write_text(json.dumps(dict(annotator='A (rule, Wikidata labels and aliases)', methods=methods,
                                              spans=out), ensure_ascii=False, indent=0),
                              encoding='utf-8', newline='\n')
    print(methods)


if __name__ == '__main__':
    main(*sys.argv[1:4])
