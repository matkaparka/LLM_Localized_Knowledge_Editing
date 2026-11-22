"""FP-12 span decisions (PLAN section 3).

  python finalize.py prepare MANIFEST SPANS_A SPANS_B LABELS DISAGREEMENTS.json
        items where A and B do not put the anchor on the same token (or one or both gave no span) -> the blind
        adjudicator's input: question, evidence, the two candidate spans; nothing else
  python finalize.py final MANIFEST SPANS_A SPANS_B ADJUDICATION.json OUT.json
        the final span of every edit, its anchor token, whether it moved, and the unlocatable flags

Agreement is judged on the anchor: the token of the bare write prompt that holds the span's last character (the
writer keys an entry on that token only). With agreement the final span is A's. Otherwise the adjudicator's
decision: 'A', 'B', 'other' (an exact substring of the question; its last occurrence is used, as the writer's rfind
would) or 'none' / 'ambiguous' (unlocatable: the entry becomes a refused slot in every paired cell).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import fp_data  # noqa: E402


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def anchor(tok, question, span):
    return fp_data.span_anchor(tok, question, tuple(span)).token if span else None


def b_spans(rows, b):
    """B's spans; where B's answer is not a substring of the question, locate it with annotator A's matching (case and
    punctuation insensitive, word boundaries, last occurrence). Amendment 2 (2026-11-21, after the first comparison,
    before any adjudication): it changes how B's chosen words are found, never which words B chose."""
    import annotator_a as aa
    out = {}
    for r in rows:
        v = dict(b[r['uid']])
        if v.get('span') is None and v.get('kind') == 'not a substring' and v.get('raw'):
            ans = v['raw'].strip().strip('"“”').strip()
            qn, idx = aa.normalise(r['question'])
            cn, _ = aa.normalise(ans)
            occ = aa.find_all(qn, cn)
            if cn and occ:
                s, e = idx[occ[-1]], idx[occ[-1] + len(cn) - 1] + 1
                v.update(span=[s, e], text=r['question'][s:e], kind='normalised match')
        out[r['uid']] = v
    return out


def compare(tok, rows, a, b):
    out = {}
    for r in rows:
        sa, sb = a[r['uid']].get('span'), b[r['uid']].get('span')
        ta, tb = anchor(tok, r['question'], sa), anchor(tok, r['question'], sb)
        out[r['uid']] = dict(a_token=ta, b_token=tb, agree=ta is not None and ta == tb)
    return out


def prepare(manifest, spans_a, spans_b, labels, out_path):
    tok = fp_data.load_tokenizer()
    rows = load(manifest)['rows']
    a = load(spans_a)['spans']
    b = b_spans(rows, load(spans_b)['spans'])
    ents = load(labels)['entities']
    cmp_ = compare(tok, rows, a, b)
    items = []
    for r in rows:
        if cmp_[r['uid']]['agree']:
            continue
        ev = ents.get(r['subject_id']) or {}
        labs = sorted(set((ev.get('labels') or {}).values()))[:6]
        desc = (ev.get('descriptions') or {}).get('en') or next(iter((ev.get('descriptions') or {}).values()), '')
        items.append(dict(uid=r['uid'], question=r['question'], entity_labels=labs, entity_description=desc,
                          candidate_A=a[r['uid']].get('text'), candidate_B=b[r['uid']].get('text')))
    Path(out_path).write_text(json.dumps(dict(n=len(items), items=items), ensure_ascii=False, indent=1),
                              encoding='utf-8', newline='\n')
    agree = sum(v['agree'] for v in cmp_.values())
    print(f'{len(rows)} edits; anchors agree on {agree} ({agree / len(rows):.3f}); to adjudicate {len(items)}')


def final(manifest, spans_a, spans_b, adjudication, out_path):
    tok = fp_data.load_tokenizer()
    rows = load(manifest)['rows']
    a = load(spans_a)['spans']
    b = b_spans(rows, load(spans_b)['spans'])
    adj = {d['uid']: d for d in load(adjudication)['decisions']} if adjudication else {}
    cmp_ = compare(tok, rows, a, b)
    final, counts = {}, Counter()
    for r in rows:
        u, q = r['uid'], r['question']
        c = cmp_[u]
        if c['agree']:
            span, src = a[u]['span'], 'agree'
        else:
            d = adj.get(u)
            if d is None:
                raise SystemExit(f'no adjudication for {u}')
            dec = d['decision']
            if dec == 'A':
                span = a[u]['span']
            elif dec == 'B':
                span = b[u]['span']
            elif dec == 'other':
                k = q.rfind(d['text'])
                if k < 0:
                    raise SystemExit(f'{u}: adjudicated text is not in the question')
                span = [k, k + len(d['text'])]
            elif dec in ('none', 'ambiguous'):
                span = None
            else:
                raise SystemExit(f'{u}: unknown decision {dec}')
            if dec in ('A', 'B') and not span:
                raise SystemExit(f'{u}: decision {dec} points at an empty candidate')
            src = f'adjudicated:{dec}'
        counts[src] += 1
        if span:
            fa = fp_data.span_anchor(tok, r['question'], tuple(span))
            final[u] = dict(span=span, text=q[span[0]:span[1]], unlocatable=False, source=src,
                            anchor_token=fa.token, orig_token=r['orig']['token'], changed=fa.token != r['orig']['token'])
        else:
            final[u] = dict(span=None, text=None, unlocatable=True, source=src, anchor_token=None,
                            orig_token=r['orig']['token'], changed=False)
    per_stream = {}
    for r in rows:
        f = final[r['uid']]
        s = per_stream.setdefault(r['stream'], Counter())
        s['edits'] += 1
        s['changed'] += f['changed']
        s['unlocatable'] += f['unlocatable']
    body = json.dumps(final, ensure_ascii=False, sort_keys=True)
    out = dict(n=len(final), sources=dict(counts), per_stream={k: dict(v) for k, v in sorted(per_stream.items())},
               inputs={k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in
                       (('manifest', manifest), ('spans_a', spans_a), ('spans_b', spans_b),
                        ('adjudication', adjudication)) if p},
               sha256_final=hashlib.sha256(body.encode()).hexdigest(), final=final)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding='utf-8', newline='\n')
    print(json.dumps(dict(sources=out['sources'], per_stream=out['per_stream']), indent=1))


if __name__ == '__main__':
    cmd, *args = sys.argv[1:]
    if cmd == 'prepare':
        prepare(*args)
    elif cmd == 'final':
        final(*args)
    else:
        raise SystemExit('prepare | final')
