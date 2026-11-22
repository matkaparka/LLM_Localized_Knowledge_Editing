"""XR-13 streams: events + writer W/R questions + extractor spans -> one <stream>.json per stream and mu_corpus.json.

  python xr_build.py extract-input PLAN.json QUESTIONS_W.json OUT.json     the write questions for the extractor
  python xr_build.py streams EVENTS_DIR PLAN.json QUESTIONS_W.json QUESTIONS_R.json SPANS.json OUT_DIR

Edits whose write question or rephrasing failed the writer's checks are dropped and counted (never repaired by hand);
hidden and probe questions likewise. The format shots are fixed general-knowledge pairs that state no fact about any
package. Text audits: no hidden or probe question may equal (normalised) any edit wording or format shot of the same
stream, and the routing-centre corpus may not contain any stream's text.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'self_observation_008'))
from so_data import norm  # noqa: E402

SHOTS = [['What is the chemical symbol for gold?', 'Au'], ['How many days are there in a leap year?', '366']]


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def extract_input(plan_path, qw_path, out_path):
    plan, qw = load(plan_path), load(qw_path)['questions']
    items = [dict(uid=i['uid'], question=qw[i['id']]) for i in plan['items'] if i['role'] == 'write' and qw.get(i['id'])]
    Path(out_path).write_text(json.dumps(dict(n=len(items), items=items), ensure_ascii=False), encoding='utf-8')
    print(len(items), 'write questions')


def streams(events_dir, plan_path, qw_path, qr_path, spans_path, out_dir):
    plan = load(plan_path)['items']
    qw, qr = load(qw_path)['questions'], load(qr_path)['questions']
    spans = load(spans_path)['spans']
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_stream = {}
    for i in plan:
        if i['role'] != 'mu':
            by_stream.setdefault(i['stream'], []).append(i)
    report, all_texts = {}, set()
    for p in sorted(Path(events_dir).glob('*_events.json')):
        name = p.name[:-len('_events.json')]
        ev = load(p)
        its = {i['id']: i for i in by_stream.get(name, [])}
        edits, dropped = [], Counter()
        for b in ev['batches']:
            for e in b['edits']:
                w, r = qw.get(f'{e["uid"]}|w'), (qr.get(f'{e["uid"]}|r') or [None, None])
                r1, r2 = (r + [None, None])[:2]
                if not w:
                    dropped['write question'] += 1
                    continue
                if not r1 or not r2:
                    dropped['rephrasing'] += 1
                    continue
                sp = spans.get(e['uid'], {})
                edits.append(dict(uid=e['uid'], step=e['step'], entity=e['entity'], attribute=e['attribute'],
                                  value=e['value'], update=w, rephrase=r1, retention=r2, span=sp.get('span'),
                                  span_text=sp.get('text'), extractor=sp.get('kind')))
        edit_text = {norm(x) for e in edits for x in (e['update'], e['rephrase'], e['retention'])}
        shot_text = {norm(q) for q, _ in SHOTS}
        hidden, probes, clash = [], [], 0
        for k, i in its.items():
            q = qw.get(k)
            if i['role'] == 'hidden':
                if not q:
                    dropped['hidden question'] += 1
                elif norm(q) in edit_text or norm(q) in shot_text:
                    clash += 1
                else:
                    hidden.append([q, i['answer']])
            elif i['role'] == 'probe':
                if not q:
                    dropped['probe question'] += 1
                elif norm(q) in edit_text or norm(q) in shot_text:
                    clash += 1
                else:
                    probes.append(q)
        all_texts |= edit_text | {norm(q) for q, _ in hidden} | {norm(q) for q in probes}
        rec = dict(stream=name, family=ev['family'], pool=ev.get('pool'), window=ev['window'], steps=len(ev['window']),
                   role=ROLE.get(name, 'test'), shots=SHOTS, edits=edits, hidden=hidden, probes=probes,
                   dropped=dict(dropped), text_clashes=clash)
        (out_dir / f'{name}.json').write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
        report[name] = dict(edits=len(edits), hidden=len(hidden), probes=len(probes), dropped=dict(dropped),
                            refused=sum(1 for e in edits if not e['span']), clashes=clash)
    mu = []
    for i in plan:
        if i['role'] == 'mu':
            q = qw.get(i['id'])
            if q and norm(q) not in all_texts:
                mu.append(f'Q: {q.strip()}\nA:')
    (out_dir / 'mu_corpus.json').write_text(json.dumps(dict(n=len(mu), texts=mu), ensure_ascii=False), encoding='utf-8')
    report['mu_corpus'] = len(mu)
    (out_dir / 'build_report.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))


ROLE = {}


def main():
    cmd, *a = sys.argv[1:]
    if cmd == 'extract-input':
        extract_input(*a)
    elif cmd == 'streams':
        global ROLE
        roles = HERE / 'streams_roles.json'
        if roles.exists():
            ROLE = load(roles)
        streams(*a)
    else:
        raise SystemExit('extract-input | streams')


if __name__ == '__main__':
    main()
