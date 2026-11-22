"""FP-12 annotator B: a pinned instruct model copies the question's mention of the subject entity.

Qwen2.5-7B-Instruct (revision pinned in REVISION), greedy decoding, chat template, four made-up few-shot examples
that come from no stream. The model sees the question and the entity's Wikidata label and description, nothing else
(no subject field, model outputs, risk rankings, hidden questions or intervention results). Its answer must be a
substring of the question (exact, else case-insensitive); the last occurrence gives the span, as the writer's rfind
would. Otherwise B gives no span. Raw outputs are kept.

  python annotator_b.py MANIFEST.json LABELS.json MODEL_DIR OUT.json [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
SYSTEM = 'You are a careful annotator. You copy text exactly as it is written.'
TASK = ('Entity: {label}\nDescription: {desc}\nQuestion: {question}\n\n'
        'Copy, character for character, the part of the question that names this entity. Leave out quotation marks '
        'around the name. If the question does not name this entity, write NONE. Write only the copied text.')
SHOTS = [
    (dict(label='Eiffel Tower', desc='tower in Paris, France',
          question='Which engineering firm built the Eiffel Tower in Paris?'), 'Eiffel Tower'),
    (dict(label='Blue Morning', desc='song by the Harbour Lights',
          question='Which album includes the song "Blue Morning" by the Harbour Lights?'), 'Blue Morning'),
    (dict(label='Lake Constance', desc='lake in Central Europe',
          question='What is the population of Zurich?'), 'NONE'),
    (dict(label='Marie Curie', desc='Polish-French physicist and chemist (1867-1934)',
          question='Where was the physicist Marie Curie born?'), 'Marie Curie'),
]


def pick(d: dict, question: str | None = None) -> str:
    d = d or {}
    if question:
        ql = question.lower()
        for v in d.values():
            if v and v.lower() in ql:
                return v
    for k in ('en', 'mul'):
        if d.get(k):
            return d[k]
    return next(iter(d.values()), '')


def messages(label: str, desc: str, question: str) -> list[dict]:
    msgs = [dict(role='system', content=SYSTEM)]
    for ex, ans in SHOTS:
        msgs.append(dict(role='user', content=TASK.format(**ex)))
        msgs.append(dict(role='assistant', content=ans))
    msgs.append(dict(role='user', content=TASK.format(label=label or '(no label)', desc=desc or '(no description)',
                                                      question=question)))
    return msgs


def to_span(question: str, answer: str):
    a = answer.strip().strip('"“”').strip()
    if not a or a.upper() == 'NONE':
        return None, 'none'
    k = question.rfind(a)
    if k >= 0:
        return [k, k + len(a)], 'exact'
    k = question.lower().rfind(a.lower())
    if k >= 0:
        return [k, k + len(a)], 'case'
    return None, 'not a substring'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest')
    ap.add_argument('labels')
    ap.add_argument('model_dir')
    ap.add_argument('out')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--batch', type=int, default=48)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = json.loads(Path(args.manifest).read_text(encoding='utf-8'))['rows'][:args.limit]
    ents = json.loads(Path(args.labels).read_text(encoding='utf-8'))['entities']
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.bfloat16).to('cuda')
    model.eval()
    prompts = []
    for r in rows:
        ev = ents.get(r['subject_id']) or {}
        label, desc = pick(ev.get('labels'), r['question']), pick(ev.get('descriptions'))
        text = tok.apply_chat_template(messages(label, desc, r['question']), tokenize=False, add_generation_prompt=True)
        prompts.append((r['uid'], r['question'], label, desc, text))
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i][4]))
    raw = {}
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(order), args.batch):
            part = [prompts[i] for i in order[s:s + args.batch]]
            enc = tok([p[4] for p in part], return_tensors='pt', padding=True).to('cuda')
            out = model.generate(**enc, max_new_tokens=48, do_sample=False, temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.eos_token_id)
            for p, o in zip(part, out[:, enc['input_ids'].shape[1]:]):
                raw[p[0]] = tok.decode(o, skip_special_tokens=True).split('\n', 1)[0]
            if (s // args.batch) % 20 == 0:
                print(f'{s + len(part)}/{len(order)} {time.time() - t0:.0f} s', flush=True)
    spans, kinds = {}, {}
    for uid, q, label, desc, text in prompts:
        span, kind = to_span(q, raw[uid])
        kinds[kind] = kinds.get(kind, 0) + 1
        spans[uid] = dict(span=span, text=q[span[0]:span[1]] if span else None, kind=kind, raw=raw[uid],
                          label=label, description=desc)
    meta = dict(annotator='B (Qwen2.5-7B-Instruct, greedy)', revision=REVISION, kinds=kinds, n=len(spans),
                prompt_sha256=hashlib.sha256(''.join(p[4] for p in prompts).encode()).hexdigest(),
                seconds=time.time() - t0, torch=torch.__version__)
    Path(args.out).write_text(json.dumps(dict(meta, spans=spans), ensure_ascii=False, indent=0), encoding='utf-8',
                              newline='\n')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
