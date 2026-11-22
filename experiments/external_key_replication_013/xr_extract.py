"""XR-13 entity-span extractor (GPU, locked in the dev phase): reads only the write question.

A pinned instruct model (OLMo-2-1124-7B-Instruct, greedy) is asked which software package, application or product the
question is about and must copy its name from the question. The answer must be a substring of the question (exact,
else case-insensitive); its last occurrence gives the span, as RV-09's writer would take it. Otherwise, or on NONE,
the edit is refused: its UID and its place in every denominator stay, and its entry can never fire. The extractor
never sees the entity field, an entity id, a gold span, the answer or any hidden question. Gold spans (where the
structured entity name occurs) are used afterwards only to report extraction quality.

  python xr_extract.py QUESTIONS.json MODEL_DIR OUT.json [--batch 48]
QUESTIONS.json: {"items": [{"uid": ..., "question": ...}, ...]}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

SYSTEM = 'You are a careful annotator. You copy text exactly as it is written.'
TASK = ('Question: {question}\n\nWhich software package is this question about? Copy the package\'s own name character '
        'for character from the question, without quotation marks or backticks around it. Do not give the name of the '
        'registry, website or package manager it is published on (for example PyPI, npm, crates.io, CRAN, NuGet, '
        'Packagist, Hackage, Homebrew), and do not give the programming language. If no single package is named, '
        'write NONE. Write only the name.')
SHOTS = [('What is the newest release of the Go module github.com/acme/widget?', 'github.com/acme/widget'),
         ('Which licence does the Ruby gem "sparkle-utils" use on RubyGems?', 'sparkle-utils'),
         ('What is the latest stable version of the fastblob crate available on crates.io?', 'fastblob'),
         ('Can you tell me how big the ocean is?', 'NONE')]


def messages(q: str) -> list[dict]:
    m = [dict(role='system', content=SYSTEM)]
    for sq, sa in SHOTS:
        m += [dict(role='user', content=TASK.format(question=sq)), dict(role='assistant', content=sa)]
    return m + [dict(role='user', content=TASK.format(question=q))]


def to_span(q: str, ans: str):
    a = ans.strip().strip('"“”`\'').strip()
    if not a or a.upper() == 'NONE':
        return None, 'none'
    k = q.rfind(a)
    if k < 0:
        k = q.lower().rfind(a.lower())
        if k < 0:
            return None, 'not a substring'
    return [k, k + len(a)], 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('questions')
    ap.add_argument('model_dir')
    ap.add_argument('out')
    ap.add_argument('--batch', type=int, default=48)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    items = json.loads(Path(a.questions).read_text(encoding='utf-8'))['items']
    tok = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, local_files_only=True, dtype=torch.bfloat16).to('cuda')
    model.eval()
    texts = [tok.apply_chat_template(messages(i['question'].strip()), tokenize=False, add_generation_prompt=True)
             for i in items]
    order = sorted(range(len(items)), key=lambda k: len(texts[k]))
    raw, t0 = {}, time.time()
    with torch.no_grad():
        for s in range(0, len(order), a.batch):
            part = order[s:s + a.batch]
            enc = tok([texts[k] for k in part], return_tensors='pt', padding=True).to('cuda')
            gen = model.generate(**enc, max_new_tokens=40, do_sample=False, temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id)
            for k, o in zip(part, gen[:, enc['input_ids'].shape[1]:]):
                raw[items[k]['uid']] = tok.decode(o, skip_special_tokens=True).split('\n', 1)[0]
    spans, kinds = {}, {}
    for i in items:
        q = i['question'].strip()
        span, kind = to_span(q, raw[i['uid']])
        kinds[kind] = kinds.get(kind, 0) + 1
        spans[i['uid']] = dict(span=span, text=q[span[0]:span[1]] if span else None, kind=kind, raw=raw[i['uid']])
    meta = dict(extractor='OLMo-2-1124-7B-Instruct, greedy', kinds=kinds, n=len(items), seconds=time.time() - t0,
                prompt_sha256=hashlib.sha256(''.join(texts).encode()).hexdigest())
    Path(a.out).write_text(json.dumps(dict(meta, spans=spans), ensure_ascii=False), encoding='utf-8')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
