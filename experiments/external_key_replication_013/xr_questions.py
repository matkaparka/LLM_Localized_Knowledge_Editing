"""XR-13 question writing (GPU): two different instruct models write the questions from structured facts.

  writer W (Qwen2.5-7B-Instruct, pinned): the write question of every edit; the questions of the stable old facts
           (hidden pool); unlabelled questions about spare entities (public probes per stream, routing-centre corpus)
  writer R (OLMo-2-1124-7B-Instruct, pinned): two further wordings of every edit (a rephrasing for Q and the coverage,
           a second one for R), from the structured fact only; R never sees W's sentence

Both decode greedily. Variety comes from a style hint chosen by a fixed hash per item (not from sampling), so the
output is reproducible. Every question must name the entity exactly and its ecosystem; answers never appear in a
prompt to W for unlabelled questions, and W is told not to state the answer. Outputs that break the format, contain
the answer, or do not contain the entity's name are dropped and reported, never repaired by hand.

  python xr_questions.py plan EVENTS_DIR OUT.json            all prompts (no model)
  python xr_questions.py write PLAN.json MODEL_DIR WRITER OUT.json [--batch 48]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

ECOSYSTEM = {
    'pypi': ('PyPI', 'Python package', 'Python'), 'npm': ('npm', 'npm package', 'Node.js'),
    'crates': ('crates.io', 'Rust crate', 'Rust'), 'cran': ('CRAN', 'R package', 'R'),
    'nuget': ('NuGet', 'NuGet package', None), 'packagist': ('Packagist', 'PHP package', 'PHP'),
    'hackage': ('Hackage', 'Haskell package', None), 'winget': ('winget', 'Windows application', None),
    'homebrew': ('Homebrew', 'Homebrew formula', None), 'endoflife': ('endoflife.date', 'software product', None)}
STYLES = ['a plain, direct question', 'a casual question a developer might type', 'a precise, formal question',
          'a short question', 'a question that starts with "Which" or "What"', 'a question that mentions why you need it']
ATTR = {'latest version': 'its latest stable version number',
        'runtime requirement': 'the minimum {runtime} version its latest release requires',
        'licence': 'the licence it is distributed under',
        'repository owner': 'the account or organisation that owns its source code repository',
        'publisher': 'the publisher of the package'}
SYSTEM = 'You write questions for a quiz about software packages. You follow the format exactly.'


def hsh(*p) -> int:
    return int(hashlib.sha256('|'.join(map(str, p)).encode()).hexdigest()[:8], 16)


def describe(family: str, entity: str, attribute: str) -> str:
    eco, kind, runtime = ECOSYSTEM[family]
    want = ATTR[attribute].format(runtime=runtime or 'runtime')
    return f'the {kind} "{entity}" on {eco}', want


def w_prompt(family, entity, attribute, key) -> str:
    what, want = describe(family, entity, attribute)
    style = STYLES[hsh('W', key) % len(STYLES)]
    return (f'Write {style} about {what}. It should ask for {want}.\n'
            f'The question must contain the exact name {entity} and must make clear that it is on '
            f'{ECOSYSTEM[family][0]}. Do not answer it. Write only the question, on one line, ending with "?".')


def r_prompt(family, entity, attribute, key) -> str:
    what, want = describe(family, entity, attribute)
    return (f'Fact to ask about: {what}, and {want}.\n'
            'Write two questions that ask for exactly this fact, worded differently from each other and from a '
            'plain template: for example change the sentence structure, the verb, or add a reason for asking. Each '
            f'question must still identify {entity} on {ECOSYSTEM[family][0]} unambiguously. Do not answer. '
            'Write the two questions on two lines, numbered "1." and "2.", nothing else.')


def plan(events_dir: Path, out: Path, n_probes: int = 2000, n_mu: int = 20000, n_hidden: int = 4000):
    items = []
    spare_all = []
    for p in sorted(events_dir.glob('*_events.json')):
        ev = json.loads(p.read_text(encoding='utf-8'))
        fam = ev['family']
        stream = p.name[:-len('_events.json')]
        for b in ev['batches']:
            for e in b['edits']:
                k = e['uid']
                items.append(dict(id=f'{k}|w', role='write', writer='W', family=fam, stream=stream, uid=k,
                                  prompt=w_prompt(fam, e['entity'], e['attribute'], k), entity=e['entity'],
                                  answer=e['value']))
                items.append(dict(id=f'{k}|r', role='rephrase', writer='R', family=fam, stream=stream, uid=k,
                                  prompt=r_prompt(fam, e['entity'], e['attribute'], k), entity=e['entity'],
                                  answer=e['value']))
        ranked = sorted(ev['stable'], key=lambda f: (f['rank'] or 0, f['attribute']))
        stable = ranked[:n_hidden]
        spare_all += [(fam, f['entity']) for f in ranked[n_hidden:]]   # beyond the hidden cap: corpus material
        for f in stable:
            k = f'{stream}:hidden:{f["entity"]}'
            items.append(dict(id=k, role='hidden', writer='W', family=fam, stream=stream,
                              prompt=w_prompt(fam, f['entity'], f['attribute'], k),
                              entity=f['entity'], answer=f['value'], attribute=f['attribute']))
        spares = sorted(ev['spares'], key=lambda s: hsh('spare', stream, s['entity']))
        for s in spares[:n_probes]:
            attr = ['latest version', 'licence'][hsh('pattr', s['entity']) % 2]
            k = f'{stream}:probe:{s["entity"]}'
            items.append(dict(id=k, role='probe', writer='W', family=fam, stream=stream,
                              prompt=w_prompt(fam, s['entity'], attr, k), entity=s['entity']))
        spare_all += [(fam, s['entity']) for s in spares[n_probes:]]
    spare_all = sorted(set(spare_all), key=lambda x: hsh('mu', *x))
    for fam, ent in spare_all[:n_mu]:
        attr = ['latest version', 'licence'][hsh('mattr', ent) % 2]
        k = f'{fam}:mu:{ent}'
        items.append(dict(id=k, role='mu', writer='W', family=fam, prompt=w_prompt(fam, ent, attr, k), entity=ent))
    out.write_text(json.dumps(dict(n=len(items), items=items), ensure_ascii=False), encoding='utf-8')
    print({r: sum(1 for i in items if i['role'] == r) for r in ('write', 'rephrase', 'hidden', 'probe', 'mu')})


def clean_question(text: str, entity: str, answer: str | None) -> tuple[str | None, str]:
    q = text.strip().strip('"').strip()
    q = re.sub(r'^\s*(\d+[.)]|Q:|Question:)\s*', '', q).strip().strip('"').strip()
    if not q.endswith('?'):
        return None, 'no question mark'
    if entity not in q:
        return None, 'entity name missing'
    if answer and len(answer) >= 2 and re.search(rf'(?<![\w.]){re.escape(answer)}(?![\w.])', q):
        return None, 'states the answer'
    if '\n' in q or len(q) > 300:
        return None, 'format'
    return q, 'ok'


def write(plan_path: Path, model_dir: str, writer: str, out: Path, batch: int = 48):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    items = [i for i in json.loads(plan_path.read_text(encoding='utf-8'))['items'] if i['writer'] == writer]
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16).to('cuda')
    model.eval()
    texts = [tok.apply_chat_template([dict(role='system', content=SYSTEM), dict(role='user', content=i['prompt'])],
                                     tokenize=False, add_generation_prompt=True) for i in items]
    order = sorted(range(len(items)), key=lambda k: len(texts[k]))
    raw = {}
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(order), batch):
            part = order[s:s + batch]
            enc = tok([texts[k] for k in part], return_tensors='pt', padding=True).to('cuda')
            gen = model.generate(**enc, max_new_tokens=96 if writer == 'R' else 64, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
            for k, o in zip(part, gen[:, enc['input_ids'].shape[1]:]):
                raw[items[k]['id']] = tok.decode(o, skip_special_tokens=True)
            if (s // batch) % 50 == 0:
                print(f'{writer}: {s + len(part)}/{len(order)} {time.time() - t0:.0f} s', flush=True)
    res, why = {}, {}
    for i in items:
        r = raw[i['id']]
        if writer == 'R':
            lines = [x for x in r.splitlines() if x.strip()]
            qs = [clean_question(x, i['entity'], i.get('answer')) for x in lines[:2]]
            res[i['id']] = [q for q, _ in qs]
            for _, w in qs:
                why[w] = why.get(w, 0) + 1
        else:
            q, w = clean_question(r.split('\n', 1)[0], i['entity'], i.get('answer'))
            res[i['id']] = q
            why[w] = why.get(w, 0) + 1
    out.write_text(json.dumps(dict(writer=writer, model_dir=model_dir, n=len(items), outcome=why, seconds=time.time() - t0,
                                   questions=res, raw=raw), ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(writer=writer, n=len(items), outcome=why)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('plan', 'write'))
    ap.add_argument('args', nargs='+')
    ap.add_argument('--batch', type=int, default=48)
    a = ap.parse_args(argv)
    if a.command == 'plan':
        plan(Path(a.args[0]), Path(a.args[1]))
    else:
        write(Path(a.args[0]), a.args[1], a.args[2], Path(a.args[3]), a.batch)


if __name__ == '__main__':
    main()
