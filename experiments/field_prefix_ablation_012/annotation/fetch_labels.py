"""FP-12 annotation evidence: Wikidata labels, aliases and descriptions of every subject entity of KT-10's D0-D7.

Source evidence for the span annotators (PLAN section 3). Labels are today's Wikidata revisions (recorded per entity
as lastrevid / modified), not the 2024 dumps WikiBigEdit was built from; they only help to find the entity's mention
in the fixed question text. Nothing else (answers, rephrasings, hidden pools) is sent anywhere.

  python fetch_labels.py OUT.json
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fp_data  # noqa: E402

API = 'https://www.wikidata.org/w/api.php'
LANGS = 'en|mul|de|fr|es|it|nl|pl|pt|sv|da|nb|fi|cs|hu|ro|ca|tr|ru|ja|zh'
UA = 'research-annotation-check/0.1 (span evidence for a knowledge-editing study)'


def fetch(ids: list[str]) -> dict:
    q = urllib.parse.urlencode(dict(action='wbgetentities', ids='|'.join(ids), props='labels|aliases|descriptions|info',
                                    languages=LANGS, languagefallback=1, format='json'))
    req = urllib.request.Request(f'{API}?{q}', headers={'User-Agent': UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode('utf-8'))
        except Exception as e:  # network hiccup: back off and retry
            if attempt == 4:
                raise
            print('retry', attempt, repr(e)[:120], flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError('unreachable')


def main(out_path: str, manifest: str | None = None):
    """Without a manifest: every subject of D0-D7. With one: only its subjects missing from OUT, merged into OUT."""
    if manifest:
        out = json.loads(Path(out_path).read_text(encoding='utf-8'))
        rows = json.loads(Path(manifest).read_text(encoding='utf-8'))['rows']
        ids = sorted({r['subject_id'] for r in rows} - set(out['entities']))
        out.setdefault('merged', []).append(dict(at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), n=len(ids)))
        out['n_ids'] = out['n_ids'] + len(ids)
    else:
        streams = fp_data.kt10_streams()
        ids = sorted({e.subject_id for s in streams.values() for e in s.edits})
        out = dict(source=API, params=dict(props='labels|aliases|descriptions|info', languages=LANGS,
                                           languagefallback=1),
                   fetched_from=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), n_ids=len(ids), entities={})
    for s in range(0, len(ids), 50):
        part = ids[s:s + 50]
        d = fetch(part)
        for qid in part:
            ent = d.get('entities', {}).get(qid, {})
            out['entities'][qid] = dict(
                lastrevid=ent.get('lastrevid'), modified=ent.get('modified'), missing='missing' in ent,
                redirect=ent.get('redirects'),
                labels={k: v['value'] for k, v in (ent.get('labels') or {}).items()},
                aliases={k: [a['value'] for a in v] for k, v in (ent.get('aliases') or {}).items()},
                descriptions={k: v['value'] for k, v in (ent.get('descriptions') or {}).items()})
        if (s // 50) % 20 == 0:
            print(f'{s + len(part)}/{len(ids)}', flush=True)
        time.sleep(0.3)
    out['fetched_until'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding='utf-8', newline='\n')
    print('done', len(out['entities']))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
