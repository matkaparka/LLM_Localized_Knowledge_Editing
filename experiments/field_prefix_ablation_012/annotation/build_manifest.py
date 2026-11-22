"""FP-12 annotation manifest: every edit of KT-10's D0-D7 (cap 150 per step) and of the smoke stream P0 (cap 12),
with the question the writer sees, the subject field and where RV-09's lookup puts the key today ('orig').

  python build_manifest.py OUT.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fp_data  # noqa: E402
from rv_policy import cap_stream  # noqa: E402
from so_data import load_edits  # noqa: E402

SMOKE = ('P0', 12)


def main(out_path: str):
    tok = fp_data.load_tokenizer()
    edits, windows, _ = load_edits(fp_data.data_dir(), expect_steps=8)
    streams, _, _ = fp_data.kt10_half(edits, len(windows))
    chosen = {n: cap_stream(s, fp_data.KT10_CAP) for n, s in streams.items() if n.startswith('D')}
    chosen[SMOKE[0]] = cap_stream(streams[SMOKE[0]], SMOKE[1])
    rows = []
    for name in sorted(chosen):
        for e in chosen[name].edits:
            q = (e.update or '').strip()
            a = fp_data.orig_anchor(tok, e.update, e.subject)
            text = fp_data.bare_prompt(e.update)
            rows.append(dict(uid=e.uid, stream=name, step=e.step, question=q, subject=e.subject,
                             subject_id=e.subject_id, relation=e.relation, relation_id=e.relation_id,
                             orig=dict(char=a.char, token=a.token, found=a.found,
                                       token_text=tok.decode([tok(text, add_special_tokens=False)['input_ids'][a.token]]))))
    body = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    out = dict(n=len(rows), per_stream={n: len(s.edits) for n, s in sorted(chosen.items())},
               smoke=dict(stream=SMOKE[0], cap=SMOKE[1]), sha256_rows=hashlib.sha256(body.encode()).hexdigest(),
               rows=rows)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding='utf-8', newline='\n')
    print(out['n'], out['per_stream'], out['sha256_rows'][:16])


if __name__ == '__main__':
    main(sys.argv[1])
