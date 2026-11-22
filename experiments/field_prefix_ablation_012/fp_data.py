"""FP-12 data: KT-10's D0-D7 streams rebuilt unchanged, and where an edit's key sits (its anchor).

Streams. The SO-08 builder on the dev half with KT-10's settings (config_run_001.json: 2 pilot + 8 dev streams,
1,600 edits, 2,000 public probes, 6,000 hidden candidates, 2 format shots), through EC-11's prepare_half, which
reproduces KT-10's streams and RV-09's routing-centre corpus on the dev half (checked in EC-11's data survey). The
per-step cap of 150 is applied by the caller (rv_policy.cap_stream), as in KT-10.

Anchors. RV-09's writer keys an entry on the token holding the last character of the last occurrence of the subject
field in the question part of the prompt (rv_editor.lookup; exact case, then case-insensitive; without a match the
question's last non-space character). FP-12's corrected row keys it on the token holding the last character of an
annotated character span of the entity in the question. Spans are offsets into question.strip(), the text that
so_model.fmt_prompt puts after 'Q: '. Both anchors are reported as token indices in the bare write prompt
fmt_prompt(question) = 'Q: <question>\\nA:'.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for d in (EXP / 'editor_compare_011', EXP / 'key_thresholds_010', EXP / 'reversal_pilot_009',
          EXP / 'self_observation_008'):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import ec_data  # noqa: E402
from rv_policy import cap_stream  # noqa: E402
from so_data import load_edits  # noqa: E402
from so_model import fmt_prompt  # noqa: E402

DATA = Path('E:/research/pause2/data/wikibigedit')
MODEL = Path('E:/research/pause2/models/qwen3_0.6b_base')
KT10_ARGS = dict(n_pilot=2, n_dev=8, edits_per_stream=1600, loc_probe=2000, loc_hidden=6000, n_shots=2)
KT10_CAP = 150
MU = dict(buckets=[6, 7, 8, 9], n=20000)
Q_PREFIX = 'Q: '


def data_dir() -> Path:
    return Path(os.environ.get('RV_DATA_DIR') or DATA)


def kt10_half(edits, steps: int, args: dict | None = None, mu: dict | None = None):
    """(uncapped dev-half streams by name, routing-centre corpus, filter report) exactly as KT-10 had them (other
    settings only in the CPU tests)."""
    args, mu = dict(args or KT10_ARGS), dict(mu or MU)
    streams, corpus, filters = ec_data.prepare_half(edits, steps, 'dev', args['loc_hidden'], mu['buckets'], mu['n'],
                                                    stream_args=args, filter_other_half=False)
    return streams, corpus, filters


def kt10_streams(path=None) -> dict:
    """D0-D7, capped at 150 edits per step (KT-10's streams)."""
    edits, windows, _ = load_edits(path or data_dir(), expect_steps=8)
    streams, _, _ = kt10_half(edits, len(windows))
    return {n: cap_stream(s, KT10_CAP) for n, s in streams.items() if n.startswith('D')}


# ---- anchors ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Anchor:
    char: int          # index of the anchor character in the bare prompt
    token: int         # index of the token holding it (rv_editor.lookup's rule)
    found: bool        # orig: the subject field was found; corr: always True


def bare_prompt(question: str) -> str:
    return fmt_prompt(question, '')


def offsets(tok, text: str):
    """Character offsets of tok.encode(text)'s tokens, or None for a tokenizer without them (the CPU tests' character
    tokenizer), as in rv_editor.lookup."""
    try:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    except TypeError:
        return None
    if 'offset_mapping' not in enc:
        return None
    if list(enc['input_ids']) != list(tok.encode(text, add_special_tokens=False)):
        raise RuntimeError('tokenizer call and encode() disagree')
    return [tuple(x) for x in enc['offset_mapping']]


def token_at(tok, text: str, char: int) -> int:
    """rv_editor.lookup's char -> token rule: the token whose offsets contain the character, else the last token
    starting before it; without offsets, the last token of text[:char + 1]."""
    offs = offsets(tok, text)
    if offs is None:
        return len(tok.encode(text[:char + 1], add_special_tokens=False)) - 1
    idx = None
    for i, (a, b) in enumerate(offs):
        if a <= char < b:
            idx = i
    if idx is None:
        idx = max(i for i, (a, _) in enumerate(offs) if a <= char)
    return idx


def orig_char(text: str, subject: str | None, start: int = 0) -> tuple[int, bool]:
    """The character rv_editor.lookup keys on (same code path)."""
    pos = text.rfind(subject, start) if subject else -1
    if pos < 0 and subject:
        pos = text.lower().rfind(subject.lower(), start)
    if pos >= 0:
        return pos + len(subject) - 1, True
    q_end = text.rfind('\nA:')
    body = text[start:q_end] if q_end >= start else text[start:]
    return start + len(body.rstrip()) - 1, False


def orig_anchor(tok, question: str, subject: str | None) -> Anchor:
    text = bare_prompt(question)
    c, found = orig_char(text, subject, 0)
    return Anchor(c, token_at(tok, text, c), found)


def span_anchor(tok, question: str, span: tuple[int, int]) -> Anchor:
    """span = (start, end) character offsets into question.strip(), end exclusive."""
    q = question.strip()
    a, b = span
    if not 0 <= a < b <= len(q):
        raise ValueError(f'span {span} outside a question of {len(q)} characters')
    text = bare_prompt(question)
    if text[len(Q_PREFIX):len(Q_PREFIX) + len(q)] != q:
        raise RuntimeError('prompt layout changed')
    c = len(Q_PREFIX) + b - 1
    return Anchor(c, token_at(tok, text, c), True)


def load_tokenizer(path=None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path or os.environ.get('RV_MODEL_PATH') or MODEL), local_files_only=True)
