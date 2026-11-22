"""EC-11 data: stream preparation for either relation half, the leakage filters and the public subject-hygiene rule.
No model calls.

The SO-08 builder is reused unchanged. Development streams are the dev half's D0-D7 exactly as RV-09/KT-10 built them;
test streams are the `confirm` half's, renamed CP0, CP1, C0-C7 before anything else touches them (names feed every
hash-derived choice: the per-step cap, panels and write seeds).

Locality pools are enlarged to `loc_hidden_eval` candidates (the builder's order is kept, so the first 6,000 are the
RV-09/KT-10 pool). The routing centre's corpus follows RV-09's rule on the 6,000-candidate pools, so it is RV-09's
own corpus on the dev half.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for d in (EXP / 'self_observation_008', EXP / 'reversal_pilot_009'):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from rv_policy import corpus, pool_texts, spare_rows  # noqa: E402
from so_data import build_streams, norm, split_halves  # noqa: E402

RENAME = {'P0': 'CP0', 'P1': 'CP1', **{f'D{i}': f'C{i}' for i in range(8)}}

# The RV-09 / KT-10 stream settings (config_run_001.json of both).
STREAM_ARGS = dict(n_pilot=2, n_dev=8, edits_per_stream=1600, loc_probe=2000, loc_hidden=6000, n_shots=2)


def confirm_streams(edits, steps: int, **over) -> dict:
    """The ten SO-08-style streams of the `confirm` half, renamed CP0, CP1, C0-C7."""
    out = {}
    for s in build_streams(edits, steps, half='confirm', **dict(STREAM_ARGS, **over)):
        out[RENAME[s.name]] = dataclasses.replace(s, name=RENAME[s.name])
    return out


def _edit_texts(e):
    return [norm(t) for t in (e.update, e.rephrase, e.personas) if t]


def prepare_half(edits, steps: int, half: str, loc_hidden_eval: int, mu_buckets, mu_n: int,
                 stream_args: dict | None = None, filter_other_half: bool | None = None):
    """Streams of one half for evaluation, the routing-centre corpus, and what the filters removed.

    - Streams: the builder with `stream_args` (default STREAM_ARGS) but loc_hidden = loc_hidden_eval; confirm-half
      streams renamed.
    - Corpus: rv_policy.corpus over the half's spare rows, computed against the streams built with the base
      loc_hidden (RV-09's rule; on the dev half this is RV-09's corpus).
    - Filters: hidden candidates whose text is in the corpus are dropped (both halves; only the enlargement can hit
      them). With filter_other_half (default: only for the confirm half), probe questions and hidden candidates whose
      text occurs in any stream or pool of the other half are dropped, and so are edits whose update, rephrase or
      persona text does.
    """
    args = dict(STREAM_ARGS if stream_args is None else stream_args)
    base = build_streams(edits, steps, half=half, **args)
    halves = split_halves(edits)
    mu_corpus = corpus(spare_rows(edits, base, halves, half), set(mu_buckets), int(mu_n))
    ext = build_streams(edits, steps, half=half, **dict(args, loc_hidden=int(loc_hidden_eval)))
    if filter_other_half is None:
        filter_other_half = half == 'confirm'
    other = 'confirm' if half == 'dev' else 'dev'
    other_texts = (pool_texts(build_streams(edits, steps, half=other, **dict(args, loc_hidden=int(loc_hidden_eval))))
                   if filter_other_half else set())
    corpus_texts = {norm(t[len('Q: '):-len('\nA:')]) for t in mu_corpus}
    streams, info = {}, {}
    for s in ext:
        name = RENAME[s.name] if half == 'confirm' else s.name
        hidden = [(q, a) for q, a in s.loc_hidden if norm(q) not in corpus_texts]
        n_corpus = len(s.loc_hidden) - len(hidden)
        probes, edits_kept = list(s.loc_probe), list(s.edits)
        probe_rel = list(s.probe_rel)
        n_probe = n_hidden = n_edit = 0
        if filter_other_half:
            keep = [i for i, q in enumerate(probes) if norm(q) not in other_texts]
            n_probe = len(probes) - len(keep)
            probes, probe_rel = [probes[i] for i in keep], [probe_rel[i] for i in keep] if probe_rel else []
            before = len(hidden)
            hidden = [(q, a) for q, a in hidden if norm(q) not in other_texts]
            n_hidden = before - len(hidden)
            edits_kept = [e for e in edits_kept if not any(t in other_texts for t in _edit_texts(e))]
            n_edit = len(s.edits) - len(edits_kept)
        streams[name] = dataclasses.replace(s, name=name, edits=tuple(edits_kept), loc_probe=tuple(probes),
                                            loc_hidden=tuple(hidden), probe_rel=tuple(probe_rel))
        info[name] = dict(hidden_in_corpus=n_corpus, probes_other_half=n_probe, hidden_other_half=n_hidden,
                          edits_other_half=n_edit, probes=len(probes), hidden=len(hidden), edits=len(edits_kept))
    return streams, mu_corpus, info


# ---- subject hygiene ---------------------------------------------------------------------------------------
# Public: reads only the edit record (subject field and update question). Fixed on dev-half streams.
FUNCTION = frozenset('a an the of in on at for to by from with and or as into onto about after before during since '
                     'until over under via per than'.split())
QUESTION = frozenset('what who whom whose which when where why how is are was were does do did has have had can '
                     'could will would'.split())
RELATION = (' located in ', ' part of ', ' example of ', ' instance of ', ' member of ', ' occurred on ',
            ' occurred in ', ' is a ', ' is the ', ' was a ', ' was the ', ' known as ')


def hygiene_flags(subject: str | None, question: str) -> list[str]:
    """Reasons the subject field looks malformed (empty list when it looks like an entity label).

    - short_or_function: empty, at most two characters, or only function/question words ('What', 'In');
    - lead_function: starts with a lower-case function word ('of solar eclipse occurred on ...');
    - trail_function: ends with a lower-case function word ('... an example of'; not titles like 'Roll On');
    - relation_phrase: contains a lower-case relation phrase ('... located in Whitsome as part of ...'; not
      titles like 'Time Is a Machine');
    - not_in_question: does not occur in the update question, so the writer keys the question's end instead
      (reported only; not part of the rule).
    """
    s = (subject or '').strip()
    words = [w.strip(',.;:!?()"\'') for w in s.split()]
    words = [w for w in words if w]
    low = [w.casefold() for w in words]
    flags = []
    if len(s) <= 2 or not low or all(w in FUNCTION or w in QUESTION for w in low):
        flags.append('short_or_function')
    if words and words[0] in FUNCTION:
        flags.append('lead_function')
    if words and words[-1] in FUNCTION:
        flags.append('trail_function')
    if any(r in f' {s} ' for r in RELATION):
        flags.append('relation_phrase')
    if s and s.casefold() not in (question or '').casefold():
        flags.append('not_in_question')
    return flags


HYGIENE_RULE = ('short_or_function', 'lead_function', 'trail_function', 'relation_phrase')


def malformed(subject: str | None, question: str, rule=HYGIENE_RULE) -> bool:
    return any(f in rule for f in hygiene_flags(subject, question))
