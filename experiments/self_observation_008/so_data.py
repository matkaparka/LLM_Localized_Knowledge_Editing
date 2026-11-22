"""SO-08 data layer: WikiBigEdit dated files -> fact index, entity-grouped streams and disjoint pools.

No model calls. Field names follow the Hub schema (all strings, nullable):
tag, subject, subject_id, relation, relation_id, object, object_id, rephrase, loc, loc_ans,
mhop, mhop_ans, update, personas, ans.
Time comes only from the eight file names (wiki_big_edit_YYYYMMDD_YYYYMMDD.json); rows carry no
timestamps and no old object values.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

FILE_RE = re.compile(r'^wiki_big_edit_(\d{8})_(\d{8})\.json$')
REQUIRED = ('subject_id', 'relation_id', 'update')
_STRIP = ' \t\n.,;:!?"\'`“”‘’()[]'


def norm(text) -> str:
    """Matching key: NFKC, casefold, collapsed spaces, outer punctuation and quotes removed."""
    if text is None:
        return ''
    t = unicodedata.normalize('NFKC', str(text)).casefold()
    return ' '.join(t.split()).strip(_STRIP)


def h(*parts) -> int:
    """Stable 64-bit hash for deterministic ordering and assignment."""
    d = hashlib.sha256('\x1f'.join(map(str, parts)).encode('utf-8')).digest()
    return int.from_bytes(d[:8], 'big')


@dataclass(frozen=True)
class Edit:
    uid: str
    step: int
    tag: str | None
    subject: str
    subject_id: str
    relation: str
    relation_id: str
    answer: str
    object_id: str | None
    update: str
    rephrase: str | None
    personas: str | None
    loc: str | None
    loc_ans: str | None
    mhop: str | None
    mhop_ans: str | None

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject_id, self.relation_id)


def read_rows(path) -> list[dict]:
    """A file may be a JSON array or JSON lines."""
    text = Path(path).read_text(encoding='utf-8')
    body = text.lstrip()
    rows = json.loads(body) if body.startswith('[') else [json.loads(x) for x in text.splitlines() if x.strip()]
    if not all(isinstance(r, dict) for r in rows):
        raise ValueError(f'{path}: every row must be a JSON object')
    return rows


def find_steps(data_dir) -> list[tuple[str, str, Path]]:
    found = sorted((m.group(1), m.group(2), p) for p in Path(data_dir).iterdir() if (m := FILE_RE.match(p.name)))
    for (a0, a1, _), (b0, b1, _) in zip(found, found[1:]):
        if a1 != b0:
            raise ValueError(f'dated windows are not contiguous: {a0}_{a1} then {b0}_{b1}')
    return found


def _s(v):
    """A field value as a non-blank string, else None. The released files encode missing values as null, as an
    empty string, or as JSON NaN (a float)."""
    return v if isinstance(v, str) and v.strip() else None


def load_edits(data_dir, expect_steps: int | None = 8):
    """Edits in file order, the date windows, and rows skipped per step for missing essentials."""
    steps = find_steps(data_dir)
    if expect_steps is not None and len(steps) != expect_steps:
        raise ValueError(f'expected {expect_steps} dated files, found {len(steps)}')
    edits, skipped = [], Counter()
    for i, (_, _, path) in enumerate(steps):
        for j, raw in enumerate(read_rows(path)):
            r = {k: _s(v) for k, v in raw.items()}
            answer = r.get('ans') or r.get('object')
            if any(not r.get(k) for k in REQUIRED) or not answer:
                skipped[i] += 1
                continue
            digest = hashlib.sha1(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:12]
            edits.append(Edit(f'{i}:{j}:{digest}', i, r.get('tag'), r.get('subject') or '', r['subject_id'],
                              r.get('relation') or '', r['relation_id'], answer, r.get('object_id'), r['update'],
                              r.get('rephrase'), r.get('personas'), r.get('loc'), r.get('loc_ans'), r.get('mhop'),
                              r.get('mhop_ans')))
    return edits, [(a, b) for a, b, _ in steps], dict(skipped)


class FactIndex:
    """Truth over the whole benchmark, per (subject_id, relation_id).

    A key is multi-valued when one step lists two different normalised answers for it; all answers seen so far
    then stay valid. Otherwise a later different answer supersedes the earlier one from its step on. This is a
    heuristic: rows carry no old values and no relation cardinality.
    """

    def __init__(self, edits):
        self.by_key: dict[tuple[str, str], list[Edit]] = defaultdict(list)
        for e in edits:
            self.by_key[e.key].append(e)
        self.multi = set()
        for k, es in self.by_key.items():
            es.sort(key=lambda e: (e.step, e.uid))
            per_step = defaultdict(set)
            for e in es:
                per_step[e.step].add(norm(e.answer))
            if any(len(v) > 1 for v in per_step.values()):
                self.multi.add(k)

    def valid(self, key, step) -> set[str]:
        """Normalised answers in force at `step`; empty if the key has no edit up to then."""
        es = [e for e in self.by_key.get(key, ()) if e.step <= step]
        if not es:
            return set()
        if key in self.multi:
            return {norm(e.answer) for e in es}
        last = max(e.step for e in es)
        return {norm(e.answer) for e in es if e.step == last}

    def superseded(self, edit: Edit, step: int) -> bool:
        return norm(edit.answer) not in self.valid(edit.key, step)

    def repeat(self, edit: Edit) -> bool:
        """The key already had an edit at an earlier step of the benchmark."""
        return any(e.step < edit.step for e in self.by_key[edit.key])

    def category(self, edit: Edit) -> str:
        if self.repeat(edit):
            return 'repeat'
        return (edit.tag or 'untagged').lower()


def split_halves(edits) -> dict[str, str]:
    """Relation families to 'dev' or 'confirm', balanced by row count, deterministic."""
    counts = Counter(e.relation_id for e in edits)
    sizes, half = [0, 0], {}
    for r in sorted(counts, key=lambda r: (-counts[r], h('so08-rel', r))):
        i = 0 if sizes[0] <= sizes[1] else 1
        half[r] = ('dev', 'confirm')[i]
        sizes[i] += counts[r]
    return half


@dataclass(frozen=True)
class Stream:
    name: str
    half: str
    steps: int
    edits: tuple[Edit, ...]
    loc_probe: tuple[str, ...]
    loc_hidden: tuple[tuple[str, str], ...]
    shots: tuple[tuple[str, str], ...]
    probe_rel: tuple[str, ...] = ()  # relation id of each probe question (public: readable from its wording)

    def at(self, step: int) -> list[Edit]:
        return [e for e in self.edits if e.step == step]


def build_streams(edits, steps: int, n_pilot=2, n_dev=8, edits_per_stream=1600, loc_probe=128, loc_hidden=256,
                  n_shots=2, half='dev'):
    """Entity-grouped streams in one half. All edits of a subject, hence of each fact key, stay in one stream.

    Locality pools come from rows of the same half that no stream uses; they are deduplicated by normalised
    question and kept apart from every update, rephrase and persona text of the half.
    """
    halves = split_halves(edits)
    pool = [e for e in edits if halves[e.relation_id] == half]
    n = n_pilot + n_dev
    names = [f'P{i}' for i in range(n_pilot)] + [f'D{i}' for i in range(n_dev)]
    keys_by_bucket = defaultdict(lambda: defaultdict(list))
    for e in pool:
        keys_by_bucket[h('so08-entity', e.subject_id) % n][e.key].append(e)
    chosen, used = {}, set()
    for b in range(n):
        picked = []
        for k in sorted(keys_by_bucket[b], key=lambda k: h('so08-key', *k)):
            if len(picked) >= edits_per_stream:
                break
            picked += keys_by_bucket[b][k]
        picked.sort(key=lambda e: (e.step, e.uid))
        chosen[b] = picked
        used.update(e.uid for e in picked)
    taken_text = {norm(t) for e in pool if e.uid in used for t in (e.update, e.rephrase, e.personas) if t}
    spare = [e for e in pool if e.uid not in used]
    loc_items, seen = [], set()
    for e in sorted(spare, key=lambda e: h('so08-loc', e.uid)):
        q = norm(e.loc)
        if e.loc and e.loc_ans and q and q not in seen and q not in taken_text:
            seen.add(q)
            loc_items.append((e.loc, e.loc_ans, e.relation_id))  # the pipeline matches loc to the row's relation
    loc_by_bucket = defaultdict(list)
    for q, a, r in loc_items:
        loc_by_bucket[h('so08-locbucket', norm(q)) % n].append((q, a, r))
    loc_texts = {norm(q) for q, _, _ in loc_items}
    shot_pool = [e for e in sorted(spare, key=lambda e: h('so08-shot', e.uid))
                 if norm(e.update) not in loc_texts and norm(e.update) not in taken_text]
    streams = []
    for b, name in enumerate(names):
        locs = loc_by_bucket[b]
        shots = tuple((e.update, e.answer) for e in shot_pool[b * n_shots:(b + 1) * n_shots])
        streams.append(Stream(name, half, steps, tuple(chosen[b]), tuple(q for q, _, _ in locs[:loc_probe]),
                              tuple((q, a) for q, a, _ in locs[loc_probe:loc_probe + loc_hidden]), shots,
                              tuple(r for _, _, r in locs[:loc_probe])))
    return streams


def audit_stream(s: Stream) -> list[str]:
    """Leakage checks; returns a list of problems (empty when clean)."""
    problems = []
    probe = {norm(q) for q in s.loc_probe}
    hidden = {norm(q) for q, _ in s.loc_hidden}
    if probe & hidden:
        problems.append('locality probe and hidden pools overlap')
    texts = {norm(t) for e in s.edits for t in (e.update, e.rephrase, e.personas) if t}
    if (probe | hidden) & texts:
        problems.append('locality questions repeat an edit text')
    shot_q = {norm(q) for q, _ in s.shots}
    if shot_q & (texts | probe | hidden):
        problems.append('format shots repeat a stream or pool question')
    return problems


def dataset_report(edits, index: FactIndex, steps: int) -> dict:
    """Benchmark-wide counts that the P0 thresholds and the stream sizes depend on."""
    halves = split_halves(edits)
    per_step = Counter(e.step for e in edits)
    keys_per_half = defaultdict(set)
    for e in edits:
        keys_per_half[halves[e.relation_id]].add(e.key)
    return dict(
        rows=len(edits), rows_per_step=[per_step.get(t, 0) for t in range(steps)],
        tags=dict(Counter((e.tag or 'untagged') for e in edits).most_common()),
        keys=len(index.by_key), multi_valued_keys=len(index.multi),
        repeat_edits=sum(1 for e in edits if index.repeat(e)),
        keys_with_changed_answer=sum(1 for k, es in index.by_key.items()
                                     if k not in index.multi and len({norm(e.answer) for e in es}) > 1),
        missing=dict(rephrase=sum(1 for e in edits if not e.rephrase), personas=sum(1 for e in edits if not e.personas),
                     loc=sum(1 for e in edits if not (e.loc and e.loc_ans)), mhop=sum(1 for e in edits if not e.mhop)),
        relations=len(halves), half_rows=dict(Counter(halves[e.relation_id] for e in edits)),
        half_keys={k: len(v) for k, v in keys_per_half.items()},
        subjects=len({e.subject_id for e in edits}))


def stream_report(s: Stream, index: FactIndex) -> dict:
    per_step = Counter(e.step for e in s.edits)
    cats = Counter(index.category(e) for e in s.edits)
    return dict(name=s.name, half=s.half, edits=len(s.edits), per_step=[per_step.get(t, 0) for t in range(s.steps)],
                subjects=len({e.subject_id for e in s.edits}), keys=len({e.key for e in s.edits}),
                multi_valued_keys=sum(1 for k in {e.key for e in s.edits} if k in index.multi),
                categories=dict(cats), loc_probe=len(s.loc_probe), loc_hidden=len(s.loc_hidden),
                missing_rephrase=sum(1 for e in s.edits if not e.rephrase),
                missing_personas=sum(1 for e in s.edits if not e.personas))


def shots_text(shots) -> str:
    """Format demonstrations prepended to every generated answer (evaluation, readouts, self-labels), so that
    answer format and knowledge are not confounded. They come from rows no stream or pool uses."""
    return ''.join(f'Q: {q.strip()}\nA: {a.strip()}\n' for q, a in shots)


class PublicView:
    """What a strategy may see when deciding at `step`.

    Labelled: the batch arriving now (edits of step - lag) and earlier arrived edits.
    Unlabelled: locality probe questions. Public format: the stream's format demonstrations.
    Rephrase, persona, multi-hop texts, locality answers and future steps are never stored here.
    """
    __slots__ = ('step', 'lag', '_batch', '_history', '_probes', '_probe_rel', '_shots')

    def __init__(self, stream: Stream, step: int, lag: int = 0):
        self.step, self.lag = step, lag
        src = step - lag
        self._batch = tuple((e.update, e.answer, e.key, e.uid) for e in stream.edits if e.step == src)
        self._history = tuple((e.update, e.answer, e.key, e.uid, e.step) for e in stream.edits if e.step < src)
        self._probes = tuple(stream.loc_probe)
        self._probe_rel = tuple(stream.probe_rel) if stream.probe_rel else ('',) * len(stream.loc_probe)
        self._shots = tuple(stream.shots)

    def batch_relations(self) -> set[str]:
        return {k[1] for _, _, k, _ in self._batch}

    def probe_questions_for(self, relations, k: int) -> list[str]:
        """Up to k probe questions whose relation is in `relations`, taken round-robin over the relations."""
        by_rel = defaultdict(list)
        for q, r in zip(self._probes, self._probe_rel):
            if r in relations:
                by_rel[r].append(q)
        out, order = [], sorted(by_rel, key=lambda r: h('so08-relorder', self.step, r))
        while len(out) < k and any(by_rel[r] for r in order):
            for r in order:
                if by_rel[r] and len(out) < k:
                    out.append(by_rel[r].pop(0))
        return out

    def history_items_for(self, relations, k: int) -> list[tuple[str, str]]:
        """Up to k arrived (question, stored label) pairs of latest_history whose relation is in `relations`."""
        items = sorted((x for x in self.latest_history() if x[2][1] in relations), key=lambda x: h('so08-histr', x[3]))
        return [(q, a) for q, a, *_ in items[:k]]

    def history_questions_for(self, relations, k: int) -> list[str]:
        return [q for q, _ in self.history_items_for(relations, k)]

    def context(self) -> str:
        return shots_text(self._shots)

    def batch(self) -> list[tuple[str, str, tuple[str, str]]]:
        return [(q, a, k) for q, a, k, _ in self._batch]

    def latest_history(self) -> list[tuple[str, str, tuple[str, str], str, int]]:
        """Latest arrived label per key, excluding keys rewritten by the current batch."""
        now = {k for _, _, k, _ in self._batch}
        latest = {}
        for item in self._history:
            if item[2] not in now:
                latest[item[2]] = item
        return list(latest.values())

    def replay(self, k: int, seed: int) -> list[tuple[str, str]]:
        """Up to k labels from latest_history, stratified evenly over past steps."""
        by_step = defaultdict(list)
        for q, a, _, uid, st in self.latest_history():
            by_step[st].append((h('so08-replay', seed, uid), q, a))
        for v in by_step.values():
            v.sort()
        out, order = [], sorted(by_step)
        while len(out) < k and any(by_step[s] for s in order):
            for s in order:
                if by_step[s] and len(out) < k:
                    _, q, a = by_step[s].pop(0)
                    out.append((q, a))
        return out

    def acceptance(self, k: int) -> list[tuple[str, str]]:
        """A fixed public check set for this decision state: same for every candidate."""
        items = sorted(self.latest_history(), key=lambda x: h('so08-accept', self.step, x[3]))
        return [(q, a) for q, a, *_ in items[:k]]

    def cache(self, k: int) -> list[tuple[str, str]]:
        """The k most recently arrived labels (label-cache readout)."""
        items = sorted(self.latest_history(), key=lambda x: (-x[4], h('so08-cache', x[3])))
        return [(q, a) for q, a, *_ in items[:k]]

    def probe_questions(self, k: int) -> list[str]:
        return list(self._probes[:k])

    def history_questions(self, k: int) -> list[str]:
        items = sorted(self.latest_history(), key=lambda x: h('so08-histq', x[3]))
        return [q for q, *_ in items[:k]]


def sample(items, k: int, *salt):
    """Deterministic subsample by hash; keeps input order among the selected."""
    if k is None or len(items) <= k:
        return list(items)
    keep = set(sorted(range(len(items)), key=lambda i: h('so08-sample', *salt, i))[:k])
    return [x for i, x in enumerate(items) if i in keep]


def shuffled(items, seed):
    out = list(items)
    random.Random(seed).shuffle(out)
    return out
