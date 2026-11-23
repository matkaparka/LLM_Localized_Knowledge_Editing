"""SP-16 core (PLAN.md of 2026-11-23): CounterFact inputs from the frozen role manifest, the author's SHARD editor
(single-fact, single-slot version at the pinned commit, upstream/sfib) with an instrumented read-out, greedy decoding
exactly as in the author's evaluator, the answer matchers, probe scores, rankings and the threshold arms.

The author's code is imported unchanged. Writing uses AblatedSHARDMethod.insert as published. Reading uses SPWrapper,
a subclass of the author's AblatedSHARDWrapper: with no read-out set it returns the parent's forward unchanged (the
native rule: at the last position of every forward pass, the most cosine-similar key wins and adds its value only if
the similarity is strictly above tau), and only observes the winner. The other read-outs (routing off, per-slot or
uniform thresholds, suppressed values, one candidate slot per row) reuse the same similarity computation.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import sys
import unicodedata
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / 'upstream' / 'sfib'
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
import ablations  # noqa: E402  (the author's AblatedSHARDWrapper and AblatedSHARDMethod)
import ablations_realdata_patches  # noqa: E402,F401  (the author's CounterFact rewrite builder)
import counterfact_data  # noqa: E402
import evaluate as upstream_eval  # noqa: E402  (upstream/sfib/evaluate.py, first on sys.path)

SOURCE_SHA = 'd017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f'   # ROME's counterfact.json
MANIFEST_SHA = '7a29631114b1da04531defd325ae0c15172dd5810f82e59f31c0a7e2b739866a'  # counterfact_roles.csv
BLOCK = 50   # threshold permutations stay within the 5 consecutive 50-write blocks of a stream (PLAN 4.5)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tie_key(case_id) -> str:
    """Tie-break of every ranking: SHA-256 of the case_id's decimal string (PLAN 4.2)."""
    return sha256(str(case_id).encode())


# ---- data: three readers, protect() only after the stream's ranking is sealed ----------------------------------------
class Source:
    """counterfact.json and the role manifest, each checked against its SHA-256. edits(), probes() and protect() expose
    only the fields that role may use; protect() refuses a stream until seal() recorded its ranking digest."""

    def __init__(self, counterfact_path, manifest_path, verify=True):
        raw = Path(counterfact_path).read_bytes()
        man = Path(manifest_path).read_bytes()
        self.source_sha, self.manifest_sha = sha256(raw), sha256(man)
        if verify and self.source_sha != SOURCE_SHA:
            raise ValueError('counterfact.json does not match the published SHA-256')
        if verify and self.manifest_sha != MANIFEST_SHA:
            raise ValueError('counterfact_roles.csv does not match the frozen manifest')
        self._records = {int(r['case_id']): r for r in json.loads(raw)}
        self._rows = list(csv.DictReader(io.StringIO(man.decode('utf-8'))))
        self._sealed: dict[str, str] = {}

    def _ids(self, partition, stream, role):
        rows = [r for r in self._rows if (r['partition'], r['stream'], r['role']) == (partition, stream, role)]
        if not rows:
            raise KeyError((partition, stream, role))
        return [int(r['case_id']) for r in sorted(rows, key=lambda r: int(r['ordinal']))]

    def _rw(self, cid):
        return self._records[cid]['requested_rewrite']

    def edits(self, partition, stream):
        out = []
        for cid in self._ids(partition, stream, 'edit'):
            rr = self._rw(cid)
            out.append(dict(case_id=cid, subject=rr['subject'], template=rr['prompt'],
                            prompt=rr['prompt'].format(rr['subject']), target_new=rr['target_new']['str'],
                            paraphrases=[p for p in self._records[cid]['paraphrase_prompts'] if isinstance(p, str)],
                            relation_id=rr['relation_id']))
        return out

    def probes(self, partition):
        return [dict(case_id=cid, prompt=self._rw(cid)['prompt'].format(self._rw(cid)['subject']))
                for cid in self._ids(partition, 'shared', 'probe')]

    def seal(self, stream, digest):
        self._sealed[stream] = digest

    def protect(self, partition, stream):
        if stream not in self._sealed:
            raise PermissionError(f'{stream}: the ranking must be sealed before protected questions are read')
        return [dict(case_id=cid, prompt=self._rw(cid)['prompt'].format(self._rw(cid)['subject']),
                     target_true=self._rw(cid)['target_true']['str'], relation_id=self._rw(cid)['relation_id'])
                for cid in self._ids(partition, stream, 'protect')]


# ---- editor -----------------------------------------------------------------------------------------------------------
class SPWrapper(ablations.AblatedSHARDWrapper):
    """The author's single-slot wrapper (cosine routing, additive values) with an instrumented read-out.

    readout None   : the parent's forward, unchanged (native); the winner is only observed.
    readout dict   : off=True (routing off) | tau=float or per-slot tensor | suppress=bool tensor (a suppressed winner
                     adds nothing) | single=long tensor, one candidate slot per row (replays).
    trace          : when a list, each forward appends (winner, similarity, passed threshold, value added) per row.
    capture        : when a list, each forward appends the normalized last-position input (probe states).
    """

    def __init__(self, base_mlp, sim_threshold):
        super().__init__(base_mlp, routing='cosine', write_mode='additive', sim_threshold=sim_threshold, eps_init=1.0)
        self.readout = None
        self.trace = None
        self.capture = None
        self._cache = None

    def _stacks(self, device, dtype):
        n = len(self.keys)
        c = self._cache
        if c is None or c[0] != n or c[1] != device or c[2] != dtype:
            K = torch.stack(self.keys, dim=0).to(device=device)
            V = torch.stack(self.values, dim=0).to(dtype=dtype, device=device)
            self._cache = c = (n, device, dtype, F.normalize(K.float(), dim=-1), V)
        return c[3], c[4]

    def _log(self, x_n, idx, sim, hit, add):
        if self.trace is not None:
            self.trace.append((idx.cpu(), sim.cpu(), hit.cpu(), add.cpu()))
        if self.capture is not None:
            self.capture.append(x_n.cpu())

    def forward(self, x):
        ro = self.readout
        if ro is None:
            out = super().forward(x)
            if self.n_slots and (self.trace is not None or self.capture is not None):
                x_n = F.normalize(x[:, -1, :].float(), dim=-1)      # the parent's own computation, repeated
                K_n, _ = self._stacks(x.device, out.dtype)
                best_sim, best_idx = (x_n @ K_n.t()).max(dim=-1)
                hit = best_sim > self.sim_threshold
                self._log(x_n, best_idx, best_sim, hit, hit)
            return out
        base_out = self.base_mlp(x)
        if ro.get('off') or self.n_slots == 0:
            return base_out
        x_n = F.normalize(x[:, -1, :].float(), dim=-1)
        K_n, V = self._stacks(x.device, base_out.dtype)
        single = ro.get('single')
        if single is not None:
            best_idx = single.to(x.device)
            best_sim = (x_n * K_n[best_idx]).sum(dim=-1)
        else:
            best_sim, best_idx = (x_n @ K_n.t()).max(dim=-1)
        tau = ro.get('tau', self.sim_threshold)
        thr = tau.to(x.device)[best_idx] if torch.is_tensor(tau) else tau
        hit = best_sim > thr
        sup = ro.get('suppress')
        add = hit & ~sup.to(x.device)[best_idx] if sup is not None else hit
        self._log(x_n, best_idx, best_sim, hit, add)
        if not bool(add.any()):
            return base_out
        last = base_out[:, -1, :]
        out = base_out.clone()
        out[:, -1, :] = torch.where(add.unsqueeze(-1), last + V[best_idx], last)
        return out


class Editor:
    """AblatedSHARDMethod with the author's 'shard' preset (cosine, additive, vstar), installed by the author's setup()
    and then given SPWrapper in place of the author's wrapper. insert() is the author's, unchanged."""

    def __init__(self, model, tokenizer, layer, tau, v_steps, v_lr, norm_cap):
        self.model, self.tok, self.layer, self.tau = model, tokenizer, int(layer), float(tau)
        self.method = ablations.AblatedSHARDMethod(
            layer_idx=layer, sim_threshold=tau, v_steps=v_steps, v_lr=v_lr, v_weight_decay=0.0,
            v_norm_constraint=norm_cap, eps_init=1.0, **ablations.ABLATION_PRESETS['shard'])
        self.method.setup(model, tokenizer, kb=None)
        self.block = model.model.layers[self.layer]
        self.base_mlp = self.method.wrapped_mlp.base_mlp
        self.w = SPWrapper(self.base_mlp, tau).to(next(model.parameters()).device)
        self.block.mlp = self.w
        self.method.wrapped_mlp = self.w
        self.case_ids: list[int] = []

    def insert(self, edit: dict):
        self.method.insert(counterfact_data.CounterFactTriple(
            case_id=edit['case_id'], subject=edit['subject'], prompt_template=edit['template'],
            target_new=edit['target_new'], target_true='', relation_id=edit['relation_id']))
        self.case_ids.append(edit['case_id'])

    def detach(self):
        """Put the base MLP back, leaving the model as it was before setup()."""
        self.block.mlp = self.base_mlp

    def snapshot(self) -> dict:
        return dict(keys=torch.stack(self.w.keys).detach().cpu(), values=torch.stack(self.w.values).detach().cpu(),
                    case_ids=list(self.case_ids), layer=self.layer, tau=self.tau)

    def load(self, snap: dict):
        dev = next(self.model.parameters()).device
        self.w.keys = [k.to(dev) for k in snap['keys']]
        self.w.values = [v.to(dev) for v in snap['values']]
        self.w._cache = None
        self.case_ids = list(snap['case_ids'])


def table_digest(snap: dict) -> str:
    h = hashlib.sha256()
    h.update(snap['keys'].float().numpy().tobytes())
    h.update(snap['values'].float().numpy().tobytes())
    h.update(json.dumps(snap['case_ids']).encode())
    return h.hexdigest()


# ---- decoding: the author's generate_greedy, with read-out, trace and capture ------------------------------------------
def eos_ids(model, tok) -> set:
    ids = set()
    e = getattr(model.generation_config, 'eos_token_id', None)
    ids.update(e if isinstance(e, (list, tuple)) else [e] if e is not None else [])
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


def steps_used(new_ids, eos) -> int:
    """Forward calls that shaped a row's text: up to the one that produced its first end token (inclusive)."""
    for t, i in enumerate(new_ids):
        if i in eos:
            return t + 1
    return len(new_ids)


@torch.no_grad()
def generate(model, tok, wrapper, prompts, max_new_tokens=12, batch_size=32, readout=None, single=None,
             trace=False, capture=False):
    """Greedy decoding with the same tokenizer call and generate() arguments as the author's
    evaluate.generate_greedy (left padding, truncation at 256 tokens, prompt stripped from the result)."""
    device = next(model.parameters()).device
    eos = eos_ids(model, tok)
    texts, ids, traces, caps = [], [], [], []
    orig_side = getattr(tok, 'padding_side', 'right')
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tok(chunk, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
            ro = None if readout is None else dict(readout)
            if single is not None:
                ro = dict(ro or {})
                ro['single'] = torch.tensor(single[i:i + batch_size], dtype=torch.long)
            wrapper.readout = ro
            wrapper.trace = [] if trace else None
            wrapper.capture = [] if capture else None
            gen = model.generate(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'],
                                 max_new_tokens=max_new_tokens, max_length=None, do_sample=False, num_beams=1,
                                 pad_token_id=tok.pad_token_id)
            L = enc['input_ids'].shape[1]
            for j, full in enumerate(gen):
                new = full[L:]
                texts.append(tok.decode(new, skip_special_tokens=True))
                ids.append(new.tolist())
                if trace:
                    used = steps_used(ids[-1], eos)
                    traces.append([(int(c[0][j]), round(float(c[1][j]), 6), bool(c[2][j]), bool(c[3][j]))
                                   for c in wrapper.trace[:used]])
            if capture:
                caps.append(wrapper.capture[0] if wrapper.capture else None)
    finally:
        tok.padding_side = orig_side
        wrapper.readout = None
        wrapper.trace = None
        wrapper.capture = None
    return dict(texts=texts, ids=ids, traces=traces if trace else None,
                capture=torch.cat(caps) if capture and all(c is not None for c in caps) else None)


# ---- answers ---------------------------------------------------------------------------------------------------------
def norm_answer(s: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())


def match_main(generated: str, reference: str) -> bool:
    """Primary scorer (PLAN 4.3): the generation starts with the normalized reference, ending on a word boundary."""
    g, r = norm_answer(generated), norm_answer(reference)
    return bool(r) and g.startswith(r) and (len(g) == len(r) or not g[len(r)].isalnum())


def match_author(generated: str, reference: str) -> bool:
    """Sensitivity scorer: the author's evaluate._is_match (normalized reference anywhere in the generation)."""
    return upstream_eval._is_match(generated, reference)


# ---- probe scores, rankings, thresholds --------------------------------------------------------------------------------
def rank(scores, case_ids) -> list[int]:
    """Slots by descending score; ties by SHA-256 of the case_id's decimal string."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], tie_key(case_ids[i])))


def matched_thresholds(maxcos, tau, margin) -> list[float]:
    return [max(tau, m + margin) for m in maxcos]


def block_permutation(n, stream, tag, p, fixed=frozenset()) -> list[int]:
    """Slot i receives the threshold of slot perm[i]; slots in `fixed` keep their own; blocks of BLOCK writes."""
    rng = random.Random(int(sha256(f'SP16|{stream}|{tag}|{p}'.encode()), 16))
    perm = list(range(n))
    for b in range(0, n, BLOCK):
        idx = [i for i in range(b, min(b + BLOCK, n)) if i not in fixed]
        src = idx[:]
        rng.shuffle(src)
        for i, j in zip(idx, src):
            perm[i] = j
    return perm


def random_hit(N, k, m) -> float:
    """Chance that k slots drawn uniformly from N include at least one of m sufficient slots (PLAN 4.4)."""
    if m <= 0:
        return 0.0
    if N - m < k:
        return 1.0
    return 1.0 - math.comb(N - m, k) / math.comb(N, k)


def arm_readout(arm, n, tau, matched, perms, top):
    """Read-out of an evaluation arm (all arms read the same frozen table, written at tau)."""
    if arm == 'none':
        return {'off': True}
    if arm == 'native':
        return None
    if arm.startswith('uniform:'):
        return {'tau': float(arm.split(':', 1)[1])}
    if arm == 'matched':
        return {'tau': torch.tensor(matched, dtype=torch.float32)}
    for tag in ('shuffled', 'active_shuffled'):
        if arm.startswith(tag + ':'):
            perm = perms[tag][int(arm.split(':', 1)[1])]
            return {'tau': torch.tensor([matched[perm[i]] for i in range(n)], dtype=torch.float32)}
    if arm.startswith('suppress_top'):
        k = int(arm[len('suppress_top'):])
        sup = torch.zeros(n, dtype=torch.bool)
        sup[top[:k]] = True
        return {'suppress': sup}
    raise ValueError(f'unknown arm {arm!r}')
