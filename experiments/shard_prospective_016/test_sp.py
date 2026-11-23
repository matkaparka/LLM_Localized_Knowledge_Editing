"""CPU tests for SP-16: the instrumented read-out against the author's wrapper, rankings, permutations, the random
expectation, matchers, the sealed protect reader, and an end-to-end run (smoke, dev, select, test, analysis, recount)
on a tiny random Qwen2 model with a character tokenizer and synthetic CounterFact records."""
import csv
import itertools
import json
import math
import string
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sp_core as C  # noqa: E402


def char_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    chars = ['<pad>', '<eos>', '<unk>'] + sorted(set(string.printable) - set('\t\n\r\x0b\x0c'))
    tk = Tokenizer(models.WordLevel({c: i for i, c in enumerate(chars)}, unk_token='<unk>'))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex('.'), behavior='isolated')
    tk.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')


def tiny_model(tok, seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    m = Qwen2ForCausalLM(cfg).eval()
    m.generation_config.eos_token_id = tok.eos_token_id
    m.generation_config.pad_token_id = tok.pad_token_id
    return m


def filled_wrappers(n=5, d=32, seed=1):
    torch.manual_seed(seed)
    base = torch.nn.Linear(d, d)
    author = C.ablations.AblatedSHARDWrapper(base, routing='cosine', write_mode='additive', sim_threshold=0.1)
    ours = C.SPWrapper(base, 0.1)
    for _ in range(n):
        k, v = torch.randn(d), torch.randn(d)
        author.add_entry(k, v)
        ours.add_entry(k, v)
    return author, ours, base


def test_native_readout_is_the_author_forward_and_is_observed():
    author, ours, _ = filled_wrappers()
    x = torch.randn(4, 3, 32)
    ours.trace = []
    assert torch.equal(author(x), ours(x))
    idx, sim, hit, add = ours.trace[0]
    s = torch.nn.functional.normalize(x[:, -1].float(), dim=-1) @ torch.nn.functional.normalize(
        torch.stack(author.keys).float(), dim=-1).t()
    assert torch.equal(idx, s.argmax(-1)) and torch.equal(hit, s.max(-1).values > 0.1) and torch.equal(hit, add)


def test_other_readouts():
    _, ours, base = filled_wrappers()
    x = torch.randn(6, 2, 32)
    native = ours(x)
    ours.readout = {'tau': torch.full((5,), 0.1)}
    assert torch.equal(ours(x), native)                              # per-slot tensor at tau = the parent's rule
    ours.readout = {'tau': 0.1}
    assert torch.equal(ours(x), native)
    ours.readout = {'off': True}
    assert torch.equal(ours(x), base(x))
    ours.readout = {'suppress': torch.ones(5, dtype=torch.bool)}
    assert torch.equal(ours(x), base(x))                             # every winner suppressed: nothing added
    ours.readout = {'tau': -2.0, 'single': torch.tensor([3] * 6)}
    out = ours(x)
    assert torch.allclose(out[:, -1], base(x)[:, -1] + ours.values[3])  # the only candidate always fires at tau -2
    ours.readout = None


def test_rankings_permutations_and_random_expectation():
    ids = [5, 9, 7, 11]
    order = C.rank([0.5, 0.9, 0.5, 0.1], ids)
    assert order[0] == 1 and order[-1] == 3
    assert order[1:3] == sorted([0, 2], key=lambda i: C.tie_key(ids[i]))
    taus = [0.7 + 0.001 * i for i in range(120)]
    for p in range(3):
        perm = C.block_permutation(120, 'T01', 'shuffled', p)
        for b in range(0, 120, C.BLOCK):
            blk = range(b, min(b + C.BLOCK, 120))
            assert sorted(taus[perm[i]] for i in blk) == sorted(taus[i] for i in blk)
            assert all(b <= perm[i] < b + C.BLOCK for i in blk)
        fixed = frozenset({3, 60, 61})
        act = C.block_permutation(120, 'T01', 'active_shuffled', p, fixed)
        assert all(act[i] == i for i in fixed) and sorted(act) == list(range(120))
    assert C.block_permutation(120, 'T01', 'shuffled', 0) != C.block_permutation(120, 'T02', 'shuffled', 0)
    N, k = 7, 3
    for m in range(0, 5):
        hits = [any(s < m for s in combo) for combo in itertools.combinations(range(N), k)]
        assert math.isclose(C.random_hit(N, k, m), sum(hits) / len(hits))


def test_matchers():
    assert C.match_main(' French. She was born', 'French')
    assert C.match_main('New York City', 'New York')
    assert not C.match_main('Frenchman', 'French')
    assert not C.match_main('the French language', 'French')
    assert C.match_main('ＰＡＲＩＳ, France', 'Paris')                    # NFKC and case folding
    assert C.match_author('the French language', 'French')
    assert not C.match_main('anything', '')


def synthetic_counterfact(n):
    recs = []
    for i in range(n):
        recs.append(dict(case_id=1000 + i, requested_rewrite=dict(
            prompt='{} works in the city of', subject=f'Person{i}', relation_id='P937',
            target_new=dict(str=f'Town{i % 7}', id='Q1'), target_true=dict(str=f'Home{i}', id='Q2')),
            paraphrase_prompts=[f'Note {i}. Person{i} is employed in', f'Person{i} has a job in the town of'],
            neighborhood_prompts=[]))
    return recs


def write_manifest(path, layout):
    rows, cid = [], 1000
    for partition, stream, role, count in layout:
        for o in range(1, count + 1):
            rows.append(dict(partition=partition, stream=stream, role=role, ordinal=o, case_id=cid, relation_id='P937'))
            cid += 1
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['partition', 'stream', 'role', 'ordinal', 'case_id', 'relation_id'],
                           lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    return cid - 1000


LAYOUT = [('dev', 'shared', 'probe', 4), ('dev', 'D01', 'edit', 4), ('dev', 'D01', 'protect', 6),
          ('dev', 'D02', 'edit', 4), ('dev', 'D02', 'protect', 6), ('test', 'shared', 'probe', 4),
          ('test', 'T01', 'edit', 4), ('test', 'T01', 'protect', 6)]


def test_source_readers_and_seal(tmp_path):
    n = write_manifest(tmp_path / 'm.csv', LAYOUT)
    (tmp_path / 'cf.json').write_text(json.dumps(synthetic_counterfact(n)))
    with pytest.raises(ValueError):
        C.Source(tmp_path / 'cf.json', tmp_path / 'm.csv')
    src = C.Source(tmp_path / 'cf.json', tmp_path / 'm.csv', verify=False)
    e = src.edits('dev', 'D01')
    assert [x['case_id'] for x in e] == [1004, 1005, 1006, 1007] and 'target_true' not in e[0]
    assert e[0]['prompt'] == 'Person4 works in the city of' and len(e[0]['paraphrases']) == 2
    assert set(src.probes('dev')[0]) == {'case_id', 'prompt'}
    with pytest.raises(PermissionError):
        src.protect('dev', 'D01')
    src.seal('D01', 'x')
    assert src.protect('dev', 'D01')[0]['target_true'] == 'Home8'


def test_end_to_end_tiny(tmp_path, monkeypatch):
    import sp_analysis as A
    import sp_run as R
    tok = char_tokenizer()
    model = tiny_model(tok)
    mdir = tmp_path / 'model'
    model.save_pretrained(mdir)
    tok.save_pretrained(mdir)
    n = write_manifest(tmp_path / 'm.csv', LAYOUT)
    recs = synthetic_counterfact(n)
    (tmp_path / 'cf.json').write_text(json.dumps(recs))
    # make every protected answer 'correct' for the unedited model: its own greedy continuation, in the same batches
    src = C.Source(tmp_path / 'cf.json', tmp_path / 'm.csv', verify=False)
    ed = C.Editor(model, tok, 1, -1.0, 2, 1.0, 20.0)
    for part, s in (('dev', 'D01'), ('dev', 'D02'), ('test', 'T01')):
        src.seal(s, 'x')
        prot = src.protect(part, s)
        g = C.generate(model, tok, ed.w, [q['prompt'] for q in prot], max_new_tokens=6, batch_size=4,
                       readout={'off': True})
        for q, t in zip(prot, g['texts']):
            r = next(r for r in recs if r['case_id'] == q['case_id'])
            r['requested_rewrite']['target_true']['str'] = t.strip() or 'x'
    ed.detach()
    (tmp_path / 'cf.json').write_text(json.dumps(recs))
    cfg = dict(status='test', model=dict(path=str(mdir), dtype='float32'),
               data=dict(counterfact=str(tmp_path / 'cf.json'), manifest=str(tmp_path / 'm.csv'), verify=False),
               editor=dict(layer=1, tau=-1.0, v_steps=2, v_lr=1.0, norm_cap=20.0), eval=dict(max_new_tokens=6, batch_size=4),
               margin=0.02, n_perms=2, budgets=[1, 2], primary_budget=2,
               gates=dict(min_damaged=1, min_streams=1, min_gain=-1.0), uniform_grid=[0.0, 0.5],
               selection=dict(damage_cap=0.5), partition='dev', streams=['D01', 'D02'],
               arms=['none', 'native', 'uniform:0.00', 'uniform:0.50', 'matched', 'shuffled:0', 'shuffled:1',
                     'active_shuffled:0', 'active_shuffled:1', 'suppress_top2'],
               test_streams=['T01'], test_arms=['none', 'native', 'uniform:SEL', 'matched', 'shuffled:0', 'shuffled:1',
                                                'active_shuffled:0', 'active_shuffled:1', 'suppress_top2'],
               checks=dict(reversed_native=True), smoke=dict(stream='D01', writes=3, probes=4))
    cdev = tmp_path / 'config_dev.json'
    cdev.write_text(json.dumps(cfg))
    R.cmd_smoke(cdev, tmp_path / 'smoke')
    sm = json.loads((tmp_path / 'smoke' / 'smoke.json').read_text())
    assert sm['passed'] and sm['keys_equal'] and sm['values_equal']
    R.cmd_run(cdev, tmp_path / 'dev')
    rec = A.load_run(tmp_path / 'dev')['D01']
    assert rec['checks']['seal_before_protect'] and rec['checks']['reversed_native_differs'] == 0
    assert rec['checks']['trigger_counts_match_recomputed']
    assert len(rec['seal']['ranking']) == 4 and sorted(rec['seal']['ranking']) == [0, 1, 2, 3]
    R.cmd_select(tmp_path / 'dev', cdev, tmp_path / 'config_test.json')
    ctest = json.loads((tmp_path / 'config_test.json').read_text())
    assert ctest['partition'] == 'test' and any(a.startswith('uniform:0.') for a in ctest['arms'])
    R.cmd_run(tmp_path / 'config_test.json', tmp_path / 'test')
    A.write_analysis(tmp_path / 'test', tmp_path / 'dev')
    an = json.loads((tmp_path / 'test' / 'analysis.json').read_text())
    assert an['main']['events']['damaged'] >= 1          # tau -1 fires everywhere, so native spoils answers
    sys.path.insert(0, str(HERE / 'checks'))
    import recount
    recount.main(tmp_path / 'test', tmp_path / 'test' / 'recount.json')
    rc = json.loads((tmp_path / 'test' / 'recount.json').read_text())
    assert rc['max_abs_diff_vs_analysis'] < 1e-9
