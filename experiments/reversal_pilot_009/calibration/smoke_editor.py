import sys, json, time, torch
sys.path.insert(0, r'E:\research\pause2\experiments\reversal_pilot_009')
from rv_editor import EditWritable, LocalEditor, LocalCfg, Fact
from rv_policy import spare_rows, corpus, cap_stream, SubjView, bucket_of, NAMES
from so_data import load_edits, FactIndex, build_streams, split_halves, norm, shots_text
from so_policy import EvalCfg, screen_locality
from transformers import AutoModelForCausalLM, AutoTokenizer
cfg = json.load(open(r'E:\research\pause2\experiments\self_observation_008\config_run_002.json'))
t0 = time.time()
edits, windows, _ = load_edits(cfg['data_dir'])
index = FactIndex(edits)
streams = build_streams(edits, len(windows), **cfg['streams'])
half_of = split_halves(edits)
spare = spare_rows(edits, streams, half_of)
print('spare rows', len(spare), 'by bucket', [sum(1 for e in spare if bucket_of(e.subject_id)==b) for b in range(10)], f'{time.time()-t0:.0f}s')
texts = corpus(spare, {6,7,8,9}, 4000)
print('corpus', len(texts), texts[:2])
tok = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(cfg['model_path'], local_files_only=True, dtype=torch.bfloat16).cuda()
sites = [f'model.layers.{i}.mlp.down_proj' for i in range(28)]
wm = EditWritable(model, tok, sites, rank=16, scale=1.0, clip=1.0)
ed = LocalEditor(LocalCfg())
t1 = time.time(); info = ed.prepare(wm, texts); print('prepare', f'{time.time()-t1:.1f}s', json.dumps(info['layers'], indent=0)[:600])
P0 = cap_stream([s for s in streams if s.name=='P0'][0], 150)
v = SubjView(P0, 0)
facts = v.batch_facts()[:24]
ctx = v.context()
# before
def ask(qs):
    return [t for t,_,_ in wm.generate(qs, contexts=[ctx]*len(qs), max_new=24)]
print('before', list(zip([f.answer for f in facts[:6]], ask([f.question for f in facts[:6]]))))
torch.cuda.reset_peak_memory_stats()
t1 = time.time(); rec = ed.edit(wm, facts, ctx, seed=1); print('edit', f'{time.time()-t1:.1f}s', 'peak GiB', torch.cuda.max_memory_allocated()/2**30)
print({k: v for k, v in rec.items() if k != 'layers_info'}); print(rec['layers_info'])
ans = ask([f.question for f in facts])
print('efficacy', sum(norm(a)==norm(f.answer) for a,f in zip(ans,facts)), '/', len(facts))
e_by_q = {e.update: e for e in P0.edits}
reph = [e_by_q[f.question].rephrase for f in facts if e_by_q[f.question].rephrase]
golds = [f.answer for f in facts if e_by_q[f.question].rephrase]
ans_r = ask(reph)
print('rephrase', sum(norm(a)==norm(g) for a,g in zip(ans_r,golds)), '/', len(reph))
print(list(zip(golds[:8], ans_r[:8])))
print('templates', ed.random_prefixes(wm, 1))
print('extra', {k: tuple(v.shape) for k,v in wm.extra.items()}, 'rank', {k:v for k,v in wm.committed_rank().items() if v})
