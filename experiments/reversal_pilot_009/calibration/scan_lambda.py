import sys, json, time, torch
sys.path.insert(0, r'E:\research\pause2\experiments\reversal_pilot_009')
from rv_editor import EditWritable, LocalEditor, LocalCfg, Fact
from rv_policy import spare_rows, corpus, cap_stream, SubjView
from so_data import load_edits, FactIndex, build_streams, split_halves, norm
from so_policy import EvalCfg, screen_locality
from transformers import AutoModelForCausalLM, AutoTokenizer
cfg = json.load(open(r'E:\research\pause2\experiments\self_observation_008\config_run_002.json'))
edits, windows, _ = load_edits(cfg['data_dir'])
streams = build_streams(edits, len(windows), **cfg['streams'])
spare = spare_rows(edits, streams, split_halves(edits))
texts = corpus(spare, {6,7,8,9}, 20000)
tok = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(cfg['model_path'], local_files_only=True, dtype=torch.bfloat16).cuda()
wm = EditWritable(model, tok, [f'model.layers.{i}.mlp.down_proj' for i in range(28)], rank=16, scale=1.0, clip=1.0)
P0 = [s for s in streams if s.name=='P0'][0]
ev = EvalCfg(max_cur=100, ret_per_step=25, max_loc=256, max_new=24, gen_batch=64)
t0=time.time(); P0s, info = screen_locality(wm, P0, ev); print('screen', info, f'{time.time()-t0:.0f}s', flush=True)
ctx = SubjView(P0s, 0).context()
facts = (SubjView(P0s, 0).batch_facts() + SubjView(P0s, 1).batch_facts())[:150]
e_by_q = {e.update: e for e in P0s.edits}
loc = list(P0s.loc_hidden)
def acc(qs, golds):
    out = [t for t,_,_ in wm.generate(qs, contexts=[ctx]*len(qs), max_new=24, batch_size=64)]
    return sum(norm(a)==norm(g) for a,g in zip(out,golds))/len(qs)
reph = [(e_by_q[f.question].rephrase, f.answer) for f in facts if e_by_q[f.question].rephrase]
variants = [dict(lam=2000.0), dict(lam=300.0), dict(lam=50.0), dict(mode='alpha', null_rel=1e-2, l2=10.0)]
for v in variants:
    ed = LocalEditor(LocalCfg(chunk=16, **v))
    t1=time.time(); ed.prepare(wm, texts); tp=time.time()-t1
    if v.get('mode')=='alpha': print('null dims', {l: r.get('null_dim') for l,r in ed.prep_info['layers'].items()})
    wm.restore(wm.empty_state())
    torch.cuda.reset_peak_memory_stats()
    t1=time.time(); rec = ed.edit(wm, facts, ctx, seed=1); te=time.time()-t1
    r = dict(v=v, edit_s=round(te,1), prep_s=round(tp,1), peak_gib=round(torch.cuda.max_memory_allocated()/2**30,2),
             z_nll=round(rec['z_nll_mean'],4), gap=round(rec['z_gap_after'],3), delta_rel=round(rec['delta_rel_mean'],3),
             upd=acc([f.question for f in facts],[f.answer for f in facts]), reph=acc([q for q,_ in reph],[a for _,a in reph]),
             loc=acc([q for q,_ in loc],[a for _,a in loc]))
    print(json.dumps(r), flush=True)
