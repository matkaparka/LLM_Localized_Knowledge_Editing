import sys, json, time, torch
sys.path.insert(0, r'E:\research\pause2\experiments\reversal_pilot_009')
from rv_editor import EditWritable, LocalEditor, LocalCfg
from rv_policy import spare_rows, corpus, cap_stream, Writer, RunCfg, trajectory, screen
from so_data import load_edits, FactIndex, build_streams, split_halves
from so_policy import EvalCfg, WriteCfg
from so_analysis import locality_damage, edit_forgetting
from transformers import AutoModelForCausalLM, AutoTokenizer
cfg = json.load(open(r'E:\research\pause2\experiments\self_observation_008\config_run_002.json'))
edits, windows, _ = load_edits(cfg['data_dir']); index = FactIndex(edits)
streams = build_streams(edits, len(windows), **cfg['streams'])
spare = spare_rows(edits, streams, split_halves(edits))
texts = corpus(spare, {6,7,8,9}, 20000)
tok = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(cfg['model_path'], local_files_only=True, dtype=torch.bfloat16).cuda()
wm = EditWritable(model, tok, [f'model.layers.{i}.mlp.down_proj' for i in range(28)], rank=16, scale=1.0, clip=1.0)
ev = EvalCfg(max_cur=100, ret_per_step=25, max_loc=256, max_new=24, gen_batch=64)
P0 = cap_stream([s for s in streams if s.name=='P0'][0], 150)
P0s, info, base_loc = screen(wm, P0, ev); print('screen', info, flush=True)
rc = RunCfg(write=WriteCfg(**cfg['write']), ev=ev)
for tau in (0.9, 0.93):
    ed = LocalEditor(LocalCfg(mode='route', layers=(8,), tau=tau, chunk=32))
    ed.prepare(wm, texts)
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    tr = trajectory(wm, ed, P0s, index, Writer('local'), rc, 0, base_loc, max_steps=5)
    for s in tr['steps']:
        a = s['eval']['acc']; x = s['exec']['edit']
        print(f"tau {tau} step {s['step']}: n {x['n']} upd {a['upd']:.3f} cur {a['cur']:.3f} ret {a['ret'] if a['ret'] is None else round(a['ret'],3)} loc {a['loc']:.3f} r2w {s['eval']['loc_right_to_wrong']} entries {x['entries']} znll {x['z_nll_mean']:.3f} write {s['exec']['write_seconds']:.0f}s", flush=True)
    evals = [s['eval'] for s in tr['steps']]
    print('tau', tau, 'loc damage', locality_damage(evals[-1], base_loc), 'forgetting', edit_forgetting(evals), f'total {time.time()-t0:.0f}s peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB hits {ed.hits}', flush=True)
    ed.detach()
