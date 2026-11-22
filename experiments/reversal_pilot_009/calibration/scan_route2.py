import sys, json, time, torch
sys.path.insert(0, r'E:\research\pause2\experiments\reversal_pilot_009')
from rv_editor import EditWritable, LocalEditor, LocalCfg, Fact, lookup
from rv_policy import spare_rows, corpus, cap_stream, Writer, RunCfg, trajectory, screen, SubjView
from so_data import load_edits, FactIndex, build_streams, split_halves, norm, shots_text
from so_model import fmt_prompt
from so_policy import EvalCfg, WriteCfg
from so_analysis import locality_damage, edit_forgetting
from transformers import AutoModelForCausalLM, AutoTokenizer
F = torch.nn.functional
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
P0s, info, base_loc = screen(wm, P0, ev)
rc = RunCfg(write=WriteCfg(**cfg['write']), ev=ev)
ctx = shots_text(P0s.shots)
subj = {}
for e in P0s.edits: subj[e.update] = e.subject
for margin in (0.02,):
    ed = LocalEditor(LocalCfg(mode='route', layers=(8,), tau=0.9, chunk=32, probe_margin=margin))
    ed.prepare(wm, texts); print('probes', ed.set_probes(wm, list(P0s.loc_probe), ctx), flush=True)
    t0 = time.time()
    tr = trajectory(wm, ed, P0s, index, Writer('local'), rc, 0, base_loc, max_steps=5)
    for s in tr['steps']:
        a = s['eval']['acc']; x = s['exec']['edit']
        print(f"margin {margin} step {s['step']}: upd {a['upd']:.3f} cur {a['cur']:.3f} ret {a['ret'] if a['ret'] is None else round(a['ret'],3)} loc {a['loc']:.3f} r2w {s['eval']['loc_right_to_wrong']} raised {x.get('raised_thresholds')} thr_mean {x.get('threshold_mean'):.3f}", flush=True)
    evals = [s['eval'] for s in tr['steps']]
    print('loc damage', locality_damage(evals[-1], base_loc), 'forgetting', edit_forgetting(evals), f'{time.time()-t0:.0f}s', flush=True)
    # which entries fire on the broken locality questions (final state)
    final = {k: ok for c,k,ok in evals[-1]['items'] if c=='loc'}
    broken = [q for q in final if base_loc[q][0] and not final[q]]
    K, T = wm.extra['route_k'], wm.extra['route_t']
    facts = [e for e in P0s.edits if e.step < 5]
    nctx = len(tok.encode(ctx, add_special_tokens=False))
    cap = {}
    blk = model.get_submodule('model.layers.8')
    hd = blk.register_forward_hook(lambda m,i,o: cap.__setitem__('h', (o[0] if isinstance(o, tuple) else o).detach().float()))
    ed.route_off = True
    fired_broken, fired_ok, examples = 0, 0, []
    ok_qs = [q for q in final if base_loc[q][0] and final[q]]
    for group, qs in (('broken', broken), ('ok', ok_qs)):
        n_f = 0
        for q in qs:
            with torch.no_grad(): model.model(input_ids=torch.tensor([tok.encode(fmt_prompt(q, ctx), add_special_tokens=False)]).cuda(), use_cache=False)
            h = cap['h'][0, nctx:]
            sims = F.normalize(h - ed.mu, dim=1) @ K.T
            valid = sims >= T
            if bool(valid.any()):
                n_f += 1
                if group == 'broken' and len(examples) < 12:
                    j = int(torch.where(valid, sims, torch.full_like(sims, -2)).max(0).values.argmax())
                    examples.append((q, facts[j].subject if j < len(facts) else '?', facts[j].update if j < len(facts) else '?'))
        print(group, len(qs), 'with any firing entry', n_f, flush=True)
    for x in examples: print('  ', x)
    ed.route_off = False; hd.remove(); ed.detach()
