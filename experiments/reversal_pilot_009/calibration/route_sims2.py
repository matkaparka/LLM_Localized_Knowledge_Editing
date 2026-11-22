import sys, json, time, torch
sys.path.insert(0, r'E:\research\pause2\experiments\reversal_pilot_009')
from rv_editor import EditWritable, LocalEditor, LocalCfg, Fact, lookup
from rv_policy import spare_rows, corpus, SubjView
from so_data import load_edits, FactIndex, build_streams, split_halves, norm
from so_model import fmt_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
F = torch.nn.functional
cfg = json.load(open(r'E:\research\pause2\experiments\self_observation_008\config_run_002.json'))
edits, windows, _ = load_edits(cfg['data_dir'])
streams = build_streams(edits, len(windows), **cfg['streams'])
spare = spare_rows(edits, streams, split_halves(edits))
texts = corpus(spare, {6,7,8,9}, 3000)
tok = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(cfg['model_path'], local_files_only=True, dtype=torch.bfloat16).cuda()
P0 = [s for s in streams if s.name=='P0'][0]
ctx = SubjView(P0, 0).context()
nctx = len(tok.encode(ctx, add_special_tokens=False))
es = [e for e in P0.edits if e.step <= 2 and e.rephrase][:200]
loc = [q for q,_ in P0.loc_hidden[:400]]
cap = {}
def states(text, layer):
    blk = model.get_submodule(f'model.layers.{layer}')
    hd = blk.register_forward_hook(lambda m,i,o: cap.__setitem__('h', (o[0] if isinstance(o, tuple) else o).detach().float()))
    with torch.no_grad():
        model.model(input_ids=torch.tensor([tok.encode(text, add_special_tokens=False)]).cuda(), use_cache=False)
    hd.remove(); return cap['h'][0]
def qs(v): v=sorted(v); n=len(v); return [round(v[int(p*(n-1))],3) for p in (0.05,0.25,0.5,0.75,0.95)]
for layer in (8,):
    allq = torch.cat([states(ctx + t, layer)[nctx:] for t in texts[:1500]]); mu = allq.mean(0); print("norms q", qs(allq.norm(dim=1).tolist()), "first-token norm", float(states(ctx, layer)[0].norm()))
    # token-weighted mean
    keys = []
    for e in es:
        vs = []
        for c in ('', ctx):
            text = fmt_prompt(e.update, c)
            ids, idx, found = lookup(tok, text, e.subject, len(c))
            vs.append(states(text, layer)[idx])
        keys.append(torch.stack(vs).mean(0))
    K = F.normalize(torch.stack(keys) - mu, dim=1)
    own, other, lo, own_rank1 = [], [], [], []
    for i, e in enumerate(es):
        st = states(fmt_prompt(e.rephrase, ctx), layer)[nctx:]
        s = F.normalize(st - mu, dim=1) @ K.T
        own.append(float(s[:, i].max()))
        best_pos = int(s[:, i].argmax()); own_rank1.append(int(s[best_pos].argmax()) == i)
        s[:, i] = -1; other.append(float(s.max()))
    shot_st = states(ctx, layer)[1:]; print("shots positions max centred sim", float((F.normalize(shot_st - mu, dim=1) @ K.T).max()), "uncentred", float((F.normalize(shot_st, dim=1) @ F.normalize(torch.stack(keys), dim=1).T).max()))
    for t in loc:
        st = states(fmt_prompt(t, ctx), layer)[nctx:]
        lo.append(float((F.normalize(st - mu, dim=1) @ K.T).max()))
    print('layer', layer, 'centred: own', qs(own), '| other', qs(other), '| loc', qs(lo), '| own is argmax at its best position', sum(own_rank1)/len(own_rank1))
    for tau in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        print(f'  tau {tau}: rephrase own {sum(o>=tau for o in own)/len(own):.2f}  loc any {sum(x>=tau for x in lo)/len(lo):.3f}  rephrase other {sum(o>=tau for o in other)/len(other):.2f}')
