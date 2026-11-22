import sys, json, time, torch
sys.path.insert(0, r'E:\research\pause2\experiments\reversal_pilot_009')
from rv_editor import EditWritable, LocalEditor, LocalCfg, Fact, lookup
from rv_policy import spare_rows, cap_stream, SubjView
from so_data import load_edits, FactIndex, build_streams, split_halves, norm
from so_model import fmt_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
cfg = json.load(open(r'E:\research\pause2\experiments\self_observation_008\config_run_002.json'))
edits, windows, _ = load_edits(cfg['data_dir'])
streams = build_streams(edits, len(windows), **cfg['streams'])
tok = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(cfg['model_path'], local_files_only=True, dtype=torch.bfloat16).cuda()
wm = EditWritable(model, tok, [f'model.layers.{i}.mlp.down_proj' for i in range(28)], rank=16, scale=1.0)
P0 = [s for s in streams if s.name=='P0'][0]
ctx = SubjView(P0, 0).context()
es = [e for e in P0.edits if e.step <= 2 and e.rephrase][:200]
loc = [q for q,_ in P0.loc_hidden[:400]]
ed = LocalEditor(LocalCfg())
def block_states(texts, layer):
    """block output for all positions of each text (with shots context), list of (T x d) float tensors"""
    out = []
    blk = model.get_submodule(f'model.layers.{layer}')
    cap = {}
    def hk(m, i, o):
        cap['h'] = (o[0] if isinstance(o, tuple) else o).detach().float()
    hd = blk.register_forward_hook(hk)
    with torch.no_grad():
        for t in texts:
            ids = tok.encode(t, add_special_tokens=False)
            model.model(input_ids=torch.tensor([ids]).cuda(), use_cache=False)
            out.append((ids, cap['h'][0]))
    hd.remove()
    return out
for layer in (8, 12, 16):
    # keys: subject-last token in the update question, mean over bare/shots contexts
    keys = []
    for e in es:
        vs = []
        for c in ('', ctx):
            text = fmt_prompt(e.update, c)
            ids, idx, found = lookup(tok, text, e.subject, len(c))
            st = block_states([text], layer)[0][1]
            vs.append(st[idx])
        keys.append(torch.stack(vs).mean(0))
    K = torch.nn.functional.normalize(torch.stack(keys), dim=1)
    # rephrase (with shots): max over positions of cos with its own key; and the best other key
    own, other = [], []
    for i, e in enumerate(es):
        text = fmt_prompt(e.rephrase, ctx)
        ids, st = block_states([text], layer)[0]
        q = torch.nn.functional.normalize(st[len(tok.encode(ctx, add_special_tokens=False)):], dim=1)
        sims = q @ K.T
        own.append(float(sims[:, i].max()))
        sims[:, i] = -1
        other.append(float(sims.max()))
    lo = []
    for t in loc:
        text = fmt_prompt(t, ctx)
        ids, st = block_states([text], layer)[0]
        q = torch.nn.functional.normalize(st[len(tok.encode(ctx, add_special_tokens=False)):], dim=1)
        lo.append(float((q @ K.T).max()))
    import statistics as S
    def qs(v): v=sorted(v); n=len(v); return [round(v[int(p*(n-1))],3) for p in (0.05,0.25,0.5,0.75,0.95)]
    print('layer', layer, 'own-rephrase q', qs(own), '| other-key in rephrase q', qs(other), '| locality max q', qs(lo))
    for tau in (0.7, 0.8, 0.85, 0.9, 0.95):
        print(f'  tau {tau}: rephrase own>=tau {sum(o>=tau for o in own)/len(own):.2f}  locality any>=tau {sum(x>=tau for x in lo)/len(lo):.3f}  rephrase wrong-key>=tau {sum(o>=tau for o in other)/len(other):.2f}')
