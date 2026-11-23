"""BR-14 calibration input: the reference norm for the massive-position cut-off (PLAN section 2).

RV-09 excludes positions whose block-output norm is at least 200 from routing (Qwen3-0.6B-Base, block 8: the first
token's norm is about 6,480). BR-14 keeps the same relative cut-off on OLMo-2-7B: massive = 200 x (median norm at
OLMo block 9) / (median norm at Qwen block 8), both medians over every position after the first of the same
routing-centre corpus (the dev half's 20,000 texts). This script measures the Qwen median; rc_core.RCEnv measures
the OLMo median at run time and applies the rule (config massive_rule.ref_median_file).

  python -I br_calibrate.py CONFIG QWEN_MODEL_DIR OUT_JSON
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'robustness_checks_015'))
import rc_core  # noqa: E402
from rc_core import fp_data, EditWritable  # noqa: E402
from so_data import load_edits  # noqa: E402


def main(cfg_path, qwen_dir, out_json):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(out_json)
    if out.exists():
        raise SystemExit(f'{out} exists')
    cfg = json.loads(Path(cfg_path).read_text(encoding='utf-8'))
    edits, windows, _ = load_edits(os.environ.get('RV_DATA_DIR') or cfg['data_dir'], expect_steps=cfg.get('expect_steps', 8))
    sc = cfg['streams']
    _, corpus, _ = fp_data.kt10_half(edits, len(windows), sc['args'], sc['mu_corpus'])
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(qwen_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(qwen_dir, local_files_only=True, dtype=torch.bfloat16).to('cuda')
    wm = EditWritable(model, tok, [f'model.layers.{i}.mlp.down_proj' for i in range(model.config.num_hidden_layers)],
                      rank=16, scale=1.0, clip=1.0)
    layer = int(cfg['massive_rule'].get('ref_layer_of_ref_model', 8))
    t0 = time.time()
    stats = rc_core.norm_stats(wm, wm.model.get_submodule(f'model.layers.{layer}'), corpus)
    manifest = Path(qwen_dir).parent.parent / 'acq' / 'model' / 'manifest.json'
    rev = json.loads(manifest.read_text(encoding='utf-8')).get('revision') if manifest.exists() else None
    rec = dict(model='Qwen/Qwen3-0.6B-Base', revision=rev, model_dir=str(qwen_dir), layer=layer,
               block=f'model.layers.{layer}', dtype='bfloat16', corpus=dict(
                   n=len(corpus), sha256=hashlib.sha256('\n'.join(corpus).encode('utf-8')).hexdigest()),
               stats=stats, seconds=time.time() - t0,
               written=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
