"""Pinned, hashed download of BR-14's second backbone, allenai/OLMo-2-1124-7B (base, Apache-2.0, not gated). SO-08's
so_fetch.py does the downloading and checking, unchanged; this file only adds the file list at the pinned revision
(the layout of XR-13's xr_fetch.py).

  python -I br_fetch.py listing RECORD_DIR                  official listing at the pinned revision
  python -I br_fetch.py fetch OUT_DIR RECORD_DIR [--parallel N] [--resume]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'self_observation_008'))

import so_fetch  # noqa: E402

KIND = 'olmo2_7b'
REPO = 'allenai/OLMo-2-1124-7B'
REVISION = '7df9a82518afdecae4e8c026b27adccc8c1f0032'
FILES = {'config.json', 'generation_config.json', 'merges.txt', 'model.safetensors.index.json',
         'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
         *(f'model-0000{i}-of-00006.safetensors' for i in range(1, 7))}
so_fetch.SPECS[KIND] = dict(repo=REPO, repo_type='model', pick=lambda n: n in FILES, expect=lambda names: FILES <= set(names))


def listing() -> dict:
    from huggingface_hub import HfApi
    info = HfApi(endpoint=so_fetch.OFFICIAL).repo_info(REPO, revision=REVISION, files_metadata=True, token=False)
    if info.sha != REVISION:
        raise SystemExit(f'revision {info.sha} != pinned {REVISION}')
    files = [s for s in info.siblings if s.rfilename in FILES]
    if {s.rfilename for s in files} != FILES:
        raise SystemExit(f'unexpected files: {sorted(s.rfilename for s in files)}')
    card = getattr(info, 'card_data', None) or getattr(info, 'cardData', None)
    return dict(repo=REPO, repo_type='model', revision=info.sha, endpoint=so_fetch.OFFICIAL,
                license=card.get('license') if hasattr(card, 'get') else None, gated=getattr(info, 'gated', None),
                listed_utc=so_fetch.now(),
                files=[dict(name=s.rfilename, size=s.size, blob_id=s.blob_id,
                            lfs_sha256=s.lfs.sha256 if s.lfs is not None else None)
                       for s in sorted(files, key=lambda s: s.rfilename)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('listing', 'fetch'))
    ap.add_argument('args', nargs='+')
    ap.add_argument('--parallel', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args(argv)
    if a.command == 'listing':
        rec = Path(a.args[0])
        rec.mkdir(parents=True, exist_ok=True)
        lst = listing()
        (rec / 'listing.json').write_text(json.dumps(lst, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'{lst["repo"]} @ {lst["revision"]}: {len(lst["files"])} files, '
              f'{sum(f["size"] for f in lst["files"])} bytes, licence {lst["license"]}, gated {lst["gated"]}')
    else:
        out, rec = Path(a.args[0]), Path(a.args[1])
        lst = json.loads((rec / 'listing.json').read_text(encoding='utf-8'))
        if lst['revision'] != REVISION:
            raise SystemExit('listing is not at the pinned revision')
        so_fetch.cmd_fetch(KIND, out, rec, None, a.resume, a.parallel)


if __name__ == '__main__':
    sys.exit(main())
