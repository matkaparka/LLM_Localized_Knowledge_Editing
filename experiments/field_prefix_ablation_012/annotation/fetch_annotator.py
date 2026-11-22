"""Pinned, hashed download of annotator B's model (Qwen/Qwen2.5-7B-Instruct). SO-08's so_fetch.py does the
downloading and checking, unchanged; this file only adds the model's file list at the pinned revision.

  python -I fetch_annotator.py listing RECORD_DIR                 official listing at the pinned revision
  python -I fetch_annotator.py fetch OUT_DIR RECORD_DIR [--parallel N] [--resume]

Every file is checked against the official listing (size plus the LFS SHA-256 or the git blob id).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'self_observation_008'))

import so_fetch  # noqa: E402

REPO = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
FILES = {'config.json', 'generation_config.json', 'merges.txt', 'model.safetensors.index.json', 'tokenizer.json',
         'tokenizer_config.json', 'vocab.json', 'LICENSE', *(f'model-0000{i}-of-00004.safetensors' for i in range(1, 5))}
so_fetch.SPECS['annotator'] = dict(repo=REPO, repo_type='model', pick=lambda n: n in FILES,
                                   expect=lambda names: FILES <= set(names))


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
                license=card.get('license') if hasattr(card, 'get') else None, listed_utc=so_fetch.now(),
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
              f'{sum(f["size"] for f in lst["files"])} bytes, licence {lst["license"]}')
    else:
        out, rec = Path(a.args[0]), Path(a.args[1])
        lst = json.loads((rec / 'listing.json').read_text(encoding='utf-8'))
        if lst['revision'] != REVISION:
            raise SystemExit('listing is not at the pinned revision')
        so_fetch.cmd_fetch('annotator', out, rec, None, a.resume, a.parallel)


if __name__ == '__main__':
    sys.exit(main())
