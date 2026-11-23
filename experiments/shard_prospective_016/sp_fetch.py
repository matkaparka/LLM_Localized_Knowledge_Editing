"""Pinned, hashed download of SP-16's backbone, Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0, not gated), and of ROME's
counterfact.json. SO-08's so_fetch.py downloads and checks the model files, unchanged; this file adds the file list at
the pinned revision (the layout of BR-14's br_fetch.py) and the checked CounterFact download.

  python -I sp_fetch.py listing RECORD_DIR                  official listing at the pinned revision
  python -I sp_fetch.py fetch OUT_DIR RECORD_DIR [--parallel N] [--resume]
  python -I sp_fetch.py counterfact OUT_FILE                download and check against the published SHA-256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'self_observation_008'))
import so_fetch  # noqa: E402

KIND = 'qwen25_1.5b_instruct'
REPO = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
FILES = {'config.json', 'generation_config.json', 'merges.txt', 'model.safetensors', 'tokenizer.json',
         'tokenizer_config.json', 'vocab.json'}
COUNTERFACT_URL = 'https://rome.baulab.info/data/dsets/counterfact.json'
COUNTERFACT_SHA = 'd017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f'
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


def counterfact(out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        tmp = out.with_suffix('.part')
        urllib.request.urlretrieve(COUNTERFACT_URL, tmp)
        tmp.rename(out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    if digest != COUNTERFACT_SHA:
        raise SystemExit(f'counterfact.json sha256 {digest} != {COUNTERFACT_SHA}')
    print(f'counterfact.json ok ({out.stat().st_size} bytes, sha256 {digest[:12]})')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('listing', 'fetch', 'counterfact'))
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
    elif a.command == 'counterfact':
        counterfact(Path(a.args[0]))
    else:
        out, rec = Path(a.args[0]), Path(a.args[1])
        lst = json.loads((rec / 'listing.json').read_text(encoding='utf-8'))
        if lst['revision'] != REVISION:
            raise SystemExit('listing is not at the pinned revision')
        so_fetch.cmd_fetch(KIND, out, rec, None, a.resume, a.parallel)


if __name__ == '__main__':
    sys.exit(main())
