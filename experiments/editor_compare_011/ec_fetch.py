"""Pinned, hashed download of EC-11's retriever (sentence-transformers/all-mpnet-base-v2). SO-08's so_fetch.py does
the downloading and checking, unchanged; this file only adds the retriever's file list at the frozen revision.

  python -I ec_fetch.py listing RECORD_DIR                  official listing at the pinned revision -> listing.json
  python -I ec_fetch.py fetch OUT_DIR RECORD_DIR [--parallel N] [--resume]
  python -I ec_fetch.py check OUT_DIR                       pooling and length settings of the downloaded model

Every file is checked against the official listing: size plus the LFS SHA-256 (model.safetensors) or the git blob id
(small files). The listing must carry the revision and model.safetensors hash frozen in config_dev.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'self_observation_008'))

import so_fetch  # noqa: E402

REPO = 'sentence-transformers/all-mpnet-base-v2'
REVISION = 'e8c3b32edf5434bc2275fc9bab85f82640a19130'
SAFETENSORS_SHA256 = '78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658'
FILES = {'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
         'vocab.txt', 'modules.json', 'sentence_bert_config.json', 'config_sentence_transformers.json',
         '1_Pooling/config.json', 'README.md'}
so_fetch.SPECS['retriever'] = dict(repo=REPO, repo_type='model', pick=lambda n: n in FILES,
                                   expect=lambda names: FILES <= set(names))


def listing() -> dict:
    from huggingface_hub import HfApi
    info = HfApi(endpoint=so_fetch.OFFICIAL).repo_info(REPO, revision=REVISION, files_metadata=True, token=False)
    if info.sha != REVISION:
        raise SystemExit(f'revision {info.sha} != pinned {REVISION}')
    files = [s for s in info.siblings if s.rfilename in FILES]
    if {s.rfilename for s in files} != FILES:
        raise SystemExit(f'unexpected files: {sorted(s.rfilename for s in files)}')
    st = next(s for s in files if s.rfilename == 'model.safetensors')
    if st.lfs is None or st.lfs.sha256 != SAFETENSORS_SHA256:
        raise SystemExit('model.safetensors hash differs from the frozen one')
    card = getattr(info, 'card_data', None) or getattr(info, 'cardData', None)
    return dict(repo=REPO, repo_type='model', revision=info.sha, endpoint=so_fetch.OFFICIAL,
                license=card.get('license') if hasattr(card, 'get') else None, listed_utc=so_fetch.now(),
                files=[dict(name=s.rfilename, size=s.size, blob_id=s.blob_id,
                            lfs_sha256=s.lfs.sha256 if s.lfs is not None else None)
                       for s in sorted(files, key=lambda s: s.rfilename)])


def check(out: Path) -> dict:
    pool = json.loads((out / '1_Pooling' / 'config.json').read_text(encoding='utf-8'))
    sb = json.loads((out / 'sentence_bert_config.json').read_text(encoding='utf-8'))
    modules = json.loads((out / 'modules.json').read_text(encoding='utf-8'))
    res = dict(mean_pooling=bool(pool.get('pooling_mode_mean_tokens')) and not pool.get('pooling_mode_cls_token')
               and not pool.get('pooling_mode_max_tokens'), max_seq_length=sb.get('max_seq_length'),
               normalize_module=any('Normalize' in m.get('type', '') for m in modules))
    ok = res['mean_pooling'] and res['max_seq_length'] == 384 and res['normalize_module']
    print(json.dumps(res), 'OK' if ok else 'MISMATCH')
    if not ok:
        raise SystemExit('the retriever is not mean-pooled, L2-normalised with 384 tokens')
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('listing', 'fetch', 'check'))
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
    elif a.command == 'fetch':
        out, rec = Path(a.args[0]), Path(a.args[1])
        lst = json.loads((rec / 'listing.json').read_text(encoding='utf-8'))
        if lst['revision'] != REVISION:
            raise SystemExit('listing is not at the pinned revision')
        so_fetch.cmd_fetch('retriever', out, rec, None, a.resume, a.parallel)
        check(out)
    else:
        check(Path(a.args[0]))


if __name__ == '__main__':
    sys.exit(main())
