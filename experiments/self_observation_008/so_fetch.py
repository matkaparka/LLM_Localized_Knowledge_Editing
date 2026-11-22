"""Pinned, hashed download of the SO-08 inputs. Run once, outside any experiment run.

  python -I so_fetch.py info                                        # revision and sizes; downloads nothing
  python -I so_fetch.py listing KIND RECORD_DIR                     # official listing -> RECORD_DIR/listing.json
  python -I so_fetch.py fetch KIND OUT_DIR RECORD_DIR [--endpoint URL] [--resume]

KIND is `data` (lukasthede/WikiBigEdit) or `model` (Qwen/Qwen3-0.6B-Base). `listing` always asks the official Hub
(https://huggingface.co) and pins the current commit. `fetch` downloads exactly the files of that listing at that
revision, optionally from a mirror given by --endpoint, and checks every file against the official listing: size
plus the LFS SHA-256 for LFS files, or the git blob id for small files. The mirror is therefore not trusted.
--resume keeps files already in OUT_DIR whose size and hash match. No token is used.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

OFFICIAL = 'https://huggingface.co'
SPECS = {
    'data': dict(repo='lukasthede/WikiBigEdit', repo_type='dataset',
                 pick=lambda n: bool(re.fullmatch(r'wiki_big_edit_\d{8}_\d{8}\.json', n)) or n == 'README.md',
                 expect=lambda names: sum(1 for n in names if n.startswith('wiki_big_edit_')) == 8),
    'model': dict(repo='Qwen/Qwen3-0.6B-Base', repo_type='model',
                  pick=lambda n: n in {'config.json', 'generation_config.json', 'model.safetensors', 'tokenizer.json',
                                       'tokenizer_config.json', 'vocab.json', 'merges.txt', 'LICENSE', 'README.md'},
                  expect=lambda names: {'config.json', 'model.safetensors', 'tokenizer.json'} <= set(names)),
}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def file_hashes(path: Path) -> tuple[str, str]:
    """(SHA-256, git blob id) of a file."""
    size = path.stat().st_size
    s256, blob = hashlib.sha256(), hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            s256.update(block)
            blob.update(block)
    return s256.hexdigest(), blob.hexdigest()


def official_listing(kind: str) -> dict:
    spec = SPECS[kind]
    info = HfApi(endpoint=OFFICIAL).repo_info(spec['repo'], repo_type=spec['repo_type'], files_metadata=True,
                                              token=False)
    files = [s for s in info.siblings if spec['pick'](s.rfilename)]
    if not spec['expect']([s.rfilename for s in files]):
        raise SystemExit(f'{spec["repo"]}: unexpected file list {[s.rfilename for s in files]}')
    card = getattr(info, 'card_data', None) or getattr(info, 'cardData', None)
    return dict(repo=spec['repo'], repo_type=spec['repo_type'], revision=info.sha, endpoint=OFFICIAL,
                license=card.get('license') if hasattr(card, 'get') else None, listed_utc=now(),
                files=[dict(name=s.rfilename, size=s.size, blob_id=s.blob_id,
                            lfs_sha256=s.lfs.sha256 if s.lfs is not None else None) for s in files])


def verify(path: Path, expected: dict) -> dict:
    s256, blob = file_hashes(path)
    size = path.stat().st_size
    if expected['lfs_sha256']:
        ok, how = size == expected['size'] and s256 == expected['lfs_sha256'], 'lfs_sha256'
    else:
        ok, how = size == expected['size'] and blob == expected['blob_id'], 'git_blob_id'
    return dict(name=expected['name'], bytes=size, sha256=s256, git_blob_id=blob, verified_by=how, match=ok)


def cmd_info():
    for kind in SPECS:
        lst = official_listing(kind)
        print(f'{lst["repo"]} @ {lst["revision"]} licence={lst["license"]} files={len(lst["files"])} '
              f'bytes={sum(f["size"] or 0 for f in lst["files"])}')
        for f in lst['files']:
            print(f'  {f["name"]:48s} {f["size"]}')


def cmd_listing(kind: str, record: Path):
    if (record / 'manifest.json').exists():
        raise SystemExit(f'{record} already holds a finished download')
    record.mkdir(parents=True, exist_ok=True)
    lst = official_listing(kind)
    (record / 'listing.json').write_text(json.dumps(lst, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{lst["repo"]} @ {lst["revision"]}: {len(lst["files"])} files listed')


def resolve_url(endpoint: str, lst: dict, name: str) -> str:
    prefix = '' if lst['repo_type'] == 'model' else f'{lst["repo_type"]}s/'
    return f'{endpoint.rstrip("/")}/{prefix}{lst["repo"]}/resolve/{lst["revision"]}/{name}'


def parallel_download(url: str, dest: Path, size: int, parts: int, log_every: float = 30.0):
    """Range-split download over parallel connections, with every proxy disabled (standard library only).

    Each part resumes from its own file under .<name>.parts/ across restarts; every request re-follows the
    endpoint's redirect, so expiring signed CDN links are refreshed. The whole file is verified by the caller.
    """
    import os
    import shutil
    import threading
    import time
    import urllib.request

    work = dest.parent / f'.{dest.name}.parts'
    work.mkdir(parents=True, exist_ok=True)
    n = max(1, min(parts, size // (8 << 20)))
    bounds = [(i * size // n, (i + 1) * size // n - 1) for i in range(n)]
    got = [(work / f'{i:03d}').stat().st_size if (work / f'{i:03d}').exists() else 0 for i in range(n)]
    errors = []

    def worker(i):
        a, b = bounds[i]
        part = work / f'{i:03d}'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # {} = no proxy, env or registry
        try:
            for attempt in range(60):
                have = part.stat().st_size if part.exists() else 0
                got[i] = have
                if have == b - a + 1:
                    return
                if have > b - a + 1:
                    raise RuntimeError(f'part {i} longer than its range')
                req = urllib.request.Request(url, headers={'Range': f'bytes={a + have}-{b}',
                                                           'User-Agent': 'so08-fetch/1.0'})
                try:
                    with opener.open(req, timeout=60) as r, part.open('ab') as f:
                        whole = r.status == 200 and a + have == 0 and b == size - 1  # server ignored a full range
                        if r.status != 206 and not whole:
                            raise RuntimeError(f'HTTP {r.status}')
                        while chunk := r.read(1 << 16):
                            f.write(chunk)
                            got[i] += len(chunk)
                except Exception as e:  # network errors: back off and resume from the part's length
                    print(f'  {dest.name} part {i}: retry {attempt + 1} after {type(e).__name__}: {e}', flush=True)
                    time.sleep(min(30, 2 + attempt))
            raise RuntimeError(f'part {i} gave up after 60 attempts')
        except Exception as e:
            errors.append(f'part {i}: {e}')

    start, t0 = sum(got), time.time()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    last = t0
    while any(t.is_alive() for t in threads):
        time.sleep(1)
        if time.time() - last >= log_every:
            done = sum(got)
            print(f'  {dest.name}: {done / 1e6:.1f}/{size / 1e6:.1f} MB, '
                  f'{(done - start) / 1e3 / (time.time() - t0):.0f} KB/s over {n} connections', flush=True)
            last = time.time()
    if errors:
        raise RuntimeError('; '.join(errors))
    tmp = dest.parent / f'.{dest.name}.assembling'
    with tmp.open('wb') as out:
        for i in range(n):
            with (work / f'{i:03d}').open('rb') as f:
                shutil.copyfileobj(f, out, 1 << 22)
    if tmp.stat().st_size != size:
        raise RuntimeError(f'assembled {tmp.stat().st_size} bytes, expected {size}')
    os.replace(tmp, dest)
    shutil.rmtree(work)


def cmd_fetch(kind: str, out: Path, record: Path, endpoint: str | None, resume: bool, parallel: int = 0):
    if (record / 'manifest.json').exists():
        raise SystemExit(f'{record} already holds a finished download')
    lst = json.loads((record / 'listing.json').read_text(encoding='utf-8'))
    if lst['repo'] != SPECS[kind]['repo']:
        raise SystemExit('listing belongs to another repository')
    if out.exists() and any(out.iterdir()) and not resume:
        raise SystemExit(f'{out} is not empty; use --resume to keep verified files')
    out.mkdir(parents=True, exist_ok=True)
    started, rows = now(), []
    for f in lst['files']:
        target = out / f['name']
        if resume and target.exists():
            row = verify(target, f)
            if row['match']:
                rows.append(dict(row, source='kept'))
                print(f'{f["name"]}: kept ({row["verified_by"]})', flush=True)
                continue
        if parallel:
            parallel_download(resolve_url(endpoint or OFFICIAL, lst, f['name']), target, f['size'], parallel)
            path = target
        else:
            path = Path(hf_hub_download(lst['repo'], f['name'], repo_type=lst['repo_type'], revision=lst['revision'],
                                        local_dir=str(out), endpoint=endpoint or OFFICIAL, token=False,
                                        force_download=target.exists()))
        row = verify(path, f)
        rows.append(dict(row, source='downloaded', parallel=parallel))
        print(f'{f["name"]}: {row["bytes"]} bytes, match={row["match"]}', flush=True)
    rec = dict(repo=lst['repo'], repo_type=lst['repo_type'], revision=lst['revision'], license=lst['license'],
               listing_endpoint=lst['endpoint'], download_endpoint=endpoint or OFFICIAL, token_used=False,
               started_utc=started, finished_utc=now(), local_dir=str(out),
               all_match=all(r['match'] for r in rows), files=rows)
    (record / 'manifest.json').write_text(json.dumps(rec, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if not rec['all_match']:
        raise SystemExit('size or hash mismatch; see manifest.json')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('info', 'listing', 'fetch'))
    ap.add_argument('args', nargs='*')
    ap.add_argument('--endpoint')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--parallel', type=int, default=0, help='range-split download over N direct connections')
    a = ap.parse_args(argv)
    if a.command == 'info':
        cmd_info()
    elif a.command == 'listing':
        if len(a.args) != 2:
            ap.error('listing KIND RECORD_DIR')
        cmd_listing(a.args[0], Path(a.args[1]))
    else:
        if len(a.args) != 3:
            ap.error('fetch KIND OUT_DIR RECORD_DIR')
        cmd_fetch(a.args[0], Path(a.args[1]), Path(a.args[2]), a.endpoint, a.resume, a.parallel)


if __name__ == '__main__':
    sys.exit(main())
