"""Check a local copy of the SHARD code (upstream/sfib) against the commit pinned in upstream/UPSTREAM.json.

SP-16 imports the SHARD authors' code unchanged (sp_core.py puts upstream/sfib on the import path). Fetch the pinned
commit from the original repository, copy its sfib directory, README.md and requirements.txt into upstream/ next to
this file (the 38 files that UPSTREAM.json pins), then verify them:

  git clone https://github.com/Minds-R-Lab/ComplexAttn.git ComplexAttn
  git -C ComplexAttn checkout 96ede1c0bf23195492fe97c7b0ecc5339184512d
  cp -r ComplexAttn/sfib upstream/sfib
  cp ComplexAttn/README.md ComplexAttn/requirements.txt upstream/
  python verify_upstream.py

Standard library only. Exit code 0 when every pinned file is present with the SHA-256 recorded in UPSTREAM.json.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    pin = json.loads((HERE / 'upstream' / 'UPSTREAM.json').read_text(encoding='utf-8'))
    bad = []
    for name, rec in sorted(pin['files'].items()):
        p = HERE / 'upstream' / name
        if not p.is_file():
            bad.append(f'missing  {name}')
        elif hashlib.sha256(p.read_bytes()).hexdigest() != rec['sha256']:
            bad.append(f'differs  {name}')
    print(f"{pin['repository']} @ {pin['commit']}: {len(pin['files']) - len(bad)} of {len(pin['files'])} files verified")
    for line in bad:
        print('  ' + line)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
