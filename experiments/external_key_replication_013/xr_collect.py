"""XR-13 collection: version histories from public package registries (one family per registry).

Every entity record keeps what the question pipeline needs (name, popularity rank, every version with its publication
time, whether it is a pre-release or yanked/unlisted, the version's runtime requirement and licence where the registry
exposes them) plus the source URL, the fetch time and the SHA-256 of the raw response, so any value can be traced
back. Personal data (authors, maintainers, e-mails) is never stored.

  python xr_collect.py FAMILY OUT_DIR [--limit N] [--workers K] [--names a,b,...]

Families: pypi, npm, cran, rubygems, nuget, crates (crates.io database dump), hackage (01-index), and the git
families homebrew, winget (commit subjects of a tree-less partial clone).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = 'research-replication-study/0.1 (knowledge-editing benchmark; no personal data stored)'


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def fetch(url: str, max_bytes: int = 64 << 20, headers: dict | None = None, tries: int = 5) -> tuple[bytes, str]:
    """GET with retries (429/5xx/network back off). Returns (body, sha256)."""
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept-Encoding': 'gzip', **(headers or {})})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    raise ValueError(f'response larger than {max_bytes} bytes')
                if r.headers.get('Content-Encoding') == 'gzip':
                    raw = gzip.decompress(raw)
                return raw, hashlib.sha256(raw).hexdigest()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if k == tries - 1:
                raise
            time.sleep(min(60, 3 * 2 ** k) if e.code == 429 else 2 * (k + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError('unreachable')


def fetch_json(url, **kw):
    raw, h = fetch(url, **kw)
    return json.loads(raw.decode('utf-8')), h


def short_license(x) -> str | None:
    """A licence identifier, not a licence text (long values are dropped)."""
    if isinstance(x, list):
        x = ' OR '.join(str(v) for v in x if v)
    if not x or not isinstance(x, str):
        return None
    x = ' '.join(x.split())
    return x if len(x) <= 60 else None


def repo_of(urls) -> str | None:
    """A normalised source repository (github/gitlab/bitbucket/codeberg owner/name), for grouping across families."""
    for u in urls:
        if not isinstance(u, str):
            continue
        m = re.search(r'(github\.com|gitlab\.com|bitbucket\.org|codeberg\.org)[/:]([\w.-]+)/([\w.-]+)', u, re.I)
        if m:
            name = re.sub(r'\.git$', '', m.group(3))
            return f'{m.group(1).lower()}/{m.group(2).lower()}/{name.lower()}'
    return None


# ---- PyPI ----------------------------------------------------------------------------------------------------------
def pypi_list(n: int) -> list[tuple[str, int]]:
    d, _ = fetch_json('https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json')
    rows = d['rows'][:n]
    return [(r['project'], int(r['download_count'])) for r in rows]


def pypi_entity(name: str) -> dict:
    from packaging.version import InvalidVersion, Version
    url = f'https://pypi.org/pypi/{urllib.parse.quote(name)}/json'
    d, h = fetch_json(url)
    info = d['info']
    versions = []
    for v, files in d.get('releases', {}).items():
        if not files:
            continue
        try:
            pv = Version(v)
        except InvalidVersion:
            continue
        times = sorted(f['upload_time_iso_8601'] for f in files if f.get('upload_time_iso_8601'))
        req = [f.get('requires_python') for f in files if f.get('requires_python')]
        spec = max(set(req), key=req.count) if req else None
        versions.append(dict(v=v, t=times[0][:19] + 'Z' if times else None, pre=pv.is_prerelease or pv.is_devrelease,
                             yanked=all(f.get('yanked') for f in files), spec=spec))
    urls = list((info.get('project_urls') or {}).values()) + [info.get('home_page')]
    return dict(entity=info['name'], versions=versions,
                license_latest=short_license(info.get('license_expression') or info.get('license')),
                repo=repo_of(urls), src=url, sha256=h, fetched=now())


# ---- npm -----------------------------------------------------------------------------------------------------------
def npm_list(n: int) -> list[tuple[str, int]]:
    raw, _ = fetch('https://raw.githubusercontent.com/wooorm/npm-high-impact/main/lib/top-download.js')
    names = re.findall(r"'((?:@[\w.-]+/)?[\w.-]+)'", raw.decode('utf-8'))
    return [(x, i) for i, x in enumerate(names[:n])]


def npm_entity(name: str) -> dict:
    url = 'https://registry.npmjs.org/' + urllib.parse.quote(name, safe='@')
    d, h = fetch_json(url, max_bytes=48 << 20)
    times = d.get('time') or {}
    versions = []
    for v, meta in (d.get('versions') or {}).items():
        eng = (meta.get('engines') or {})
        node = eng.get('node') if isinstance(eng, dict) else None
        versions.append(dict(v=v, t=(times.get(v) or '')[:19] + 'Z' if times.get(v) else None, pre='-' in v,
                             yanked=bool(meta.get('deprecated')), spec=node if isinstance(node, str) else None,
                             license=short_license(meta.get('license') if isinstance(meta.get('license'), str)
                                                  else (meta.get('license') or {}).get('type')
                                                  if isinstance(meta.get('license'), dict) else None)))
    repo = d.get('repository')
    repo = repo.get('url') if isinstance(repo, dict) else repo
    return dict(entity=d.get('name', name), versions=versions, repo=repo_of([repo, d.get('homepage')]),
                license_latest=short_license(d.get('license') if isinstance(d.get('license'), str) else None),
                src=url, sha256=h, fetched=now())


# ---- CRAN ----------------------------------------------------------------------------------------------------------
def cran_list(n: int) -> list[tuple[str, int]]:
    d, _ = fetch_json('https://crandb.r-pkg.org/-/desc')
    names = sorted(d)
    counts = {}
    for s in range(0, len(names), 50):
        part = names[s:s + 50]
        try:
            rows, _ = fetch_json('https://cranlogs.r-pkg.org/downloads/total/last-month/' + ','.join(part))
        except Exception:
            rows = []
        for r in rows if isinstance(rows, list) else []:
            counts[r['package']] = int(r.get('downloads') or 0)
        time.sleep(0.2)
    ranked = sorted(names, key=lambda x: (-counts.get(x, 0), x))
    return [(x, counts.get(x, 0)) for x in ranked[:n]]


def cran_entity(name: str) -> dict:
    url = f'https://crandb.r-pkg.org/{urllib.parse.quote(name)}/all'
    d, h = fetch_json(url)
    timeline = d.get('timeline') or {}
    versions = []
    for v, meta in (d.get('versions') or {}).items():
        dep = (meta.get('Depends') or {})
        spec = dep.get('R') if isinstance(dep, dict) else None
        t = timeline.get(v)
        versions.append(dict(v=v, t=(t[:19] + 'Z') if t else None, pre=False, yanked=False, spec=spec,
                             license=short_license(meta.get('License'))))
    latest = (d.get('versions') or {}).get(d.get('latest') or '', {})
    urls = re.split(r'[,\s]+', ' '.join(str(latest.get(k) or '') for k in ('URL', 'BugReports')))
    return dict(entity=d.get('name', name), versions=versions, repo=repo_of(urls),
                license_latest=short_license(latest.get('License')), archived=bool(d.get('archived')),
                src=url, sha256=h, fetched=now())


# ---- NuGet ---------------------------------------------------------------------------------------------------------
def nuget_list(n: int) -> list[tuple[str, int]]:
    out = []
    for skip in range(0, n, 1000):
        d, _ = fetch_json(f'https://azuresearch-usnc.nuget.org/query?q=&take={min(1000, n - skip)}&skip={skip}'
                          '&prerelease=false&semVerLevel=2.0.0')
        out += [(r['id'], int(r.get('totalDownloads') or 0)) for r in d.get('data', [])]
    out.sort(key=lambda x: -x[1])
    return out[:n]


def nuget_entity(name: str) -> dict:
    url = f'https://api.nuget.org/v3/registration5-gz-semver2/{name.lower()}/index.json'
    d, h = fetch_json(url)
    hashes = [h]
    entries = []
    for page in d.get('items', []):
        items = page.get('items')
        if items is None:
            pd, ph = fetch_json(page['@id'])
            hashes.append(ph)
            items = pd.get('items', [])
        entries += [it['catalogEntry'] for it in items if 'catalogEntry' in it]
    versions = []
    for c in entries:
        t = c.get('published') or ''
        listed = c.get('listed', True) and not t.startswith('1900')
        tfs = sorted({g.get('targetFramework') for g in c.get('dependencyGroups') or [] if g.get('targetFramework')})
        versions.append(dict(v=c['version'], t=t[:19] + 'Z' if t and not t.startswith('1900') else None,
                             pre='-' in c['version'], yanked=not listed, spec=None,
                             license=short_license(c.get('licenseExpression')), frameworks=tfs))
    last = entries[-1] if entries else {}
    return dict(entity=last.get('id', name), versions=versions, repo=repo_of([last.get('projectUrl')]),
                license_latest=short_license(last.get('licenseExpression')), src=url,
                sha256=hashlib.sha256(''.join(hashes).encode()).hexdigest(), fetched=now())


# ---- Packagist -----------------------------------------------------------------------------------------------------
def packagist_list(n: int) -> list[tuple[str, int]]:
    out = []
    page = 1
    while len(out) < n:
        d, _ = fetch_json(f'https://packagist.org/explore/popular.json?per_page=100&page={page}')
        rows = d.get('packages') or []
        if not rows:
            break
        out += [(r['name'], int(r.get('downloads') or 0)) for r in rows]
        page += 1
        time.sleep(0.3)
    return out[:n]


def _expand_minified(entries: list[dict]) -> list[dict]:
    """Composer 2 minified metadata: every entry lists only the fields that differ from the previous one."""
    out, cur = [], {}
    for e in entries:
        cur = dict(cur)
        for k, v in e.items():
            if v == '__unset':
                cur.pop(k, None)
            else:
                cur[k] = v
        out.append(cur)
    return out


def packagist_entity(name: str) -> dict:
    url = f'https://repo.packagist.org/p2/{name}.json'
    d, h = fetch_json(url)
    entries = (d.get('packages') or {}).get(name, [])
    if d.get('minified') == 'composer/2.0':
        entries = _expand_minified(entries)
    versions = []
    for e in entries:
        v = e.get('version', '')
        stab = re.search(r'(dev|alpha|beta|rc|a\d|b\d|-)', v.lower())
        req = e.get('require') or {}
        versions.append(dict(v=v, t=(e.get('time') or '')[:19] + 'Z' if e.get('time') else None, pre=bool(stab),
                             yanked=False, spec=req.get('php') if isinstance(req, dict) else None,
                             license=short_license(e.get('license'))))
    first = entries[0] if entries else {}
    src = first.get('source') or {}
    return dict(entity=name, versions=versions, repo=repo_of([src.get('url'), first.get('homepage')]),
                license_latest=short_license(first.get('license')), src=url, sha256=h, fetched=now())


# ---- bulk families: crates.io dump, Hackage index, git histories, endoflife.date ------------------------------------------
def crates_bulk(n: int, work: Path) -> list[dict]:
    """crates.io's daily database dump: crates, downloads and every version (licence, rust_version, yanked)."""
    import csv
    csv.field_size_limit(2 ** 31 - 1)
    url = 'https://static.crates.io/db-dump.tar.gz'
    dump = work / 'crates-db-dump.tar.gz'
    if not dump.exists():
        req = urllib.request.Request(url, headers={'User-Agent': UA})
        with urllib.request.urlopen(req, timeout=600) as r, open(dump, 'wb') as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
    sha = hashlib.sha256(dump.read_bytes()).hexdigest()
    crates, downloads, versions = {}, {}, {}
    with tarfile.open(dump, 'r:gz') as tar:
        for m in tar:
            base = m.name.rsplit('/', 1)[-1]
            if base not in ('crates.csv', 'crate_downloads.csv', 'versions.csv'):
                continue
            f = io.TextIOWrapper(tar.extractfile(m), encoding='utf-8', newline='')
            for row in csv.DictReader(f):
                if base == 'crates.csv':
                    crates[row['id']] = dict(name=row['name'], repo=repo_of([row.get('repository'), row.get('homepage')]),
                                             downloads=int(row.get('downloads') or 0))
                elif base == 'crate_downloads.csv':
                    downloads[row['crate_id']] = int(row.get('downloads') or 0)
                else:
                    versions.setdefault(row['crate_id'], []).append(dict(
                        v=row['num'], t=(row['created_at'][:19].replace(' ', 'T') + 'Z') if row.get('created_at') else None,
                        pre='-' in row['num'], yanked=row.get('yanked') in ('t', 'true', 'True', '1'),
                        spec=row.get('rust_version') or None, license=short_license(row.get('license'))))
    for cid, c in crates.items():
        c['downloads'] = downloads.get(cid, c['downloads'])
    top = sorted(crates, key=lambda k: -crates[k]['downloads'])[:n]
    out = []
    for i, cid in enumerate(top):
        vs = versions.get(cid, [])
        lic = sorted((v for v in vs if v['t']), key=lambda v: v['t'])
        out.append(dict(entity=crates[cid]['name'], versions=vs, repo=crates[cid]['repo'],
                        license_latest=lic[-1]['license'] if lic else None, src=url, sha256=sha, fetched=now(),
                        query=crates[cid]['name'], rank=i, popularity=crates[cid]['downloads']))
    return out


def _cabal_field(text: str, field: str) -> str | None:
    m = re.search(rf'^\s*{field}\s*:\s*(.+)$', text, re.I | re.M)
    return m.group(1).strip() if m else None


def hackage_bulk(n: int, work: Path) -> list[dict]:
    """Hackage's 01-index: every .cabal upload (tar mtime = upload time) and its later metadata revisions."""
    url = 'https://hackage.haskell.org/01-index.tar.gz'
    path = work / 'hackage-01-index.tar.gz'
    if not path.exists():
        raw, _ = fetch(url, max_bytes=1 << 30)
        path.write_bytes(raw)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    first, latest_text = {}, {}
    with tarfile.open(path, 'r:gz') as tar:
        for m in tar:
            parts = m.name.split('/')
            if len(parts) != 3 or not parts[2].endswith('.cabal'):
                continue
            key = (parts[0], parts[1])
            if key not in first:
                first[key] = m.mtime
            latest_text[parts[0]] = tar.extractfile(m).read().decode('utf-8', 'replace')
    by_pkg = {}
    for (pkg, ver), mt in first.items():
        by_pkg.setdefault(pkg, []).append(dict(v=ver, t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(mt)),
                                               pre=False, yanked=False, spec=None))
    recent = {p: max(v['t'] for v in vs) for p, vs in by_pkg.items()}
    names = sorted(by_pkg, key=lambda p: recent[p], reverse=True)[:n]
    out = []
    for i, p in enumerate(names):
        text = latest_text.get(p, '')
        out.append(dict(entity=p, versions=by_pkg[p], repo=repo_of([_cabal_field(text, 'homepage') or '',
                                                                    _cabal_field(text, 'bug-reports') or '', text]),
                        license_latest=short_license(_cabal_field(text, 'license')), src=url, sha256=sha,
                        fetched=now(), query=p, rank=i, popularity=None))
    return out


def _git_subjects(repo_url: str, work: Path, since: str) -> tuple[list[tuple[str, str]], str]:
    import subprocess
    d = work / re.sub(r'\W', '_', repo_url.rsplit('/', 1)[-1])
    if not d.exists():
        subprocess.run(['git', 'clone', '--bare', '--filter=tree:0', repo_url, str(d)], check=True)
    head = subprocess.run(['git', '-C', str(d), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()
    log = subprocess.run(['git', '-C', str(d), 'log', f'--since={since}', '--format=%cI%x09%s'],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in log.splitlines():
        t, _, s = line.partition('\t')
        rows.append((t, s))
    return rows, head


def _utc(t: str) -> str:
    from datetime import datetime, timezone
    return datetime.fromisoformat(t).astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def homebrew_bulk(n: int, work: Path, since: str = '2025-01-01') -> list[dict]:
    rows, head = _git_subjects('https://github.com/Homebrew/homebrew-core.git', work, since)
    pat = re.compile(r'^([a-z0-9][\w@.+-]*) (\d[\w.+~-]*)(?: \(new formula\))?$')
    by = {}
    for t, s in rows:
        m = pat.match(s)
        if m:
            by.setdefault(m.group(1), {})[m.group(2)] = _utc(t)
    ana, _ = fetch_json('https://formulae.brew.sh/api/analytics/install-on-request/365d.json')
    rank = {r['formula']: int(str(r['count']).replace(',', '')) for r in ana.get('items', [])}
    meta, msha = fetch_json('https://formulae.brew.sh/api/formula.json', max_bytes=256 << 20)
    info = {f['name']: f for f in meta}
    names = sorted((x for x in by if x in info), key=lambda x: -rank.get(x, 0))[:n]
    out = []
    for i, x in enumerate(names):
        vs = [dict(v=v, t=t, pre=False, yanked=False, spec=None) for v, t in sorted(by[x].items(), key=lambda z: z[1])]
        f = info[x]
        out.append(dict(entity=x, versions=vs, repo=repo_of([f.get('homepage'), (f.get('urls') or {}).get('stable', {}).get('url')]),
                        license_latest=short_license(f.get('license')), desc=f.get('desc'),
                        src=f'https://github.com/Homebrew/homebrew-core@{head}', sha256=head, fetched=now(), query=x,
                        rank=i, popularity=rank.get(x, 0)))
    return out


def winget_bulk(n: int, work: Path, since: str = '2025-01-01') -> list[dict]:
    rows, head = _git_subjects('https://github.com/microsoft/winget-pkgs.git', work, since)
    add = re.compile(r'^(?:New version|New package|Update version|Add version)s?: ([\w.+-]+) (?:version )?([\w.+~-]+)')
    rem = re.compile(r'^(?:Remove version|Automatic deletion|Deletion)s?: ([\w.+-]+) (?:version )?([\w.+~-]+)', re.I)
    by, gone = {}, set()
    for t, s in rows:
        m = add.match(s)
        if m:
            by.setdefault(m.group(1), {}).setdefault(m.group(2), _utc(t))
            continue
        m = rem.match(s)
        if m:
            gone.add((m.group(1), m.group(2)))
    names = sorted(by, key=lambda x: (-len(by[x]), x))[:n]
    out = []
    for i, x in enumerate(names):
        vs = [dict(v=v, t=t, pre=bool(re.search(r'(alpha|beta|rc|preview|nightly|insider|canary|dev)', v, re.I)),
                   yanked=(x, v) in gone, spec=None) for v, t in sorted(by[x].items(), key=lambda z: z[1])]
        out.append(dict(entity=x, versions=vs, repo=None, license_latest=None, publisher=x.split('.', 1)[0],
                        src=f'https://github.com/microsoft/winget-pkgs@{head}', sha256=head, fetched=now(), query=x,
                        rank=i, popularity=len(by[x])))
    return out


def endoflife_bulk(n: int, work: Path) -> list[dict]:
    import subprocess
    d = work / 'release-data'
    if not d.exists():
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/endoflife-date/release-data.git', str(d)],
                       check=True)
    head = subprocess.run(['git', '-C', str(d), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()
    out = []
    for p in sorted((d / 'releases').glob('*.json')):
        j = json.loads(p.read_text(encoding='utf-8'))
        vs = [dict(v=k, t=(v.get('date') or '')[:10] + 'T00:00:00Z' if v.get('date') else None,
                   pre=bool(re.search(r'(alpha|beta|rc|preview|dev)', k, re.I)), yanked=False, spec=None)
              for k, v in (j.get('versions') or {}).items()]
        out.append(dict(entity=p.stem, versions=vs, repo=None, license_latest=None,
                        src=f'https://github.com/endoflife-date/release-data@{head}', sha256=head, fetched=now()))
    for i, r in enumerate(sorted(out, key=lambda r: -len(r['versions']))):
        r.update(query=r['entity'], rank=i, popularity=len(r['versions']))
    return out[:n]


FAMILIES = {
    'pypi': (pypi_list, pypi_entity),
    'npm': (npm_list, npm_entity),
    'cran': (cran_list, cran_entity),
    'nuget': (nuget_list, nuget_entity),
    'packagist': (packagist_list, packagist_entity),
}
BULK = {'crates': crates_bulk, 'hackage': hackage_bulk, 'homebrew': homebrew_bulk, 'winget': winget_bulk,
        'endoflife': endoflife_bulk}


def collect(family: str, out: Path, limit: int, workers: int, names: list[str] | None = None):
    out.mkdir(parents=True, exist_ok=True)
    if family in BULK:
        work = out / '_work'
        work.mkdir(exist_ok=True)
        recs = BULK[family](limit, work)
        with gzip.open(out / f'{family}_entities.jsonl.gz', 'wt', encoding='utf-8') as g:
            for r in recs:
                r['family'] = family
                g.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'{family}: done, {len(recs)} entities', flush=True)
        return
    lister, getter = FAMILIES[family]
    listing = [(x, None) for x in names] if names else lister(limit)
    (out / f'{family}_list.json').write_text(json.dumps(dict(family=family, fetched=now(), n=len(listing),
                                                             items=listing)), encoding='utf-8')
    path = out / f'{family}_entities.jsonl.gz'
    done = set()
    if path.exists():
        with gzip.open(path, 'rt', encoding='utf-8') as g:
            done = {json.loads(line)['query'] for line in g}
    todo = [(i, x, p) for i, (x, p) in enumerate(listing) if x not in done]
    errors = 0
    with gzip.open(path, 'at', encoding='utf-8') as g, cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(getter, x): (i, x, p) for i, x, p in todo}
        for k, f in enumerate(cf.as_completed(futs)):
            i, x, p = futs[f]
            try:
                rec = f.result()
            except Exception as e:  # keep going; the error is recorded
                errors += 1
                rec = dict(entity=x, error=repr(e)[:200], versions=[])
            rec.update(query=x, rank=i, popularity=p, family=family)
            g.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if k % 200 == 0:
                print(f'{family}: {k + 1}/{len(todo)} errors {errors}', flush=True)
    print(f'{family}: done, {len(todo)} fetched, {errors} errors', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('family', choices=sorted(set(FAMILIES) | set(BULK)))
    ap.add_argument('out')
    ap.add_argument('--limit', type=int, default=4000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--names', default=None)
    a = ap.parse_args(argv)
    collect(a.family, Path(a.out), a.limit, a.workers, a.names.split(',') if a.names else None)


if __name__ == '__main__':
    main()
