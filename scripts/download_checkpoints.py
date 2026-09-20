"""Download the nine selected checkpoints, or install already downloaded assets."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            result.remove_header('Authorization')
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--from-dir', type=Path, help='Directory containing the named release assets')
    ap.add_argument('--output', type=Path, default=ROOT / 'run/models')
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    manifest = json.loads((ROOT / 'checkpoints/manifest.json').read_text(encoding='utf-8'))
    token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if not token and not args.from_dir and not args.verify_only and shutil.which('gh'):
        result = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True)
        if result.returncode == 0:
            token = result.stdout.strip()
    headers = {'User-Agent':'joint-sensing-proposals', 'Accept':'application/vnd.github+json'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    opener = urllib.request.build_opener(SafeRedirect())
    assets = None
    for item in manifest['models']:
        dest = args.output.resolve() / item['arm'] / str(item['model_seed']) / 'best.pt'
        if dest.is_file():
            if sha(dest) != item['sha256']:
                raise ValueError(f'Existing checkpoint differs: {dest}')
            print(f"VERIFIED {item['arm']}/{item['model_seed']}", flush=True)
            continue
        if args.verify_only:
            raise FileNotFoundError(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix('.pt.partial')
        try:
            if args.from_dir:
                shutil.copyfile(args.from_dir / item['asset'], temp)
            else:
                if assets is None:
                    url = f"https://api.github.com/repos/{manifest['repository']}/releases/tags/{manifest['release_tag']}"
                    with opener.open(urllib.request.Request(url, headers=headers), timeout=60) as r:
                        release = json.load(r)
                    assets = {a['name']:a for a in release['assets']}
                asset = assets[item['asset']]
                if token:
                    url = asset['url']
                    file_headers = {**headers, 'Accept':'application/octet-stream'}
                else:
                    url = asset['browser_download_url']
                    file_headers = {'User-Agent':'joint-sensing-proposals'}
                with opener.open(urllib.request.Request(url, headers=file_headers), timeout=120) as r, temp.open('wb') as f:
                    shutil.copyfileobj(r, f, length=4 * 1024**2)
            if temp.stat().st_size != item['bytes'] or sha(temp) != item['sha256']:
                raise ValueError(f"Downloaded asset does not match manifest: {item['asset']}")
            os.replace(temp, dest)
        finally:
            if temp.exists():
                temp.unlink()
        print(f"INSTALLED {item['arm']}/{item['model_seed']} (epoch {item['chosen_epoch']})", flush=True)


if __name__ == '__main__':
    main()
