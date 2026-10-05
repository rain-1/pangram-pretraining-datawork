"""Acquire the pinned diverse-corpus raw files without replacing mismatched files."""

import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent


def fetch():
    config = json.loads((ROOT / 'config/diverse_downloads.json').read_text())
    raw = ROOT / config['raw_directory']
    raw.mkdir(parents=True, exist_ok=True)
    for spec in config['files']:
        if Path(spec['local_path']).name != spec['local_path']:
            raise ValueError('Expected a plain raw filename')
        path = raw / spec['local_path']
        if path.exists():
            if len(path.read_bytes()) != spec['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
                raise ValueError(f'Existing raw pin differs: {path}')
            continue
        request = urllib.request.Request(spec['url'], headers={'User-Agent': 'Mozilla/5.0 (compatible; pretraining-datawork research)'})
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read()
        if len(data) != spec['bytes'] or hashlib.sha256(data).hexdigest() != spec['sha256']:
            raise ValueError(f'Upstream file changed; raw pin preserved: {spec["url"]}')
        path.write_bytes(data)
    acquisition = raw / 'acquisition.json'
    if not acquisition.exists():
        acquisition.write_text(json.dumps({'scope': 'Acquired against frozen download configuration', 'files': config['files'], 'failures': []}, indent=2) + '\n')
    print(f'Verified {len(config["files"])} pinned raw files')


if __name__ == '__main__':
    fetch()
