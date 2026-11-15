"""Download two public, annotated EEG examples without an account (about 91 MB)."""
import argparse
import json
from pathlib import Path
import requests
from neuroscope.signals import sha256_file

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'data/examples')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT/'configs/public-examples.json').read_text())
    for entry in manifest['files']:
        path = args.directory/entry['name']
        if path.exists() and sha256_file(path) == entry['sha256']:
            print('Verified', path.name, flush=True)
            continue
        partial = path.with_suffix(path.suffix+'.part')
        with requests.get(entry['url'], stream=True, timeout=(30,180)) as response:
            response.raise_for_status()
            with partial.open('wb') as output:
                for chunk in response.iter_content(1024*1024):
                    output.write(chunk)
        if sha256_file(partial) != entry['sha256']:
            partial.unlink()
            raise ValueError('Downloaded file checksum mismatch: '+path.name)
        partial.replace(path)
        print('Downloaded and verified', path.name, flush=True)

if __name__ == '__main__':
    main()
