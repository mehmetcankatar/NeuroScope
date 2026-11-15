"""Create a source-only ZIP, excluding recordings, credentials, weights, and runtime data."""
from pathlib import Path
import argparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git','.venv','venv','node_modules','dist','data','models','vendor','__pycache__','.pytest_cache','.DS_Store','.env','test-results','playwright-report'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as output:
        for path in sorted(ROOT.rglob('*')):
            rel = path.relative_to(ROOT)
            if not path.is_file() or path.resolve()==destination or set(rel.parts)&EXCLUDED:
                continue
            if any(part.endswith('.egg-info') for part in rel.parts) or path.suffix in {'.pyc','.edf','.bdf','.pth','.pt','.mat','.sqlite3','.tsbuildinfo'}:
                continue
            output.write(path,Path('NeuroScope')/rel)
    print(destination)


if __name__=='__main__':
    main()
