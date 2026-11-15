"""Download public EEGs and load their saved real-model analyses into a local workspace."""
import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from neuroscope.config import Settings
from neuroscope.db import Store, atomic_json
from neuroscope.signals import sha256_file
ROOT = Path(__file__).resolve().parents[1]

def load_examples(data_dir, examples_dir, bundle=ROOT/'examples/public-analyses.zip'):
    store=Store(data_dir)
    with zipfile.ZipFile(bundle) as archive:
        for entry in json.loads(archive.read('manifest.json')):
            filename,head=entry['filename'],entry['head']
            source=examples_dir/filename
            if sha256_file(source)!=entry['sha256']:
                raise ValueError('Public example checksum mismatch: '+filename)
            payload=json.loads(archive.read(filename+'/metadata.json'))
            record,run=payload['recording'],payload['run']
            if store.recording(record['id']) and store.run(run['id']):
                print('Already loaded',filename);continue
            path=data_dir/'recordings'/f"{record['id']}.edf"
            shutil.copyfile(source,path);record['path']=str(path.resolve())
            directory=data_dir/'runs'/run['id'];directory.mkdir(exist_ok=True)
            atomic_json(directory/'results.json',payload['results'])
            (directory/f'{head}.npz').write_bytes(archive.read(filename+'/'+head+'.npz'))
            run.update(status='complete',progress=100,message='Saved public-data analysis; rerun Analyze for new predictions')
            now=time.time()
            # Insert completed runs atomically so an active worker cannot claim them.
            with store.connect() as db:
                db.execute('INSERT OR IGNORE INTO recordings VALUES (?,?)',(record['id'],json.dumps(record)))
                db.execute('INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?,?)',
                           (run['id'],record['id'],'complete',run['created'],now,0,json.dumps(run)))
            print('Loaded',filename,'with saved',head,'analysis')
    return store

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Settings.from_env().data_dir)
    parser.add_argument('--examples-dir',type=Path,default=ROOT/'data/examples')
    args=parser.parse_args()
    subprocess.run([sys.executable,str(ROOT/'scripts/download_examples.py'),'--directory',str(args.examples_dir)],check=True)
    load_examples(args.data_dir.resolve(),args.examples_dir.resolve())
    print('Start NeuroScope with the same NEUROSCOPE_DATA_DIR and open Recordings.')

if __name__=='__main__':main()
