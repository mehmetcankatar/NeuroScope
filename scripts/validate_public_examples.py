"""Exercise the running app on public EDFs, then compare with separate annotations.
Requires dev and open-model dependencies. No fitting or threshold tuning is performed.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
import httpx
import mne
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.evaluate import event_metrics, sleep_metrics
from neuroscope.signals import sha256_file

LABELS = ['Awake','N1','N2','N3','REM']

def sleep_reference(path, duration):
    annotations = mne.read_annotations(path)
    mapping = {'Sleep stage W':'Awake','Sleep stage 1':'N1','Sleep stage 2':'N2',
               'Sleep stage 3':'N3','Sleep stage 4':'N3','Sleep stage R':'REM'}
    truth = []
    for start in np.arange(0, duration-29.999, 30.):
        stage = next((mapping[a['description']] for a in annotations
                      if a['description'] in mapping and a['onset'] <= start+15 < a['onset']+a['duration']), None)
        if stage:
            truth.append({'start_s':float(start),'end_s':float(start+30),'stage':stage})
    return truth

def agreement(predicted, truth):
    by_time = {e['start_s']:e for e in predicted}
    rows = [(e['stage'], by_time[e['start_s']]['stage']) for e in truth
            if e['start_s'] in by_time and by_time[e['start_s']]['stage'] in LABELS]
    y,z=zip(*rows)
    return {'reference_epochs':len(truth),'scored_epochs':len(rows),'coverage':len(rows)/len(truth),
            'accuracy':accuracy_score(y,z),'cohen_kappa':cohen_kappa_score(y,z),
            **sleep_metrics(predicted,truth)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    parser.add_argument('--directory',type=Path,default=ROOT/'data/examples')
    parser.add_argument('--output',type=Path,default=ROOT/'data/examples/validation.json')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'configs/public-examples.json').read_text())
    for entry in manifest['files']:
        if sha256_file(args.directory/entry['name']) != entry['sha256']:
            raise ValueError('Example checksum mismatch: '+entry['name'])
    report={'scope':'Two recordings, no tuning; not clinical validation or a guaranteed independent held-out cohort',
            'files':manifest['files'],'records':{}}
    with httpx.Client(base_url=args.base_url,timeout=120) as client:
        health=client.get('/api/health');health.raise_for_status()
        if not health.json()['worker_online']:
            raise RuntimeError('Start python -m neuroscope before running validation')
        for filename,head in [('chb01_03.edf','seizure'),('SC4001E0-PSG.edf','sleep')]:
            print('Importing',filename,flush=True)
            with (args.directory/filename).open('rb') as f:
                response=client.post('/api/recordings',files={'file':(filename,f,'application/octet-stream')})
            response.raise_for_status();record=response.json()
            response=client.post(f"/api/recordings/{record['id']}/analyses",json={'heads':[head]})
            response.raise_for_status();run=response.json();deadline=time.monotonic()+7200
            while run['status'] in ('queued','running','postprocessing'):
                if time.monotonic()>deadline: raise TimeoutError('Analysis exceeded two hours')
                time.sleep(2)
                response=client.get(f"/api/analyses/{run['id']}");response.raise_for_status();run=response.json()
            if run['status']!='complete': raise RuntimeError(run)
            response=client.get(f"/api/analyses/{run['id']}/results");response.raise_for_status();result=response.json()
            (args.directory/(filename+'.results.json')).write_text(json.dumps(result,indent=2))
            for kind in ('json','csv','html'):
                exported=client.get(f"/api/analyses/{run['id']}/export",params={'format':kind});exported.raise_for_status()
                (args.directory/(filename+'.export.'+kind)).write_text(exported.text)
            wave=client.get(f"/api/recordings/{record['id']}/waveform",params={'start':0,'end':20});wave.raise_for_status()
            item={'recording_id':record['id'],'run_id':run['id'],'duration_s':record['duration_s'],
                  'runtime_s':run['runtime_s'],'coverage':result['coverage'],'config':run['config']}
            if head=='seizure':
                summary=(args.directory/'chb01-summary.txt').read_text()
                block=summary.split('File Name: chb01_03.edf')[1].split('File Name:')[0]
                start=float(re.search(r'Seizure Start Time: (\d+)',block)[1]);end=float(re.search(r'Seizure End Time: (\d+)',block)[1])
                truth=[{'start_s':start,'end_s':end}]
                events=[e for e in result['events'] if e['label']=='seizure']
                item.update(metrics=event_metrics(events,truth,record['duration_s']),reference=truth,
                            candidates=[{k:e[k] for k in ('start_s','end_s','peak_score')} for e in events])
            else:
                truth=sleep_reference(args.directory/'SC4001EC-Hypnogram.edf',record['duration_s'])
                raw=[{**e,'stage':max(e['scores'],key=e['scores'].get)} for e in result['sleep'] if e['coverage']>0]
                sleep_times=[e['start_s'] for e in truth if e['stage']!='Awake']
                lo,hi=max(0,min(sleep_times)-1800),max(sleep_times)+1830
                night=[e for e in truth if lo<=e['start_s']<hi]
                item.update(displayed_full_record=agreement(result['sleep'],truth),raw_argmax_full_record=agreement(raw,truth),
                            displayed_sleep_period=agreement(result['sleep'],night),raw_argmax_sleep_period=agreement(raw,night),
                            sleep_period_s=[lo,hi],uncertain_epochs=sum(e['stage']=='Uncertain' for e in result['sleep']))
            report['records'][filename]=item
            print(filename,'complete',run['runtime_s'],'seconds',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Saved',args.output)

if __name__=='__main__': main()
