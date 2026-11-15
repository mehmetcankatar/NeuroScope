"""Compare chunked SPaRCNet scores to whole-record filtering at fixed seam/clinical windows."""
import argparse
import json
from pathlib import Path
import mne
import numpy as np
import torch
from neuroscope.open_channels import seizure_derivations
from neuroscope.open_inference import prepare_bipolar, preprocess_sparcnet
from neuroscope.sparcnet_network import DenseNetClassifier
from neuroscope.signals import sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',required=True,type=Path)
    parser.add_argument('--scores',required=True,type=Path)
    parser.add_argument('--registry',type=Path,default=Path('models/registry.json'))
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    cfg=json.loads(args.registry.read_text())['sparcnet']
    if sha256_file(cfg['checkpoint'])!=cfg['sha256']: raise ValueError('Checkpoint mismatch')
    torch.set_num_threads(2)
    model=DenseNetClassifier().eval()
    model.load_state_dict(torch.load(cfg['checkpoint'],weights_only=True,map_location='cpu'),strict=True)
    raw=mne.io.read_raw_edf(args.recording,preload=False,verbose='ERROR')
    try:
        values=prepare_bipolar(raw,seizure_derivations(raw.ch_names),0,raw.n_times)
        filtered=preprocess_sparcnet(values,raw.info['sfreq'])
    finally:
        raw.close()
    with np.load(args.scores,allow_pickle=False) as arrays:
        starts=arrays['starts'];saved=arrays['scores']
    chosen=[t for t in [0,2,588,590,598,600,602,1198,1200,1798,1800,2398,2400,2990,2996,2998,3000,3036,3590] if t in starts]
    windows=np.asarray([filtered[:,int(t*200):int(t*200)+2000] for t in chosen])
    with torch.inference_mode():
        probabilities=torch.softmax(model(torch.from_numpy(np.clip(windows,-500,500).astype('float32')))[0],dim=1).numpy()
    errors=np.abs(probabilities-np.asarray([saved[np.flatnonzero(starts==t)[0]] for t in chosen]))
    report={'method':'Whole-record preprocessing vs 600-second chunks with 30-second context; fixed boundary and event windows',
            'windows_s':chosen,'max_absolute_score_difference':float(errors.max()),
            'tolerance':0.01,'passed':bool(errors.max()<0.01),'input_sha256':sha256_file(args.recording),
            'conversion_max_error':cfg['conversion_max_error']}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(report)
    if not report['passed']: raise RuntimeError('Chunk equivalence tolerance exceeded')

if __name__=='__main__': main()
