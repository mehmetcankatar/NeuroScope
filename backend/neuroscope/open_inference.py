"""CPU inference in an isolated child process. Inputs never come from annotations."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import mne
from .config import CLASSES
from .db import atomic_json
from .signals import sha256_file
from .open_channels import seizure_derivations, sleep_channels


def prepare_bipolar(raw, derivations, start, stop):
    names = list(dict.fromkeys(n for p in derivations for n in [p['positive'], p['negative'], *p['duplicates']] if n))
    values = raw.get_data(picks=names, start=start, stop=stop, units='uV')
    lookup = dict(zip(names, values))
    output = []
    for p in derivations:
        signal = lookup[p['positive']]
        for duplicate in p['duplicates']:
            if not np.allclose(signal, lookup[duplicate], atol=.001, rtol=0, equal_nan=True):
                raise ValueError(f"Duplicate channels disagree: {p['positive']} and {duplicate}; export an unambiguous montage")
        if p['negative']:
            signal = signal - lookup[p['negative']]
        output.append(signal * p['sign'])
    return np.asarray(output)


def preprocess_sparcnet(values, fs):
    if not np.isfinite(values).all():
        raise ValueError("Non-finite samples in EEG; import continuous finite segments")
    if fs != 200:
        values = mne.filter.resample(values, up=200, down=fs, npad='auto', verbose='ERROR')
    # Match the authors' 60-Hz notch and 0.5–40-Hz FIR preprocessing, in microvolts.
    values = mne.filter.notch_filter(values, 200, 60, n_jobs=1, verbose='ERROR')
    return mne.filter.filter_data(values, 200, .5, 40, n_jobs=1, verbose='ERROR')


def seizure(recording, registry, progress):
    import torch
    from .sparcnet_network import DenseNetClassifier
    cfg = registry['sparcnet']
    if sha256_file(cfg['checkpoint']) != cfg['sha256']:
        raise ValueError('SPaRCNet checksum mismatch; rerun setup_open_models.py')
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    model = DenseNetClassifier().eval()
    model.load_state_dict(torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True), strict=True)
    raw = mne.io.read_raw_edf(recording['path'], preload=False, verbose='ERROR')
    try:
        fs = raw.info['sfreq']; duration = raw.n_times / fs
        derivations = seizure_derivations(raw.ch_names)
        starts = np.arange(0, duration - 10 + 1e-8, 2.)
        scores = np.full((len(starts), 6), np.nan)
        valid = np.zeros(len(starts), dtype=bool)
        for left in np.arange(0, duration, 600.):
            indices = np.flatnonzero((starts >= left) & (starts < left + 600))
            if not len(indices):
                continue
            context_left = max(0., left - 30)
            context_right = min(duration, starts[indices[-1]] + 40)
            values = prepare_bipolar(raw, derivations, round(context_left * fs), round(context_right * fs))
            filtered = preprocess_sparcnet(values, fs)
            for batch_start in range(0, len(indices), 32):
                batch_indices = indices[batch_start:batch_start + 32]
                windows = np.array([filtered[:, round((starts[i] - context_left) * 200):round((starts[i] - context_left) * 200) + 2000] for i in batch_indices])
                # Flat channels are not treated as evidence of a normal record.
                usable = np.all(np.std(windows, axis=2) > .01, axis=1)
                for annotation in raw.annotations:
                    if annotation['description'].lower().startswith(('bad', 'edge')):
                        a, b = annotation['onset'], annotation['onset'] + annotation['duration']
                        usable &= ~((starts[batch_indices] < b) & (starts[batch_indices] + 10 > a))
                if usable.any():
                    clipped = np.clip(windows[usable], -500, 500).astype('float32')
                    with torch.inference_mode():
                        logits, _ = model(torch.from_numpy(clipped))
                        scores[batch_indices[usable]] = torch.softmax(logits, dim=1).numpy()
                    valid[batch_indices[usable]] = True
                progress((batch_indices[-1] + 1) / len(starts))
        return starts, scores, valid, {"model": "SPaRCNet", **cfg, "window_s": 10, "stride_s": 2,
            "preprocessing": "bipolar-16-resample200-notch60-bandpass0.5-40-clip500-v1", "context_s": 30,
            "derivations": derivations, "classes": CLASSES['seizure'], "device": "cpu",
            "timing": "window centers; estimated intervals, not annotated onset/offset"}
    finally:
        raw.close()


def sleep(recording, registry, progress):
    # The macOS PyTorch wheel supplies libomp, also needed by LightGBM.
    # Load it before YASA rather than requiring a separate Homebrew installation.
    if sys.platform == 'darwin':
        import torch
    import yasa
    cfg = registry['yasa']
    if yasa.__version__ != cfg['version']:
        raise ValueError('YASA version changed; rerun setup_open_models.py')
    root = Path(yasa.__file__).parent / 'classifiers'
    for name, digest in cfg['classifier_hashes'].items():
        if sha256_file(root / name) != digest:
            raise ValueError('YASA classifier checksum mismatch')
    raw = mne.io.read_raw_edf(recording['path'], preload=False, verbose='ERROR')
    try:
        channels = sleep_channels(raw.ch_names, recording.get('channel_rates'))
        raw.pick([n for n in channels.values() if n]).load_data()
        raw.set_channel_types({n: t for t, n in channels.items() if n}, verbose='ERROR')
        progress(.15)
        stager = yasa.SleepStaging(raw, eeg_name=channels['eeg'], eog_name=channels['eog'], emg_name=channels['emg'])
        stager.fit()
        progress(.75)
        stager.predict()
        probabilities = stager.predict_proba()
        scores = probabilities[['W', 'N1', 'N2', 'N3', 'R']].to_numpy()
        starts = np.arange(len(scores)) * 30.
        valid = np.isfinite(scores).all(axis=1)
        eeg = raw.get_data(picks=[channels['eeg']], units='uV')[0]
        fs = raw.info['sfreq']
        for i, start in enumerate(starts):
            segment = eeg[round(start * fs):round((start + 30) * fs)]
            valid[i] &= np.isfinite(segment).all() and np.std(segment) > .01
        for a in raw.annotations:
            if a['description'].lower().startswith(('bad', 'edge')):
                valid &= ~((starts < a['onset'] + a['duration']) & (starts + 30 > a['onset']))
        scores[~valid] = np.nan
        progress(1)
        return starts, scores, valid, {"model": "YASA", "version": yasa.__version__, "window_s": 30,
            "stride_s": 30, "channels": channels, "classifier_hashes": cfg['classifier_hashes'],
            "dependencies": cfg['dependencies'], "preprocessing": "YASA-native-whole-recording-v1",
            "caveat": "Trained on full-night PSG; montage, age and pathology can affect performance"}
    finally:
        raw.close()


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    registry = json.loads(Path(request['registry']).read_text())
    def progress(value):
        atomic_json(Path(request['progress']), {'fraction': float(value)})
    function = {'seizure': seizure, 'sleep': sleep}[request['head']]
    starts, scores, valid, provenance = function(request['recording'], registry, progress)
    np.savez_compressed(request['output'], starts=starts, scores=scores, valid=valid)
    atomic_json(Path(request['provenance']), provenance)


if __name__ == '__main__':
    main()
