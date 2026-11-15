import numpy as np
from neuroscope.config import EEG_CHANNELS
from neuroscope.signals import capabilities, channel_map, demo_recording, waveform, inspect_edf


def test_bipolar_labels_are_never_mapped_to_electrodes():
    mapping, _, _ = channel_map(['FP1-F7','F7-T7','C3-REF'])
    assert mapping == {'C3':'C3-REF'}
    assert not capabilities(['FP1-F7','F7-T7'])[0]['seizure']['supported']


def test_aliases_and_reference_compatibility():
    names = [c.replace('T3','T7').replace('T4','T8').replace('T5','P7').replace('T6','P8')+'-REF' for c in EEG_CHANNELS]
    caps,_,_=capabilities(names)
    assert all(c['supported'] for c in caps.values())
    names[0] = 'FP1-A1'
    assert not capabilities(names)[0]['seizure']['supported']


def test_envelope_preserves_min_and_max_and_gaps():
    demo = demo_recording()
    data = waveform(demo,1755,1785,['FP1'],100)
    assert len(data['times']) <= 100
    assert any(v is None for v in data['channels'][0]['min'])
    assert data['times'][0] == 1755


def test_real_edf_round_trip(tmp_path):
    import mne
    fs=200; t=np.arange(4000)/fs
    data=np.array([np.sin(2*np.pi*(8+i/10)*t)*20e-6 for i in range(19)])
    raw=mne.io.RawArray(data,mne.create_info([c+'-REF' for c in EEG_CHANNELS],fs,'eeg'),verbose='ERROR')
    path=tmp_path/'signal.edf'
    raw.export(path,fmt='edf',overwrite=True,verbose='ERROR')
    result=inspect_edf(path,'test','signal.edf')
    assert result['duration_s'] == 20
    assert result['capabilities']['seizure']['supported']
    assert not result['capabilities']['abnormal']['supported']
    assert not result['capabilities']['sleep']['supported']  # shorter than 5 minutes
    displayed=waveform(result,5,6,[result['channels'][0]],300)
    assert displayed['times'][0] == 5
    assert max(displayed['channels'][0]['max']) > 19
