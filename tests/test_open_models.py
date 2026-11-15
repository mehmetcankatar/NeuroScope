import numpy as np
import pytest
from neuroscope.open_channels import PAIRS, seizure_derivations, sleep_channels, open_capabilities
from neuroscope.pipeline import build_events, build_native_sleep, quality_intervals


def test_bipolar_order_aliases_and_reversed_polarity():
    aliases = {'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}
    names = [f'{aliases.get(a,a)}-{aliases.get(b,b)}' for a,b in PAIRS]
    names[2] = 'P7-T7'
    derivations = seizure_derivations(names[::-1])
    assert derivations[0]['positive']=='FP1-F7'
    assert derivations[2]['positive']=='P7-T7' and derivations[2]['sign']==-1
    assert derivations[2]['negative'] is None
    with pytest.raises(ValueError,match='FP1-F7'):
        seizure_derivations(names[1:])


def test_sleep_ignores_low_rate_emg_and_does_not_use_seizure_montage():
    names=['EEG Fpz-Cz','EOG horizontal','EMG submental']
    selected=sleep_channels(names,dict(zip(names,[100,100,1])))
    assert selected=={'eeg':names[0],'eog':names[1],'emg':None}
    caps=open_capabilities(names,dict(zip(names,[100,100,1])),3600)
    assert caps['sleep']['supported'] and not caps['seizure']['supported']
    assert not caps['abnormal']['supported']
    assert not open_capabilities(names,dict(zip(names,[100,100,1])),60)['sleep']['supported']
    with pytest.raises(ValueError):
        sleep_channels(['FP1-F7'])


def test_native_sleep_timing_and_partial_epoch():
    starts=np.array([0.,30.,60.]);scores=np.eye(5)[[0,2,4]];valid=np.array([True,False,True])
    epochs=build_native_sleep('run',starts,scores,valid,100)
    assert [(e['start_s'],e['end_s'],e['stage']) for e in epochs]==[
        (0,30,'Awake'),(30,60,'Unscored'),(60,90,'REM'),(90,100,'Unscored')]
    gaps,covered=quality_intervals(starts,valid,100,window_s=30,stride_s=30)
    assert covered==60 and [(g['start_s'],g['end_s']) for g in gaps]==[(30,60),(90,100)]


def test_two_second_stride_events_do_not_use_one_second_timing():
    starts=np.array([0.,2.,4.,6.]);scores=np.array([[.1,.9],[.1,.9],[.9,.1],[.1,.9]])
    events=build_events('run','abnormal',starts,scores,np.array([True,True,False,True]),
        {'enter_threshold':.7,'exit_threshold':.45,'persistence_s':3,'merge_gap_s':10},10,2)
    assert len(events)==1
    assert (events[0]['start_s'],events[0]['end_s'])==(4,8)


def test_duplicate_bipolar_channels_must_agree():
    import mne
    from neuroscope.open_inference import prepare_bipolar
    raw=mne.io.RawArray(np.array([[1,2,3],[1,2,4]])*1e-6,mne.create_info(['T8-P8-0','T8-P8-1'],200,'eeg'),verbose='ERROR')
    with pytest.raises(ValueError,match='Duplicate channels disagree'):
        prepare_bipolar(raw,[{'positive':'T8-P8-0','negative':None,'sign':1,'duplicates':['T8-P8-1']}],0,3)
