import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from neuroscope.config import Settings
from neuroscope.api import create_app
from neuroscope.worker import process_run


@pytest.fixture
def env(tmp_path):
    settings=Settings(tmp_path/'data',tmp_path/'models.json',1024*1024,tmp_path/'dist')
    app=create_app(settings)
    return TestClient(app),app.state.store,settings


def analyzed(env):
    client,store,settings=env
    rec=client.post('/api/demo').json()
    run=client.post(f'/api/recordings/{rec["id"]}/analyses',json={}).json()
    process_run(store,settings,store.claim())
    return rec,run


def test_full_demo_review_export(env):
    client,store,_=env
    rec,run=analyzed(env)
    result=client.get(f'/api/analyses/{run["id"]}/results').json()
    assert result['run']['status']=='complete'
    assert result['source']=='synthetic'
    assert len(result['sleep'])==120
    event=next(e for e in result['events'] if e['label']=='seizure')
    assert 727 <= event['start_s'] <= 729
    response=client.post(f'/api/analyses/{run["id"]}/annotations',json={'target_type':'event','target_id':event['id'],'decision':'accepted','note':'Checked original waveform'})
    assert response.status_code==201
    updated=client.get(f'/api/analyses/{run["id"]}/results').json()
    assert next(e for e in updated['events'] if e['id']==event['id'])['decision']=='accepted'
    for kind in ['csv','json','html']:
        exported=client.get(f'/api/analyses/{run["id"]}/export?format={kind}')
        assert exported.status_code==200 and 'attachment' in exported.headers['content-disposition']
        assert 'Checked original waveform' in exported.text
    assert client.get(f'/api/recordings/{rec["id"]}/waveform?start=720&end=740').status_code==200


def test_invalid_uploads_and_cross_origin_writes(env):
    client,_,_=env
    assert client.post('/api/recordings',files={'file':('x.txt',b'bad')}).status_code==415
    assert client.post('/api/recordings',files={'file':('x.edf',b'bad')}).status_code==422
    assert client.post('/api/demo',headers={'Origin':'https://example.com'}).status_code==403
    assert client.post('/api/recordings',files={'file':('big.edf',b'x'*(1024*1024+1))}).status_code==413


def test_annotations_validate_target_and_bounds(env):
    client,_,_=env;rec,run=analyzed(env)
    base={'target_type':'event','target_id':'missing','decision':'accepted'}
    assert client.post(f'/api/analyses/{run["id"]}/annotations',json=base).status_code==404
    base.update(target_type='manual',label='Note',start_s=0,end_s=99999)
    assert client.post(f'/api/analyses/{run["id"]}/annotations',json=base).status_code==422


def test_uploaded_recording_never_uses_synthetic_fallback(env):
    client,store,_=env
    record=client.post('/api/demo').json();record['id']='real-file';record['kind']='edf'
    store.add_recording(record)
    response=client.post('/api/recordings/real-file/analyses',json={'heads':['seizure']})
    assert response.status_code==409
    assert 'checkpoint' in response.text.lower()


def test_cancel_queued_and_recover_after_restart(env):
    client,store,_=env
    rec=client.post('/api/demo').json()
    run=client.post(f'/api/recordings/{rec["id"]}/analyses',json={}).json()
    client.post(f'/api/analyses/{run["id"]}/cancel')
    assert store.run(run['id'])['status']=='cancelled'
    assert store.claim() is None
    client.post(f'/api/analyses/{run["id"]}/retry')
    claimed=store.claim();assert claimed['status']=='running'
    store.recover();assert store.run(run['id'])['status']=='queued'


def test_threshold_validation(env):
    client,_,_=env
    rec=client.post('/api/demo').json()
    assert client.post(f'/api/recordings/{rec["id"]}/analyses',json={'enter_threshold':.4,'exit_threshold':.8}).status_code==422


def test_real_edf_import_serializes_metadata(env, tmp_path, monkeypatch):
    import mne
    import numpy as np
    from neuroscope.config import EEG_CHANNELS
    client, store, _ = env
    raw = mne.io.RawArray(np.tile(np.sin(np.arange(4000) / 10) * 20e-6, (19, 1)),
                         mne.create_info(EEG_CHANNELS, 200, 'eeg'), verbose='ERROR')
    path = tmp_path / 'input.edf'
    raw.export(path, fmt='edf', overwrite=True, verbose='ERROR')
    # Reproduce MNE's n_times type at the reader/persistence boundary, even if
    # the reader already casts that field. Exercise nested scientific types too.
    from neuroscope.signals import inspect_edf
    def numpy_metadata(*args):
        record = inspect_edf(*args)
        record['n_samples'] = np.int64(record['n_samples'])
        record['sample_rate'] = np.float64(record['sample_rate'])
        record['capabilities']['seizure']['supported'] = np.bool_(True)
        record['channels'] = np.array(record['channels'])
        return record
    monkeypatch.setattr('neuroscope.api.inspect_edf', numpy_metadata)
    response = client.post('/api/recordings', files={'file': ('input.edf', path.read_bytes())})
    assert response.status_code == 201, response.text
    record = response.json()
    assert record['n_samples'] == 4000
    assert record['channel_rates']['C3'] == 200
    assert store.recording(record['id'])['sha256'] == record['sha256']
    assert record['capabilities']['seizure']['supported']
    assert not record['capabilities']['abnormal']['supported']
    displayed = client.get(f'/api/recordings/{record["id"]}/waveform?start=0&end=1&points=300')
    assert displayed.status_code == 200, displayed.text
    wave = displayed.json()
    assert wave['sample_rate'] == 200 and wave['times'][0] == 0
    assert wave['channels'][0]['name'] == EEG_CHANNELS[0]
    np.testing.assert_allclose(wave['channels'][0]['max'], raw.get_data()[0, :200] * 1e6, atol=.001)


def test_omitted_heads_select_only_supported_ready_models(env, monkeypatch):
    client, store, _ = env
    record = client.post('/api/demo').json()
    record['kind'] = 'edf'; record['id'] = 'compatible-sleep'
    for h in record['capabilities']:
        record['capabilities'][h]['supported'] = h == 'sleep'
    store.add_recording(record)
    monkeypatch.setattr('neuroscope.api.model_status', lambda _: {h: {'ready': h == 'sleep'} for h in record['capabilities']})
    response = client.post('/api/recordings/compatible-sleep/analyses', json={})
    assert response.status_code == 202
    assert response.json()['config']['heads'] == ['sleep']
