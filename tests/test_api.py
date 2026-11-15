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
    response=client.post('/api/recordings/real-file/analyses',json={})
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
