"""Opt-in full inference test; uses actual PhysioNet data and installed weights.

NEUROSCOPE_TEST_CHB=/path/chb01_03.edf pytest tests/test_public_recording.py -q
Optional NEUROSCOPE_MODEL_REGISTRY selects a configured public-model registry.
"""
import os
from pathlib import Path
import mne
import numpy as np
import pytest
from fastapi.testclient import TestClient
from neuroscope.api import create_app
from neuroscope.config import Settings
from neuroscope.signals import sha256_file
from neuroscope.worker import process_run


@pytest.mark.skipif(not os.getenv('NEUROSCOPE_TEST_CHB'), reason='Set NEUROSCOPE_TEST_CHB for real-data inference')
def test_chb_upload_waveform_real_analysis_and_exports(tmp_path, monkeypatch):
    monkeypatch.setenv('NEUROSCOPE_ENGINE', 'open')
    path = Path(os.environ['NEUROSCOPE_TEST_CHB']).resolve()
    expected_hash = '4c4a95a9b4331aeaadadd538763eb2e735950d9aa615b85ee6246c784be8ae90'
    assert sha256_file(path) == expected_hash
    registry = Settings.from_env().registry
    settings = Settings(tmp_path / 'data', registry, 100 * 1024 * 1024, tmp_path / 'dist')
    app = create_app(settings)
    store = app.state.store
    with TestClient(app) as client:
        with path.open('rb') as source:
            response = client.post('/api/recordings', files={'file': (path.name, source, 'application/octet-stream')})
        assert response.status_code == 201, response.text
        record = response.json()
        assert record['n_samples'] == 921600
        assert record['sample_rate'] == 256 and record['duration_s'] == 3600
        assert record['sha256'] == expected_hash
        assert sha256_file(store.recording(record['id'])['path']) == expected_hash
        with mne.io.read_raw_edf(path, preload=False, verbose='ERROR') as raw:
            assert record['channels'] == raw.ch_names
            assert all(rate == 256 for rate in record['channel_rates'].values())
            response = client.get(f'/api/recordings/{record["id"]}/waveform',
                                  params={'start': 2996, 'end': 3006, 'points': 3000, 'channels': 'FP1-F7'})
            assert response.status_code == 200, response.text
            wave = response.json()
            np.testing.assert_array_equal(wave['times'], np.arange(2996 * 256, 3006 * 256) / 256)
            expected = raw.get_data(picks=['FP1-F7'], start=2996 * 256, stop=3006 * 256, units='uV')[0]
            np.testing.assert_array_equal(wave['channels'][0]['min'], expected)
            np.testing.assert_array_equal(wave['channels'][0]['max'], expected)
        response = client.post(f'/api/recordings/{record["id"]}/analyses', json={'heads': ['seizure']})
        assert response.status_code == 202, response.text
        run = response.json()
        process_run(store, settings, store.claim())  # Real subprocess/model, no mocks.
        response = client.get(f'/api/analyses/{run["id"]}/results')
        assert response.status_code == 200, response.text
        results = response.json()
        assert results['run']['status'] == 'complete', results['run']
        assert results['source'] == 'pretrained-models'
        assert results['provenance']['seizure']
        assert results['coverage']['seizure']['fraction'] > .99
        response = client.get(f'/api/analyses/{run["id"]}/scores?head=seizure&start=2996&end=3036')
        assert response.status_code == 200, response.text
        displayed = response.json()
        with np.load(settings.data_dir / 'runs' / run['id'] / 'seizure.npz', allow_pickle=False) as scores:
            centers = scores['starts'] + float(scores['window_s']) / 2
            mask = (centers >= 2996) & (centers < 3036)
            np.testing.assert_array_equal(displayed['times'], centers[mask])
            np.testing.assert_array_equal(displayed['scores'], scores['scores'][mask])
            np.testing.assert_array_equal(displayed['valid'], scores['valid'][mask])
        for kind in ('json', 'csv', 'html'):
            exported = client.get(f'/api/analyses/{run["id"]}/export?format={kind}')
            assert exported.status_code == 200, exported.text
            assert 'attachment' in exported.headers['content-disposition']
        assert sha256_file(path) == expected_hash
