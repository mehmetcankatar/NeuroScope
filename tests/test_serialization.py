import json
import numpy as np
import pytest
from neuroscope.db import Store, atomic_json
from neuroscope.serialization import json_dumps, to_jsonable


def test_numpy_values_round_trip_without_mutating_arrays():
    samples = np.array([[1, 2], [3, 4]], dtype=np.int64)
    payload = {'count': np.int64(921600), 'unsigned': np.uint64(2**63 + 1),
               'rate': np.float64(256), 'score': np.float32(.875),
               'valid': np.bool_(True), 'samples': samples,
               'nested': (np.array(True), {'start_s': np.int32(2996)})}
    with pytest.raises(TypeError, match='int64'):
        json.dumps(payload)  # The original persistence failure.
    decoded = json.loads(json_dumps(payload))
    assert decoded == {'count': 921600, 'unsigned': 2**63 + 1, 'rate': 256.0,
                       'score': .875, 'valid': True, 'samples': [[1, 2], [3, 4]],
                       'nested': [True, {'start_s': 2996}]}
    assert type(decoded['count']) is int and type(decoded['valid']) is bool
    np.testing.assert_array_equal(samples, [[1, 2], [3, 4]])
    assert payload['samples'] is samples


def test_invalid_numbers_and_unsupported_objects_are_not_hidden():
    for value in (np.float32(np.nan), np.float64(np.inf), -float('inf')):
        with pytest.raises(ValueError):
            json_dumps({'value': value})
    with pytest.raises(TypeError):
        json_dumps({'value': object()})


def test_recording_run_annotation_and_result_persistence(tmp_path):
    store = Store(tmp_path)
    store.add_recording({'id': 'edf', 'n_samples': np.int64(921600),
                         'channel_rates': {'FP1-F7': np.float64(256)}})
    assert store.recording('edf')['n_samples'] == 921600
    store.add_run({'id': 'run', 'recording_id': 'edf', 'config': {'heads': ['seizure']},
                   'tasks': {}, 'source': 'pretrained-models', 'samples': np.int64(921600)})
    assert store.claim()['samples'] == 921600
    store.update_run('run', progress=np.float32(50), valid=np.bool_(True))
    assert store.run('run')['valid'] is True
    store.annotate('run', {'start_s': np.int64(2996), 'end_s': np.float64(3036)})
    assert store.annotations('run')[0]['start_s'] == 2996
    results = {'events': [{'start_s': np.int64(2996), 'end_s': np.int64(3062),
                           'peak_score': np.float32(.875)}], 'valid': np.array([True, False])}
    path = tmp_path / 'runs' / 'run' / 'results.json'
    atomic_json(path, results)
    assert json.loads(path.read_text()) == to_jsonable(results)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        atomic_json(path, {'bad': np.float64(np.nan)})
    assert path.read_bytes() == before
