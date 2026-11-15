import numpy as np
import pytest
from neuroscope.pipeline import build_events, build_sleep, quality_intervals, validate_predictions, apply_reviews

CONFIG = {"enter_threshold": .7, "exit_threshold": .4, "persistence_s": 2, "merge_gap_s": 3}


def test_events_use_recording_offsets_and_merge_without_invalid_gaps():
    starts = np.arange(100, 109, dtype=float)
    p = np.array([.1,.8,.9,.3,.8,.9,.1,.1,.1])
    scores = np.column_stack([1-p,p]); valid = np.ones(9,dtype=bool)
    events = build_events("r", "abnormal", starts, scores, valid, CONFIG)
    assert len(events) == 1
    assert events[0]["start_s"] == 105.5
    assert events[0]["end_s"] == 110.5
    assert events[0]["support_start_s"] == 101
    valid[3] = False
    assert len(build_events("r", "abnormal", starts, scores, valid, CONFIG)) == 2


def test_missing_window_prevents_merge():
    starts = np.array([0,1,4,5],float); scores = np.tile([.1,.9], (4,1))
    assert len(build_events('r','abnormal',starts,scores,np.ones(4,bool),CONFIG)) == 2


def test_sleep_missing_coverage_is_not_awake():
    starts = np.arange(0,81,dtype=float)
    scores = np.tile([.9,.025,.025,.025,.025], (81,1)); valid = np.ones(81,bool)
    valid[(starts+5>=30)&(starts+5<60)] = False
    epochs = build_sleep('r',starts,scores,valid,90)
    assert [e['stage'] for e in epochs] == ['Awake','Unscored','Awake']
    assert epochs[1]['coverage'] == 0


def test_quality_includes_boundaries_and_all_invalid():
    gaps, covered = quality_intervals(np.arange(11),np.ones(11,bool),20)
    assert covered == 11
    assert gaps == [dict(start_s=0,end_s=4.5,reason='Unscored signal or insufficient window support'),dict(start_s=15.5,end_s=20,reason='Unscored signal or insufficient window support')]
    assert quality_intervals(np.arange(11),np.zeros(11,bool),20)[1] == 0


def test_invalid_scores_and_timestamp_order_rejected():
    with pytest.raises(ValueError, match='increasing'):
        validate_predictions(np.array([1,1]),np.array([[.2,.8],[.1,.9]]),np.ones(2,bool),2)
    with pytest.raises(ValueError, match='non-finite'):
        validate_predictions(np.array([1]),np.array([[np.nan,np.nan]]),np.ones(1,bool),2)


def test_reviews_do_not_overwrite_model_values():
    events=[dict(id='e',start_s=10,end_s=20,label='seizure',decision='unreviewed')]
    notes=[dict(target_type='event',target_id='e',decision='corrected',start_s=11,end_s=19,label='other')]
    output,_=apply_reviews(events,[],notes)
    assert output[0]['start_s'] == 10 and output[0]['reviewed_start_s'] == 11
    assert events[0]['decision'] == 'unreviewed'
