from evaluation.evaluate import event_metrics, sleep_metrics
from neuroscope.exporting import safe_cell, html_export


def test_one_to_one_matching_prevents_double_counting():
    predicted=[dict(start_s=5,end_s=25),dict(start_s=8,end_s=15)]
    truth=[dict(start_s=10,end_s=20)]
    metrics=event_metrics(predicted,truth,3600)
    assert metrics['true_positives']==1
    assert metrics['false_alarms_per_hour']==1
    assert metrics['event_sensitivity']==1


def test_sleep_reports_unscored_coverage():
    truth=[dict(start_s=0,end_s=30,stage='N2'),dict(start_s=30,end_s=60,stage='REM')]
    predicted=[dict(start_s=0,end_s=30,stage='N2'),dict(start_s=30,end_s=60,stage='Unscored')]
    assert sleep_metrics(predicted,truth)['scored_reference_fraction']==.5


def test_csv_formula_injection_is_escaped():
    assert safe_cell('=1+2')=="'=1+2"
    assert safe_cell('Plain text')=='Plain text'
