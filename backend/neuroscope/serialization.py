"""Convert scientific values at JSON boundaries without changing signal arrays.

Finite NumPy scalars retain their numeric/Boolean values; arrays become nested
lists. Unsupported objects and non-finite JSON numbers fail explicitly rather
than being silently stringified. Waveform gaps are handled by the waveform
reader's existing explicit null policy. Model arrays remain lossless NPZ files.
"""
import json
import numpy as np


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {to_jsonable(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def json_dumps(value, **kwargs):
    return json.dumps(to_jsonable(value), allow_nan=False, **kwargs)
