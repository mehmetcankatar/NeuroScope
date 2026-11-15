import numpy as np
from .signals import DEMO_EVENTS, DEMO_GAP


def predictions(head, duration_s):
    """Scripted UI fixtures only. Never invoked on uploaded files."""
    starts = np.arange(0, duration_s - 10 + 1, dtype=float)
    centers = starts + 5
    valid = ~((starts < DEMO_GAP[1]) & (starts + 10 > DEMO_GAP[0]))
    if head == "abnormal":
        p = .12 + .025 * np.sin(centers / 37)
        for a, b, _ in DEMO_EVENTS:
            p[(centers >= a) & (centers < b)] = .88
        scores = np.column_stack([1 - p, p])
    elif head == "seizure":
        scores = np.full((len(starts), 6), .012)
        scores[:, 0] = .94
        for a, b, kind in DEMO_EVENTS:
            if kind not in ("seizure", "LPD"):
                continue
            mask = (centers >= a) & (centers < b)
            ci = 1 if kind == "seizure" else 2
            scores[mask] = .014
            scores[mask, ci] = .93
    else:
        scores = np.full((len(starts), 5), .025)
        boundaries = [0, 180, 360, 900, 1380, 1680, 1860, 2160, 2670, 3030, 3360, 3601]
        stages = [0, 1, 2, 3, 2, 0, 1, 2, 3, 4, 0]
        for a, b, stage in zip(boundaries, boundaries[1:], stages):
            scores[(centers >= a) & (centers < b), stage] = .9
    scores = scores / scores.sum(axis=1, keepdims=True)
    scores[~valid] = np.nan
    return starts, scores, valid
