"""Pure temporal postprocessing. Unknown intervals are never negative predictions."""
import hashlib
import numpy as np
from .config import CLASSES


def stable_id(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:20]


def validate_predictions(starts, scores, valid, n_classes):
    if starts.ndim != 1 or scores.shape != (len(starts), n_classes) or valid.shape != starts.shape:
        raise ValueError("Prediction dimensions do not match the head contract")
    if len(starts) and (not np.all(np.isfinite(starts)) or np.any(np.diff(starts) <= 0) or starts[0] < 0):
        raise ValueError("Prediction timestamps must be finite, unique, and increasing")
    if np.any(valid & (~np.isfinite(scores).all(axis=1))):
        raise ValueError("Valid predictions contain non-finite scores")
    if np.any(scores[valid] < 0) or np.any(scores[valid] > 1):
        raise ValueError("Scores must be in [0,1]")
    if np.any(valid) and not np.allclose(scores[valid].sum(axis=1), 1, atol=1e-4):
        raise ValueError("Class scores must sum to one")


def build_events(run_id, head, starts, scores, valid, config, window_s=10, stride_s=1):
    centers = starts + window_s / 2
    labels = CLASSES[head]
    events = []
    for ci in range(1, len(labels)):
        active = None
        segments = []
        for i, t in enumerate(centers):
            contiguous = i == 0 or t - centers[i - 1] <= stride_s * 1.01
            if active is not None and (not valid[i] or not contiguous or scores[i, ci] < config["exit_threshold"]):
                segments.append((active, i - 1))
                active = None
            if valid[i] and active is None and scores[i, ci] >= config["enter_threshold"]:
                active = i
        if active is not None:
            segments.append((active, len(centers) - 1))
        # Never bridge invalid or absent windows, even when the gap is short.
        merged = []
        for a, b in segments:
            if merged:
                pa, pb = merged[-1]
                gap = centers[a] - centers[pb] - stride_s
                contiguous_gap = np.all(np.diff(centers[pb:a + 1]) <= stride_s * 1.01)
                if gap <= config["merge_gap_s"] and np.all(valid[pb:a + 1]) and contiguous_gap:
                    merged[-1] = (pa, b)
                    continue
            merged.append((a, b))
        for a, b in merged:
            if (b - a + 1) * stride_s < config["persistence_s"]:
                continue
            start, end = float(centers[a] - stride_s / 2), float(centers[b] + stride_s / 2)
            events.append({"id": stable_id(run_id, head, ci, start, end), "head": head,
                           "label": labels[ci], "start_s": start, "end_s": end,
                           "support_start_s": float(starts[a]), "support_end_s": float(starts[b] + window_s),
                           "peak_score": float(scores[a:b + 1, ci].max()),
                           "mean_score": float(scores[a:b + 1, ci].mean()),
                           "prediction_start_index": a, "prediction_end_index": b,
                           "timing": "estimated", "decision": "unreviewed"})
    return sorted(events, key=lambda e: (e["start_s"], e["label"]))


def build_sleep(run_id, starts, scores, valid, duration_s, epoch_s=30):
    centers = starts + 5
    result = []
    for start in np.arange(0, duration_s, epoch_s):
        end = min(float(start + epoch_s), duration_s)
        mask = (centers >= start) & (centers < end)
        usable = mask & valid
        # Denominator includes edge windows we could not score.
        coverage = min(1, int(usable.sum()) / (end - start))
        probs = scores[usable].mean(axis=0) if usable.any() else np.zeros(5)
        stage = "Unscored"
        if coverage >= .8:
            ordered = np.sort(probs)
            stage = CLASSES["sleep"][int(probs.argmax())] if ordered[-1] >= .5 and ordered[-1] - ordered[-2] >= .1 else "Uncertain"
        result.append({"id": stable_id(run_id, "sleep", start), "start_s": float(start), "end_s": end,
                       "stage": stage, "scores": dict(zip(CLASSES["sleep"], map(float, probs))),
                       "coverage": coverage, "decision": "unreviewed"})
    return result


def quality_intervals(starts, valid, duration_s, window_s=10, stride_s=1):
    # Coverage cells partition the score timeline; edges remain explicitly unscored.
    covered = []
    for start in starts[valid]:
        left = max(0, float(start + window_s / 2 - stride_s / 2))
        right = min(duration_s, left + stride_s)
        if covered and left <= covered[-1][1] + 1e-6:
            covered[-1][1] = max(right, covered[-1][1])
        else:
            covered.append([left, right])
    gaps, cursor = [], 0
    for a, b in covered:
        if a > cursor + 1e-6:
            gaps.append({"start_s": cursor, "end_s": a, "reason": "Unscored signal or insufficient window support"})
        cursor = b
    if cursor < duration_s:
        gaps.append({"start_s": cursor, "end_s": duration_s, "reason": "Unscored signal or insufficient window support"})
    return gaps, sum(b - a for a, b in covered)


def build_native_sleep(run_id, starts, scores, valid, duration_s):
    """Preserve YASA's native 30-second grid, without artificial 1-Hz expansion."""
    lookup = {float(t): i for i, t in enumerate(starts)}
    result = []
    for start in np.arange(0, duration_s, 30.):
        i = lookup.get(float(start))
        usable = i is not None and valid[i] and start + 30 <= duration_s
        probabilities = scores[i] if usable else np.zeros(5)
        rank = np.sort(probabilities)
        stage = CLASSES['sleep'][int(probabilities.argmax())] if usable else 'Unscored'
        if usable and (rank[-1] < .5 or rank[-1] - rank[-2] < .1):
            stage = 'Uncertain'
        result.append({'id': stable_id(run_id, 'sleep', start), 'start_s': float(start),
                       'end_s': min(float(start + 30), duration_s), 'stage': stage,
                       'scores': dict(zip(CLASSES['sleep'], map(float, probabilities))),
                       'coverage': 1. if usable else 0., 'decision': 'unreviewed'})
    return result


def apply_reviews(events, sleep, annotations):
    events = [dict(e) for e in events]
    sleep = [dict(e) for e in sleep]
    lookup = {("event", e["id"]): e for e in events} | {("sleep", e["id"]): e for e in sleep}
    for review in annotations:
        key = (review["target_type"], review["target_id"])
        if review["target_type"] == "manual" and key not in lookup:
            event = {"id": review["target_id"], "head": "manual", "label": review["label"],
                     "start_s": review["start_s"], "end_s": review["end_s"], "peak_score": None,
                     "mean_score": None, "timing": "reviewer", "decision": review["decision"]}
            events.append(event)
            lookup[key] = event
        target = lookup.get(key)
        if target is not None:
            target.update(decision=review["decision"], review=review)
            # Model timestamps/stages are retained; reviewed values are separate.
            target["reviewed_start_s"] = review["start_s"]
            target["reviewed_end_s"] = review["end_s"]
            target["reviewed_label"] = review["label"]
    return events, sleep
