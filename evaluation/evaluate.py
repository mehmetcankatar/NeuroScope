"""Evaluate seizure intervals and sleep stages from existing annotations.

CLI: python -m evaluation.evaluate --predictions export.json --truth labels.json
Reference intervals use recording-relative seconds. Model outputs are evaluated,
never reviewer-corrected values. This is a declared overlap matcher, not SzCORE.
"""
import argparse
import json
from pathlib import Path


def _interval(item, duration):
    if not 0 <= item["start_s"] < item["end_s"] <= duration:
        raise ValueError("Invalid reference or predicted interval")


def event_metrics(predicted, truth, duration_s):
    if duration_s <= 0:
        raise ValueError("Recording duration must be positive")
    for item in predicted + truth:
        _interval(item, duration_s)
    graph = [[j for j, target in enumerate(truth)
              if min(pred["end_s"], target["end_s"]) > max(pred["start_s"], target["start_s"])] for pred in predicted]
    matched = {}

    def assign(i, seen):
        for j in graph[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or assign(matched[j], seen):
                matched[j] = i
                return True
        return False

    for i in range(len(predicted)):
        assign(i, set())
    tp = len(matched); fp = len(predicted) - tp
    onset = [predicted[i]["start_s"] - truth[j]["start_s"] for j, i in matched.items()]
    offset = [predicted[i]["end_s"] - truth[j]["end_s"] for j, i in matched.items()]
    return {"matching_rule": "one-to-one maximum matching; strictly positive temporal overlap; no tolerance",
            "true_positives": tp, "false_positives": fp, "false_negatives": len(truth) - tp,
            "event_sensitivity": tp / len(truth) if truth else None,
            "precision": tp / len(predicted) if predicted else None,
            "false_alarms_per_hour": fp / (duration_s / 3600),
            "onset_errors_s": onset, "offset_errors_s": offset,
            "mean_absolute_onset_error_s": sum(map(abs, onset)) / tp if tp else None}


def sleep_metrics(predicted, truth):
    labels = ["Awake", "N1", "N2", "N3", "REM"]
    matrix = [[0] * 5 for _ in labels]
    lookup = {(x["start_s"], x["end_s"]): x["stage"] for x in predicted}
    scored = 0
    for row in truth:
        stage = lookup.get((row["start_s"], row["end_s"]))
        if row["stage"] not in labels:
            continue
        if stage not in labels:
            continue
        matrix[labels.index(row["stage"])][labels.index(stage)] += 1
        scored += 1
    per_stage = {}
    f1s = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]; actual = sum(matrix[i]); found = sum(row[i] for row in matrix)
        f1 = 2 * tp / (actual + found) if actual + found else None
        if f1 is not None:
            f1s.append(f1)
        per_stage[label] = {"precision": tp / found if found else None, "recall": tp / actual if actual else None, "f1": f1}
    valid_reference = sum(x["stage"] in labels for x in truth)
    return {"labels": labels, "confusion_matrix_true_rows": matrix, "per_stage": per_stage,
            "macro_f1_present_classes": sum(f1s) / len(f1s) if f1s else None,
            "scored_reference_fraction": scored / valid_reference if valid_reference else None,
            "epoch_alignment": "Exact interval match; unscored/uncertain predictions excluded and counted in coverage"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prediction = json.loads(args.predictions.read_text()); truth = json.loads(args.truth.read_text())
    if prediction["recording"]["id"] != truth["recording_id"]:
        raise ValueError("Ground truth is for a different recording")
    if prediction["source"] != "morgoth":
        raise ValueError("Synthetic demonstrations are not model performance evaluations")
    result = {"recording_id": truth["recording_id"], "provenance": prediction["provenance"], "coverage": prediction["coverage"]}
    if "seizures" in truth:
        result["seizures"] = event_metrics([e for e in prediction["events"] if e["label"] == "seizure" and e["head"] != "manual"], truth["seizures"], prediction["recording"]["duration_s"])
    if "sleep" in truth:
        result["sleep"] = sleep_metrics(prediction["sleep"], truth["sleep"])
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
