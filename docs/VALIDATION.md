# Validation and limitations

## What the tests establish

The automated suite covers timestamp offsets, event merging, invalid-window boundaries, missing-window gaps, sleep coverage, score validation, preservation of original predictions, channel aliases, rejection of bipolar reinterpretation, a generated EDF round trip, API upload validation, background job recovery, persistent reviews, exports, and synthetic-fallback prevention.

Frontend tests cover timestamp formatting and parsing. The production frontend build performs TypeScript checking. Browser review exercises the actual app separately from unit tests.

Run:

```bash
python -m pytest
cd frontend
pnpm test
pnpm build
```

See `BUILD_REPORT.md` for checks actually completed during delivery. This file defines validation scope rather than asserting a test run occurred.

## Public-model validation

Real inference was exercised through the running API and worker on one CHB-MIT recording and one Sleep-EDF recording. See [BUILD_REPORT.md](BUILD_REPORT.md), [public-validation.json](public-validation.json), and [sparcnet-parity.json](sparcnet-parity.json) for actual results. The public recordings include substantial non-event/wake time. No thresholds were tuned on these examples.

Reproduce after installing the public models and starting the app:

```bash
python scripts/download_examples.py
python scripts/validate_public_examples.py
```

The script verifies dataset hashes before import. Seizure annotations are read from the separate CHB summary after model inference. Sleep annotations are read from the separate hypnogram; old stages 3 and 4 are combined as N3, while movement/unknown reference epochs are excluded. Intervals use recording-relative seconds.

Sleep results report both raw argmax stages and displayed stages after uncertainty filtering, with coverage. Full-record accuracy is dominated by wake time, so the report separately evaluates the reference sleep period plus 30 minutes before/after. The latter crop is used only for evaluation; inference processes the whole recording. Poor N1/REM results must not be hidden by the full-record aggregate.

## Remaining limitations

- Two records establish functional operation, not clinical performance or robust generalization. Training-cohort overlap has not been audited; this is not a guaranteed independent test set.
- The seizure example produces seven false alarms per hour at default thresholds. The matched event ends 26 seconds after the reference. Review is required, and these settings are not validated for automated alerts.
- N1 and REM classification is substantially weaker than overall sleep agreement. Montage, age, pathology, artifacts and domain differences can affect both models.
- Broad abnormality is unavailable with public models. SPaRCNet Other is not normal. LPD/GPD/LRDA/GRDA outputs are supported technically but not assessed against labels in these two examples.
- The architecture conversion probe and selected full-record/chunk seam comparisons pass; this is not exhaustive equivalence testing across all EDF rates and montages. NeuroScope includes the final complete seizure window that the author's loop omits.
- Authorized MORGOTH weights, GPU inference, Docker deployment, clinician usability, clinical outcomes and time savings were not validated here.
- Diagnosis, seizure prediction before onset, emergency alerts, apnea, arousals, and limb-movement detection are outside scope.

## Reference annotations

Use existing dataset annotations, with patients separated between tuning and final evaluation. Check overlap with the pretrained model's development cohorts; report any uncertainty. A dataset with recording-level labels cannot validate precise event boundaries. Separate sleep and seizure evaluation may require different compatible datasets.

Example reference format (illustrative values only):

```json
{
  "recording_id": "same-id-as-the-export",
  "seizures": [{"start_s": 120.0, "end_s": 145.0}],
  "sleep": [
    {"start_s": 0.0, "end_s": 30.0, "stage": "Awake"},
    {"start_s": 30.0, "end_s": 60.0, "stage": "N1"}
  ]
}
```

Omit unsupported task labels rather than treating absent annotation fields as negative evidence. The evaluator rejects synthetic predictions.

```bash
python -m evaluation.evaluate --predictions results.json --truth labels.json --output metrics.json
```

Seizure event matching is one-to-one maximum matching with strictly positive overlap and no onset tolerance. One prediction can match at most one annotated event. False alarms/hour uses total supplied recording duration. Report unscored coverage alongside metrics; missing signal can cause missed events. Onset/offset errors are signed seconds for matched pairs. This evaluator is not an implementation of the complete SzCORE benchmark.

Sleep evaluation matches exact epoch intervals, reports per-stage scores and a confusion matrix, and excludes uncertain/unscored predictions while reporting the scored reference fraction. Use this coverage value when interpreting scores. It does not silently interpolate mismatched ground-truth epochs.

For a submission, report sensitivity, false alarms/hour, precision, timing error, sleep per-stage metrics, coverage, runtime/hardware, and sample count. Do not advertise one overall accuracy number as proof of clinical reliability.
