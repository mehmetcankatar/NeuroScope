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

## What remains unverified

- Authorized MORGOTH weights were not available during initial development. Real inference, output parity, model class polarity, runtime, GPU memory, and clinical performance are not claimed as tested.
- Chunk context and preprocessing seams require comparison against unchunked inference with real checkpoints.
- The strict loader may expose upstream checkpoint/version incompatibilities. These must be resolved against the pinned author source; do not disable checks to obtain plausible-looking output.
- Event thresholds and sleep aggregation rules are provisional engineering settings.
- No independent clinician usability study or time-saving study has been performed.
- Real patient diagnosis, seizure prediction before onset, emergency alerts, apnea, arousals, and limb-movement detection are outside scope.

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
