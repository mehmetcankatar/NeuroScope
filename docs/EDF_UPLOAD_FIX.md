# EDF upload serialization fix · v0.1.1

## Root cause

MNE 1.8 returns `raw.n_times` as a NumPy `int64`. For the actual PhysioNet
`chb01_03.edf`, its value is 921600. The original reader assigned it directly to
`record['n_samples']`. The upload handler called `Store.add_recording`, whose
`json.dumps(body)` failed before inserting the recording into SQLite. The upload
handler translated that failure into HTTP 422, `Cannot read recording: Object
of type int64 is not JSON serializable`, and removed the incomplete upload.

The previous real-model update already cast this particular field to `int`.
An older server left running can still execute the original reader even after
the repository has been updated. The observed server at port 8000 still reported
the old MORGOTH-only model configuration. A freshly started current server
successfully imported the same file. Updating GitHub does not restart a local
Python process.

## Boundary protection

`serialization.py` recursively converts NumPy integers, floating-point scalars,
Booleans, arrays, and nested containers into JSON-compatible Python values.
It does not stringify numbers, modify the input arrays, replace predictions,
or silently suppress unsupported values. Strict JSON serialization continues
to reject non-finite numbers; the waveform reader retains its explicit null
representation for missing signal values.

The converter covers EDF metadata, waveform payloads, public recording/results
responses, all SQLite JSON writes, atomic result/provenance files, and JSON/HTML
exports. Score responses already convert arrays to lists and non-finite scores
to null; model arrays still use NPZ storage without passing through JSON.

The UI now displays the backend's actual version from `/api/health`, rather than
a hard-coded version. An unsupported sleep montage also shows its explanation
while a seizure-only run is active.

## Verification

- The strengthened EDF upload regression fails against the previous persistence
  code with the exact reported HTTP 422 error and passes after this fix.
- Core backend: 29 passed; the separate real-data test is opt-in.
- Frontend: 4 passed; TypeScript and production build pass.
- Full actual CHB-MIT test: passed in 91.96 seconds, including real SPaRCNet
  subprocess inference, result persistence, score retrieval, and all exports.
- Browser verification also passed: importing the original EDF, selecting the supported
  seizure task, completing background inference, and viewing timestamped candidates
  with the original EEG waveform in v0.1.1.
- Source and stored EDF SHA-256 agree with the pinned PhysioNet checksum.
  Duration remains 3600 seconds, rate 256 Hz, sample count 921600, and channel
  names/order are unchanged. The 2996–3006 second waveform and timestamps match
  MNE samples exactly. Returned class scores match the saved NPZ arrays exactly.

Reproduce from the repository root after installing and configuring the public
models and downloading the example files:

```bash
python -m pytest -q
NEUROSCOPE_TEST_CHB="$PWD/data/examples/chb01_03.edf" \
  python -m pytest tests/test_public_recording.py -q
```

Stop the previous local NeuroScope process with Ctrl+C in its Terminal, then
restart `python -m neuroscope` from the updated environment. Open the URL printed
by that process and confirm **v0.1.1** in the sidebar. Refresh an already-open
browser tab after restarting.

CHB-MIT's bipolar montage supports the SPaRCNet seizure/related-pattern task.
It does not satisfy this YASA adapter's sleep-channel contract. Use the included
Sleep-EDF example for sleep staging. No broad normal/abnormal model has been
added, and these software tests do not establish clinical accuracy.
