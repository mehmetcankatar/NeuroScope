# NeuroScope

**v0.1.1:** NumPy-safe EDF uploads and persistence. If an old local server still
reports an `int64` JSON error, restart it and confirm v0.1.1 in the sidebar.
See [the fix and real-recording test results](docs/EDF_UPLOAD_FIX.md).

An EEG review workspace that runs pretrained models locally, turns their scores into timestamped candidates, and links every finding to the original waveform.

**Real EDF analysis now uses public SPaRCNet and YASA models. No account, API key, model training, or manual labeling is required.** The synthetic demo remains separately labeled. Uploaded EEG never falls back to scripted predictions.

Research prototype, not a diagnostic device. The app has been exercised on public seizure and sleep recordings; this is functional evidence, not clinical validation. Read the [measured results](docs/BUILD_REPORT.md): the seizure example also produced false alarms.

## Capabilities

| Task | Model | Output |
|---|---|---|
| Possible seizures and specific rhythmic/periodic patterns | SPaRCNet | Seizure, LPD, GPD, LRDA, GRDA candidates with estimated start/end times and scores |
| Sleep staging | YASA 0.6.5 | Awake, N1, N2, N3, REM in native 30-second epochs, with uncertain/unscored intervals |
| Broad normal/abnormal interpretation | Unavailable in the public-model configuration | Explicitly disabled; SPaRCNet “Other” does not mean normal |

A recording may support only one task. The app checks channels, references, duration, and sampling rates before enabling analysis. These models do not provide a complete neurological interpretation of every EDF.

The interface includes multichannel traces, a clickable event queue and timeline, sleep hypnogram, original model scores, review decisions, boundary/stage corrections, notes, and JSON/CSV/printable HTML exports. Background jobs survive page reloads, report progress, support cancellation/retry, and retain per-task errors.

## Install and run

Use **Python 3.12**, **Node.js 22 or 24**, and **pnpm 11**. Commands run from the repository root. On macOS use an ARM64 Python on Apple Silicon; this configuration was tested on the development Mac. The public inference path uses CPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,open-models]'
python scripts/setup_open_models.py
cd frontend
pnpm install --frozen-lockfile
pnpm build
cd ..
python -m neuroscope
```

If pnpm is missing, install with `npm install -g pnpm@11`. On Windows activate with `.venv\Scripts\Activate.ps1`. Linux LightGBM may require the distribution's OpenMP runtime (`libgomp1`). The macOS adapter loads the OpenMP runtime bundled with the pinned PyTorch wheel.

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), choose **Import EDF**, then **Start analysis**. Keep the launcher running. If your local `.env` specifies another port, use that port instead. Model setup shows which models are configured. A fresh clone needs the setup command because weights and machine-specific registries are intentionally excluded from Git.

For UI/demo development only, install `.[dev]` and skip model setup. **Explore the demo** creates synthetic signals and scripted scores.

## Open the two real examples with saved analyses

After installing dependencies and building the interface, run:

```bash
python scripts/load_examples.py
python -m neuroscope
```

Open **Recordings** to select the CHB-MIT seizure example or the Sleep-EDF example. Their saved real-model analyses are included in the repository; the original EDF files download from PhysioNet with verified hashes. Loading saved results does not need model weights. Install the models to analyze new uploads or rerun these examples. For a custom data directory, pass `--data-dir` to the loader and use the same `NEUROSCOPE_DATA_DIR` for the app. See [example details](examples/README.md).

## Try public recordings

```bash
python scripts/download_examples.py
```

This downloads about 91 MB to `data/examples/`, verifying pinned SHA-256 hashes. No verified account is needed.

- **chb01_03.edf**: one-hour CHB-MIT recording with a separately annotated seizure. Import it for seizure/pattern analysis.
- **SC4001E0-PSG.edf**: Sleep-EDF recording with Fpz-Cz EEG and EOG. Import it for sleep staging. Its low-frequency EMG envelope is excluded from the model input.

The annotation files are only used by the evaluator, never as inference inputs. With NeuroScope running:

```bash
python scripts/validate_public_examples.py
# For a different local port:
python scripts/validate_public_examples.py --base-url http://127.0.0.1:8001
```

The script imports both EDFs through the real API/worker, checks exports and waveforms, and writes metrics to `data/examples/validation.json`. It creates new recordings in the workspace. See [validation methods](docs/VALIDATION.md) and [dataset provenance](configs/public-examples.json).

## Input contracts and timing

- Continuous EDF, at least 10 seconds, default upload ceiling 512 MB. Discontinuous EDF+D is rejected.
- SPaRCNet requires all 16 longitudinal bipolar derivations, supplied directly or computed from consistently referenced electrodes. T7/T8/P7/P8 aliases and reversed pairs are handled explicitly. Missing electrodes are never filled with zeros. Duplicate derivations must agree.
- Seizure input must be sampled at least 100 Hz. Processing uses 200 Hz, 0.5–40 Hz filtering, the authors' 60 Hz notch, microvolt scaling and clipping, 10-second windows with a 2-second stride.
- YASA accepts supported central EEG references or the explicit Fpz-Cz test montage, sampled at least 80 Hz, with optional suitable EOG/EMG. At least five minutes is required; full-night sleep recordings are preferred. Unknown reference labels require reviewer confirmation of suitability.
- All timestamps are elapsed recording time. Seizure boundaries are estimates from overlapping model windows, not exact onset labels. The first/last four seconds are unscored on the seizure score timeline. YASA uses native 30-second epochs; partial final epochs stay unscored.
- Flat or marked bad intervals are excluded; this is not a comprehensive artifact detector. A completed run is not proof that a recording is normal.

## Review and export

Click an event or sleep epoch to inspect the original signal and model score. Accept, reject, mark uncertain, edit boundaries/stages, or add a manual finding. Exports preserve original predictions separately from reviewer edits and include model versions, checksums, timing, coverage, and run settings.

Scores are not calibrated clinical probabilities. The default event thresholds (0.70 enter, 0.45 exit, 3-second persistence, 2-second merge gap) are exploratory, not clinically validated operating points.

## Configuration

The launcher reads `.env` without overriding existing environment variables. Default data and models live in the checkout.

| Variable | Default | Purpose |
|---|---|---|
| `NEUROSCOPE_DATA_DIR` | `./data` | Recordings, database, runs, reviews |
| `NEUROSCOPE_HOST` | `127.0.0.1` | Local-only binding |
| `NEUROSCOPE_PORT` | `8000` | API and built UI |
| `NEUROSCOPE_MAX_UPLOAD_MB` | `512` | Upload ceiling |
| `NEUROSCOPE_MODEL_REGISTRY` | `./models/registry.json` | Local model paths/checksums |
| `NEUROSCOPE_ENGINE` | `open` | Input contract; optional `morgoth` requires its own registry |

Data remains on the computer. The app has no authentication for public hosting. See [data handling](docs/DATA_HANDLING.md). Do not expose this local research workspace to the internet with private recordings.

## Development and verification

```bash
python -m pytest
cd frontend
pnpm test
pnpm build
# Optional separate Vite dev server:
pnpm dev
```

The built UI is served by FastAPI. Vite development uses port 5173 and proxies to port 8000; adjust its proxy if you change the backend port. API docs are at `/api/docs`.

Source: `backend/neuroscope/`, `frontend/`, `scripts/`, `evaluation/`, `tests/`, `configs/`, and `docs/`. Read the [architecture](docs/ARCHITECTURE.md) and [implementation notes](docs/IMPLEMENTATION.md).

## Optional MORGOTH adapter

The original adapter is retained for users with authorized checkpoints. It is not needed for the public-model workflow and was not validated with real MORGOTH weights. See [MORGOTH setup](docs/MORGOTH.md). Switching engines requires matching registry and `NEUROSCOPE_ENGINE=morgoth`, restarting, and reimporting recordings so their input contracts are recalculated.

## Attribution and licensing

SPaRCNet: [bdsp-core/IIIC-SPaRCNet](https://github.com/bdsp-core/IIIC-SPaRCNet). YASA: [Vallat and Walker, 2021](https://doi.org/10.7554/eLife.70092). Public recordings: [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) and [Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/).

Original NeuroScope application code uses the [MIT license](LICENSE). This does not relicense third-party architecture, weights, software, or data. See [third-party notices](THIRD_PARTY_NOTICES.md), including the scope of SPaRCNet's upstream license. Code and documentation were developed with OpenAI Codex assistance.
