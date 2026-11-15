# NeuroScope

**An evidence-linked EEG review workspace.** Explore multichannel recordings, jump to timestamped candidates, review sleep stages, and preserve your decisions alongside original predictions.

NeuroScope integrates pretrained **MORGOTH** task checkpoints for abnormality, seizure-related patterns, and five-stage sleep classification. The application includes a fully usable **synthetic demonstration**; it is explicitly labeled and never used as a fallback for real EEG analysis.

> Research prototype. Not a diagnostic device, emergency monitor, or source of medical advice. Model access, input compatibility, and independent evaluation are required before drawing conclusions from predictions.

## What is included

- EDF import with channel/reference validation and bounded waveform reads.
- A responsive EEG workspace: event queue, clickable recording timeline, waveform navigation, score track, and sleep hypnogram.
- Recording-relative start/end timestamps, including explicit unscored gaps.
- Frozen MORGOTH checkpoint adapter, source pinning, checksum verification, and isolated inference.
- Persistent background jobs with progress, cancellation, restart recovery, per-task failures, and chunk caching.
- Accept/reject/uncertain review, boundary corrections, sleep-stage corrections, manual findings, and append-only notes.
- JSON, CSV, and printable HTML export containing model and reviewed values separately.
- Evaluation tools, automated tests, local packaging, and CI configuration.

**What is not bundled:** MORGOTH weights, restricted datasets, clinical validation results, or patient recordings. Real inference is implemented but has not been verified with authorized checkpoints in this delivery. See [validation status](docs/VALIDATION.md).

## Quick start

Requirements: **Python 3.12**, **Node.js 22 or 24**, and **pnpm 11**. Python 3.11 is also supported by the application; optional model dependencies should use Python 3.12. Run commands from the repository root unless shown otherwise.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. If pnpm is missing, install it with `npm install -g pnpm@11`.

Build the interface:

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm build
cd ..
```

Start both API and background worker:

```bash
python -m neuroscope
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** and choose **Explore the demo**. The demo creates a one-hour synthetic recording with scripted findings and sleep scores. No model download, API key, or manual labeling is needed for this demonstration.

Keep the launcher running while using the application. Ctrl+C stops the API and worker. To resume after a restart, rerun the same command; persisted jobs and reviews remain in `data/`.

## Real MORGOTH analysis

1. Obtain authorized checkpoints from the [MORGOTH authors](https://github.com/bdsp-core/morgoth) / [BDSP release](https://bdsp.io/content/morgoth1/1.0.0/).
2. Follow [docs/MORGOTH.md](docs/MORGOTH.md) to install optional dependencies and register the checkpoints.
3. Confirm the `NORMAL` checkpoint's binary positive-class meaning from its metadata or authors.
4. Validate native output parity and timing on authorized examples.
5. Import a compatible EDF, choose supported tasks, and start analysis.

The UI's **Model setup** panel shows configuration status. Without weights, uploaded EEG can still be viewed, but real analysis returns an actionable configuration error. Configured checkpoints are checksum-verified when execution begins; configuration does not establish clinical performance.

### Input requirements

- Continuous EDF recordings, at least 10 seconds; default upload limit 512 MB.
- General EEG tasks require the supported 19-electrode layout with a consistent reference.
- Five-stage sleep uses F3, C3, O1, F4, C4, O2 or supported mastoid-referenced equivalents.
- Known T7/T8/P7/P8 aliases are normalized. Bipolar pairs are not relabeled as single electrodes.
- EDF+D, unsupported reference layouts, absent required electrodes, and sampling rates below 128 Hz are rejected for the affected analysis. Fractional sampling rates are not supported by the current model adapter.
- Many public recordings use different montages. Check compatibility before choosing a dataset; do not assume CHB-MIT can be passed directly into this model.

The strict initial channel policy deliberately avoids silently filling missing electrodes. Waveform viewing can remain available even when model analysis is unsupported.

## Review workflow

1. **Import** an EDF or open the demo.
2. **Analyze** abnormality, possible seizures/related patterns, and/or sleep. Each task reports its own status.
3. **Navigate** using the overview, position slider, event cards, or sleep epochs. Time is elapsed recording time, not a guessed wall clock.
4. **Inspect** the original signal, the estimated model interval, and the score track. Display scale, channel selection, and display re-referencing do not change predictions.
5. **Review** a candidate: accept, reject, or mark uncertain. Edit boundaries, correct sleep stages, or add a missed finding.
6. **Export** JSON for complete provenance/history, CSV for tabular review, or HTML to print/save as PDF in your browser.

Sleep stages are context, not seizure alerts. Non-seizure rhythmic/periodic classes are kept separately. Broad abnormality and seizure candidates may overlap. Scores are not validated clinical probabilities. Event thresholds are exploratory settings stored with each run.

## Development

Terminal 1:

```bash
source .venv/bin/activate
python -m neuroscope
```

Terminal 2:

```bash
cd frontend
pnpm dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Vite proxies `/api` to port 8000. The built frontend is served directly by FastAPI in the one-command launcher.

Interactive API documentation: [http://127.0.0.1:8000/api/docs](http://127.0.0.1:8000/api/docs).

```bash
python -m pytest
cd frontend
pnpm test
pnpm build
```

Application dependencies are pinned in `pyproject.toml`; the frontend includes `pnpm-lock.yaml`. Model dependencies are optional so a GPU stack is not required for UI development. See [architecture](docs/ARCHITECTURE.md) and [implementation notes](docs/IMPLEMENTATION.md).

## Configuration and local storage

Copy `.env.example` to `.env` if needed. The launcher loads it without replacing existing environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `NEUROSCOPE_DATA_DIR` | `./data` in the checkout | SQLite, recordings, results, and reviews |
| `NEUROSCOPE_HOST` | `127.0.0.1` | Local API binding |
| `NEUROSCOPE_PORT` | `8000` | API/UI port |
| `NEUROSCOPE_MAX_UPLOAD_MB` | `512` | Per-file upload ceiling |
| `NEUROSCOPE_MODEL_REGISTRY` | `./models/registry.json` | Trusted checkpoint registry |
| `NEUROSCOPE_MORGOTH_PYTHON` | Current Python | Optional isolated model interpreter |
| `NEUROSCOPE_DEVICE` | `cpu` | `cpu` or `cuda` |

```text
data/
  neuroscope.sqlite3    # recordings, runs, and append-only reviews
  recordings/          # original uploaded EDF files
  runs/<run-id>/       # scores, validity masks, results, provenance, logs
models/registry.json  # local checkpoint paths and checksums
vendor/morgoth/        # separately downloaded, ignored upstream source
```

This is a local, single-user application. It does not include production authentication, multi-tenant authorization, or managed encrypted storage. Do not expose it publicly with private recordings. See [data handling](docs/DATA_HANDLING.md).

## Evaluation

Existing annotations are used as references; no manual labeling is required. Import an exported JSON result and a reference file using recording-relative seconds:

```bash
python -m evaluation.evaluate --predictions results.json --truth labels.json --output metrics.json
```

Reference format and matching rules are documented in [docs/VALIDATION.md](docs/VALIDATION.md). Synthetic exports are intentionally rejected by the clinical metric CLI. Do not use demo scores as model performance or transplant accuracy numbers from the MORGOTH paper into NeuroScope results.

## Repository layout

```text
backend/neuroscope/   # API, persistent worker, EDF handling, model adapter, postprocessing
frontend/            # React/TypeScript interface and canvas/SVG signal views
scripts/             # model registration, inference bridge, parity verification, packaging
evaluation/          # seizure-event and sleep-stage metrics
tests/               # backend, timing, EDF, queue, review, and export tests
configs/             # registry example
docs/                # architecture, model setup, validation, and handoff
.github/workflows/   # checks that run only after you upload the repository
```

## Uploading to your own GitHub repository

Everything here is local. No GitHub repository was created, modified, or pushed by the project authoring process. See [docs/GITHUB_HANDOFF.md](docs/GITHUB_HANDOFF.md) for your upload steps. `.gitignore` excludes recordings, credentials, model weights, generated data, installed dependencies, and upstream source.

## Credits and license

- EEG models: [MORGOTH, bdsp-core](https://github.com/bdsp-core/morgoth), Sun and colleagues. Code license: **CC BY-NC 4.0**, which prohibits commercial use under that license. Obtain appropriate rights before commercial deployment. Check checkpoint and dataset terms separately.
- Signal IO: [MNE-Python](https://mne.tools/).
- Application: FastAPI, NumPy, SciPy, React, TypeScript, Vite, and Lucide icons.
- Synthetic demonstration signals and scores are generated by NeuroScope and contain no patient data.
- Code and documentation were developed with OpenAI Codex assistance. Disclose this and other tools as required by your hackathon rules.

Original NeuroScope application code is licensed under [MIT](LICENSE). That does **not** relicense MORGOTH or grant commercial rights to its code, weights, or datasets. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
