# Implementation notes

The architecture document is the original design baseline. This file records concrete implementation choices and current boundaries.

- **Frontend:** React/TypeScript/Vite. Canvas draws waveform min/max envelopes, and SVG draws score/sleep tracks. This replaces the proposed Plotly dependency to keep startup and interaction lightweight; model data never uses the display reduction.
- **Predictions:** compressed NPZ blocks with `allow_pickle=False` instead of Parquet. JSON contains events, sleep epochs, coverage, and provenance. Native arrays remain available for evaluation.
- **Jobs:** one worker process with a cross-platform exclusive file lock, a persistent SQLite queue, heartbeat, cancellation, and restart recovery. The launcher starts API and worker together.
- **Model isolation:** a subprocess per task/chunk; this favors dependency isolation and bounded memory over lowest latency. Checkpoint reload overhead is expected. It can be optimized after real profiling.
- **EDF input:** continuous EDF only. EDF+D and boundary annotations are explicitly rejected. This release does not claim support for discontinuous records. Epochs at unsupported edges and quality gaps remain unscored.
- **Time:** floating-point seconds on the recording's sample grid, with exact native window start positions and per-chunk offsets. Source sample indices can be reconstructed from the stored sampling rate and offsets. No wall-clock timezone is fabricated.
- **Channels:** strict required sets; no silent missing-electrode imputation. Known aliases are handled. Mixed references and bipolar-only inputs are not guessed into the expected montage.
- **Detection:** hysteresis thresholds, persistence, and gap merging are run configuration. Invalid/missing windows always interrupt merging. Default settings are exploratory, not validated clinical operating points.
- **Sleep:** 30-second display epochs, score averaging by window center, 80% minimum coverage, score ≥0.5 and top-two margin ≥0.1; otherwise uncertain/unscored. These are transparent engineering defaults requiring evaluation. Native scores are preserved.
- **Review:** append-only annotations; model intervals and classifications are retained. The latest review is projected into separate `reviewed_*` fields. Notes and corrections survive reloads.
- **Identity:** a local reviewer label, not authenticated clinician identity. Multi-user authorization and clinical audit certification are outside this local prototype.
- **Exports:** JSON is the full fidelity record. CSV includes model and reviewed fields. HTML is printable from a browser and includes coverage, provenance, and review history. No generated LLM clinical interpretation.
- **Scope:** first release implements abnormality, seizure/IIIC, and five-stage sleep. Dedicated spike/slowing/burst-suppression tracks remain extensions, not silently emulated outputs.

The synthetic demonstration is a functional integration fixture. Its signals and scores are scripted independently of MORGOTH. It must not be used for clinical model performance claims.
