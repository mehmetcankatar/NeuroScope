# Build and real-data verification

15 November 2026 · local macOS ARM64 CPU execution · Python 3.12 / PyTorch 2.4.1 / YASA 0.6.5. CPU model and peak memory were not recorded. Runtime is measured on this machine, not a performance guarantee.

## Working end-to-end path

Public checkpoint download/verification → EDF upload → capability check → real background inference → timestamped results → waveform/score access → JSON, CSV and HTML exports. The real-data integration checks also exercised saving an uncertain review in an isolated test workspace. Browser checks confirmed model status and event/sleep navigation in the running app.

No model training or manual labeling was performed. Annotations are separate evaluation inputs. Raw EDFs and model weights remain outside Git.

## Seizure example

CHB-MIT `chb01_03.edf`: 3,600 seconds, 256 Hz, bipolar EEG. Runtime **80.57 seconds**. Score-timeline coverage **99.78%**.

- Reference seizure: **00:49:56–00:50:36** (2996–3036 seconds).
- Matched candidate: **00:49:56–00:51:02** (2996–3062 seconds).
- One reference event detected; **seven additional candidates**, equivalent to **7 false alarms/hour** under the declared overlap matcher. Event precision **12.5%**.
- Matched onset error: 0 seconds. Offset error: +26 seconds.
- Defaults unchanged: enter 0.70, exit 0.45, persistence 3 seconds, merge gap 2 seconds.

This is one seizure, not evidence of population-level sensitivity. The false alarms and late endpoint prevent treating these outputs as reliable automatic diagnoses or alerts.

## Sleep example

Sleep-EDF `SC4001E0-PSG.edf`: 79,500 seconds, EEG Fpz-Cz and EOG horizontal at 100 Hz. Its 1 Hz EMG envelope is omitted. Runtime **9.88 seconds**. **2,650 epochs**; **116** are displayed as uncertain. Raw model scores cover all full epochs.

| Evaluation | Accuracy | Cohen's kappa | Reference coverage |
|---|---:|---:|---:|
| Full 22-hour record, raw argmax | 92.38% | 0.813 | 100.00% |
| Full record, displayed confident stages | 94.28% | 0.852 | 95.62% |
| Sleep period ±30 minutes, raw argmax | 78.36% | 0.714 | 100.00% |
| Sleep period ±30 minutes, displayed confident stages | 81.32% | 0.750 | 90.37% |

The full record contains extensive wake time. Sleep-period raw argmax macro F1 is **0.705**. Raw N1 recall is **34.5%**, and REM recall is **50.4%**. The detailed confusion matrices and per-stage precision/recall/F1 are in [public-validation.json](public-validation.json). Excluding uncertain predictions raises displayed-stage accuracy while reducing coverage; both are reported.

## Numerical and software verification

- Official legacy-checkpoint → tensor-only checkpoint conversion: maximum output difference **0.0** on the deterministic setup probe.
- Whole-record filtering versus chunked filtering: maximum absolute class-score difference **0.00000569** over 19 fixed windows, including chunk boundaries and the seizure. Tolerance: 0.01. See [sparcnet-parity.json](sparcnet-parity.json).
- Core backend suite: **26 passed**. Frontend: **4 tests passed**, TypeScript and production build passed. Dependency consistency check passed. These run separately from the real-model evaluation.
- Real-model runs use no synthetic fallback. Unsupported heads remain disabled; no broad normal/abnormal conclusion is generated.

These checks do not establish clinical safety, regulatory status, or general accuracy on arbitrary EDFs. Docker and optional MORGOTH weights remain unverified.
