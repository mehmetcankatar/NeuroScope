# MORGOTH integration

## Access and supported tasks

Use [the authors' repository](https://github.com/bdsp-core/morgoth) and [BDSP](https://bdsp.io/content/morgoth1/1.0.0/) to obtain authorized checkpoints. The current README points to credentialed S3 storage; the release page also contains an older Dropbox reference. This project does not bypass access controls or redistribute weights.

Required for all three capabilities:

| Checkpoint | NeuroScope task | Output |
|---|---|---|
| `NORMAL.pth` | abnormal | Binary normal/abnormal, polarity explicitly configured |
| `IIIC.pth` | seizure | Other, Seizure, LPD, GPD, LRDA, GRDA |
| `SLEEPPSG.pth` | sleep | Awake, N1, N2, N3, REM |

`SLEEP.pth` is a different three-stage model and is not substituted for `SLEEPPSG.pth`.

## Installation

Use Python 3.12. The application can use the same environment or a separate model environment.

```bash
python -m pip install -e '.[morgoth]'
```

For a CUDA host, install the appropriate PyTorch 2.4.1 / torchvision 0.19.1 wheels for that machine according to [PyTorch's instructions](https://pytorch.org/get-started/previous-versions/), then install the remaining extras. Set `NEUROSCOPE_DEVICE=cuda`. CPU runs are supported in the adapter but have not been benchmarked with authorized weights. MPS is not currently a supported model device.

Register trusted author-provided checkpoint files using the supplied script. **Replace the binary positive-class value below with the value confirmed from checkpoint metadata.** The file name `NORMAL` is insufficient to establish polarity.

```bash
python scripts/setup_morgoth.py \
  --normal /absolute/path/NORMAL.pth \
  --iiic /absolute/path/IIIC.pth \
  --sleep /absolute/path/SLEEPPSG.pth \
  --normal-positive-class normal \
  --trust-checkpoints
```

The example's `normal` is a configuration illustration, not a verified statement about your checkpoint. If polarity is unknown, omit `--normal` and configure only the other tasks until it is confirmed. Checkpoints use Python serialization; load only trusted files. SHA-256 detects later changes, but registering an untrusted file does not make it safe.

The script downloads public source pinned to **`3e449003287465127b441f492263579c805c741d`**, keeps it in ignored `vendor/morgoth/`, and records source/checkpoint hashes in ignored `models/registry.json`. It does not download weights. Restart the app if environment variables changed, then refresh **Model setup**.

## How execution works

1. The API checks channel/reference compatibility and registry configuration.
2. The worker verifies source and checkpoint checksums.
3. Continuous data is scheduled in 600-second cores with 30-second context on either side. Core ownership is based on each native window's center.
4. An isolated subprocess calls the pinned upstream dataset, montage, clipping, scaling, model architecture, and channel-index logic.
5. Predictions retain the native 10-second support, a one-second step, and an independent validity mask. No CSV row is assumed to be a clinical onset timestamp.
6. Context results are trimmed, and unique global timestamps are validated before event/sleep postprocessing.

No task-specific model is trained or fine-tuned. Each checkpoint runs separately. The bridge uses strict state-dictionary loading, so unexpected checkpoint structure fails visibly instead of leaving uninitialized parameters in a partially loaded model.

The bridge calls the model in float32, using upstream preprocessing and reshape/scaling. It does not use optional test-time augmentation or fit a calibration model. Native probability parity can differ from upstream automatic mixed precision and must be measured.

## Validation before interpreting outputs

Authorized checkpoint execution is an outstanding validation dependency. For each head:

1. Run a known compatible, annotated recording through the official pinned implementation.
2. Capture native window scores **before** upstream CSV resizing, interpolation, or default filling.
3. Use the same class order and NORMAL polarity as NeuroScope. Save `starts`, `scores`, `valid` in an NPZ file.
4. Compare with the corresponding NeuroScope chunk NPZ:

```bash
python scripts/verify_parity.py reference.npz candidate.npz --atol 0.00001
```

5. Compare chunked and unchunked output around seams and true signal gaps. Increase/rework context only with a documented preprocessing version; the current 30 seconds is an engineering default, not a proven parity guarantee.
6. Measure runtime, memory, class performance, false alarms, and timing errors on separate annotated recordings.

If resampling creates a different number of native predictions than valid window positions, the bridge rejects the result instead of interpolating it into a misleading timeline. Missing checkpoints, incompatible inputs, and runtime exceptions appear as per-task errors. Synthetic scores are never substituted.

## License

The upstream code is CC BY-NC 4.0. Keep its license and attribution, and obtain appropriate rights for commercial use. Check dataset and checkpoint redistribution permissions separately. The application MIT license covers only original NeuroScope code.
