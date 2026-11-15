# Implementation notes

The original MORGOTH-only plan was changed at the user's request to avoid verified-account access. The default configuration now uses publicly downloadable SPaRCNet weights and YASA's bundled classifiers. It provides real seizure/pattern scoring and sleep staging; broad abnormality is explicitly unavailable.

Important engineering decisions:

- SPaRCNet class order and bipolar order follow the pinned author source. Values remain in microvolts, with no additional normalization. The network definitions in `sparcnet_network.py` preserve the author architecture; see third-party notices.
- The only legacy pickle load occurs during setup, after exact SHA-256 verification of the pinned official checkpoint. Runtime loads the converted tensor-only state dictionary with `weights_only=True`.
- Duplicate channel labels are checked for signal equality before accepting a derivation. Missing channels are never fabricated. Reverse bipolar pairs have their sign corrected.
- Native EDF signal counts include annotation channels, whereas MNE's exposed signal list does not. Import explicitly accounts for that difference and serializes native integer metadata.
- YASA runs in a child process with version and classifier checksum verification. On macOS, PyTorch loads its bundled OpenMP runtime before LightGBM/YASA to avoid requiring Homebrew.
- SPaRCNet uses chunked signal processing; YASA uses whole-recording features. Both share the existing worker, cancellation, provenance, review and export interfaces.
- Seizure timing uses a 2-second score stride; sleep timing uses native 30-second epochs. Partial and invalid intervals remain unscored. Synthetic fixtures retain their own separate timing contract.
- Public validation uses independent annotation files only after inference. No model fitting, feature labeling, or threshold optimization is performed.

The Dockerfile builds a demo/core installation by default. Build with `--build-arg INSTALL_OPEN_MODELS=1` to install public inference dependencies. Run model setup inside that environment and preserve the `/app/models` volume; registries contain absolute paths specific to their environment. Docker execution itself has not been validated in this delivery.

Optional MORGOTH remains documented in MORGOTH.md. To use it, select its registry and set `NEUROSCOPE_ENGINE=morgoth`, then reimport EDFs. Its real weights have not been tested here.
