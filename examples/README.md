# Public EEG examples with saved model analyses

`public-analyses.zip` contains real SPaRCNet/YASA outputs, score arrays and model provenance for the two public recordings listed in `configs/public-examples.json`. It contains no raw EDF, private recordings, credentials or machine-specific paths. These are previously computed outputs, not new inference and not synthetic fixtures.

Run `python scripts/load_examples.py` to download the matching EDFs from PhysioNet, verify their hashes, and load the saved results. Then start `python -m neuroscope`. To use a custom workspace, pass `--data-dir /path/to/data` and use the same `NEUROSCOPE_DATA_DIR` for the app. Repeating the loader does not duplicate records or overwrite reviews. To generate new predictions, install the public models and click Analyze, or run `scripts/validate_public_examples.py`.

The seizure sample includes seven false alarms in an hour. See `docs/BUILD_REPORT.md` for full metrics and limits. Original data attribution and terms are linked in `THIRD_PARTY_NOTICES.md`; the archive does not grant additional rights to the source recordings or models.
