# Repository and local files

Repository: [mehmetcankatar/NeuroScope](https://github.com/mehmetcankatar/NeuroScope).

The source repository contains application code, scripts, tests, model/dataset provenance and measured validation reports. Model weights, EDFs, local reviews, dependencies and machine-specific registries remain on the computer and are excluded by `.gitignore` and `scripts/package_source.py`.

A fresh clone must follow the README to install dependencies, fetch public models and build the interface. `scripts/download_examples.py` retrieves the two open recordings and annotations with pinned hashes. GitHub Actions runs core backend tests and frontend tests/build; it does not establish model accuracy or execute the public model evaluation.

To make a source-only archive:

```bash
python scripts/package_source.py /path/to/NeuroScope-source.zip
```

Review the archive before sharing. Do not upload the ZIP as the repository's code; upload its extracted contents or commit with Git while preserving directories and hidden configuration files.
