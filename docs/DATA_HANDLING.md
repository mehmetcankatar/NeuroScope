# Local data handling

NeuroScope is a local, single-user research application. The default server listens only on `127.0.0.1`. It has no telemetry and sends no EEG data to an external AI API.

Uploaded recordings are saved under generated IDs. The original upload filename is used as a display label. EDF patient headers are not returned by the metadata API, but the original EDF on disk may contain identifying information. Do not upload private clinical recordings to a public demo or commit them to Git.

The API validates extension, size, EDF parsing, channels, time ranges, and annotation targets. Cross-origin browser writes from non-local websites are rejected. This is not a substitute for authentication or a hardened public file-upload service.

Local storage is not application-encrypted. Access to the machine and filesystem grants access to recordings and reviews. Use appropriate operating-system account and disk protections for any sensitive research data. The application is not advertised as compliant with any healthcare regulation.

`data/`, `models/`, `vendor/`, `.env`, model weights, and EDF files are ignored by Git. Before sharing, review both tracked and staged files. Do not include your local source/weight registry because it contains machine-specific paths.

To remove all local recordings and reviews, stop NeuroScope and remove the configured `data/` directory after making any desired exports. This action cannot be undone. Back up the entire stopped data directory, including its database, if retaining a research workspace.

For a future hosted deployment, add real authentication, per-recording authorization, upload rate/resource limits, encrypted storage, retention/deletion controls, secure backup, and deployment-specific review. This repository does not enable public hosting by default.
