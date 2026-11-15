"""Register authorized, trusted checkpoints and fetch a pinned public source revision."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

COMMIT = "3e449003287465127b441f492263579c805c741d"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, help="Authorized NORMAL.pth path")
    parser.add_argument("--iiic", type=Path, help="Authorized IIIC.pth path")
    parser.add_argument("--sleep", type=Path, help="Authorized SLEEPPSG.pth path")
    parser.add_argument("--normal-positive-class", choices=["normal", "abnormal"],
                        help="Confirm from checkpoint label metadata; do not guess")
    parser.add_argument("--trust-checkpoints", action="store_true", help="Confirm these are trusted author-provided pickle checkpoints")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "vendor/morgoth")
    parser.add_argument("--registry", type=Path, default=ROOT / "models/registry.json")
    args = parser.parse_args()
    paths = {"abnormal": args.normal, "seizure": args.iiic, "sleep": args.sleep}
    if not any(paths.values()):
        parser.error("Provide at least one checkpoint")
    if not args.trust_checkpoints:
        parser.error("PyTorch checkpoints may execute Python when loaded. Register only trusted author-provided files with --trust-checkpoints.")
    if args.normal and not args.normal_positive_class:
        parser.error("--normal-positive-class is required with --normal")
    for path in paths.values():
        if path and not path.is_file():
            parser.error(f"Checkpoint not found: {path}")
    source = args.source_dir.resolve()
    source.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/bdsp-core/morgoth/zip/{COMMIT}"
    print(f"Fetching public MORGOTH source at {COMMIT}")
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for member in archive.infolist():
        relative = Path(*Path(member.filename).parts[1:])
        if len(relative.parts) != 1:
            continue
        if relative.suffix == ".py" or relative.name in ("LICENSE", "README.md", "requirements.txt"):
            (source / relative.name).write_bytes(archive.read(member))
    hashes = {p.name: digest(p) for p in sorted(source.glob("*.py"))}
    if not {"utils.py", "backbone.py", "task_model.py"}.issubset(hashes):
        raise RuntimeError("Downloaded source is incomplete")
    heads = {}
    for head, path in paths.items():
        if path:
            heads[head] = {"checkpoint": str(path.resolve()), "sha256": digest(path),
                           "classes": {"abnormal": ["normal", "abnormal"], "seizure": ["other", "seizure", "LPD", "GPD", "LRDA", "GRDA"], "sleep": ["Awake", "N1", "N2", "N3", "REM"]}[head]}
    if "abnormal" in heads:
        heads["abnormal"]["positive_class"] = args.normal_positive_class
    registry = {"upstream_commit": COMMIT, "source_dir": str(source), "source_hashes": hashes, "heads": heads}
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(json.dumps(registry, indent=2) + "\n")
    print(f"Registered {', '.join(heads)}. Registry: {args.registry.resolve()}")
    print("No weights were downloaded. Run the parity validation in docs/MORGOTH.md before relying on predictions.")


if __name__ == "__main__":
    main()
