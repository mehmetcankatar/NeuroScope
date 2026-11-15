"""Install public SPaRCNet weights and register bundled YASA classifiers. No account required."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

COMMIT = "d374d5137fe30e595d686ecc1b584a2e551f1aa0"
SHA256 = "e2ec2c02fe5c9d7b4189c8c79826db1e61710fc1e40f4e4468b41f9f4fd0f419"
URL = f"https://raw.githubusercontent.com/bdsp-core/IIIC-SPaRCNet/{COMMIT}/SPaRCNet/model_1130.pt"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    args = parser.parse_args()
    target = args.directory.resolve()
    target.mkdir(parents=True, exist_ok=True)
    original = target / "sparcnet.pt"
    if not original.exists():
        import requests
        print("Downloading public SPaRCNet checkpoint…", flush=True)
        with requests.get(URL, stream=True, timeout=180) as response:
            response.raise_for_status()
            partial = original.with_suffix(".part")
            with partial.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
            partial.replace(original)
    if digest(original) != SHA256:
        raise ValueError("Official checkpoint checksum mismatch; refusing to deserialize")
    import torch
    # Legacy checkpoint refers to classes in __main__. Only the pinned, hashed author file is loaded.
    from neuroscope import sparcnet_network as network
    import __main__
    for name in ("DenseNetClassifier", "DenseNetEnconder", "_DenseLayer", "_DenseBlock", "_Transition"):
        setattr(__main__, name, getattr(network, name))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.serialization.SourceChangeWarning)
        old = torch.load(original, map_location="cpu", weights_only=False).eval()
    model = network.DenseNetClassifier().eval()
    model.load_state_dict(old.state_dict(), strict=True)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(123)
    probe = torch.randn((2, 16, 2000), generator=generator) * 20
    with torch.inference_mode():
        difference = (old(probe)[0] - model(probe)[0]).abs().max().item()
    if difference > 1e-6:
        raise ValueError(f"Checkpoint conversion changed output: {difference}")
    converted = target / "sparcnet-state.pt"
    torch.save(model.state_dict(), converted)
    import yasa
    import importlib.metadata
    classifiers = sorted((Path(yasa.__file__).parent / "classifiers").glob("*.joblib"))
    if not classifiers:
        raise ValueError("YASA package contains no pretrained classifiers")
    registry = {"engine": "open", "sparcnet": {"checkpoint": str(converted), "sha256": digest(converted),
        "original_sha256": SHA256, "source_url": URL, "source_commit": COMMIT, "conversion_max_error": difference},
        "yasa": {"version": yasa.__version__, "classifier_hashes": {p.name: digest(p) for p in classifiers},
                 "dependencies": {n: importlib.metadata.version(n) for n in ["lightgbm", "scikit-learn", "antropy", "numpy", "scipy"]}}}
    (target / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    print("SPaRCNet and YASA configured. Restart NeuroScope and refresh Model setup.")


if __name__ == "__main__":
    main()
