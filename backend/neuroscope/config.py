from dataclasses import dataclass
import os
from pathlib import Path

_source_root = Path(__file__).resolve().parents[2]
ROOT = Path(os.getenv("NEUROSCOPE_PROJECT_ROOT", str(
    _source_root if (_source_root / "scripts/morgoth_bridge.py").is_file() else Path.cwd()
))).resolve()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    registry: Path
    max_upload_bytes: int
    frontend: Path

    @classmethod
    def from_env(cls):
        return cls(
            Path(os.getenv("NEUROSCOPE_DATA_DIR", str(ROOT / "data"))).resolve(),
            Path(os.getenv("NEUROSCOPE_MODEL_REGISTRY", str(ROOT / "models/registry.json"))).resolve(),
            int(os.getenv("NEUROSCOPE_MAX_UPLOAD_MB", "512")) * 1024 * 1024,
            ROOT / "frontend/dist",
        )


HEADS = ("abnormal", "seizure", "sleep")
EEG_CHANNELS = ["FP1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "FZ", "CZ", "PZ",
                "FP2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]
SLEEP_CHANNELS = ["F3", "C3", "O1", "F4", "C4", "O2"]
CLASSES = {
    "abnormal": ["normal", "abnormal"],
    "seizure": ["other", "seizure", "LPD", "GPD", "LRDA", "GRDA"],
    "sleep": ["Awake", "N1", "N2", "N3", "REM"],
}
UPSTREAM_COMMIT = "3e449003287465127b441f492263579c805c741d"
