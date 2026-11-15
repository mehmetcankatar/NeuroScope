"""MORGOTH registry, checksum checks, and isolated inference bridge."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.io import savemat
from .config import EEG_CHANNELS, SLEEP_CHANNELS, CLASSES, ROOT, UPSTREAM_COMMIT
from .signals import read_signal, sha256_file
from .db import atomic_json
from .pipeline import validate_predictions


class Cancelled(Exception):
    pass


def morgoth_status(settings):
    status = {h: {"ready": False, "reason": "Configure authorized MORGOTH checkpoints; see docs/MORGOTH.md"} for h in CLASSES}
    if not settings.registry.is_file():
        return status
    try:
        registry = json.loads(settings.registry.read_text())
        source = Path(registry["source_dir"])
        if registry.get("upstream_commit") != UPSTREAM_COMMIT or not (source / "utils.py").is_file():
            raise ValueError("Pinned MORGOTH source is missing or mismatched")
        if not registry.get("source_hashes"):
            raise ValueError("Source hashes are missing; rerun setup_morgoth.py")
        for name, digest in registry["source_hashes"].items():
            if sha256_file(source / name) != digest:
                raise ValueError("MORGOTH source changed; restore the pinned source before analysis")
        for head in status:
            cfg = registry.get("heads", {}).get(head)
            if not cfg:
                continue
            if not Path(cfg["checkpoint"]).is_file():
                status[head]["reason"] = "Checkpoint file is missing"
                continue
            if not cfg.get("sha256") or len(cfg["sha256"]) != 64:
                status[head]["reason"] = "Checkpoint checksum has not been registered"
                continue
            if head == "abnormal" and cfg.get("positive_class") not in ("normal", "abnormal"):
                status[head]["reason"] = "Confirm NORMAL checkpoint positive-class meaning first"
                continue
            status[head] = {"ready": True, "reason": "Configured; checkpoint verified when analysis starts",
                            "checkpoint_sha256": cfg["sha256"], "upstream_commit": UPSTREAM_COMMIT}
    except (ValueError, KeyError, OSError) as exc:
        status = {h: {"ready": False, "reason": str(exc)} for h in CLASSES}
    return status


def run_morgoth(settings, recording, head, run_dir, cancelled, progress):
    registry = json.loads(settings.registry.read_text())
    cfg = registry["heads"][head]
    digest = sha256_file(cfg["checkpoint"])
    if digest != cfg["sha256"]:
        raise ValueError("Checkpoint checksum mismatch; analysis stopped")
    required = SLEEP_CHANNELS if head == "sleep" else EEG_CHANNELS
    mapped = [recording["mapping"][c] for c in required]
    duration = recording["duration_s"]
    all_starts, all_scores, all_valid = [], [], []
    chunk_count = int(np.ceil(duration / 600))
    for chunk_index in range(chunk_count):
        if cancelled():
            raise Cancelled()
        core_start, core_end = chunk_index * 600, min(duration, (chunk_index + 1) * 600)
        # Integer-second starts maintain a single global 1-second grid.
        left, right = max(0, core_start - 30), min(duration, core_end + 30)
        cache = run_dir / f"{head}-{chunk_index:05d}.npz"
        manifest = run_dir / f"{head}-{chunk_index:05d}.manifest.json"
        key = {"input": recording["sha256"], "model": digest, "source": registry["upstream_commit"],
               "preprocessing": "upstream-continuous-v1", "left": left, "right": right, "head": head,
               "positive_class": cfg.get("positive_class")}
        cached = cache.exists() and manifest.exists() and json.loads(manifest.read_text()) == key
        if not cached:
            mat_path = run_dir / f"input-{head}-{chunk_index}.mat"
            times, values, _ = read_signal(recording, left, right, mapped)
            if len(times) < int(10 * recording["sample_rate"]):
                continue
            # A canonical internal MAT avoids upstream EDF alias heuristics.
            savemat(mat_path, {"data": values, "fs": recording["sample_rate"]})
            output = cache.with_suffix(".pending.npz")
            request = run_dir / f"request-{head}-{chunk_index}.json"
            atomic_json(request, {"source_dir": registry["source_dir"], "checkpoint": cfg["checkpoint"],
                                 "head": head, "input": str(mat_path), "output": str(output),
                                 "sample_rate": recording["sample_rate"], "positive_class": cfg.get("positive_class"),
                                 "device": os.getenv("NEUROSCOPE_DEVICE", "cpu")})
            command = [os.getenv("NEUROSCOPE_MORGOTH_PYTHON", sys.executable),
                       str(ROOT / "scripts/morgoth_bridge.py"), str(request)]
            log_path = run_dir / f"{head}-{chunk_index}.log"
            try:
                with log_path.open("w") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=log)
                    started = time.monotonic()
                    try:
                        while process.poll() is None:
                            if cancelled():
                                raise Cancelled()
                            if time.monotonic() - started > 3600:
                                raise RuntimeError("MORGOTH chunk exceeded the one-hour timeout")
                            progress(chunk_index / chunk_count)
                            time.sleep(.3)
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                if process.returncode:
                    raise RuntimeError(f"MORGOTH inference failed. See {log_path.name} in the local run folder.")
                if not output.is_file():
                    raise RuntimeError("MORGOTH bridge returned no predictions")
                output.replace(cache)
                atomic_json(manifest, key)
            finally:
                mat_path.unlink(missing_ok=True)
                request.unlink(missing_ok=True)
                output.unlink(missing_ok=True)
        with np.load(cache, allow_pickle=False) as data:
            starts, scores, valid = data["starts"] + left, data["scores"], data["valid"].astype(bool)
        validate_predictions(starts, scores, valid, len(CLASSES[head]))
        centers = starts + 5
        keep = (centers >= core_start) & (centers < core_end)
        all_starts.append(starts[keep]); all_scores.append(scores[keep]); all_valid.append(valid[keep])
        progress((chunk_index + 1) / chunk_count)
    if not all_starts:
        raise ValueError("No complete model windows were available")
    result = (np.concatenate(all_starts), np.concatenate(all_scores), np.concatenate(all_valid))
    validate_predictions(*result, len(CLASSES[head]))
    return result, {"checkpoint_sha256": digest, "upstream_commit": registry["upstream_commit"],
                    "window_s": 10, "stride_s": 1, "context_s": 30, "device": os.getenv("NEUROSCOPE_DEVICE", "cpu"),
                    "positive_class": cfg.get("positive_class"), "parity_status": "Requires validation with authorized checkpoint"}


def model_status(settings):
    import importlib.util
    status = {
        "abnormal": {"ready": False, "model": "Unavailable", "reason": "No validated broad abnormality checkpoint. SPaRCNet Other does not mean normal."},
        "seizure": {"ready": False, "model": "SPaRCNet", "reason": "Run python scripts/setup_open_models.py to install the public checkpoint"},
        "sleep": {"ready": False, "model": "YASA", "reason": "Install open-model dependencies and run python scripts/setup_open_models.py"}}
    if not settings.registry.is_file():
        return status
    try:
        registry = json.loads(settings.registry.read_text())
        if registry.get("engine") != "open":
            return morgoth_status(settings)
        for head, key, dependency in [("seizure", "sparcnet", "torch"), ("sleep", "yasa", "yasa")]:
            if key not in registry or importlib.util.find_spec(dependency) is None:
                continue
            cfg = registry[key]
            if key == "sparcnet" and (not Path(cfg['checkpoint']).is_file() or sha256_file(cfg['checkpoint']) != cfg['sha256']):
                status[head]['reason'] = 'SPaRCNet checkpoint missing or checksum mismatch'
                continue
            status[head].update(ready=True, reason="Configured for local inference; outputs require review")
    except (ValueError, KeyError, OSError) as exc:
        for head in ('seizure', 'sleep'):
            status[head]['reason'] = str(exc)
    return status


def run_model(settings, recording, head, run_dir, cancelled, progress):
    registry = json.loads(settings.registry.read_text())
    if registry.get('engine') != 'open':
        return run_morgoth(settings, recording, head, run_dir, cancelled, progress)
    output = run_dir / f'{head}-native.npz'
    provenance_path = run_dir / f'{head}-provenance.json'
    progress_path = run_dir / f'{head}-progress.json'
    request = run_dir / f'{head}-request.json'
    atomic_json(request, {'registry': str(settings.registry), 'recording': recording, 'head': head,
                         'output': str(output), 'provenance': str(provenance_path), 'progress': str(progress_path)})
    env = dict(os.environ)
    env.setdefault('MPLCONFIGDIR', str(settings.data_dir / 'matplotlib'))
    env.setdefault('NUMBA_CACHE_DIR', str(settings.data_dir / 'numba'))
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[name] = '2'
    log_path = run_dir / f'{head}-inference.log'
    with log_path.open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'neuroscope.open_inference', str(request)], stdout=log, stderr=log, env=env)
        began = time.monotonic()
        try:
            while process.poll() is None:
                if cancelled():
                    raise Cancelled()
                if time.monotonic() - began > 7200:
                    raise RuntimeError('Inference exceeded two hours; use a shorter recording')
                try:
                    fraction = json.loads(progress_path.read_text())['fraction']
                except (OSError, ValueError, KeyError):
                    fraction = 0
                progress(fraction)
                time.sleep(.3)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            request.unlink(missing_ok=True)
    if process.returncode:
        lines = log_path.read_text(errors='replace').strip().splitlines()
        raise RuntimeError(f"{head} analysis failed: {lines[-1] if lines else 'unknown error'}. See {log_path.name}.")
    with np.load(output, allow_pickle=False) as data:
        result = data['starts'], data['scores'], data['valid'].astype(bool)
    validate_predictions(*result, len(CLASSES[head]))
    return result, json.loads(provenance_path.read_text())
