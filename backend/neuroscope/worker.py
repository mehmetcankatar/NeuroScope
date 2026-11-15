"""Single persistent analysis worker. Launch with python -m neuroscope.worker."""
from contextlib import contextmanager
import json
import os
import signal
import sys
from pathlib import Path
import time
import traceback
import numpy as np
from .config import Settings, CLASSES
from .db import Store, atomic_json
from .demo import predictions as demo_predictions
from .models import Cancelled, model_status, run_morgoth
from .pipeline import build_events, build_sleep, quality_intervals, validate_predictions


@contextmanager
def worker_lock(directory):
    path = directory / "worker.lock"
    f = path.open("a+b")
    if os.name == "nt":
        import msvcrt
        f.write(b"0"); f.flush(); f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield
    finally:
        f.close()


def process_run(store, settings, run):
    run_id = run["id"]
    recording = store.recording(run["recording_id"])
    directory = settings.data_dir / "runs" / run_id
    directory.mkdir(exist_ok=True)
    heads = run["config"]["heads"]
    results = {"events": [], "sleep": [], "quality": {}, "coverage": {}, "provenance": {},
               "source": "synthetic" if recording["kind"] == "synthetic" else "morgoth",
               "notice": "Synthetic demonstration; not MORGOTH predictions" if recording["kind"] == "synthetic" else "Research predictions; review required"}
    statuses = {h: {"status": "queued", "error": None} for h in heads}
    start_time = time.monotonic()
    last_progress = 0.0

    def cancelled():
        return store.run(run_id)["cancel_requested"]

    def progress(value):
        nonlocal last_progress
        now = time.monotonic()
        if now - last_progress > .3:
            store.update_run(run_id, progress=round(value * 100, 1), tasks=statuses)
            atomic_json(settings.data_dir / "worker-heartbeat.json", {"time": time.time(), "pid": os.getpid()})
            last_progress = now

    try:
        for hi, head in enumerate(heads):
            if cancelled():
                raise Cancelled()
            statuses[head]["status"] = "running"
            store.update_run(run_id, tasks=statuses, message=f"Analyzing {head}")
            try:
                if recording["kind"] == "synthetic":
                    starts, scores, valid = demo_predictions(head, recording["duration_s"])
                    provenance = {"model": "scripted-synthetic-fixture-v1", "window_s": 10, "stride_s": 1}
                else:
                    readiness = model_status(settings)[head]
                    if not readiness["ready"]:
                        raise ValueError(readiness["reason"])
                    (starts, scores, valid), provenance = run_morgoth(
                        settings, recording, head, directory, cancelled,
                        lambda x: progress((hi + x) / len(heads)))
                validate_predictions(starts, scores, valid, len(CLASSES[head]))
                target = directory / f"{head}.npz"
                tmp = target.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, starts=starts, scores=scores, valid=valid)
                tmp.replace(target)
                gaps, coverage_s = quality_intervals(starts, valid, recording["duration_s"])
                results["quality"][head] = gaps
                results["coverage"][head] = {"scored_s": coverage_s, "total_s": recording["duration_s"],
                                             "fraction": coverage_s / recording["duration_s"]}
                results["provenance"][head] = provenance
                if head == "sleep":
                    results["sleep"] = build_sleep(run_id, starts, scores, valid, recording["duration_s"])
                else:
                    results["events"].extend(build_events(run_id, head, starts, scores, valid, run["config"]))
                statuses[head]["status"] = "complete"
                atomic_json(directory / "results.json", results)
            except Cancelled:
                raise
            except Exception as exc:
                statuses[head].update(status="failed", error=str(exc))
                (directory / f"{head}-error.log").write_text(traceback.format_exc())
            progress((hi + 1) / len(heads))
        completed = sum(s["status"] == "complete" for s in statuses.values())
        status = "complete" if completed == len(heads) else "partial" if completed else "failed"
        atomic_json(directory / "results.json", results)
        store.update_run(run_id, status=status, tasks=statuses, progress=100,
                         runtime_s=round(time.monotonic() - start_time, 2),
                         message="Analysis finished" if status == "complete" else "Some analyses could not finish; inspect task details")
    except Cancelled:
        for value in statuses.values():
            if value["status"] in ("running", "queued"):
                value["status"] = "cancelled"
        atomic_json(directory / "results.json", results)
        store.update_run(run_id, status="cancelled", tasks=statuses, message="Cancelled; completed task results retained")


def main():
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    settings = Settings.from_env()
    store = Store(settings.data_dir)
    with worker_lock(settings.data_dir):
        store.recover()
        print("NeuroScope worker ready", flush=True)
        try:
            while True:
                atomic_json(settings.data_dir / "worker-heartbeat.json", {"time": time.time(), "pid": os.getpid()})
                run = store.claim()
                if run:
                    try:
                        process_run(store, settings, run)
                    except Exception as exc:
                        store.update_run(run["id"], status="failed", message=str(exc))
                else:
                    time.sleep(.5)
        finally:
            (settings.data_dir / "worker-heartbeat.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
