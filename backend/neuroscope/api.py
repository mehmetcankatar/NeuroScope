from contextlib import asynccontextmanager
import json
from pathlib import Path
import time
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from .config import Settings, CLASSES
from .db import Store, new_id
from .signals import demo_recording, inspect_edf, waveform
from .models import model_status
from .schemas import AnalysisRequest, AnnotationRequest
from .pipeline import apply_reviews
from .exporting import csv_export, html_export


def public_recording(record):
    return {k: v for k, v in record.items() if k not in ("path", "mapping")}


def create_app(settings=None):
    settings = settings or Settings.from_env()
    store = Store(settings.data_dir)
    app = FastAPI(title="NeuroScope API", version="0.1.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.store = store
    app.state.settings = settings

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        # Browser cross-origin writes are disabled; this is a local single-user app.
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            from urllib.parse import urlparse
            allowed = {"localhost", "127.0.0.1", "[::1]", "::1"}
            if urlparse(origin).hostname not in allowed:
                return Response("Cross-origin writes are disabled", status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def recording_or_404(id):
        record = store.recording(id)
        if record is None:
            raise HTTPException(404, "Recording not found")
        return record

    def run_or_404(id):
        run = store.run(id)
        if run is None:
            raise HTTPException(404, "Analysis not found")
        return run

    def results_for(id):
        run = run_or_404(id)
        path = settings.data_dir / "runs" / id / "results.json"
        results = json.loads(path.read_text()) if path.is_file() else {
            "events": [], "sleep": [], "coverage": {}, "quality": {}, "provenance": {},
            "source": run["source"], "notice": "Analysis has not produced results yet"}
        annotations = store.annotations(id)
        results["events"], results["sleep"] = apply_reviews(results["events"], results["sleep"], annotations)
        return {**results, "annotations": annotations, "run": run,
                "recording": public_recording(recording_or_404(run["recording_id"]))}

    @app.get("/api/health")
    def health():
        heartbeat_path = settings.data_dir / "worker-heartbeat.json"
        try:
            heartbeat = json.loads(heartbeat_path.read_text())["time"]
        except (OSError, ValueError, KeyError):
            heartbeat = 0
        return {"status": "ok", "worker_online": time.time() - heartbeat < 15, "version": "0.1.0"}

    @app.get("/api/models")
    def models():
        return {"heads": model_status(settings), "research_only": True,
                "license": "MORGOTH code: CC BY-NC 4.0", "demo_available": True}

    @app.get("/api/recordings")
    def recordings():
        return [public_recording(r) for r in store.recordings()]

    @app.post("/api/demo")
    def demo():
        record = demo_recording()
        if not store.recording(record["id"]):
            store.add_recording(record)
        return public_recording(store.recording(record["id"]))

    @app.post("/api/recordings", status_code=201)
    async def upload(file: UploadFile = File(...)):
        name = Path((file.filename or "recording.edf").replace("\\", "/")).name[:180]
        if not name.lower().endswith(".edf"):
            raise HTTPException(415, "Upload an EDF recording (.edf)")
        id = new_id()
        path = settings.data_dir / "recordings" / f"{id}.edf"
        total = 0
        try:
            with path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > settings.max_upload_bytes:
                        raise HTTPException(413, f"Upload exceeds {settings.max_upload_bytes // 1024 // 1024} MB limit")
                    output.write(chunk)
            from starlette.concurrency import run_in_threadpool
            record = await run_in_threadpool(inspect_edf, path, id, name)
            store.add_recording(record)
            return public_recording(record)
        except HTTPException:
            path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(422, f"Cannot read recording: {exc}") from exc
        finally:
            await file.close()

    @app.get("/api/recordings/{id}/capabilities")
    def caps(id: str):
        record = recording_or_404(id)
        return {"recording": record["capabilities"], "models": model_status(settings)}

    @app.get("/api/recordings/{id}/analyses")
    def analyses(id: str):
        recording_or_404(id)
        return store.runs(id)

    @app.post("/api/recordings/{id}/analyses", status_code=202)
    def analyze(id: str, config: AnalysisRequest):
        record = recording_or_404(id)
        readiness = model_status(settings)
        for head in config.heads:
            if not record["capabilities"][head]["supported"]:
                raise HTTPException(422, f"{head}: {record['capabilities'][head]['reason']}")
            if record["kind"] != "synthetic" and not readiness[head]["ready"]:
                raise HTTPException(409, f"{head}: {readiness[head]['reason']}")
        active = [r for r in store.runs(id) if r["status"] in ("queued", "running", "postprocessing")]
        if active:
            raise HTTPException(409, "An analysis is already active for this recording")
        run = {"id": new_id(), "recording_id": id, "config": config.model_dump(), "tasks": {},
               "source": "synthetic" if record["kind"] == "synthetic" else "morgoth", "message": "Waiting for worker"}
        store.add_run(run)
        return store.run(run["id"])

    @app.get("/api/analyses/{id}")
    def analysis(id: str):
        return run_or_404(id)

    @app.post("/api/analyses/{id}/cancel")
    def cancel(id: str):
        run_or_404(id)
        store.cancel(id)
        return store.run(id)

    @app.post("/api/analyses/{id}/retry")
    def retry(id: str):
        run = run_or_404(id)
        if run["status"] not in ("failed", "partial", "cancelled"):
            raise HTTPException(409, "Only failed, partial, or cancelled analyses can be retried")
        if any(r["status"] in ("queued", "running") for r in store.runs(run["recording_id"])):
            raise HTTPException(409, "Another analysis is already active")
        with store.connect() as db:
            db.execute("UPDATE runs SET cancel=0 WHERE id=?", (id,))
        store.update_run(id, status="queued", progress=0, message="Retry queued; verified chunk caches will be reused")
        return store.run(id)

    @app.get("/api/analyses/{id}/results")
    def results(id: str):
        return results_for(id)

    @app.get("/api/analyses/{id}/events")
    def events(id: str, label: str | None = None):
        events = results_for(id)["events"]
        return [e for e in events if label is None or e["label"] == label]

    @app.get("/api/analyses/{id}/sleep")
    def sleep(id: str):
        return results_for(id)["sleep"]

    @app.get("/api/analyses/{id}/scores")
    def scores(id: str, head: str, start: float = Query(0, ge=0), end: float = Query(60, gt=0)):
        run_or_404(id)
        if head not in CLASSES or end <= start or end - start > 3600:
            raise HTTPException(422, "Invalid score range or head")
        path = settings.data_dir / "runs" / id / f"{head}.npz"
        if not path.is_file():
            return {"times": [], "scores": [], "classes": CLASSES[head], "valid": []}
        with np.load(path, allow_pickle=False) as data:
            centers = data["starts"] + 5
            mask = (centers >= start) & (centers < end)
            values = data["scores"][mask]
            return {"times": centers[mask].tolist(), "scores": [[float(v) if np.isfinite(v) else None for v in row] for row in values],
                    "valid": data["valid"][mask].tolist(), "classes": CLASSES[head]}

    @app.get("/api/recordings/{id}/waveform")
    def wave(id: str, start: float = Query(0, ge=0), end: float = Query(20, gt=0),
             channels: str | None = None, points: int = Query(1200, ge=100, le=3000), montage: str = "recorded"):
        record = recording_or_404(id)
        if end <= start or start >= record["duration_s"] or end - start > 120:
            raise HTTPException(422, "Waveform ranges must be 0–120 seconds within the recording")
        selected = channels.split(",") if channels else record["channels"][:19]
        if not selected or len(selected) > 32 or any(c not in record["channels"] for c in selected):
            raise HTTPException(422, "Unknown channel or too many channels")
        if montage not in ("recorded", "average"):
            raise HTTPException(422, "Unsupported display montage")
        return waveform(record, start, min(end, record["duration_s"]), selected, points, montage)

    @app.post("/api/analyses/{id}/annotations", status_code=201)
    def annotate(id: str, annotation: AnnotationRequest):
        payload = results_for(id)
        if annotation.end_s is not None and annotation.end_s > payload["recording"]["duration_s"]:
            raise HTTPException(422, "Annotation exceeds recording duration")
        if annotation.target_type != "manual":
            collection = payload["events"] if annotation.target_type == "event" else payload["sleep"]
            if not any(e["id"] == annotation.target_id for e in collection):
                raise HTTPException(404, "Annotation target does not exist")
        if annotation.target_type == "sleep" and annotation.label is not None and annotation.label not in CLASSES["sleep"] + ["Unscored", "Uncertain"]:
            raise HTTPException(422, "Unknown sleep stage")
        return store.annotate(id, annotation.model_dump())

    @app.get("/api/analyses/{id}/export")
    def export(id: str, format: str = "json"):
        payload = results_for(id)
        if format == "json":
            body, media = json.dumps(payload, indent=2, allow_nan=False), "application/json"
        elif format == "csv":
            body, media = csv_export(payload), "text/csv"
        elif format == "html":
            body, media = html_export(payload), "text/html"
        else:
            raise HTTPException(422, "Choose json, csv, or html")
        return Response(body, media_type=media, headers={"Content-Disposition": f'attachment; filename="NeuroScope-{id[:8]}.{format}"'})

    if settings.frontend.is_dir():
        app.mount("/assets", StaticFiles(directory=settings.frontend / "assets"), name="assets")

        @app.get("/favicon.svg")
        def icon():
            return FileResponse(settings.frontend / "favicon.svg")

        @app.get("/")
        def index():
            return FileResponse(settings.frontend / "index.html")
    else:
        @app.get("/")
        def index():
            return {"app": "NeuroScope", "message": "Build the frontend or start Vite on port 5173", "api_docs": "/api/docs"}
    return app


app = create_app()
