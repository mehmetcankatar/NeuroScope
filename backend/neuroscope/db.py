"""Small persistent queue and append-only review history. No model arrays in SQLite."""
from contextlib import contextmanager
import json
import sqlite3
import time
import uuid
from .serialization import json_dumps, to_jsonable
from pathlib import Path


def new_id():
    return uuid.uuid4().hex


class Store:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("recordings", "runs", "tmp"):
            (directory / name).mkdir(exist_ok=True)
        self.path = directory / "neuroscope.sqlite3"
        with self.connect() as db:
            db.executescript("""
              PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS recordings(id TEXT PRIMARY KEY, body TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS runs(
                id TEXT PRIMARY KEY, recording_id TEXT NOT NULL,
                status TEXT NOT NULL, created REAL NOT NULL, heartbeat REAL NOT NULL,
                cancel INTEGER NOT NULL DEFAULT 0, body TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS annotations(
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, created REAL NOT NULL, body TEXT NOT NULL);
              CREATE INDEX IF NOT EXISTS run_recording ON runs(recording_id,created);
              CREATE INDEX IF NOT EXISTS annotation_run ON annotations(run_id,created);
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def add_recording(self, body):
        with self.connect() as db:
            db.execute("INSERT INTO recordings VALUES (?,?)", (body["id"], json_dumps(body)))

    def recordings(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT body FROM recordings ORDER BY rowid DESC")]

    def recording(self, id):
        with self.connect() as db:
            row = db.execute("SELECT body FROM recordings WHERE id=?", (id,)).fetchone()
            return json.loads(row[0]) if row else None

    def add_run(self, body):
        now = time.time()
        body.update(created=now, updated=now, progress=0, status="queued")
        with self.connect() as db:
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
                       (body["id"], body["recording_id"], "queued", now, now, 0, json_dumps(body)))

    def run(self, id):
        with self.connect() as db:
            row = db.execute("SELECT body,cancel FROM runs WHERE id=?", (id,)).fetchone()
            if not row:
                return None
            return {**json.loads(row[0]), "cancel_requested": bool(row[1])}

    def runs(self, recording_id):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute(
                "SELECT body FROM runs WHERE recording_id=? ORDER BY created DESC", (recording_id,))]

    def update_run(self, id, **changes):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM runs WHERE id=?", (id,)).fetchone()
            body = json.loads(row[0])
            body.update(to_jsonable(changes), updated=time.time())
            db.execute("UPDATE runs SET status=?,heartbeat=?,body=? WHERE id=?",
                       (body["status"], time.time(), json_dumps(body), id))
        return body

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # Only one worker is supported; its launcher enforces a process lock.
            row = db.execute("SELECT id,body FROM runs WHERE status='queued' AND cancel=0 ORDER BY created LIMIT 1").fetchone()
            if not row:
                return None
            body = json.loads(row["body"])
            body.update(status="running", updated=time.time())
            db.execute("UPDATE runs SET status='running', heartbeat=?, body=? WHERE id=?",
                       (time.time(), json_dumps(body), row["id"]))
            return body

    def recover(self):
        # Called after acquiring the exclusive worker process lock.
        with self.connect() as db:
            rows = db.execute("SELECT id,body,cancel FROM runs WHERE status IN ('running','postprocessing')").fetchall()
            for row in rows:
                body = json.loads(row["body"])
                status = "cancelled" if row["cancel"] else "queued"
                body.update(status=status, message="Recovered after worker restart")
                db.execute("UPDATE runs SET status=?,body=? WHERE id=?", (status, json_dumps(body), row["id"]))

    def cancel(self, id):
        with self.connect() as db:
            db.execute("UPDATE runs SET cancel=1 WHERE id=?", (id,))
        run = self.run(id)
        if run and run["status"] == "queued":
            self.update_run(id, status="cancelled")

    def annotate(self, run_id, body):
        body = {**to_jsonable(body), "id": new_id(), "created": time.time(), "run_id": run_id}
        with self.connect() as db:
            db.execute("INSERT INTO annotations VALUES (?,?,?,?)",
                       (body["id"], run_id, body["created"], json_dumps(body)))
        return body

    def annotations(self, run_id):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute(
                "SELECT body FROM annotations WHERE run_id=? ORDER BY created,rowid", (run_id,))]


def atomic_json(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json_dumps(body), encoding="utf-8")
    tmp.replace(path)
