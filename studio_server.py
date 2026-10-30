#!/usr/bin/env python3
"""Local page for the dataset studio."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from studio_catalog import catalog, expand

ROOT = Path(__file__).resolve().parent


def _load_local_env():
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        name, value = text.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


_load_local_env()
HOST = os.environ.get("STUDIO_SSH_HOST", "")
KEY = Path(os.environ.get("STUDIO_SSH_KEY", ""))
REMOTE_ROOT = os.environ.get("STUDIO_REMOTE_ROOT", "")
PORT = int(os.environ.get("STUDIO_PORT", "8787"))
JOBS = {}
LOCK = threading.Lock()


def ssh_base():
    if not HOST or not KEY.is_file() or not REMOTE_ROOT:
        raise RuntimeError("Set STUDIO_SSH_HOST, STUDIO_SSH_KEY, and STUDIO_REMOTE_ROOT in the local environment.")
    return [
        "ssh", "-4", "-i", str(KEY), "-o", "ConnectTimeout=20", "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=200", HOST,
    ]


def scp(source, destination):
    subprocess.run(
        ["scp", "-i", str(KEY), "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", source, destination],
        check=True,
    )


def saved_runs():
    folder = ROOT / "results" / "studio"
    if not folder.exists():
        return []
    runs = []
    for job_path in sorted(folder.glob("*/job.json"), reverse=True):
        try:
            record = json.loads(job_path.read_text())
        except json.JSONDecodeError:
            continue
        labels_path = job_path.parent / "labels.json"
        count = 0
        if labels_path.exists():
            try:
                count = len(json.loads(labels_path.read_text()).get("images") or [])
            except json.JSONDecodeError:
                count = 0
        settings = record.get("settings") or {}
        runs.append({
            "id": record.get("id") or job_path.parent.name,
            "pipeline": settings.get("pipeline") or "anomalygen",
            "count": count,
        })
    return [run for run in runs if run["count"]][:12]


def running_job():
    with LOCK:
        return next((job for job in JOBS.values() if job["status"] == "running"), None)


def public_job(job):
    return {
        "id": job["id"],
        "status": job["status"],
        "events": job["events"],
        "prompt": job["prompt"],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[studio] {self.address_string()} {fmt % args}", flush=True)

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 20_000_000:
            raise ValueError("Upload is larger than 20 MB")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode() or "{}")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._file(ROOT / "studio" / "index.html", "text/html; charset=utf-8")
        if path == "/api/catalog":
            return self.send_json(catalog())
        if path == "/api/runs":
            return self.send_json({"runs": saved_runs()})
        if path.startswith("/api/runs/"):
            return self._run(path.split("/")[-1])
        if path == "/api/jobs":
            with LOCK:
                jobs = [public_job(job) for job in JOBS.values()]
            return self.send_json({"jobs": jobs})
        if path.startswith("/api/jobs/"):
            job_id = path.split("/")[-1]
            with LOCK:
                job = JOBS.get(job_id)
            if not job:
                return self.send_json({"error": "No such job"}, 404)
            return self.send_json(public_job(job))
        if path.startswith("/media/"):
            return self._media(path[len("/media/"):])
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self.read_json()
        except ValueError as error:
            return self.send_json({"error": str(error)}, 400)
        except json.JSONDecodeError:
            return self.send_json({"error": "The request was not valid JSON"}, 400)
        if path == "/api/preview":
            return self.send_json(self._preview(payload.get("settings") or {}))
        if path == "/api/jobs":
            return self._start(payload)
        if path == "/api/jobs/continue":
            return self._continue()
        self.send_error(404)

    def _run(self, job_id):
        if not job_id.replace("-", "").isdigit():
            return self.send_json({"error": "No such run"}, 404)
        folder = ROOT / "results" / "studio" / job_id
        job_path = folder / "job.json"
        labels_path = folder / "labels.json"
        if not job_path.exists() or not labels_path.exists():
            return self.send_json({"error": "No such run"}, 404)
        record = json.loads(job_path.read_text())
        labels = json.loads(labels_path.read_text())
        return self.send_json({
            "id": job_id,
            "pipeline": (record.get("settings") or {}).get("pipeline") or "anomalygen",
            "images": labels.get("images") or [],
        })

    def _preview(self, settings):
        plan = expand(settings)
        return {
            "pipeline": plan["pipeline"],
            "summary": plan["summary"],
            "total": plan["total"],
            "full_total": plan["full_total"],
            "held_total": plan["held_total"],
            "scene_total": plan["scene_total"],
            "minutes": plan["minutes"],
            "varieties": plan["varieties"],
            "positions": plan["positions"],
            "backgrounds": plan["backgrounds"],
        }

    def _start(self, payload):
        active = running_job()
        if active or remote_studio_running():
            return self.send_json({
                "error": "A pipeline job is already running on the GPU. Press Continue to reattach.",
                "id": active["id"] if active else None,
            }, 409)
        settings = payload.get("settings") or {}
        plan = expand(settings)
        if not plan["generate"] and not plan["held"]:
            return self.send_json({"error": "Choose a machine and a damage the grid can record."}, 400)
        job_id = time_id()
        prompt = (payload.get("prompt") or "").strip() or catalog()["sample_prompt"]
        folder = ROOT / "results" / "studio" / job_id
        folder.mkdir(parents=True, exist_ok=True)
        reference = save_photo(payload.get("photo"), folder) or "inputs/reference.jpg"
        job = {
            "id": job_id,
            "status": "running",
            "prompt": prompt,
            "events": [],
            "settings": settings,
        }
        record = {
            "id": job_id,
            "prompt": prompt,
            "reference": reference,
            "settings": settings,
        }
        (folder / "job.json").write_text(json.dumps(record, indent=2))
        with LOCK:
            JOBS[job_id] = job
        threading.Thread(target=launch, args=(job, folder / "job.json"), daemon=True).start()
        return self.send_json({"id": job_id, "status": "running"}, 202)

    def _continue(self):
        active = running_job()
        if active:
            return self.send_json({"id": active["id"], "status": "running"})
        record = latest_saved_job()
        if not record:
            return self.send_json({"error": "There is no job to continue."}, 404)
        job = {
            "id": record["id"],
            "status": "running",
            "prompt": record.get("prompt", ""),
            "events": [],
            "settings": record.get("settings") or {},
        }
        with LOCK:
            JOBS[job["id"]] = job
        sync_events(job)
        if remote_studio_running():
            with LOCK:
                if job["status"] != "done":
                    job["status"] = "running"
            threading.Thread(target=follow, args=(job,), daemon=True).start()
        else:
            threading.Thread(target=launch, args=(job, ROOT / "results" / "studio" / job["id"] / "job.json"), daemon=True).start()
        return self.send_json({"id": job["id"], "status": job["status"]}, 202)

    def _file(self, path: Path, content_type: str):
        if not path.exists():
            return self.send_error(404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _media(self, relative: str):
        if relative.startswith("inputs/"):
            path = (ROOT / relative).resolve()
            allowed = (ROOT / "inputs").resolve()
        elif relative.startswith("results/studio/"):
            path = (ROOT / relative).resolve()
            allowed = (ROOT / "results" / "studio").resolve()
        else:
            return self.send_error(404)
        if not str(path).startswith(str(allowed)) or not path.is_file():
            return self.send_error(404)
        kind = "image/png" if path.suffix == ".png" else "image/jpeg"
        return self._file(path, kind)


def time_id():
    import time
    return time.strftime("%Y%m%d-%H%M%S")


def save_photo(photo, folder: Path):
    if not photo:
        return None
    ext = "png" if photo.startswith("data:image/png") else "jpg"
    if "," in photo:
        photo = photo.split(",", 1)[1]
    destination = folder / f"reference.{ext}"
    destination.write_bytes(base64.b64decode(photo))
    return str(destination.relative_to(ROOT))


def add_event(job, event):
    with LOCK:
        job["events"].append(event)
        kind = event.get("kind")
        if kind == "done":
            job["status"] = "done"
        elif kind == "error":
            job["status"] = "error"
        elif job["status"] != "done":
            job["status"] = "running"


def remote_studio_running():
    completed = subprocess.run(
        ssh_base() + ["pgrep -f '[p]ython3 -u studio_'"],
        capture_output=True, text=True,
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


def latest_saved_job():
    files = sorted((ROOT / "results" / "studio").glob("*/job.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def sync_events(job):
    remote = f"{REMOTE_ROOT}/results/studio/{job['id']}/events.jsonl"
    completed = subprocess.run(ssh_base() + [f"cat {remote}"], capture_output=True, text=True)
    if completed.returncode:
        return
    seen = {event.get("t") for event in job["events"]}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("t") in seen:
            continue
        paths = image_paths(event)
        if paths:
            failed = False
            for relative in paths:
                local = ROOT / relative
                if local.exists() and local.stat().st_size > 10000:
                    continue
                try:
                    pull(relative)
                except subprocess.CalledProcessError:
                    failed = True
            if failed:
                event = {
                    "kind": "message", "role": "pipeline", "t": event.get("t"),
                    "text": "An image is on the GPU machine. Copying it here failed, and Continue will try again.",
                }
        add_event(job, event)


def follow(job):
    idle = 0
    while job["status"] == "running":
        before = len(job["events"])
        sync_events(job)
        if job["status"] != "running":
            return
        if len(job["events"]) != before:
            idle = 0
        elif not remote_studio_running():
            idle += 1
            if idle >= 2:
                add_event(job, {
                    "kind": "error",
                    "text": "The GPU job stopped. Press Continue to resume from the scenes already saved.",
                })
                return
        else:
            idle = 0
        time.sleep(3)


def image_paths(event):
    if event.get("kind") == "image" and event.get("path"):
        return [event["path"]]
    if event.get("kind") == "result":
        return [event[key] for key in ("clean", "mask", "damaged") if event.get(key)]
    return []


def pull(relative: str):
    local = ROOT / relative
    local.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for _ in range(5):
        try:
            scp(f"{HOST}:{REMOTE_ROOT}/{relative}", str(local))
            return
        except subprocess.CalledProcessError as error:
            last = error
            time.sleep(3)
    raise last


def launch(job, job_path: Path):
    try:
        remote_dir = f"{REMOTE_ROOT}/results/studio/{job['id']}"
        subprocess.run(ssh_base() + [f"mkdir -p {remote_dir}/logs"], check=True)
        for name in ("studio_catalog.py", "studio_run.py", "studio_sunburst.py", "shape_on_part.py"):
            scp(str(ROOT / name), f"{HOST}:{REMOTE_ROOT}/{name}")
        scp(str(job_path), f"{HOST}:{remote_dir}/job.json")
        record = json.loads(job_path.read_text())
        reference = record["reference"]
        if reference.startswith("results/"):
            scp(str(ROOT / reference), f"{HOST}:{REMOTE_ROOT}/{reference}")
        script = "studio_sunburst.py" if (record.get("settings") or {}).get("pipeline") in {"sunburst", "gptclean"} else "studio_run.py"
        command = ssh_base() + [
            f"cd {REMOTE_ROOT} && python3 -u {script} {remote_dir}/job.json",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            if not line.startswith("EVENT "):
                continue
            event = json.loads(line[len("EVENT "):])
            copied = True
            for relative in image_paths(event):
                try:
                    pull(relative)
                except subprocess.CalledProcessError:
                    copied = False
            if not copied:
                add_event(job, {
                    "kind": "message", "role": "pipeline",
                    "text": "Copying an image to this Mac failed because the SSH connection dropped. The GPU job keeps going.",
                })
            add_event(job, event)
        code = process.wait()
        if code and job["status"] == "running":
            add_event(job, {"kind": "error", "text": f"The GPU job exited with status {code}."})
    except Exception as error:
        add_event(job, {"kind": "error", "text": str(error)})


def reattach_if_remote():
    if not remote_studio_running():
        return
    record = latest_saved_job()
    if not record:
        return
    job = {
        "id": record["id"],
        "status": "running",
        "prompt": record.get("prompt", ""),
        "events": [],
        "settings": record.get("settings") or {},
    }
    with LOCK:
        JOBS[job["id"]] = job
    sync_events(job)
    with LOCK:
        if job["status"] != "done":
            job["status"] = "running"
    threading.Thread(target=follow, args=(job,), daemon=True).start()
    print(f"Reattached to GPU job {job['id']}", flush=True)


def main():
    threading.Thread(target=reattach_if_remote, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Dataset studio: http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
