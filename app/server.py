"""Recover Studio web server: the browser UI + JSON API around the restoration pipeline.

    python app/server.py [--host 127.0.0.1] [--port 8793] [--no-browser]

Only the standard library is used for HTTP. One job runs at a time (the GPU is shared), the others wait in a queue.
"""
import argparse
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import traceback
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

APP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP)
WEB = os.path.join(APP, "web")
sys.path.insert(0, APP)

from studio import analyze as AN  # noqa: E402
from studio import audio_io as A  # noqa: E402
from studio import device as D  # noqa: E402
from studio import dsp  # noqa: E402
from studio import models as M  # noqa: E402
from studio import pipeline as P  # noqa: E402

VERSION = "1.0.0"
os.makedirs(P.PROJECTS, exist_ok=True)
for ext, t in ((".mp3", "audio/mpeg"), (".flac", "audio/flac"), (".wav", "audio/wav"), (".ogg", "audio/ogg"),
               (".txt", "text/plain; charset=utf-8"), (".js", "application/javascript"), (".json", "application/json")):
    mimetypes.add_type(t, ext)

JOBS = {}            # pid -> latest Job of that project
QUEUE = queue.Queue()
DOWNLOADS = {}       # model key -> {status, done, total, error}
SYSTEM = {}


def worker():
    while True:
        job = QUEUE.get()
        if job._cancel:
            job.status = "cancelled"
            continue
        try:
            job.run()
        except Exception:
            traceback.print_exc()


def system_info(refresh=False):
    if refresh or not SYSTEM:
        d = D.detect("auto")
        SYSTEM.update(d, label=D.label(d), version=VERSION, python=sys.version.split()[0],
                      colab="COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ)
    return dict(SYSTEM, models=M.status(), downloads=DOWNLOADS, queue=QUEUE.qsize(), projects_dir=P.PROJECTS,
                sources={k: v for k, v in AN.SOURCES.items()}, defaults=P.DEFAULT_OPTIONS,
                grid=[round(float(f), 1) for f in dsp.GRID], target=[round(float(v), 1) for v in dsp.TARGET_BALANCED],
                tilts={s: [round(float(v), 2) for v in dsp.style_tilt(dsp.GRID, s)] for s in ("warm", "balanced", "open")})


def project_list():
    out = []
    for pid in sorted(os.listdir(P.PROJECTS), reverse=True):
        try:
            m = P.load_manifest(pid)
        except Exception:
            continue
        j = JOBS.get(pid)
        a = m.get("analysis") or {}
        res = m.get("results") or {}
        out.append(dict(id=pid, name=m["name"], created=m["created"],
                        status=j.status if j and j.status in ("queued", "running") else m.get("status"),
                        duration=m.get("duration"), source=a.get("source_guess"),
                        findings=len([f for f in a.get("findings", []) if f["severity"] > 0]),
                        full=bool(res.get("full")), preview=bool(res.get("preview")),
                        tone=(res.get("full") or {}).get("after", {}).get("tone_error_db")))
    return out


def safe_path(pid, rel):
    base = os.path.realpath(P.project_dir(pid))
    p = os.path.realpath(os.path.join(base, rel))
    if not (p == base or p.startswith(base + os.sep)):
        raise PermissionError(rel)
    return p


def clean_name(s):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", s).strip(" .") or "track"


def export_wav(pid, kind, what, sr, bits):
    """Converts a stored float master to the requested WAV format. -> (path, download name)."""
    man = P.load_manifest(pid)
    res = (man.get("results") or {}).get(kind)
    if not res:
        raise FileNotFoundError("нет результата - сначала нажмите «Восстановить»")
    src = os.path.join(P.project_dir(pid), res["files"][what])
    x, sr0 = A.read_any(src)
    if res.get("mono_out"):
        x = x.mean(1, keepdims=True)
    y = A.resample(x, sr0, sr)
    if sr > sr0:   # resampled peaks can overshoot by a hair
        pk = A.true_peak_db(y, sr) if y.shape[1] else -99
        if pk > -0.9:
            y = y * 10 ** ((-1.0 - pk) / 20)
    tag = {"restored": "restored", "repaired": "repaired (no mastering)"}[what]
    name = clean_name(f"{man['name']}{' (preview)' if kind == 'preview' else ''} [{tag}] {sr // 1000 if sr % 1000 == 0 else sr / 1000}k {bits}bit.wav")
    out = os.path.join(P.project_dir(pid), "export", f"{what}_{kind}_{sr}_{bits}.wav")
    A.write_wav(out, y, sr, str(bits))
    return out, name


def export_zip(pid, kind, sr, bits):
    man = P.load_manifest(pid)
    res = (man.get("results") or {}).get(kind)
    if not res:
        raise FileNotFoundError("нет результата")
    pdir = P.project_dir(pid)
    out = os.path.join(pdir, "export", f"{clean_name(man['name'])}_{kind}.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with zipfile.ZipFile(out + ".tmp", "w", allowZip64=True) as z:
        for what in ("restored", "repaired"):
            p, name = export_wav(pid, kind, what, sr, bits)
            z.write(p, name, compress_type=zipfile.ZIP_STORED)
            os.remove(p)
        for s in res.get("stems") or []:
            z.write(os.path.join(pdir, s["file"]), f"stems/{os.path.basename(s['file'])}", compress_type=zipfile.ZIP_STORED)
        rp = os.path.join(pdir, kind, "report.txt")
        if os.path.exists(rp):
            z.write(rp, "report.txt")
    os.replace(out + ".tmp", out)
    return out, os.path.basename(out)


# ================================================================ HTTP
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RecoverStudio/" + VERSION

    def log_message(self, fmt, *args):
        if args and isinstance(args[1], str) and args[1].startswith(("4", "5")):
            sys.stderr.write("  http %s\n" % (fmt % args))

    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def err(self, code, msg):
        self.send_json(dict(error=msg), code)

    def body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def read_body_to(self, dst):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            raise ValueError("пустой файл")
        left = n
        with open(dst, "wb") as f:
            while left > 0:
                b = self.rfile.read(min(1 << 20, left))
                if not b:
                    raise IOError("загрузка оборвалась")
                f.write(b)
                left -= len(b)

    def send_file(self, path, download=False, name=None, remove_after=False):
        if not os.path.isfile(path):
            return self.err(404, "файл не найден")
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and not download:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        if download:
            fn = name or os.path.basename(path)
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(fn)}")
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    b = f.read(min(1 << 20, left))
                    if not b:
                        break
                    self.wfile.write(b)
                    left -= len(b)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass
        finally:
            if remove_after:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        path = unquote(u.path)
        try:
            if path in ("/", "/index.html"):
                return self.send_file(os.path.join(WEB, "index.html"))
            if path.startswith("/static/"):
                p = os.path.realpath(os.path.join(WEB, path[len("/static/"):]))
                if not p.startswith(os.path.realpath(WEB) + os.sep):
                    return self.err(403, "forbidden")
                return self.send_file(p)
            if path == "/api/system":
                return self.send_json(system_info(q.get("refresh") == "1"))
            if path == "/api/projects":
                return self.send_json(project_list())
            m = re.fullmatch(r"/api/projects/([\w\-]+)", path)
            if m:
                pid = m.group(1)
                man = P.load_manifest(pid)
                j = JOBS.get(pid)
                man["job"] = j.state(len(j.log)) if j else None
                return self.send_json(man)
            m = re.fullmatch(r"/api/jobs/([\w\-]+)", path)
            if m:
                j = JOBS.get(m.group(1))
                return self.send_json(j.state(int(q.get("since", 0))) if j else dict(status="none"))
            m = re.fullmatch(r"/files/([\w\-]+)/(.+)", path)
            if m:
                return self.send_file(safe_path(m.group(1), m.group(2)), q.get("dl") == "1")
            m = re.fullmatch(r"/api/peaks/([\w\-]+)/(.+)", path)
            if m:
                return self.send_json(A.peaks_cached(safe_path(m.group(1), m.group(2)), int(q.get("n", 1600))))
            m = re.fullmatch(r"/api/spectro/([\w\-]+)/(.+)", path)
            if m:
                p = safe_path(m.group(1), m.group(2))
                cache = p + ".spec.json"
                if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(p):
                    return self.send_file(cache)
                x, sr = A.read_any(p)
                d = A.spectrogram(x, sr, cols=int(q.get("cols", 900)), rows=int(q.get("rows", 140)))
                with open(cache, "w") as f:
                    json.dump(d, f)
                return self.send_json(d)
            m = re.fullmatch(r"/api/download/([\w\-]+)", path)
            if m:
                pid = m.group(1)
                kind = q.get("kind", "full")
                what = q.get("what", "restored")
                sr = int(q.get("sr", 44100))
                bits = q.get("bits", "24")
                if kind not in ("full", "preview") or sr not in (44100, 48000, 88200, 96000) or bits not in ("16", "24", "32"):
                    return self.err(400, "bad format")
                if what == "zip":
                    p, name = export_zip(pid, kind, sr, bits)
                elif what in ("restored", "repaired"):
                    p, name = export_wav(pid, kind, what, sr, int(bits))
                else:
                    return self.err(400, "what?")
                return self.send_file(p, True, name, remove_after=True)
            return self.err(404, "нет такого адреса")
        except FileNotFoundError as e:
            return self.err(404, str(e))
        except PermissionError:
            return self.err(403, "forbidden")
        except Exception as e:
            traceback.print_exc()
            return self.err(500, f"{type(e).__name__}: {e}")

    def do_PUT(self):
        return self.do_POST()

    def queue_job(self, pid, opts):
        j = JOBS.get(pid)
        if j and j.status in ("queued", "running"):
            return False
        job = P.Job(pid, opts)
        JOBS[pid] = job
        QUEUE.put(job)
        return True

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        path = unquote(u.path)
        try:
            if path == "/api/upload":
                name = os.path.basename(q.get("name", "track.wav"))
                if not name.lower().endswith(A.AUDIO_EXT):
                    return self.err(400, "нужен аудиофайл: " + ", ".join(A.AUDIO_EXT))
                pid = P.create_project(name, self.read_body_to)
                self.queue_job(pid, dict(analyze_only=True))
                return self.send_json(dict(id=pid))
            if path == "/api/import":   # a local path (desktop use / Colab: the file is already on disk)
                src = (self.body_json().get("path") or "").strip().strip('"')
                if not os.path.isfile(src):
                    return self.err(404, f"файл не найден: {src}")
                if not src.lower().endswith(A.AUDIO_EXT):
                    return self.err(400, "это не аудиофайл")
                pid = P.create_project(os.path.basename(src), lambda dst: shutil.copyfile(src, dst))
                self.queue_job(pid, dict(analyze_only=True))
                return self.send_json(dict(id=pid))
            m = re.fullmatch(r"/api/projects/([\w\-]+)/reference", path)
            if m:
                name = os.path.basename(q.get("name", "reference.wav"))
                if not name.lower().endswith(A.AUDIO_EXT):
                    return self.err(400, "нужен аудиофайл")
                P.set_reference(m.group(1), name, self.read_body_to)
                return self.send_json(dict(ok=True))
            m = re.fullmatch(r"/api/projects/([\w\-]+)/reference/delete", path)
            if m:
                man = P.load_manifest(m.group(1))
                if man.get("reference"):
                    try:
                        os.remove(os.path.join(P.project_dir(m.group(1)), man["reference"]["file"]))
                    except OSError:
                        pass
                man["reference"] = None
                P.save_manifest(m.group(1), man)
                return self.send_json(dict(ok=True))
            m = re.fullmatch(r"/api/projects/([\w\-]+)/(run|analyze)", path)
            if m:
                pid = m.group(1)
                opts = self.body_json() if m.group(2) == "run" else dict(analyze_only=True)
                if not self.queue_job(pid, opts):
                    return self.err(409, "уже выполняется")
                return self.send_json(dict(ok=True, queued=QUEUE.qsize()))
            m = re.fullmatch(r"/api/projects/([\w\-]+)/cancel", path)
            if m:
                j = JOBS.get(m.group(1))
                if j:
                    j.cancel()
                return self.send_json(dict(ok=True))
            m = re.fullmatch(r"/api/projects/([\w\-]+)/rename", path)
            if m:
                man = P.load_manifest(m.group(1))
                man["name"] = (self.body_json().get("name") or man["name"]).strip()[:120]
                P.save_manifest(m.group(1), man)
                return self.send_json(dict(ok=True))
            m = re.fullmatch(r"/api/projects/([\w\-]+)/delete", path)
            if m:
                pid = m.group(1)
                j = JOBS.get(pid)
                if j and j.status in ("queued", "running"):
                    return self.err(409, "сначала отмените обработку")
                shutil.rmtree(P.project_dir(pid), ignore_errors=True)
                JOBS.pop(pid, None)
                return self.send_json(dict(ok=True))
            m = re.fullmatch(r"/api/projects/([\w\-]+)/open", path)
            if m:   # open the folder in Explorer / Finder (desktop only)
                d = P.project_dir(m.group(1))
                if os.name == "nt":
                    os.startfile(d)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", d])
                else:
                    subprocess.Popen(["xdg-open", d])
                return self.send_json(dict(ok=True))
            m = re.fullmatch(r"/api/models/(\w+)/download", path)
            if m:
                key = m.group(1)
                if key not in M.MODELS_INFO:
                    return self.err(404, "нет такой модели")
                if DOWNLOADS.get(key, {}).get("status") == "running":
                    return self.send_json(dict(ok=True))
                st = DOWNLOADS[key] = dict(status="running", done=0, total=0, file="")

                def go():
                    try:
                        M.ensure(key, lambda d, t, fn: st.update(done=d, total=t, file=fn), lambda s: None)
                        st["status"] = "done"
                    except Exception as e:
                        st.update(status="error", error=str(e))
                threading.Thread(target=go, daemon=True).start()
                return self.send_json(dict(ok=True))
            return self.err(404, "нет такого адреса")
        except FileNotFoundError as e:
            return self.err(404, str(e))
        except Exception as e:
            traceback.print_exc()
            return self.err(500, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8793)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    threading.Thread(target=worker, daemon=True).start()
    info = system_info()
    port = a.port
    for _ in range(20):
        try:
            srv = ThreadingHTTPServer((a.host, port), H)
            break
        except OSError:
            port += 1
    else:
        sys.exit("не нашёл свободный порт")
    srv.daemon_threads = True
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{port}/"
    print("=" * 64)
    print(f"  RECOVER STUDIO {VERSION}  ->  {url}")
    print(f"  устройство: {info['label']}")
    print(f"  проекты:    {P.PROJECTS}")
    print("  Закройте это окно, чтобы выключить студию.")
    print("=" * 64, flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
