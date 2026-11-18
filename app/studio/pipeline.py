"""Recover pipeline: analyse -> repair -> enhance -> master (the LANDR-style chain for old recordings).

  1. read (any format) -> 44.1 kHz stereo float
  2. analyse damage (hum, clicks, crackle, noise, bandwidth, wow, clipping, azimuth) -> auto settings
  3. REPAIR   rumble HPF, de-hum, de-click, de-crackle, de-clip, de-wow, azimuth, AI denoise (Mel-Band RoFormer)
  4. ENHANCE  stems balance + vocal de-reverb (BS-RoFormer SW, anvuew), Apollo (codec / HF restore),
              bandwidth extension (exciter), tonal rebalance to a target curve or a reference track, stereo width
  5. MASTER   glue compression, tape warmth, true-peak limiter, LUFS target; trim + fades
  6. outputs: restored master, repair-only version, optional stems, previews, report, before/after metrics
A 30-second preview runs the same chain on an excerpt, so settings can be tried in seconds.
"""
import json
import os
import re
import shutil
import threading
import time
import traceback

import numpy as np

from . import analyze as AN
from . import audio_io as A
from . import device as D
from . import dsp
from . import models as M

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP)
PROJECTS = os.environ.get("RCS_PROJECTS") or os.path.join(ROOT, "projects")
SR = A.SR

STEM_TITLES = {"vocals": "Вокал", "drums": "Ударные", "bass": "Бас", "guitar": "Гитара", "piano": "Клавишные",
               "other": "Остальное"}

DEFAULT_OPTIONS = dict(
    source="auto",                # auto | vinyl | shellac | cassette | reel | lossy | demo
    device="auto",                # auto | gpu | cpu
    engine="auto",                # auto (AI on a GPU, DSP on a CPU) | ai | dsp
    preview=False, preview_start=None, preview_len=30.0,
    rumble_hz="auto",
    hum=dict(on="auto", depth=1.0),
    clicks=dict(on="auto", sensitivity=5.0),
    crackle=dict(on="auto", sensitivity=4.0),
    declip=dict(on="auto"),
    wow=dict(on="auto", strength=1.0, flutter=True),
    azimuth=dict(on="auto"),
    denoise=dict(on="auto", model="auto", amount=1.0, polish_db=0.0),
    stems=dict(on=False, vocals_db=0.0, drums_db=0.0, bass_db=0.0, guitar_db=0.0, piano_db=0.0, other_db=0.0,
               dereverb=False, keep=True),
    apollo=dict(on="auto"),
    bandwidth=dict(on="auto", amount=1.0),
    bass=dict(on="auto", amount=1.0, kick=True),
    tone=dict(on="auto", mode="target", style="balanced", amount=0.85, max_boost=15.0),
    guard=dict(on=True, extra_db=6.0),
    width=1.0,
    master=dict(on=True, intensity="medium", lufs=None, ceiling=-1.0, warmth=None),
    trim=dict(on="auto"),
    output=dict(sr=44100, bits=24, channels="auto"),
)
INTENSITY = {"low": dict(lufs=-16.0, ratio=1.3, thr=-4.0, warmth=0.5),
             "medium": dict(lufs=-14.0, ratio=1.6, thr=-6.0, warmth=1.5),
             "high": dict(lufs=-11.0, ratio=2.2, thr=-8.0, warmth=2.5)}


def merge_options(o):
    out = json.loads(json.dumps(DEFAULT_OPTIONS))
    for k, v in (o or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


# ================================================================ projects
def new_id(name):
    base = re.sub(r"[^\w\-]+", "_", os.path.splitext(name)[0], flags=re.U).strip("_")[:40] or "track"
    return f"{time.strftime('%Y%m%d-%H%M%S')}_{base}"


def project_dir(pid):
    if not re.fullmatch(r"[\w\-]+", pid, flags=re.U):
        raise ValueError("bad project id")
    return os.path.join(PROJECTS, pid)


def load_manifest(pid):
    with open(os.path.join(project_dir(pid), "project.json"), encoding="utf-8") as f:
        return json.load(f)


_MLOCK = threading.Lock()


def save_manifest(pid, man):
    p = os.path.join(project_dir(pid), "project.json")
    with _MLOCK:
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=1)
        os.replace(p + ".tmp", p)


def create_project(src_name, write_fn):
    """write_fn(dst_path) stores the uploaded/copied file. -> project id."""
    os.makedirs(PROJECTS, exist_ok=True)
    pid = new_id(src_name)
    while os.path.exists(project_dir(pid)):
        time.sleep(1)
        pid = new_id(src_name)
    d = project_dir(pid)
    os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(src_name)[1].lower() or ".wav"
    write_fn(os.path.join(d, "source" + ext))
    man = dict(id=pid, name=os.path.splitext(os.path.basename(src_name))[0], source="source" + ext,
               source_name=os.path.basename(src_name), created=time.strftime("%Y-%m-%d %H:%M:%S"),
               status="new", options=None, analysis=None, results={}, reference=None)
    save_manifest(pid, man)
    return pid


def set_reference(pid, name, write_fn):
    d = project_dir(pid)
    for f in os.listdir(d):
        if f.startswith("reference."):
            os.remove(os.path.join(d, f))
    ext = os.path.splitext(name)[1].lower() or ".wav"
    write_fn(os.path.join(d, "reference" + ext))
    man = load_manifest(pid)
    man["reference"] = dict(file="reference" + ext, name=os.path.basename(name))
    save_manifest(pid, man)


class Cancelled(Exception):
    pass


# ================================================================ job
class Job:
    """One pipeline run. Thread-safe state for the web UI: progress 0..1, step text, steps, log lines."""

    def __init__(self, pid, options):
        self.pid = pid
        self.options = merge_options(options)
        self.kind = "analyze" if self.options.get("analyze_only") else "preview" if self.options.get("preview") else "full"
        self.progress = 0.0
        self.step = "В очереди"
        self.steps = []
        self.log = []
        self.status = "queued"
        self.error = None
        self.started = self.finished = None
        self._cancel = False
        self._lock = threading.Lock()
        self._span = (0.0, 1.0)

    def say(self, msg):
        with self._lock:
            self.log.extend(str(msg).split("\n"))
            if len(self.log) > 5000:
                del self.log[:1000]
        print(msg, flush=True)

    def cancel(self):
        self._cancel = True

    def check(self):
        if self._cancel:
            raise Cancelled()

    def state(self, since=0):
        with self._lock:
            return dict(pid=self.pid, kind=self.kind, status=self.status, progress=round(self.progress, 4),
                        step=self.step, steps=self.steps, error=self.error, log=self.log[since:],
                        log_len=len(self.log), started=self.started, finished=self.finished)

    # step helpers ------------------------------------------------------------------------------
    def plan(self, items):
        """items: [(id, title, weight)]"""
        tot = sum(w for _, _, w in items) or 1
        acc = 0.0
        self.steps = []
        for sid, title, w in items:
            self.steps.append(dict(id=sid, title=title, state="wait", note="", p0=acc / tot, p1=(acc + w) / tot))
            acc += w

    def begin(self, sid):
        self.check()
        for s in self.steps:
            if s["id"] == sid:
                s["state"] = "run"
                self._span = (s["p0"], s["p1"])
                self.progress = s["p0"]
                self.step = s["title"]
                self._t0 = time.time()
                self.say(f"\n▶ {s['title']}")

    def sub(self, frac, text=None):
        a, b = self._span
        self.progress = a + (b - a) * min(1.0, max(0.0, frac))
        if text:
            cur = next((s["title"] for s in self.steps if s["state"] == "run"), "")
            self.step = f"{cur}: {text}"
        self.check()

    def done(self, sid, note="", state="done"):
        self.check()
        for s in self.steps:
            if s["id"] == sid:
                s["state"] = state
                s["note"] = note
                if state == "done":
                    s["note"] = (note + " " if note else "") + f"({time.time() - getattr(self, '_t0', time.time()):.1f} с)"
        self.progress = self._span[1]
        if note:
            self.say(f"  {note}")

    def skip(self, sid, note=""):
        for s in self.steps:
            if s["id"] == sid:
                s["state"] = "skip"
                s["note"] = note

    # run ---------------------------------------------------------------------------------------
    def run(self):
        self.status, self.started = "running", time.strftime("%Y-%m-%d %H:%M:%S")
        man = load_manifest(self.pid)
        man["prev_status"] = man.get("status")
        man["status"] = "running"
        save_manifest(self.pid, man)
        try:
            self._run()
            self.status = "done"
            self.progress, self.step = 1.0, "Готово"
        except Cancelled:
            self.status, self.step = "cancelled", "Отменено"
            self.say("\n! Отменено пользователем")
        except Exception as e:
            traceback.print_exc()
            self.status, self.step = "error", "Ошибка"
            self.error = f"{type(e).__name__}: {e}"
            self.say("\n! ОШИБКА: " + self.error)
            self.say(traceback.format_exc())
        finally:
            for mod in ("roformer", "apollo"):
                try:
                    __import__(f"studio.{mod}", fromlist=["unload"]).unload()
                except Exception:
                    pass
        self.finished = time.strftime("%Y-%m-%d %H:%M:%S")
        man = load_manifest(self.pid)
        if self.kind == "analyze":
            man["status"] = man.get("status") if man.get("status") in ("done", "preview") else (
                "analyzed" if self.status == "done" else self.status)
        else:
            man["status"] = "done" if (man.get("results") or {}).get("full") else (
                "preview" if (man.get("results") or {}).get("preview") else self.status)
        if self.status == "error":
            man["error"] = self.error
        man["last_job"] = dict(kind=self.kind, status=self.status, error=self.error, finished=self.finished,
                               log_tail=self.log[-300:], steps=self.steps)
        save_manifest(self.pid, man)

    def _ai_ok(self, dinfo):
        eng = self.options.get("engine", "auto")
        if eng == "dsp":
            return False
        if eng == "ai":
            return True
        return dinfo["kind"] != "cpu"

    def _run(self):
        o = self.options
        pid = self.pid
        pdir = project_dir(pid)
        man = load_manifest(pid)
        want = {"gpu": "auto"}.get(o.get("device"), o.get("device") or "auto")
        dinfo = D.detect(want)
        ai = self._ai_ok(dinfo)
        preview = bool(o.get("preview"))

        if o.get("analyze_only"):
            self.kind = "analyze"
            self.plan([("load", "Чтение трека", 1), ("analyze", "Анализ дефектов", 4)])
            self.begin("load")
            x, src_sr, ch = A.load_stereo(os.path.join(pdir, man["source"]), SR)
            self.done("load", f"{len(x) / SR:.1f} с, {src_sr} Гц, {'моно' if ch == 1 else 'стерео'}")
            self.begin("analyze")
            rep = AN.analyze(x, SR, os.path.splitext(man["source"])[1].lower(), ch, src_sr, log=self.say)
            rep["version"] = 2
            man = load_manifest(pid)
            man["analysis"], man["duration"] = rep, round(len(x) / SR, 2)
            save_manifest(pid, man)
            op = os.path.join(pdir, "original.mp3")
            if not os.path.exists(op) and not os.path.exists(os.path.splitext(op)[0] + ".ogg"):
                A.write_mp3(op, x)
            for f in rep["findings"]:
                self.say("  • " + f["text"])
            self.done("analyze", AN.SOURCES.get(rep["source_guess"], {}).get("title", ""))
            return
        self.plan([("load", "Чтение трека", 2), ("analyze", "Анализ дефектов", 4),
                   ("hum", "Рокот и гул сети", 2), ("clicks", "Щелчки", 5), ("crackle", "Треск", 5),
                   ("declip", "Клиппинг", 2), ("wow", "Детонация (wow / flutter)", 4), ("azimuth", "Азимут ленты", 1),
                   ("denoise", "Шумоподавление", 14 if ai else 4), ("stems", "Стэмы и баланс партий", 18),
                   ("apollo", "Apollo: восстановление верхов", 14), ("bandwidth", "Расширение полосы (верх и низ)", 3),
                   ("tone", "Тональный баланс", 2), ("guard", "Контроль шума после улучшения", 3),
                   ("width", "Стерео", 1), ("master", "Мастеринг", 4),
                   ("save", "Сохранение", 4)])
        self.say(f"Recover Studio · {man['name']} · {'ПРЕВЬЮ' if preview else 'весь трек'}")
        self.say(f"Устройство: {D.label(dinfo)} · нейросети: {'да' if ai else 'нет (DSP-режим)'}")

        # 1. load ---------------------------------------------------------------------------------
        self.begin("load")
        src = os.path.join(pdir, man["source"])
        x, src_sr, ch = A.load_stereo(src, SR)
        full_dur = len(x) / SR
        self.done("load", f"{full_dur:.1f} с, {src_sr} Гц, {'моно' if ch == 1 else 'стерео'}")

        # 2. analyse (full track, cached in the manifest) -------------------------------------------
        self.begin("analyze")
        rep = man.get("analysis")
        if not rep or rep.get("version") != 2:
            rep = AN.analyze(x, SR, os.path.splitext(man["source"])[1].lower(), ch, src_sr, log=self.say)
            rep["version"] = 2
            man = load_manifest(pid)
            man["analysis"] = rep
            man["duration"] = round(full_dur, 2)
            save_manifest(pid, man)
        for f in rep["findings"]:
            self.say("  • " + f["text"])
        o, why = AN.auto_options(rep, o, self.say)
        if why:
            self.say("  авто: " + "; ".join(why))
        self.done("analyze", f"носитель: {AN.SOURCES.get(o['source_resolved'], {}).get('title', o['source_resolved'])}")

        # preview excerpt
        off = 0.0
        if preview:
            L = float(o.get("preview_len") or 30.0)
            if o.get("preview_start") not in (None, ""):
                off = max(0.0, min(float(o["preview_start"]), full_dur - L))
            else:   # the loudest stretch, but not the very start
                lv = dsp.frame_rms_db(x.mean(1), SR, 0.5, win=L)
                lo = int(min(10, full_dur * 0.15) / 0.5)
                cand = lv[lo: max(lo + 1, len(lv) - int(L / 0.5))]
                off = (lo + int(np.argmax(cand))) * 0.5 - L / 2 if len(cand) else 0.0
                off = max(0.0, min(off, full_dur - L))
            pad = 2.0      # context for the filters, removed at the end
            a = max(0, int((off - pad) * SR))
            b = min(len(x), int((off + L + pad) * SR))
            x = x[a:b]
            cut_a, cut_b = int((off - a / SR) * SR), int((off - a / SR + L) * SR)
            ocut = (cut_a, cut_b)
            self.say(f"  превью: {off:.1f}–{off + L:.1f} с")
        orig = x.copy()

        # 3. repair ---------------------------------------------------------------------------------
        self.begin("hum")
        notes0 = []
        if rep["mono"] and ch == 2:
            x = dsp.mono_sum(x)                      # pro practice: a mono disc / tape is summed to mono first
            notes0.append("моно-запись сведена в моно (шум −3 дБ)")
        x = dsp.highpass(x, SR, float(o.get("rumble_hz") or 20), order=4)
        if o["hum"]["on"] and rep["hum"].get("f0"):
            x, nh = dsp.dehum(x, SR, rep["hum"], float(o["hum"].get("depth", 1.0)))
            self.done("hum", "; ".join(notes0 + [f"рокот < {o['rumble_hz']} Гц убран",
                                                 f"{nh} режекторов на {rep['hum']['freq']:.2f} Гц и гармониках"]))
        else:
            self.done("hum", "; ".join(notes0 + [f"рокот < {o['rumble_hz']} Гц убран"]))

        self.begin("clicks")
        if o["clicks"]["on"]:
            x, n = dsp.declick(x, SR, float(o["clicks"].get("sensitivity", 5)), progress=lambda f: self.sub(f))
            self.done("clicks", f"исправлено щелчков: {n}")
        else:
            self.skip("clicks", "выкл.")

        self.begin("crackle")
        if o["crackle"]["on"]:
            x, n = dsp.declick(x, SR, float(o["crackle"].get("sensitivity", 4)), crackle=True,
                               progress=lambda f: self.sub(f))
            self.done("crackle", f"исправлено: {n}")
        else:
            self.skip("crackle", "выкл.")

        self.begin("declip")
        if o["declip"]["on"]:
            x, n = dsp.declip(x, SR)
            self.done("declip", f"восстановлено пиков: {n}")
        else:
            self.skip("declip", "выкл.")

        self.begin("wow")
        if o["wow"]["on"]:
            x, info = dsp.dewow(x, SR, float(o["wow"].get("strength", 1.0)), bool(o["wow"].get("flutter", True)),
                                progress=lambda f: self.sub(f))
            if info.get("applied"):
                self.done("wow", f"скорость выровнена: ±{info['rms_cents']} цента (макс. {info['max_cents']})")
            else:
                self.done("wow", "нечего исправлять")
            if preview:   # time map may change the length a little: keep the cut points proportional
                k = len(x) / len(orig)
                cut_a, cut_b = int(cut_a * k), int(cut_b * k)
        else:
            self.skip("wow", "выкл.")

        self.begin("azimuth")
        if o["azimuth"]["on"] and not rep["mono"]:
            x = dsp.fix_azimuth(x, rep["azimuth_lag"])
            self.done("azimuth", f"правый канал сдвинут на {-rep['azimuth_lag']:+.2f} семпла")
        else:
            self.skip("azimuth", "не требуется")

        self.begin("denoise")
        if o["denoise"]["on"]:
            amt = float(o["denoise"].get("amount", 1.0))
            side_note = ""
            if o["source_resolved"] in ("vinyl", "shellac") and not rep["mono"] and o["denoise"].get("side", True):
                x = dsp.side_denoise(x, SR, 9.0 * amt)
                side_note = " + шум поверхности в side-канале (−9 дБ)"
            if ai:
                from . import roformer
                key = o["denoise"].get("model") or "denoise"
                if key not in ("denoise", "denoise_aggr"):
                    key = "denoise"
                M.ensure(key, lambda d, t, fn: self.sub(0.0, f"загрузка модели {d / 2**20:.0f}/{t / 2**20:.0f} МБ"), self.say)
                self.sub(0.02, "загрузка нейросети")
                st, info = roformer.run(x.T.copy(), key, want, overlap=2 if not preview else 2,
                                        progress=lambda f, t: self.sub(f, t), log=self.say)
                clean = st["dry"].T
                x = np.nan_to_num(clean * amt + x * (1 - amt)).astype(np.float32)
                note = f"{M.MODELS_INFO[key]['title'].split('·')[-1].strip()}, {info['seconds']} с{side_note}"
            else:
                prof = None
                if rep["lead_quiet_s"] > 1.0 or rep["tail_quiet_s"] > 1.0:
                    note = "спектральный (профиль из тихих участков)"
                else:
                    note = "спектральный"
                x = dsp.denoise(x, SR, 10 + 8 * amt, 1.0, prof)
                note += side_note
            pol = float(o["denoise"].get("polish_db") or 0)
            if pol > 0:
                x = dsp.denoise(x, SR, pol, 0.8)
                note += f" + доочистка {pol:.0f} дБ"
            self.done("denoise", note)
        else:
            self.skip("denoise", "выкл.")
        repaired = x.copy()
        floor0 = dsp.noise_floor(x, SR)        # noise reference for the hiss guard

        # 4. enhance --------------------------------------------------------------------------------
        self.begin("stems")
        stems_out = None
        so = o["stems"]
        lowend = bool(o["bass"]["on"]) and ai          # the low end is rebuilt from the bass / drum stems
        if (_on_(so.get("on")) or lowend) and (ai or o.get("engine") == "ai" or so.get("force_cpu")):
            from . import roformer
            M.ensure("stems", lambda d, t, fn: self.sub(0.0, f"загрузка модели {d / 2**20:.0f}/{t / 2**20:.0f} МБ"), self.say)
            st, info = roformer.run(x.T.copy(), "stems", want, overlap=2,
                                    progress=lambda f, t: self.sub(f * (0.75 if so.get("dereverb") else 1), t), log=self.say)
            names = [k for k in st if not k.endswith("_rest")]
            total = sum(st[k] for k in names)
            resid = x.T - total
            if so.get("dereverb") and "vocals" in st:
                M.ensure("dereverb", None, self.say)
                dv, _ = roformer.run(st["vocals"].copy(), "dereverb", want, overlap=2,
                                     progress=lambda f, t: self.sub(0.75 + 0.25 * f, "вокал без комнаты: " + t), log=self.say)
                st["vocals"] = dv["noreverb"]
            lowl = []
            if lowend:
                amt = float(o["bass"].get("amount", 1.0))
                if "bass" in st:
                    sb = dsp.octave_down(st["bass"].mean(0), SR) * 0.9 * amt
                    st["bass"] = st["bass"] + sb[None, :]
                    lowl.append("суббас из стэма баса")
                if "drums" in st and o["bass"].get("kick", True):
                    kk, nk = dsp.kick_sub(st["drums"].T, SR)
                    st["drums"] = st["drums"] + kk[None, :] * amt
                    lowl.append(f"саб бочки на {nk} ударах")
                o["bass"]["done"] = True
            mix = resid.copy()
            stems_out = {}
            gains = []
            for k in names:
                g = float(so.get(f"{k}_db", 0.0) or 0.0)
                mix += st[k] * dsp.undb(g)
                stems_out[k] = st[k].T
                if abs(g) > 0.05:
                    gains.append(f"{STEM_TITLES.get(k, k)} {g:+.1f} дБ")
            x = np.nan_to_num(mix.T).astype(np.float32)
            self.done("stems", "; ".join([", ".join(gains) or "баланс без изменений"] + lowl
                                         + (["вокал без реверберации"] if so.get("dereverb") else [])))
            if not _on_(so.get("on")):
                stems_out = None        # stems were only a tool for the low end
        elif _on_(so.get("on")):
            self.skip("stems", "нужна видеокарта (или «Нейросети: всегда»)")
        else:
            self.skip("stems", "выкл.")

        self.begin("apollo")
        if o["apollo"]["on"] and ai:
            from . import apollo
            M.ensure("apollo", None, self.say)
            y, info = apollo.restore(x.T.copy(), want, progress=lambda f, t: self.sub(f, t), log=self.say)
            x = np.nan_to_num(y.T).astype(np.float32)
            self.done("apollo", f"{info['seconds']} с")
        elif o["apollo"]["on"]:
            self.skip("apollo", "только с видеокартой")
        else:
            self.skip("apollo", "выкл.")

        self.begin("bandwidth")
        bw = o["bandwidth"]["on"]
        fc = dsp.detect_cutoff(x, SR) if bw else 0
        if bw == "auto":
            bw = fc < 15500                  # music bandwidth measured after the hiss is gone
        if bw:
            x, gdb = dsp.excite(x, SR, fc, float(o["bandwidth"].get("amount", 1.0)))
            notes = [f"верха выше {fc / 1000:.1f} кГц дорисованы" if gdb is not None else "верхи на месте"]
        else:
            notes = []
        if o["bass"]["on"] and not o["bass"].get("done"):
            x, sdb = dsp.bass_restore(x, SR, float(o["bass"].get("amount", 1.0)))
            notes.append("суббас восстановлен (октава вниз от 70-180 Гц)")
        if notes:
            self.done("bandwidth", "; ".join(notes))
        else:
            self.skip("bandwidth", "выкл.")

        self.begin("tone")
        to = o["tone"]
        if to["on"] and to.get("mode") != "off":
            target = None
            if to.get("mode") == "reference" and man.get("reference"):
                rx, _, _ = A.load_stereo(os.path.join(pdir, man["reference"]["file"]), SR)
                rc, _ = dsp.band_curve(rx, SR)
                target = dsp.norm_curve(rc)
                label = f"по референсу «{man['reference']['name']}»"
            else:
                target = dsp.TARGET_BALANCED.copy()
                label = "по эталонной кривой"
            target = target + dsp.style_tilt(dsp.GRID, to.get("style", "balanced"))
            ref_mode = label.startswith("по референсу")
            x, g = dsp.match_eq(x, SR, target, float(to.get("amount", 0.85)),
                                max_boost=float(to.get("max_boost", 15)) + (5 if ref_mode else 0))
            if o["hum"]["on"] and rep["hum"].get("f0") and g[:12].max() > 6:   # the low boost lifts leftover hum
                x, _ = dsp.dehum(x, SR, rep["hum"], float(o["hum"].get("depth", 1.0)))
            big = sorted(zip(dsp.GRID, g), key=lambda t: -abs(t[1]))[:3]
            self.done("tone", label + " · " + ", ".join(f"{f:.0f} Гц {v:+.0f} дБ" for f, v in sorted(big)))
        else:
            self.skip("tone", "выкл.")

        self.begin("guard")
        if o.get("guard", {}).get("on", True) is not False:
            x, red = dsp.hiss_guard(x, SR, floor0, float(o.get("guard", {}).get("extra_db", 6.0)))
            self.done("guard", f"пол шума возвращён к уровню после ремонта −{o.get('guard', {}).get('extra_db', 6.0):.0f} дБ "
                               f"(минимальная статистика), средняя коррекция верхов −{red} дБ")
        else:
            self.skip("guard", "выкл.")

        self.begin("width")
        w = float(o.get("width") or 1.0)
        mono_out = o["output"].get("channels") == "mono" or (o["output"].get("channels") == "auto" and ch == 1)
        if not rep["mono"] and abs(w - 1) > 0.01 and not mono_out:
            x = dsp.width(x, w)
            self.done("width", f"ширина {w * 100:.0f}%")
        else:
            self.skip("width", "моно" if rep["mono"] or mono_out else "без изменений")

        # 5. master ---------------------------------------------------------------------------------
        self.begin("master")
        mo = o["master"]
        if mono_out:
            x = np.repeat(x.mean(1, keepdims=True), 2, 1)
        if mo["on"]:
            it = INTENSITY.get(mo.get("intensity", "medium"), INTENSITY["medium"])
            x, gr = dsp.compress(x, SR, it["ratio"], it["thr"], 0.03, 0.3)
            warm = mo.get("warmth")
            warm = it["warmth"] if warm in (None, "") else float(warm)
            if o["tone"].get("style") == "warm":
                warm += 1.0
            if warm > 0:
                x = dsp.saturate(x, SR, warm, 0.5)
            lufs = float(mo.get("lufs") or it["lufs"])
            x = dsp.finalize(x, SR, lufs, float(mo.get("ceiling", -1.0)), A.loudness, A.true_peak_db)
            self.done("master", f"компрессия до {gr:.1f} дБ, {lufs:.0f} LUFS, потолок {mo.get('ceiling', -1.0)} dBTP")
        else:
            x = dsp.limiter(x * dsp.undb(-1 - A.true_peak_db(x, SR)) if A.true_peak_db(x, SR) > -1 else x, SR, -1.0)
            self.done("master", "только защита от перегруза (-1 dBTP)")

        # 6. save -----------------------------------------------------------------------------------
        self.begin("save")
        kind = "preview" if preview else "full"
        od = os.path.join(pdir, kind)
        shutil.rmtree(od, ignore_errors=True)
        os.makedirs(od, exist_ok=True)
        rep_out = repaired
        if preview:
            x, rep_out, orig = x[cut_a:cut_b].copy(), repaired[cut_a:cut_b].copy(), orig[ocut[0]:ocut[1]].copy()
            if stems_out:
                stems_out = {k: v[cut_a:cut_b] for k, v in stems_out.items()}
            fade = int(0.05 * SR)
            for arr in (x, rep_out, orig):
                arr[:fade] *= np.linspace(0, 1, fade)[:, None]
                arr[-fade:] *= np.linspace(1, 0, fade)[:, None]
        cut = (0.0, 0.0)
        if not preview and o["trim"]["on"] is not False:
            ea, eb = dsp.find_edges(rep_out, SR)
            if ea > 0.3 * SR or len(rep_out) - eb > 0.3 * SR:
                x, c0, c1 = dsp.apply_edges(x, SR, ea, eb)
                a_, b_ = int(round(c0 * SR)), len(rep_out) - int(round(c1 * SR))
                rep_out, _, _ = dsp.apply_edges(rep_out, SR, ea, eb)
                cut = (c0, c1)
                if stems_out:
                    stems_out = {k: v[a_:b_] for k, v in stems_out.items()}
                if c0 + c1 > 0.2:
                    self.say(f"  обрезано: {c0:.1f} с в начале, {c1:.1f} с в конце (тишина / шум без музыки)")
        # repair-only version: just peak-normalised to -1 dBFS
        rep_out = (rep_out * dsp.undb(-1.0) / (np.abs(rep_out).max() + 1e-9)).astype(np.float32)
        A.write_wav(os.path.join(od, "restored.wav"), x, SR, "32")
        A.write_wav(os.path.join(od, "repaired.wav"), rep_out, SR, "32")
        prev = dict(restored=os.path.relpath(A.write_mp3(os.path.join(od, "restored.mp3"), x), pdir).replace("\\", "/"),
                    repaired=os.path.relpath(A.write_mp3(os.path.join(od, "repaired.mp3"), rep_out), pdir).replace("\\", "/"))
        if preview:
            prev["original"] = os.path.relpath(A.write_mp3(os.path.join(od, "original.mp3"), orig), pdir).replace("\\", "/")
        else:
            op = os.path.join(pdir, "original.mp3")
            if not os.path.exists(op) and not os.path.exists(os.path.splitext(op)[0] + ".ogg"):
                op = A.write_mp3(op, orig)
            elif not os.path.exists(op):
                op = os.path.splitext(op)[0] + ".ogg"
            prev["original"] = os.path.relpath(op, pdir).replace("\\", "/")
        stems_files = []
        if stems_out and so.get("keep", True):
            for k, v in stems_out.items():
                fn = f"stems/{k}.wav"
                A.write_wav(os.path.join(od, fn), v.astype(np.float32), SR, "24")
                stems_files.append(dict(key=k, title=STEM_TITLES.get(k, k), file=f"{kind}/{fn}"))
        self.sub(0.5, "замеры до / после")
        before = AN.measure(orig, SR)
        after = AN.measure(x, SR)
        res = dict(kind=kind, created=time.strftime("%Y-%m-%d %H:%M:%S"), offset=round(off, 2),
                   duration=round(len(x) / SR, 2), files=dict(restored=f"{kind}/restored.wav", repaired=f"{kind}/repaired.wav"),
                   previews=prev, stems=stems_files, before=before, after=after, trimmed=[round(c, 2) for c in cut],
                   options=o, steps=[{k: v for k, v in s.items() if k not in ("p0", "p1")} for s in self.steps],
                   device=D.label(dinfo), ai=ai, mono_out=bool(mono_out),
                   repaired_lufs=_r1(A.loudness(rep_out, SR)))
        with open(os.path.join(od, "report.txt"), "w", encoding="utf-8-sig") as f:
            f.write(report_text(man, rep, res))
        man = load_manifest(pid)
        man.setdefault("results", {})[kind] = res
        man["options"] = self.options
        save_manifest(pid, man)
        self.done("save", f"{len(x) / SR:.1f} с")
        self.say(f"\nГотово: тональная ошибка {before['tone_error_db']} → {after['tone_error_db']} дБ, "
                 f"сигнал/шум {before['snr_db']} → {after['snr_db']} дБ, щелчков/мин {before['clicks_per_min']} → "
                 f"{after['clicks_per_min']}, {after['lufs']} LUFS")


def _r1(v):
    return round(float(v), 1) if np.isfinite(v) else None


def _on_(v):
    return v is True or v == "on"


def report_text(man, rep, res):
    L = [f"RECOVER STUDIO - {man['name']}", "=" * 64, f"Исходник: {man['source_name']}  ({rep['duration']:.1f} с, "
         f"{rep['source_sr']} Гц, {'моно' if rep['mono'] else 'стерео'})",
         f"Обработка: {'ПРЕВЬЮ ' + str(res['offset']) + ' с' if res['kind'] == 'preview' else 'весь трек'}, {res['device']}",
         "", "НАЙДЕННЫЕ ДЕФЕКТЫ:"]
    L += [f"  - {f['text']}" for f in rep["findings"]]
    L += ["", "ЧТО СДЕЛАНО:"]
    for s in res["steps"]:
        mark = {"done": "+", "skip": "-", "error": "!"}.get(s["state"], " ")
        L.append(f"  [{mark}] {s['title']}: {s.get('note', '')}")
    b, a = res["before"], res["after"]
    L += ["", "ДО / ПОСЛЕ:",
          f"  Громкость            {b['lufs']} -> {a['lufs']} LUFS",
          f"  True peak            {b['true_peak']} -> {a['true_peak']} dBTP",
          f"  Сигнал/шум           {b['snr_db']} -> {a['snr_db']} дБ",
          f"  Щелчков в минуту     {b['clicks_per_min']} -> {a['clicks_per_min']}",
          f"  Гул (индекс)         {b['hum_score']} -> {a['hum_score']}",
          f"  Полоса               {b['cutoff_hz']} -> {a['cutoff_hz']} Гц",
          f"  Отклонение тона      {b['tone_error_db']} -> {a['tone_error_db']} дБ (от эталонной кривой)"]
    if res.get("trimmed") and any(res["trimmed"]):
        L.append(f"  Обрезано: {res['trimmed'][0]} с в начале, {res['trimmed'][1]} с в конце")
    return "\n".join(L) + "\n"
