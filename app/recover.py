"""Recover Studio from the command line (no browser).

    python app/recover.py "old tape.wav" [--source auto|vinyl|shellac|cassette|reel|lossy|demo]
        [--preview] [--style warm|balanced|open] [--intensity low|medium|high] [--lufs -14]
        [--reference ref.wav] [--stems] [--vocals 2] [--dereverb] [--apollo on|off|auto]
        [--engine auto|ai|dsp] [--device auto|cuda|cpu] [--sr 44100|48000|96000] [--bits 16|24|32] [--out folder]

The restored WAV (and the repair-only version + report) is written to projects/<date>_<name>/full/
and, with --out, also copied there in the requested format.
"""
import argparse
import os
import shutil
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from studio import audio_io as A  # noqa: E402
from studio import pipeline as P  # noqa: E402


def export(pid, rel, sr, bits, dst):
    x, sr0 = A.read_any(os.path.join(P.project_dir(pid), rel))
    A.write_wav(dst, A.resample(x, sr0, sr), sr, bits)
    return dst


def main():
    ap = argparse.ArgumentParser(description="Recover Studio: restore an old recording")
    ap.add_argument("input")
    ap.add_argument("--source", default="auto")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--start", type=float, default=None, help="preview start, seconds")
    ap.add_argument("--style", default="balanced", choices=["warm", "balanced", "open"])
    ap.add_argument("--intensity", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--lufs", type=float, default=None)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--stems", action="store_true")
    ap.add_argument("--vocals", type=float, default=0.0, help="vocal level change, dB (needs --stems)")
    ap.add_argument("--dereverb", action="store_true")
    ap.add_argument("--apollo", default="auto", choices=["on", "off", "auto"])
    ap.add_argument("--denoise", default="auto", choices=["on", "off", "auto"])
    ap.add_argument("--no-master", action="store_true")
    ap.add_argument("--engine", default="auto", choices=["auto", "ai", "dsp"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--bits", default="24", choices=["16", "24", "32"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not os.path.isfile(a.input):
        sys.exit(f"нет файла: {a.input}")
    pid = P.create_project(os.path.basename(a.input), lambda dst: shutil.copyfile(a.input, dst))
    tone = dict(style=a.style)
    if a.reference:
        P.set_reference(pid, os.path.basename(a.reference), lambda dst: shutil.copyfile(a.reference, dst))
        tone["mode"] = "reference"
    onoff = {"on": True, "off": False, "auto": "auto"}
    opts = dict(source=a.source, preview=a.preview, preview_start=a.start, engine=a.engine, device=a.device,
                tone=tone, apollo=dict(on=onoff[a.apollo]), denoise=dict(on=onoff[a.denoise]),
                stems=dict(on=a.stems, vocals_db=a.vocals, dereverb=a.dereverb),
                master=dict(on=not a.no_master, intensity=a.intensity, lufs=a.lufs))
    job = P.Job(pid, opts)
    job.run()
    if job.status != "done":
        sys.exit(1)
    man = P.load_manifest(pid)
    res = man["results"]["preview" if a.preview else "full"]
    print("\nПроект:", P.project_dir(pid))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        base = os.path.join(a.out, man["name"] + (" (preview)" if a.preview else "") + " [restored].wav")
        print("Сохранено:", export(pid, res["files"]["restored"], a.sr, a.bits, base))
    else:
        print("Результат:", os.path.join(P.project_dir(pid), res["files"]["restored"]))


if __name__ == "__main__":
    main()
