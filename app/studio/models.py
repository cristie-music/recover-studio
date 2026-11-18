"""Model registry + downloader. All weights live in <studio>/models (override: env RCS_MODELS).

    python app/studio/models.py                 download every model
    python app/studio/models.py denoise apollo  download only these
"""
import os
import shutil
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.environ.get("RCS_MODELS") or os.path.join(ROOT, "models")

HF = "https://huggingface.co"
SEP_RELEASE = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs"

# arch: melband | bsroformer | apollo
MODELS_INFO = {
    "denoise": dict(
        title="Шумоподавление · Mel-Band RoFormer Denoise (aufr33)",
        about="Нейросеть, обученная убирать шум из МУЗЫКИ (шипение плёнки, шум винила, фон). "
              "Сохраняет инструменты и вокал лучше спектральных шумодавов.",
        arch="melband", stems_keep="dry",
        files={"denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt":
               [f"{SEP_RELEASE}/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"],
               "denoise_mel_band_roformer_aufr33_sdr_27.9959_config.yaml":
               [f"{SEP_RELEASE}/denoise_mel_band_roformer_aufr33_sdr_27.9959_config.yaml"]},
        ckpt="denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        config="denoise_mel_band_roformer_aufr33_sdr_27.9959_config.yaml", size_mb=870, license="GPL-3.0 (UVR community)"),
    "denoise_aggr": dict(
        title="Шумоподавление (агрессивное) · Mel-Band RoFormer Denoise Aggr (aufr33)",
        about="Та же нейросеть, обученная давить шум сильнее. Для очень шумных кассет и заезженного винила.",
        arch="melband", stems_keep="dry",
        files={"denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt":
               [f"{SEP_RELEASE}/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt"],
               "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768_config.yaml":
               [f"{SEP_RELEASE}/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768_config.yaml"]},
        ckpt="denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
        config="denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768_config.yaml", size_mb=870,
        license="GPL-3.0 (UVR community)"),
    "apollo": dict(
        title="Восстановление верхов · Apollo (Tsinghua / Tencent AI Lab)",
        about="Генеративная модель восстановления музыки после сжатия (MP3, плохие оцифровки): "
              "дорисовывает срезанные верха и убирает «бульканье» кодека.",
        arch="apollo",
        files={"apollo_JusperLee.bin": [f"{HF}/JusperLee/Apollo/resolve/main/pytorch_model.bin"]},
        ckpt="apollo_JusperLee.bin", size_mb=66, license="CC BY-SA 4.0"),
    "stems": dict(
        title="Стэмы · BS-RoFormer SW 6 stems",
        about="Разделение на вокал, ударные, бас, гитару, клавиши и остальное: баланс партий, "
              "вытягивание вокала, отдельная обработка каждой партии.",
        arch="bsroformer",
        files={"BS-Rofo-SW-Fixed.ckpt": [f"{HF}/enerjazzer/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt",
                                         f"{HF}/ChanTrail/BS-RoFormer/resolve/main/BS-Rofo-SW-Fixed.ckpt"],
               "BS-Rofo-SW-Fixed.yaml": [f"{HF}/ChanTrail/BS-RoFormer/resolve/main/BS-Rofo-SW-Fixed.yaml",
                                         f"{HF}/enerjazzer/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.yaml"]},
        ckpt="BS-Rofo-SW-Fixed.ckpt", config="BS-Rofo-SW-Fixed.yaml", size_mb=700, license="see model card"),
    "dereverb": dict(
        title="Комната · Mel-Band RoFormer De-Reverb (anvuew)",
        about="Убирает гулкую комнату с вокала домашних записей и концертов (применяется к стэму вокала).",
        arch="melband", stems_keep="noreverb",
        files={"dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt":
               [f"{HF}/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"],
               "dereverb_mel_band_roformer_anvuew.yaml":
               [f"{HF}/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml"]},
        ckpt="dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", config="dereverb_mel_band_roformer_anvuew.yaml",
        size_mb=870, license="GPL-3.0"),
}
# downloaded by the installer; the rest on first use (or from the UI)
DEFAULT_SET = ["denoise", "apollo", "stems"]


def path(name):
    return os.path.join(MODELS, name)


def installed(key):
    return all(os.path.exists(path(f)) for f in MODELS_INFO[key]["files"])


def status():
    return {k: dict(title=v["title"], about=v["about"], installed=installed(k), size_mb=v.get("size_mb"))
            for k, v in MODELS_INFO.items()}


def download(key, progress=None, log=print):
    """Fetches missing files of a model (tries mirrors in order). progress(done_bytes, total_bytes, filename)."""
    os.makedirs(MODELS, exist_ok=True)
    for fn, urls in MODELS_INFO[key]["files"].items():
        dst = path(fn)
        if os.path.exists(dst):
            continue
        err = None
        for url in urls:
            try:
                log(f"  загрузка {fn} <- {url.split('/')[2]}")
                req = urllib.request.Request(url, headers={"User-Agent": "recover-studio/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(dst + ".part", "wb") as f:
                    total = int(r.headers.get("Content-Length") or 0)
                    done = 0
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        done += len(b)
                        if progress:
                            progress(done, total, fn)
                    if total and done != total:
                        raise IOError(f"оборвалось: {done}/{total} байт")
                shutil.move(dst + ".part", dst)
                err = None
                break
            except Exception as e:  # next mirror
                err = e
                try:
                    os.remove(dst + ".part")
                except OSError:
                    pass
                time.sleep(1)
        if err:
            raise RuntimeError(f"не удалось скачать {fn}: {err}")


def ensure(key, progress=None, log=print):
    if not installed(key):
        download(key, progress, log)


if __name__ == "__main__":
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    keys = sys.argv[1:] or list(MODELS_INFO)
    for k in keys:
        print(f"[{k}] {MODELS_INFO[k]['title']}")

        def p(d, t, fn):
            print(f"\r    {fn}: {d / 2**20:.0f}/{t / 2**20:.0f} MB", end="", flush=True)
        ensure(k, p)
        print("\n    OK")
