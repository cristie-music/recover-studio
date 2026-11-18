"""Generates colab/RecoverStudio_Colab.ipynb (kept as code here so the notebook stays easy to edit).

    python tools/make_colab.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_URL = "https://github.com/USER/recover-studio"

MD_INTRO = """# 📀 Recover Studio — Google Colab

**Восстановление старых записей** с кассет, пластинок, бобин и домашних демо: гул, щелчки, треск, шум, детонация, срезанные верха и низ — и мастеринг как в LANDR. Результат скачивается в WAV.

Тот же веб-интерфейс, что и на компьютере: открывается через прокси Colab или по публичной ссылке **ngrok** (шаг 4, нужен бесплатный токен с dashboard.ngrok.com).

**Перед запуском:** `Среда выполнения → Сменить среду выполнения → T4 GPU`. Без GPU студия тоже работает (быстрые DSP-алгоритмы вместо нейросетей).

Шаги: **1** → **2** → **3** → **4** (по порядку, каждую ячейку кнопкой ▶). Шаг 5 — восстановление без интерфейса."""

C_GPU = """#@title 1. Проверка GPU
import subprocess
r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                   capture_output=True, text=True)
print("GPU:", r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else
      "НЕТ — будет CPU (DSP-режим). Для нейросетей: Среда выполнения → Сменить среду → T4 GPU")"""

C_CODE = """#@title 2. Код студии и Google Drive
#@markdown **Откуда взять код**: `git` — ваш репозиторий (REPO_URL); `upload` / `drive` / `url` — архив `recover-studio-app.zip` (собирается `python tools/make_dist.py`):
CODE_SOURCE = "git"  #@param ["git", "upload", "drive", "url"]
REPO_URL = "%REPO_URL%"  #@param {type:"string"}
DRIVE_ZIP = "/content/drive/MyDrive/RecoverStudio/recover-studio-app.zip"  #@param {type:"string"}
CODE_URL = ""  #@param {type:"string"}
#@markdown **Хранить проекты и нейросети на Google Drive** (не пропадут после отключения Colab, ~1.6 ГБ моделей не качаются заново):
USE_DRIVE_STORAGE = True  #@param {type:"boolean"}
import os, shutil, zipfile, urllib.request, subprocess
RC = "/content/recover-studio"
if USE_DRIVE_STORAGE or CODE_SOURCE == "drive":
    from google.colab import drive
    drive.mount("/content/drive")
zip_path = "/content/recover-studio-app.zip"
if CODE_SOURCE == "git":
    shutil.rmtree(RC, ignore_errors=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, RC], check=True)
elif CODE_SOURCE == "upload":
    from google.colab import files
    print("Выберите recover-studio-app.zip ...")
    up = files.upload()
    name = next(iter(up))
    if name != os.path.basename(zip_path):
        shutil.move(name, zip_path)
elif CODE_SOURCE == "drive":
    shutil.copy(DRIVE_ZIP, zip_path)
else:
    urllib.request.urlretrieve(CODE_URL, zip_path)
if CODE_SOURCE != "git":
    shutil.rmtree(RC, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RC)
STORE = "/content/drive/MyDrive/RecoverStudio" if USE_DRIVE_STORAGE else RC
os.makedirs(f"{STORE}/projects", exist_ok=True)
os.makedirs(f"{STORE}/models", exist_ok=True)
os.environ["RCS_PROJECTS"] = f"{STORE}/projects"
os.environ["RCS_MODELS"] = f"{STORE}/models"
print("Код:", RC, "| проекты:", os.environ["RCS_PROJECTS"], "| нейросети:", os.environ["RCS_MODELS"])"""

C_INSTALL = """#@title 3. Установка (≈3–5 минут, один раз за сессию)
#@markdown Отдельное окружение Python 3.11 (не трогает пакеты Colab), PyTorch под GPU (или CPU), нейросети.
#@markdown Дополнительно скачать агрессивный шумодав и de-reverb для вокала (+1.7 ГБ):
EXTRA_MODELS = False  #@param {type:"boolean"}
import os, subprocess
RC = "/content/recover-studio"
VENV = "/content/rc-venv"
PY = f"{VENV}/bin/python"
def sh(cmd):
    print("$", cmd[:140]); r = subprocess.run(cmd, shell=True, cwd=RC)
    if r.returncode: raise SystemExit(f"ошибка: {cmd}")
sh("pip -q install uv")
if not os.path.exists(PY):
    sh(f"uv venv -q --python 3.11 {VENV}")
gpu = subprocess.run("nvidia-smi -L", shell=True, capture_output=True).returncode == 0
idx = "https://download.pytorch.org/whl/cu126" if gpu else "https://download.pytorch.org/whl/cpu"
sh(f"uv pip install -q --python {PY} torch==2.14.1 --index-url {idx}")
sh(f"uv pip install -q --python {PY} -r requirements.txt")
sh(f"{PY} app/studio/models.py denoise apollo stems" + (" denoise_aggr dereverb" if EXTRA_MODELS else ""))
sh(f"{PY} -W ignore -c \\"import sys; sys.path.insert(0,'app'); from studio import device as D; print('Устройство:', D.label(D.detect()))\\"")
print("\\n✅ Установлено")"""

C_RUN = """#@title 4. Запустить студию (веб-интерфейс)
#@markdown **Как открыть студию:** `colab` — встроенный прокси Colab (только в этом браузере); `ngrok` — ссылка `https://….ngrok-free.app`, открывается в любом браузере и на телефоне; `both` — обе ссылки.
TUNNEL = "colab"  #@param ["colab", "ngrok", "both"]
#@markdown Токен ngrok — https://dashboard.ngrok.com/get-started/your-authtoken . Можно оставить пустым и сохранить токен в секретах Colab (🔑 слева) под именем `NGROK_AUTHTOKEN`.
NGROK_TOKEN = ""  #@param {type:"string"}
#@markdown Ваш статический домен ngrok (необязательно, например `my-name.ngrok-free.app`):
NGROK_DOMAIN = ""  #@param {type:"string"}
#@markdown Пароль на ссылку ngrok (необязательно; логин `rc`, не короче 8 символов). Без пароля студию откроет любой, у кого есть ссылка:
NGROK_PASSWORD = ""  #@param {type:"string"}
import os, sys, subprocess, time, urllib.request
RC = "/content/recover-studio"; PY = "/content/rc-venv/bin/python"; PORT = 8793
if "rc_server" in globals() and rc_server.poll() is None:
    rc_server.terminate(); time.sleep(1)
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1")
log = open("/content/recover-studio.log", "w")
rc_server = subprocess.Popen([PY, "-u", "app/server.py", "--no-browser", "--port", str(PORT)], cwd=RC, env=env,
                             stdout=log, stderr=subprocess.STDOUT)
for _ in range(90):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/system", timeout=2); break
    except Exception:
        time.sleep(1)
else:
    print(open("/content/recover-studio.log").read()); raise SystemExit("сервер не запустился")
print("✅ Студия запущена.")
if TUNNEL in ("ngrok", "both"):
    token = NGROK_TOKEN.strip()
    if not token:
        try:
            from google.colab import userdata
            token = (userdata.get("NGROK_AUTHTOKEN") or "").strip()
        except Exception:
            token = ""
    if not token:
        raise SystemExit("Нужен токен ngrok: вставьте его в NGROK_TOKEN или в секреты Colab (🔑) как NGROK_AUTHTOKEN")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "pyngrok"], check=True)
    from pyngrok import ngrok
    ngrok.set_auth_token(token)
    for t in ngrok.get_tunnels():          # повторный запуск ячейки: закрыть старый туннель
        ngrok.disconnect(t.public_url)
    opts = {}
    if NGROK_DOMAIN.strip():
        opts["domain"] = NGROK_DOMAIN.strip().replace("https://", "").strip("/")
    if NGROK_PASSWORD:
        if len(NGROK_PASSWORD) < 8:
            raise SystemExit("ngrok требует пароль не короче 8 символов")
        opts["basic_auth"] = [f"rc:{NGROK_PASSWORD}"]
    rc_tunnel = ngrok.connect(PORT, "http", **opts)
    print("🌐 ngrok:", rc_tunnel.public_url, "(логин rc)" if NGROK_PASSWORD else "")
    print("   Ссылка работает, пока эта сессия Colab активна. На первой странице ngrok нажмите «Visit Site».")
if TUNNEL in ("colab", "both"):
    from google.colab import output
    print("Откройте студию по ссылке (лучше в отдельной вкладке):")
    output.serve_kernel_port_as_window(PORT, anchor_text="▶ Открыть Recover Studio")
    output.serve_kernel_port_as_iframe(PORT, height=900)"""

MD_CLI = """### 5. (необязательно) Восстановление без интерфейса
Загрузите оцифровку — получите восстановленный WAV (настройки определяются автоматически)."""

C_CLI = """#@title 5. Восстановить запись и скачать WAV
SOURCE = "auto"  #@param ["auto", "vinyl", "shellac", "cassette", "reel", "lossy", "demo"]
STYLE = "balanced"  #@param ["warm", "balanced", "open"]
INTENSITY = "medium"  #@param ["low", "medium", "high"]
SAMPLE_RATE = 44100  #@param [44100, 48000, 96000]
BITS = "24"  #@param ["16", "24", "32"]
PREVIEW_ONLY = False  #@param {type:"boolean"}
import os, subprocess, glob
from google.colab import files
RC = "/content/recover-studio"; PY = "/content/rc-venv/bin/python"
up = files.upload()
src = os.path.abspath(next(iter(up)))
out = "/content/restored"
cmd = [PY, "-u", "app/recover.py", src, "--source", SOURCE, "--style", STYLE, "--intensity", INTENSITY,
       "--sr", str(SAMPLE_RATE), "--bits", BITS, "--out", out] + (["--preview"] if PREVIEW_ONLY else [])
env = dict(os.environ, PYTHONIOENCODING="utf-8")
p = subprocess.Popen(cmd, cwd=RC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    if " окон" not in line:
        print(line, end="")
p.wait()
res = sorted(glob.glob(out + "/*.wav"), key=os.path.getmtime)
if res:
    print("Файл:", res[-1]); files.download(res[-1])"""


def cell_md(src):
    return dict(cell_type="markdown", metadata={}, source=src.splitlines(True))


def cell_code(src):
    return dict(cell_type="code", metadata={"cellView": "form"}, execution_count=None, outputs=[],
                source=src.splitlines(True))


def main():
    nb = dict(nbformat=4, nbformat_minor=0,
              metadata=dict(colab=dict(provenance=[], gpuType="T4", name="RecoverStudio_Colab.ipynb"),
                            kernelspec=dict(name="python3", display_name="Python 3"), accelerator="GPU",
                            language_info=dict(name="python")),
              cells=[cell_md(MD_INTRO), cell_code(C_GPU), cell_code(C_CODE.replace("%REPO_URL%", REPO_URL)),
                     cell_code(C_INSTALL), cell_code(C_RUN), cell_md(MD_CLI), cell_code(C_CLI)])
    os.makedirs(os.path.join(ROOT, "colab"), exist_ok=True)
    dst = os.path.join(ROOT, "colab", "RecoverStudio_Colab.ipynb")
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    # every code cell must at least be valid Python
    import ast
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    print(dst)


if __name__ == "__main__":
    main()
