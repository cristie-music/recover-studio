#!/usr/bin/env bash
# RECOVER STUDIO - installer for Linux and macOS.
# Creates .venv (Python 3.11 via uv) inside this folder and picks the PyTorch build for this machine:
#   NVIDIA (nvidia-smi) -> CUDA 12.6 | AMD (rocm-smi / /opt/rocm) -> ROCm | Apple Silicon -> MPS | else CPU
#   ./install.sh                  auto
#   ./install.sh cpu|cuda|rocm    force a build
#   RCS_MODELS_PRELOAD="denoise_aggr dereverb" ./install.sh   - also download the extra models
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
TORCH_VER="2.14.1"

echo "==============================================================="
echo "  RECOVER STUDIO - установка ($(uname -s) $(uname -m))"
echo "==============================================================="

if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"
  else
    echo "[1/5] uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
echo "[1/5] Python 3.11 -> .venv"
[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
PY="$HERE/.venv/bin/python"

MODE="${1:-auto}"
if [ "$MODE" = "auto" ]; then
  if [ "$(uname -s)" = "Darwin" ]; then MODE="mac"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then MODE="cuda"; nvidia-smi -L
  elif command -v rocm-smi >/dev/null 2>&1 || [ -d /opt/rocm ]; then MODE="rocm"
  else MODE="cpu"; fi
fi
echo "[2/5] PyTorch: $MODE"
case "$MODE" in
  cuda) uv pip install --python "$PY" "torch==$TORCH_VER" --index-url https://download.pytorch.org/whl/cu126 ;;
  rocm) uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/rocm6.4 \
          || uv pip install --python "$PY" "torch==$TORCH_VER" --index-url https://download.pytorch.org/whl/cpu ;;
  mac)  uv pip install --python "$PY" "torch==$TORCH_VER" ;;
  cpu)  uv pip install --python "$PY" "torch==$TORCH_VER" --index-url https://download.pytorch.org/whl/cpu ;;
  *) echo "неизвестный режим $MODE"; exit 1 ;;
esac

echo "[3/5] Библиотеки ..."
uv pip install --python "$PY" -r requirements.txt

echo "[4/5] Нейросети ..."
"$PY" app/studio/models.py denoise apollo stems ${RCS_MODELS_PRELOAD:-}

echo "[5/5] Проверка ..."
"$PY" -W ignore -c "import sys; sys.path.insert(0,'app'); from studio import device as D, dsp, analyze, pipeline, roformer, apollo; print('  устройство:', D.label(D.detect())); print('  OK')"
chmod +x run.sh 2>/dev/null || true
echo
echo "ГОТОВО. Запуск: bash run.sh   (без браузера: .venv/bin/python app/recover.py track.wav)"
