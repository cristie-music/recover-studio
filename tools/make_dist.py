"""Builds dist/recover-studio-app.zip: code only (app, tools, colab, installers), well under 1 MB.
Upload it to Colab, or unpack on another PC and run install.bat (Windows) / install.sh (Linux, macOS).

    python tools/make_dist.py
"""
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")
INCLUDE_DIRS = ["app", "tools", "colab"]
INCLUDE_FILES = ["install.bat", "install.sh", "run.sh", "RecoverStudio.bat", "recover.bat", "check-gpu.bat",
                 "requirements.txt", "README.md", "README.txt", "THIRD_PARTY_NOTICES.txt", ".gitignore", ".gitattributes"]
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}


def main():
    os.makedirs(DIST, exist_ok=True)
    dst = os.path.join(DIST, "recover-studio-app.zip")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for f in INCLUDE_FILES:
            p = os.path.join(ROOT, f)
            if os.path.exists(p):
                z.write(p, f)
        for d in INCLUDE_DIRS:
            for root, dirs, files in os.walk(os.path.join(ROOT, d)):
                dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
                for f in files:
                    if f.endswith((".pyc", ".peaks.json", ".spec.json")):
                        continue
                    p = os.path.join(root, f)
                    z.write(p, os.path.relpath(p, ROOT))
    print("  ", dst, f"{os.path.getsize(dst) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
