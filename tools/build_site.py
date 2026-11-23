#!/usr/bin/env python3
"""Monta build_src/ con los ficheros del repo (misma estructura, para que los enlaces relativos
de los README funcionen) y compila el sitio con MkDocs en site/.   Uso: python tools/build_site.py"""
import re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "build_src"
SKIP_FILES = {"CLAUDE.md", "GEMINI.md", "mkdocs.yml", ".gitattributes", ".gitignore", ".dockerignore"}
SKIP_PREFIX = ("tools/", ".github/")

DISPLAY = re.compile(r"^(\s*)\$\$(.+?)\$\$\s*$")


def isolate_display_math(text: str) -> str:
    """`$$...$$` escrito en una sola línea (típico dentro de listas) se convierte en un bloque
    independiente con líneas en blanco, que es lo que arithmatex reconoce como fórmula en bloque.
    Solo afecta a la copia de build_src/; el documento original no se modifica."""
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else DISPLAY.match(line)
        if m:
            ind = m.group(1)
            out += ["", f"{ind}$$", f"{ind}{m.group(2).strip()}", f"{ind}$$", ""]
        else:
            out.append(line)
    return "\n".join(out)


if SRC.exists():
    shutil.rmtree(SRC)
files = subprocess.check_output(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True
).splitlines()
n = 0
for rel in files:
    if rel in SKIP_FILES or rel.startswith(SKIP_PREFIX) or not (ROOT / rel).is_file():
        continue
    dst = SRC / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if rel.endswith(".md"):
        dst.write_text(isolate_display_math((ROOT / rel).read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    else:
        shutil.copy2(ROOT / rel, dst)
    n += 1
print(f"[site] {n} ficheros copiados a {SRC.name}/")
sys.exit(subprocess.call([sys.executable, "-m", "mkdocs", "build", "--strict" if "--strict" in sys.argv else "-q"], cwd=ROOT))
