#!/usr/bin/env python3
"""Builds .zip packages for all plugins and environments in theta-plugins, computing SHA-256."""
import hashlib
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

COMPONENTS = [
    {"id": "obsidian", "name": "obsidian-0.1.0.zip", "src": ROOT / "plugins" / "obsidian"},
    {"id": "doom-sim", "name": "doom_sim-0.1.0.zip", "src": ROOT / "plugins" / "doom-sim"},
    {"id": "antigravity", "name": "antigravity-0.1.4.zip", "src": ROOT / "plugins" / "antigravity"},
    {"id": "doom-env", "name": "doom_env-1.0.0.zip", "src": ROOT / "envs" / "doom-env"},
]


def package_component(src_dir: Path, output_zip: Path) -> str:
    """Compress directory into zip archive and return hex SHA-256."""
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in sorted(src_dir.rglob("*")):
            if file_path.is_file() and not file_path.name.startswith(".") and "__pycache__" not in file_path.parts:
                arcname = file_path.relative_to(src_dir)
                zf.write(file_path, arcname=arcname)

    data = output_zip.read_bytes()
    return hashlib.sha256(data).hexdigest()


def main():
    DIST.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Packaging Theta Plugins & Components")
    print("=" * 70)

    for comp in COMPONENTS:
        src = comp["src"]
        if not src.exists():
            print(f"⚠️ Skipping {comp['id']}: directory {src} not found")
            continue

        out = DIST / comp["name"]
        sha = package_component(src, out)
        size_kb = out.stat().st_size / 1024
        print(f"📦 {comp['id']:<15} -> dist/{comp['name']}")
        print(f"   Size:   {size_kb:.1f} KB")
        print(f"   SHA256: {sha}")
        print()

    print(f"Artifacts created in: {DIST}")


if __name__ == "__main__":
    main()
