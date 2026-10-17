#!/usr/bin/env python3
"""Builds .zip packages for all plugins, models, methods, and environments in theta-plugins, computing SHA-256."""
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent.parent
HUB_ROOT = ROOT.parent / "theta-hub"
DIST = ROOT / "dist"
HUB_PACKAGES = HUB_ROOT / "packages"

COMPONENTS = [
    # UI Plugins
    {"id": "obsidian-notes", "name": "obsidian-0.1.0.zip", "src": ROOT / "plugins" / "obsidian", "kind": "plugin"},
    {"id": "doom-sim", "name": "doom_sim-0.1.0.zip", "src": ROOT / "plugins" / "doom-sim", "kind": "plugin"},
    {"id": "antigravity", "name": "antigravity-0.1.5.zip", "src": ROOT / "plugins" / "antigravity", "kind": "plugin"},
    # Environments
    {"id": "doom-env", "name": "doom_env-1.0.0.zip", "src": ROOT / "envs" / "doom-env", "kind": "env", "paradigms": ["online_rl"]},
    {"id": "gym_classic_control", "name": "gym_classic_control-1.0.0.zip", "src": ROOT / "envs" / "gym_classic_control", "kind": "env", "paradigms": ["online_rl"]},
    {"id": "atari-env", "name": "atari_env-1.0.0.zip", "src": ROOT / "envs" / "atari-env", "kind": "env", "paradigms": ["online_rl"]},
    # Models
    {"id": "dnn", "name": "dnn-1.0.0.zip", "src": ROOT / "models" / "dnn", "kind": "model", "paradigms": ["online_rl", "offline_rl", "supervised"]},
    {"id": "dueling_resnet", "name": "dueling_resnet-1.0.0.zip", "src": ROOT / "models" / "dueling_resnet", "kind": "model", "paradigms": ["online_rl", "offline_rl", "supervised"]},
    {"id": "nature_cnn", "name": "nature_cnn-1.0.0.zip", "src": ROOT / "models" / "nature_cnn", "kind": "model", "paradigms": ["online_rl", "offline_rl", "supervised"]},
    {"id": "cew", "name": "cew-1.0.0.zip", "src": ROOT / "models" / "cew", "kind": "model", "paradigms": ["online_rl", "offline_rl", "supervised"]},
    # Methods (Real Benchmark Suites)
    {"id": "sb3-suite", "name": "sb3_suite-1.0.0.zip", "src": ROOT / "methods" / "sb3", "kind": "method", "paradigms": ["online_rl"]},
    {"id": "d3rlpy-suite", "name": "d3rlpy_suite-1.0.0.zip", "src": ROOT / "methods" / "d3rlpy", "kind": "method", "paradigms": ["offline_rl"]},
    # Suites / Meta-Packages
    {"id": "standard_rl_suite", "name": "standard_rl_suite-1.0.0.zip", "src": ROOT / "suites" / "standard_rl_suite", "kind": "suite", "paradigms": ["online_rl", "offline_rl"]},
    {"id": "basic_suite", "name": "basic_suite-1.0.0.zip", "src": ROOT / "suites" / "basic_suite", "kind": "suite", "paradigms": ["online_rl", "offline_rl", "supervised"]},
]


def package_component(src_dir: Path, output_zip: Path) -> str:
    """Compress directory into zip archive and return hex SHA-256."""
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in sorted(src_dir.rglob("*")):
            if (
                file_path.is_file()
                and (not file_path.name.startswith(".") or file_path.name == ".theta_component.json")
                and "__pycache__" not in file_path.parts
                and not file_path.name.startswith("test_")
            ):
                arcname = file_path.relative_to(src_dir)
                zf.write(file_path, arcname=arcname)

    data = output_zip.read_bytes()
    return hashlib.sha256(data).hexdigest()


def update_registry_entry(comp: dict, sha256: str, zip_name: str):
    """Update or create registry metadata entry in theta-hub/registry/."""
    if not HUB_ROOT.exists():
        return

    kind_plural = comp["kind"] + "s" if not comp["kind"].endswith("s") else comp["kind"]
    reg_dir = HUB_ROOT / "registry" / kind_plural
    reg_dir.mkdir(parents=True, exist_ok=True)
    reg_file = reg_dir / f"{comp['id']}.json"

    meta_file = comp["src"] / ".theta_component.json"
    comp_meta = {}
    if meta_file.exists():
        try:
            comp_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    ver = comp_meta.get("version", "1.0.0")
    raw_url = f"https://raw.githubusercontent.com/CameronEgb/theta-hub/main/packages/{zip_name}"

    if reg_file.exists():
        try:
            entry = json.loads(reg_file.read_text(encoding="utf-8"))
        except Exception:
            entry = {}
    else:
        entry = {
            "id": comp["id"],
            "name": comp_meta.get("name", comp["id"].replace("_", " ").title()),
            "kind": comp["kind"],
            "version": ver,
            "description": comp_meta.get("description", f"{comp['id']} component for Theta-IDE."),
            "author": comp_meta.get("author", {"name": "Theta-IDE Team"}),
            "repository": comp_meta.get("repository", "https://github.com/CameronEgb/theta-plugins"),
            "license": comp_meta.get("license", "MIT"),
            "tags": [comp["kind"], comp["id"]],
            "releases": {},
        }

    if "repository" not in entry:
        entry["repository"] = comp_meta.get("repository", "https://github.com/CameronEgb/theta-plugins")
    if "license" not in entry:
        entry["license"] = comp_meta.get("license", "MIT")

    # Record supported paradigms
    paradigms = comp_meta.get("paradigms") or comp.get("paradigms")
    if paradigms:
        entry["paradigms"] = paradigms

    entry["version"] = ver
    if "releases" not in entry:
        entry["releases"] = {}

    entry["releases"][ver] = {
        "tag": f"v{ver}",
        "url": raw_url,
        "sha256": sha256,
        "published_at": "2026-10-16T19:00:00Z",
    }

    reg_file.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")


def main():
    DIST.mkdir(parents=True, exist_ok=True)
    if HUB_ROOT.exists():
        HUB_PACKAGES.mkdir(parents=True, exist_ok=True)

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
        print(f"📦 {comp['id']:<20} -> dist/{comp['name']}")
        print(f"   Size:   {size_kb:.1f} KB")
        print(f"   SHA256: {sha}")

        # Sync to theta-hub
        if HUB_ROOT.exists():
            hub_zip = HUB_PACKAGES / comp["name"]
            shutil.copy2(str(out), str(hub_zip))
            update_registry_entry(comp, sha, comp["name"])
            print(f"   Synced to theta-hub: packages/{comp['name']}")
        print()

    print(f"Artifacts created in: {DIST}")


if __name__ == "__main__":
    main()
