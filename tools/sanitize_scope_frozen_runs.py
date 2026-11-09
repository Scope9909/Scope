"""One-time migration of final Scope run metadata and its local asset bundle.

The model weights are not rewritten. The original training records remain in
their historical directories; this produces a clean, frozen inference copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "tools/final_runs.json"
FINAL = ROOT / "tools/final_assets_manifest.json"
RELEASE = ROOT / "tools/release_assets_manifest.json"
BUNDLE_NAME = "scope_final_frozen_assets_v2.zip"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def renamed(value: str) -> str:
    return value.replace("TLAST_improved", "Scope").replace("TLAST", "Scope")


def encoded(data: dict) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    options = parser.parse_args()
    runs = json.loads(RUNS.read_text(encoding="utf-8"))
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    planned: dict[Path, bytes] = {}

    for dataset, relative in runs.items():
        folder = ROOT / relative
        folder.resolve().relative_to((ROOT / "EXP_experiments/Scope").resolve())
        config_path = folder / "config.json"
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        if config["dataset_name"] != dataset or config["modelid"] != "Scope":
            raise ValueError(f"Unexpected final config: {config_path}")
        new_config = config_bytes.replace(b"TLAST_improved", b"Scope")
        if b"tlast" in new_config.lower():
            raise ValueError(f"Unresolved old name: {config_path}")
        planned[config_path] = new_config

        pickle_path = folder / "config.pkl"
        args = pickle.loads(pickle_path.read_bytes())  # verified local artifact
        if args.dataset_name != dataset or args.modelid != "TLAST_improved":
            raise ValueError(f"Unexpected frozen pickle: {pickle_path}")
        for key, value in vars(args).items():
            if isinstance(value, str):
                setattr(args, key, renamed(value))
        if args.modelid != "Scope" or args.note != "Scope":
            raise ValueError(f"Model or note rename failed: {pickle_path}")
        new_pickle = pickle.dumps(args, protocol=4)
        if b"tlast" in new_pickle.lower():
            raise ValueError(f"Unresolved old name: {pickle_path}")
        planned[pickle_path] = new_pickle

    for row in final["files"]:
        path = ROOT / row["path"]
        path.resolve().relative_to(ROOT.resolve())
        before = path.read_bytes()
        if digest(before) != row["sha256"] or len(before) != row["size"]:
            raise ValueError(f"Final asset changed before migration: {path}")
        after = planned.get(path, before)
        row["sha256"] = digest(after)
        row["size"] = len(after)

    scope = ROOT / "EXP_experiments/Scope"
    for path in scope.rglob("*"):
        if path.is_file() and b"tlast" in planned.get(path, path.read_bytes()).lower():
            raise ValueError(f"Scope still contains an old name: {path}")

    bundle = ROOT / "release/staging" / BUNDLE_NAME
    if bundle.exists():
        raise FileExistsError(f"Refusing to overwrite bundle: {bundle}")
    print(f"Scope runs: {len(runs)}; metadata files updated: {len(planned)}")
    print(f"Assets for replacement bundle: {len(final['files'])}")
    print(f"New local bundle: {bundle}")
    if not options.apply:
        return

    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary_bundle = bundle.with_suffix(".tmp")
    if temporary_bundle.exists():
        raise FileExistsError(f"Refusing to overwrite temporary bundle: {temporary_bundle}")
    try:
        with zipfile.ZipFile(temporary_bundle, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            for row in final["files"]:
                path = ROOT / row["path"]
                archive.writestr(row["path"], planned.get(path, path.read_bytes()))
        with zipfile.ZipFile(temporary_bundle) as archive:
            if len(archive.namelist()) != len(final["files"]):
                raise ValueError("Bundle member count mismatch")
            for row in final["files"]:
                data = archive.read(row["path"])
                if digest(data) != row["sha256"] or len(data) != row["size"]:
                    raise ValueError(f"Bundle member mismatch: {row['path']}")
        release["bundle"].update(filename=BUNDLE_NAME,
                                 size_bytes=temporary_bundle.stat().st_size,
                                 sha256=digest(temporary_bundle.read_bytes()),
                                 file_count=len(final["files"]))
        for path, data in planned.items():
            replacement = path.with_name(path.name + ".scope-tmp")
            if replacement.exists():
                raise FileExistsError(f"Refusing to overwrite: {replacement}")
            replacement.write_bytes(data)
            os.replace(replacement, path)
        FINAL.write_bytes(encoded(final))
        RELEASE.write_bytes(encoded(release))
        os.replace(temporary_bundle, bundle)
    finally:
        if temporary_bundle.exists():
            temporary_bundle.unlink()
    print(f"Verified bundle SHA-256: {release['bundle']['sha256']}")


if __name__ == "__main__":
    main()
