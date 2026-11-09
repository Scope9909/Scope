"""Verify and optionally install the final frozen bundle and five caches.

This does not obtain raw data, prepare dataset splits, or run evaluation.
No destination file is overwritten. Preview is the default.
"""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tools/release_assets_manifest.json"
FINAL = ROOT / "tools/final_assets_manifest.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def destination_for(rel):
    path = ROOT / rel
    path.resolve().relative_to(ROOT.resolve())
    return path


def bundle_member_for(rel):
    """The current verified bundle uses the final destination names."""
    return rel


def verify_existing(path, size, sha256):
    if path.exists() and (
        not path.is_file() or path.stat().st_size != size or digest(path) != sha256
    ):
        raise RuntimeError(f"Destination conflict; refusing overwrite: {path}")
    return path.exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="Directory containing datasets/<name>/<cache-file>.npz")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    bundle = args.bundle.resolve()
    cache_root = args.cache_root.resolve()
    if bundle.stat().st_size != manifest["bundle"]["size_bytes"] or digest(bundle) != manifest["bundle"]["sha256"]:
        raise RuntimeError("Frozen bundle size or SHA-256 mismatch")
    pending_assets = []
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        if len(final["files"]) != manifest["bundle"]["file_count"]:
            raise RuntimeError("Final-asset manifest count mismatch")
        for row in final["files"]:
            rel = row["path"]
            member = bundle_member_for(rel)
            if member not in names:
                raise RuntimeError(f"Bundle member missing: {member}")
            data = archive.read(member)
            if len(data) != row["size"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise RuntimeError(f"Bundle member mismatch: {member}")
            destination = destination_for(rel)
            if not verify_existing(destination, row["size"], row["sha256"]):
                pending_assets.append(rel)
        pending_caches = []
        for row in manifest["retrieval_caches"]:
            rel = row["install_path"]
            source = cache_root / rel
            source.resolve().relative_to(cache_root)
            if not source.is_file() or source.stat().st_size != row["size_bytes"] or digest(source) != row["sha256"]:
                raise RuntimeError(f"Cache source mismatch: {source}")
            destination = destination_for(rel)
            if not verify_existing(destination, row["size_bytes"], row["sha256"]):
                pending_caches.append(row)
        print(json.dumps({
            "final_assets_checked": len(final["files"]),
            "retrieval_caches_checked": len(manifest["retrieval_caches"]),
            "assets_to_install": len(pending_assets),
            "caches_to_install": len(pending_caches),
            "cache_bytes_to_install": sum(row["size_bytes"] for row in pending_caches),
            "apply": args.apply,
            "raw_data_included": False,
        }, indent=2))
        if not args.apply:
            return
        for rel in pending_assets:
            destination = destination_for(rel)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(bundle_member_for(rel)) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, 8 << 20)
        for row in pending_caches:
            source = cache_root / row["install_path"]
            destination = destination_for(row["install_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as input_stream, destination.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, 8 << 20)
    for row in final["files"]:
        if not verify_existing(destination_for(row["path"]), row["size"], row["sha256"]):
            raise RuntimeError(f"Final asset missing after installation: {row['path']}")
    for row in manifest["retrieval_caches"]:
        if not verify_existing(destination_for(row["install_path"]), row["size_bytes"], row["sha256"]):
            raise RuntimeError(f"Retrieval cache missing after installation: {row['install_path']}")
    print("Frozen assets and caches installed and verified; raw data is still required.")


if __name__ == "__main__":
    main()
