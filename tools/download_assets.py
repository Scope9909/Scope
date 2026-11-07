"""Download release assets (checkpoints, SRM states, teacher predictions)
into their expected in-repo locations, with SHA-256 verification.

Usage (after the GitHub release is published):
    python tools/download_assets.py --base-url https://github.com/<owner>/<repo>/releases/download/v1.0.0

The manifest (tools/assets_manifest.json) lists every asset with its target
relative path and expected SHA-256. Each asset on the release is a single
file named after its relative path with '/' replaced by '__'.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets_manifest.json")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True,
                    help="Release download base URL, e.g. "
                         "https://github.com/<owner>/<repo>/releases/download/<tag>")
    ap.add_argument("--only", default=None,
                    help="Optional group filter: checkpoints | srm_states | teacher_predictions | misc")
    args = ap.parse_args()

    with open(MANIFEST, "r", encoding="utf-8") as fh:
        files = json.load(fh)["files"]

    n_ok = n_dl = n_fail = 0
    for item in files:
        if args.only and item["group"] != args.only:
            continue
        rel = item["path"]
        dest = os.path.join(REPO_ROOT, rel.replace("/", os.sep))
        if os.path.exists(dest) and sha256_of(dest) == item["sha256"]:
            n_ok += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        asset_name = rel.replace("/", "__")
        url = f"{args.base_url.rstrip('/')}/{asset_name}"
        print(f"downloading {rel} ({item['size']/1024/1024:.1f} MB)")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}")
            n_fail += 1
            continue
        if sha256_of(dest) != item["sha256"]:
            print("  FAILED: sha256 mismatch")
            os.remove(dest)
            n_fail += 1
            continue
        n_dl += 1

    print(f"\ndone: {n_dl} downloaded, {n_ok} already present, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
