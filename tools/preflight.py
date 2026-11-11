"""Read-only preflight for the five frozen Scope replay configurations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from prepare_local_inputs import required_paths

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> int:
    failures = []
    for package in ("numpy", "torch", "einops", "pandas"):
        if importlib.util.find_spec(package) is None:
            failures.append(f"Missing Python package: {package}")
    manifest = json.loads(
        (ROOT / "configs/replay_assets_manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = (ROOT / row["path"]).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError:
            failures.append(f"Replay asset escapes repository: {row['path']}")
            continue
        if not path.is_file():
            failures.append(f"Missing replay asset: {row['path']}")
        elif digest(path) != row["sha256"]:
            failures.append(f"Replay asset hash mismatch: {row['path']}")
    for rel in required_paths():
        if not (ROOT / rel).is_file():
            failures.append(f"Missing prepared data: {rel}")
    for failure in failures:
        print("FAIL:", failure)
    print("Frozen replay assets checked:", len(manifest["files"]))
    print("Prepared dataset files checked:", len(required_paths()))
    print("NOT A NUMERICAL REPRODUCTION CHECK: run Validation, then Test.")
    print("BLOCKED" if failures else "STATIC CHECKS PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
