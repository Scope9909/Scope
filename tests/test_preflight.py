"""Small structural tests for the public replay inventory."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from prepare_local_inputs import required_paths  # noqa: E402


class ReplayInventoryTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(
            (ROOT / "configs/replay_assets_manifest.json").read_text(encoding="utf-8"))

    def test_exactly_fifteen_prepared_dataset_files(self):
        paths = required_paths()
        self.assertEqual(len(paths), 15)
        self.assertEqual(len(paths), len(set(paths)))

    def test_manifest_paths_are_relative_and_contained(self):
        for row in self.manifest["files"]:
            path = ROOT / row["path"]
            path.resolve().relative_to(ROOT.resolve())
            self.assertFalse(Path(row["path"]).is_absolute())

    def test_manifest_has_unique_sha256_rows(self):
        paths = [row["path"] for row in self.manifest["files"]]
        self.assertEqual(len(paths), len(set(paths)))
        for row in self.manifest["files"]:
            self.assertRegex(row["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
