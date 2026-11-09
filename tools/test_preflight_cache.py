"""Small, self-contained checks for frozen retrieval-cache preflight."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from preflight import check_retrieval_caches


class RetrievalCachePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / 'tools').mkdir()
        (self.root / 'tools/final_runs.json').write_text(
            json.dumps({'PEMS03': 'runs/frozen'}), encoding='utf-8')
        (self.root / 'runs/frozen').mkdir(parents=True)
        (self.root / 'runs/frozen/config.json').write_text(json.dumps({
            'dataset_name': 'PEMS03',
            'analog_retrieval_cache': 'datasets/PEMS03/retrieval.npz',
            'analog_retrieval_k': 4,
            'input_length': 12,
            'predict_length': 12,
            'index_scaler_path': 'datasets/PEMS03/index.npz',
        }), encoding='utf-8')
        (self.root / 'datasets/PEMS03').mkdir(parents=True)
        self.index_path = self.root / 'datasets/PEMS03/index.npz'
        self.cache_path = self.root / 'datasets/PEMS03/retrieval.npz'
        self.indices = {
            'idx_train': np.array([12, 13], dtype=np.int64),
            'idx_val': np.array([14], dtype=np.int64),
            'idx_test': np.array([15], dtype=np.int64),
        }
        np.savez(self.index_path, **self.indices)
        arrays = dict(self.indices, k=4, input_length=12, predict_length=12)
        for split in ('train', 'val', 'test'):
            for prefix in ('retrieval', 'exact_retrieval', 'exact_available'):
                arrays[f'{prefix}_{split}'] = np.zeros(1, dtype=np.float32)
        np.savez(self.cache_path, **arrays)

    def tearDown(self):
        self.temporary.cleanup()

    def test_matching_cache(self):
        failures, checked = check_retrieval_caches(self.root)
        self.assertEqual((failures, checked), ([], 1))

    def test_split_mismatch(self):
        changed = dict(self.indices, idx_val=np.array([99], dtype=np.int64))
        np.savez(self.index_path, **changed)
        failures, checked = check_retrieval_caches(self.root)
        self.assertEqual(checked, 1)
        self.assertTrue(any('Retrieval val split mismatch' in item for item in failures))

    def test_missing_cache(self):
        self.cache_path.unlink()
        failures, checked = check_retrieval_caches(self.root)
        self.assertEqual(checked, 0)
        self.assertTrue(any('Missing frozen retrieval cache' in item for item in failures))


if __name__ == '__main__':
    unittest.main()
