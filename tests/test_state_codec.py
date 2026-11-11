"""Lossless edge cases and independent layout/decoder checks."""
import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from replay import state_codec as codec  # noqa: E402


class CodecTests(unittest.TestCase):
    def test_every_reversible_layout(self):
        rng = np.random.default_rng(46)
        arrays = [rng.integers(0, 65536, (31, 7), dtype=np.uint16),
                  np.array([[0, 65535], [65535, 0], [0, 65535]], np.uint16),
                  rng.integers(0, 256, (31, 7), dtype=np.uint8)]
        for a in arrays:
            for predictor in ('none', 'delta'):
                for order in ('C', 'F'):
                    for layout in ('plain', 'bytes', 'planes'):
                        data, dtype = codec.transform(a, order, predictor, layout)
                        for compressor, blob in codec.compressions(data):
                            row = dict(shape=list(a.shape), dtype=a.dtype.str,
                                       encoded_dtype=dtype, predictor=predictor,
                                       layout=layout, order=order,
                                       compression=compressor, sha256=codec.digest(a))
                            restored = codec.restore(blob, row)
                            self.assertEqual(restored.tobytes(), a.tobytes())

    def test_metadata_and_empty_streams(self):
        state = {'scalar': np.array(3), 'tail': np.arange(1200, dtype='>f8').reshape(30, 40)[:, ::2],
                 'train_raw_mixed_q_codes': np.zeros((20, 0), np.uint8),
                 'float': np.array([np.nan, np.inf, -0.0], dtype=np.float64)}
        blob, generic, _ = codec.encode(state)
        for package in (blob, generic):
            restored = codec.decode(package)
            self.assertEqual(state.keys(), restored.keys())
            for k in state:
                self.assertEqual(state[k].shape, restored[k].shape)
                self.assertEqual(state[k].dtype, restored[k].dtype)
                self.assertEqual(state[k].tobytes(), restored[k].tobytes())

    def test_corruption_rejected(self):
        blob, _, _ = codec.encode({'small': np.array([4, 5], np.int16)})
        for bad in (blob[:-1], blob + b'junk', b'broken', blob[:-1] + bytes([blob[-1] ^ 1])):
            with self.assertRaises(Exception):
                codec.decode(bad)


if __name__ == '__main__':
    unittest.main()
