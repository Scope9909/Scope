"""Lossless codec for the analog candidate-index cache.

The cache stores ``(query, K, node)`` int32 global time endpoints, ``-1``
meaning "no candidate available".  Empirical range is tiny (all endpoints come
from the train pool) and, more importantly, the TOD partition is a *deterministic
function* of ``(idx_train, time_data)``:

    candidate_group(tod) = sorted(idx_train[time_data[idx_train, 0] == tod])

Every endpoint therefore belongs to exactly one group, so it suffices to store
its **position inside that group**.  Group sizes are 35-59 across the four
PEMS datasets, i.e. 6 bits.

Two codecs are offered:

    offset8   one uint8 per slot (simplest, -75%)
    offset6   the same offsets bit-packed at ``bits`` bits (here -81.25%)

Both are exactly invertible.  Nothing here changes a single predicted value:
:meth:`TodIndexCodec.decode` returns an array that compares equal to the
original under ``np.array_equal``.
"""

import numpy as np


class TodIndexCodec:
    """Encode/decode candidate endpoint indices as TOD-group offsets."""

    def __init__(self, time_data, idx_train):
        self.time_data = np.asarray(time_data)
        self.idx_train = np.asarray(idx_train, dtype=np.int64)
        train_tod = self.time_data[self.idx_train, 0].astype(np.int64)
        self.groups = {}
        for tod in np.unique(train_tod):
            self.groups[int(tod)] = np.ascontiguousarray(
                self.idx_train[train_tod == tod])
        sizes = np.array([g.size for g in self.groups.values()])
        self.group_size_max = int(sizes.max())
        if self.group_size_max > 254:
            raise ValueError("TOD group too large for a uint8 offset code")
        self.sentinel = self.group_size_max      # first value past legal offsets
        self.bits = int(np.ceil(np.log2(self.group_size_max + 1)))

        table = np.full(int(self.time_data.shape[0]), -1, dtype=np.int64)
        for endpoints in self.groups.values():
            table[endpoints] = np.arange(endpoints.size)
        self._position = table

    # ------------------------------------------------------------------ encode
    def encode(self, endpoints):
        """(Q, K, N) int32 endpoints -> (Q, K, N) uint8 offsets."""
        endpoints = np.asarray(endpoints)
        invalid = endpoints < 0
        positions = self._position[endpoints]
        if np.any(positions > self.sentinel - 1):
            raise ValueError("endpoint not present in its TOD group")
        return np.where(invalid, self.sentinel, positions).astype(np.uint8)

    def decode(self, code, query_indices):
        """(Q, K, N) uint8 offsets -> (Q, K, N) int32 endpoints."""
        code = np.asarray(code)
        query_tod = np.asarray(
            self.time_data[np.asarray(query_indices, dtype=np.int64), 0],
            dtype=np.int64)
        if query_tod.size != code.shape[0]:
            raise ValueError("query_indices does not match the code rows")
        out = np.full(code.shape, -1, dtype=np.int32)
        for tod in np.unique(query_tod):
            rows = np.flatnonzero(query_tod == tod)
            block = code[rows]
            group = self.groups[int(tod)]
            valid = block != self.sentinel
            if np.any(block[valid] >= group.size):
                raise ValueError(f"TOD={tod}: offset outside its group")
            out[rows] = np.where(valid, group[np.where(valid, block, 0)], -1)
        return out

    # ------------------------------------------------------------- bit packing
    def pack(self, code, chunk_rows=200000):
        """Pack the K axis (last after transpose) at ``self.bits`` bits."""
        values = np.ascontiguousarray(code.transpose(0, 2, 1))
        rows = values.reshape(-1, values.shape[-1])
        count = rows.shape[1]
        weights = np.arange(self.bits - 1, -1, -1).astype(np.uint8)
        cols = (count * self.bits + 7) // 8
        packed = np.empty((rows.shape[0], cols), dtype=np.uint8)
        for start in range(0, rows.shape[0], chunk_rows):
            stop = min(start + chunk_rows, rows.shape[0])
            block = rows[start:stop]
            bits = ((block[:, :, None] >> weights) & 1).astype(np.uint8)
            packed[start:stop] = np.packbits(
                bits.reshape(block.shape[0], count * self.bits), axis=1)
        return packed.reshape(code.shape[0], code.shape[2], cols)

    def unpack(self, packed, num_candidates, chunk_rows=200000):
        """Inverse of :meth:`pack`; returns (Q, N, K) uint8 offsets."""
        rows = packed.reshape(-1, packed.shape[-1])
        need = num_candidates * self.bits
        weights = (np.uint64(1) << np.arange(
            self.bits - 1, -1, -1, dtype=np.uint64))
        out = np.empty((rows.shape[0], num_candidates), dtype=np.uint8)
        for start in range(0, rows.shape[0], chunk_rows):
            stop = min(start + chunk_rows, rows.shape[0])
            flat = np.unpackbits(rows[start:stop], axis=1)[:, :need]
            words = flat.reshape(stop - start, num_candidates, self.bits)
            out[start:stop] = (words.astype(np.uint64) * weights).sum(
                axis=2).astype(np.uint8)
        return out.reshape(packed.shape[:-1] + (num_candidates,))

    def pack_transposed_roundtrip(self, code, num_candidates):
        """pack -> unpack, returning code in the original (Q, K, N) layout."""
        packed = self.pack(code)
        back = self.unpack(packed, num_candidates)
        return np.ascontiguousarray(back.transpose(0, 2, 1)), packed
