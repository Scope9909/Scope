"""Self-contained, checksummed lossless array archive; no forecasting changes.

Each field selects a reversible layout/predictor and a standard compressor.
The complete archive includes all dtype/shape/layout/compression metadata.
"""
from __future__ import annotations

import bz2
import hashlib
import io
import json
import lzma
import struct
from pathlib import Path

import numpy as np

MAGIC = b'ALSC\x01'
HISTORY_FIELDS = {'train_raw', 'train_raw_mixed_q_codes', 'train_raw_mixed_exact'}


def digest(a):
    return hashlib.sha256(np.asarray(a).tobytes(order='C')).hexdigest()


def json_bytes(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=True).encode()


def compressions(data):
    import zlib
    yield 'raw', data
    yield 'zlib', zlib.compress(data, 9)
    if len(data) > 1024:
        yield 'xz', lzma.compress(data, preset=6)
        yield 'bz2', bz2.compress(data, compresslevel=9)


def transform(a, order, predictor, layout):
    work = np.asarray(a)
    if predictor == 'delta':
        wide = work.astype(np.int64)
        diffs = np.diff(wide, axis=0, prepend=np.zeros_like(wide[:1]))
        work = ((diffs << 1) ^ (diffs >> 63)).astype('<u4')
    flat = work.ravel(order=order)
    raw = flat.tobytes()
    width = work.dtype.itemsize
    if layout == 'bytes':
        raw = np.frombuffer(raw, np.uint8).reshape(-1, width).T.tobytes()
    elif layout == 'planes':
        bits = np.unpackbits(np.frombuffer(raw, np.uint8)).reshape(-1, width * 8)
        raw = np.packbits(bits.T.reshape(-1)).tobytes()
    return raw, work.dtype.str


def restore(data, row):
    import zlib
    decompressors = {'raw': lambda x: x, 'zlib': zlib.decompress,
                     'xz': lzma.decompress, 'bz2': bz2.decompress}
    raw = decompressors[row['compression']](data)
    dtype = np.dtype(row['encoded_dtype'])
    count = int(np.prod(row['shape'], dtype=np.int64))
    if len(raw) != count * dtype.itemsize:
        raise ValueError('Decoded field byte length mismatch')
    if row['layout'] == 'bytes':
        raw = np.frombuffer(raw, np.uint8).reshape(dtype.itemsize, count).T.copy().tobytes()
    elif row['layout'] == 'planes':
        bits = np.unpackbits(np.frombuffer(raw, np.uint8))
        raw = np.packbits(bits.reshape(dtype.itemsize * 8, count).T.reshape(-1)).tobytes()
    array = np.frombuffer(raw, dtype=dtype).reshape(row['shape'], order=row['order'])
    if row['predictor'] == 'delta':
        code = array.astype(np.int64)
        array = np.cumsum((code >> 1) ^ -(code & 1), axis=0)
    array = array.astype(np.dtype(row['dtype']), copy=True)
    if digest(array) != row['sha256']:
        raise ValueError('Decoded field checksum mismatch')
    return array


def assemble(rows, payloads):
    import zlib
    header = zlib.compress(json_bytes(rows), 9)
    return MAGIC + struct.pack('<I', len(header)) + header + b''.join(payloads)


def encode(state):
    """Return full optimized archive, raw-layout control, and field diagnostics."""
    chosen, payloads, plain, plain_payloads, diagnostics = [], [], [], [], []
    for key in sorted(state):
        a = np.asarray(state[key])
        if a.dtype.hasobject or a.dtype.fields:
            raise ValueError('Object and structured arrays are unsupported')
        history = key in HISTORY_FIELDS and a.ndim == 2 and a.size > 0
        integer_history = (history and a.dtype.kind in 'ui' and
                           a.min() >= 0 and a.max() <= 65535)
        float_history = history and a.dtype == np.dtype('float32')
        if history and not (integer_history or float_history):
            raise ValueError('History must be uint16-exact integers or float32')
        candidates = []
        # Integer history permits reversible numeric delta coding. Float32
        # history is encoded bit-exactly without numeric differencing.
        for predictor in (('none', 'delta') if integer_history else ('none',)):
            for order in (('C', 'F') if a.ndim > 1 else ('C',)):
                for layout in (('plain', 'bytes', 'planes') if a.size > 512 else ('plain',)):
                    data, encoded_dtype = transform(a, order, predictor, layout)
                    for compression, blob in compressions(data):
                        row = dict(key=key, dtype=a.dtype.str, shape=list(a.shape),
                                   encoded_dtype=encoded_dtype, order=order,
                                   predictor=predictor, layout=layout,
                                   compression=compression, length=len(blob), sha256=digest(a))
                        candidates.append((len(blob) + len(json_bytes(row)), row, blob))
        best = min(candidates, key=lambda x: x[0])
        raw_best = min((x for x in candidates if x[1]['predictor'] == 'none'
                       and x[1]['layout'] == 'plain'), key=lambda x: x[0])
        for candidate in (best, raw_best):
            restored = restore(candidate[2], candidate[1])
            if restored.dtype != a.dtype or restored.shape != a.shape or restored.tobytes() != a.tobytes():
                raise AssertionError(f'Lossless field verification failed: {key}')
        chosen.append(best[1]); payloads.append(best[2])
        plain.append(raw_best[1]); plain_payloads.append(raw_best[2])
        diagnostics.append(dict(best[1], original_bytes=a.nbytes,
                                raw_layout_compressed_bytes=raw_best[1]['length']))
    return assemble(chosen, payloads), assemble(plain, plain_payloads), diagnostics


def decode(blob):
    import zlib
    if not blob.startswith(MAGIC) or len(blob) < len(MAGIC) + 4:
        raise ValueError('Invalid archive magic/version')
    offset = len(MAGIC) + 4
    size = struct.unpack('<I', blob[len(MAGIC):offset])[0]
    rows = json.loads(zlib.decompress(blob[offset:offset + size]))
    offset += size
    result = {}
    for row in rows:
        if row['key'] in result:
            raise ValueError('Duplicate field')
        stop = offset + row['length']
        result[row['key']] = restore(blob[offset:stop], row)
        offset = stop
    if offset != len(blob):
        raise ValueError('Archive length mismatch')
    return result


def load_runtime_state(path):
    """Compatibility entry point returning the existing runtime state mapping."""
    from replay.online_codec_runtime import decode_online_state
    state = decode(Path(path).read_bytes())
    stream = io.BytesIO()
    np.savez(stream, **state)
    stream.seek(0)
    decoded = decode_online_state(stream)
    # Match the existing deployed delta/Q8 decoder's float32 history contract.
    decoded['train_raw'] = decoded['train_raw'].astype(np.float32)
    return decoded
