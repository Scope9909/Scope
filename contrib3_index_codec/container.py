"""A self-describing package for the retrieval cache -- one file, no side info.

Why this module exists
----------------------
The earlier measurement scripts wrote ``np.savez_compressed(**cfg)`` and then
decoded with ``decode(cfg, ref)`` where ``ref`` was the *original* cache: the
fired mask was read from ``ref["B"]``, the E shape from ``ref["E"].shape``, the
fd shape from ``ref["fd"]``.  That is not a codec, it is a codec plus the thing
it is supposed to replace.  This module removes every such reference:

    pack = write_package(path, *encode_fields(fields, plan))
    fields2 = decode_fields(*read_package(path))     # path is the ONLY input

Design rules
------------
1. **Bits mean bits.**  Integer codes are packed at their exact width, so a
   "6-bit knee" costs 6 bits per value, not 8.  Without this, a ladder measured
   in bits cannot be turned into bytes honestly.
2. **The header carries everything the decoder needs**: shapes, dtypes, codec
   names, codec parameters, blob offsets, payload length and SHA-256.  A
   truncated or edited file fails loudly instead of decoding to garbage.
3. **One file, one xz stream.**  All blobs are concatenated and compressed
   together, so the file size on disk is comparable to any other package written
   by this module -- that is what makes the Phase-C baseline fair.

Field plan (what a configuration *is*)::

    {"T":  {"codec": "qint", "bits": 8}      # per-(H,N) scale, exact bit pack
     "E":  {"codec": "sparse"}               # bitmap + compacted fp16 payload
     "fd": {"codec": "log",  "bits": 4}      # per-(H,N) log code
     "hd": {"codec": "qint", "bits": 16}     # per-node scale
     "B":  {"codec": "bitmap"}}              # free: shares E's bitmap

Every field also accepts ``{"codec": "off"}`` (drop it: decode returns zeros)
and the numeric fields accept ``{"codec": "f16"}`` as an uncompressed-above-fp16
reference rung.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MAGIC = b"FCLPKG1\x00"          # 8 bytes, versioned in-band as well
FORMAT_VERSION = 1
HORIZON = 12


# --------------------------------------------------------------------- bits --
def pack_bits(u, bits):
    """Pack unsigned integers (< 2**bits) into bytes at exactly ``bits`` each."""
    flat = np.ascontiguousarray(np.asarray(u).reshape(-1))
    if bits == 8:
        return flat.astype(np.uint8)
    if bits == 16:
        return flat.astype(np.uint16)
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be 1..16, got {bits}")
    weights = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    bit_matrix = ((flat.astype(np.uint64)[:, None] >> weights) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(-1))


def unpack_bits(buf, count, bits):
    buf = np.asarray(buf)
    if bits == 8:
        return buf.astype(np.uint8)[:count]
    if bits == 16:
        return buf.astype(np.uint16)[:count]
    raw = np.unpackbits(np.asarray(buf, dtype=np.uint8))[: count * bits]
    matrix = raw.reshape(count, bits).astype(np.uint64)
    multipliers = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint64)
    return matrix @ multipliers


# ------------------------------------------------------------------- scalars --
def qint_encode(x, bits, global_scale=False):
    """Symmetric per-trailing-slice quantiser; returns (packed blob, scale).

    One scale per ``x.shape[1:]`` slice, i.e. per (h, node) for T/E/fd and per
    node for hd -- the same partition the in-memory runs used, so the decoded
    array is bit-identical to the one the earlier MAE numbers were measured on.
    """
    levels = (1 << (bits - 1)) - 1
    x = np.asarray(x, dtype=np.float32)
    if global_scale:
        absmax = np.float32(np.abs(x).max()).reshape(1)
    else:
        absmax = np.abs(x).max(axis=0)
    scale = np.where(absmax > 0, absmax / float(levels), 1.0).astype(np.float32)
    shape = (1,) * (x.ndim - scale.ndim) + scale.shape
    q = np.clip(np.round(x / scale.reshape(shape)), -levels, levels).astype(np.int32)
    return pack_bits(q + levels, bits), scale, levels


def qint_decode(blob, scale, bits, shape):
    levels = (1 << (bits - 1)) - 1
    u = unpack_bits(blob, int(np.prod(shape)), bits).astype(np.int32)
    q = (u - levels).reshape(shape)
    scale_shape = (1,) * (len(shape) - scale.ndim) + scale.shape
    return (q.astype(np.float32) * scale.reshape(scale_shape)).astype(np.float32)


def fd_log_encode(fd, bits):
    """Per-(h,node) log code: fd is a dispersion, log makes error relative."""
    levels = (1 << bits) - 1
    x = np.asarray(fd, dtype=np.float32)
    top = np.maximum(x.max(axis=0), 1e-9)
    t = np.log1p(x / top[None]) / float(np.log(2.0))
    c = np.clip(np.round(t * levels), 0, levels).astype(np.int32)
    return pack_bits(c, bits), top.astype(np.float32)


def fd_log_decode(blob, top, bits, shape):
    levels = (1 << bits) - 1
    c = unpack_bits(blob, int(np.prod(shape)), bits).astype(np.float32)
    t = c.reshape(shape) / float(levels) * float(np.log(2.0))
    return (top[None] * np.expm1(t)).astype(np.float32)


# --------------------------------------------------------------- field plans --
def encode_fields(fields, plan):
    """``fields`` -> (blobs, layout).  ``layout`` is JSON-safe."""
    S, H, N = fields["T"].shape
    fired = np.asarray(fields["B"]).reshape(S, H, N) > 0.5
    b_spec = dict(plan.get("B", {"codec": "bitmap"}))
    blobs = {}
    if b_spec["codec"] == "raw":
        # the incumbent layout: one uint8 per slot, no shared bitmap
        blobs["B_raw"] = np.asarray(fields["B"], np.uint8).reshape(S, H, N, 1)
    else:
        blobs["bitmap"] = np.packbits(np.ascontiguousarray(fired.reshape(-1)))
    layout = {"shape": [S, H, N], "horizon": H, "coders": {}}

    def coder(name):
        return dict(plan.get(name, {"codec": "off"}))

    # --- T -----------------------------------------------------------------
    spec = coder("T")
    code = spec["codec"]
    layout["coders"]["T"] = spec
    if code == "qint":
        blob, scale, _ = qint_encode(fields["T"], spec["bits"],
                                       spec.get("global", False))
        blobs["T_q"], blobs["T_scale"] = blob, scale
    elif code == "f16":
        blobs["T_f16"] = np.asarray(fields["T"], np.float32).astype(np.float16)
    elif code == "raw":
        blobs["T_raw"] = np.ascontiguousarray(fields["T"], np.float32)
    elif code == "r8":
        mean, basis = plan["_t_basis"]
        centered = np.asarray(fields["T"], np.float32) - mean[None]
        blobs["T_code"] = np.einsum("shn,nhr->snr", centered,
                                    basis).astype(np.float16)
        blobs["T_mean"], blobs["T_basis"] = mean, basis
    elif code != "off":
        raise ValueError(f"unknown T codec {code}")

    # --- E -----------------------------------------------------------------
    spec = coder("E")
    layout["coders"]["E"] = spec
    if spec["codec"] == "sparse":
        blobs["E_val"] = np.asarray(fields["E"], np.float32)[fired].astype(np.float16)
    elif spec["codec"] == "raw":
        blobs["E_raw"] = np.ascontiguousarray(fields["E"], np.float32)
    elif spec["codec"] == "qint":
        blob, scale, _ = qint_encode(fields["E"], spec["bits"],
                                       spec.get("global", False))
        blobs["E_q"], blobs["E_scale"] = blob, scale
    elif spec["codec"] != "off":
        raise ValueError(f"unknown E codec {spec['codec']}")

    # --- fd ----------------------------------------------------------------
    spec = coder("fd")
    layout["coders"]["fd"] = spec
    if spec["codec"] == "log":
        blobs["fd_q"], blobs["fd_top"] = fd_log_encode(fields["fd"], spec["bits"])
    elif spec["codec"] == "qint":
        # the naive baseline: one linear grid for a dispersion spanning 3 decades
        blob, scale, _ = qint_encode(fields["fd"], spec["bits"],
                                     spec.get("global", False))
        blobs["fd_qz"], blobs["fd_scale"] = blob, scale
    elif spec["codec"] == "f16":
        blobs["fd_f16"] = np.asarray(fields["fd"], np.float32).astype(np.float16)
    elif spec["codec"] == "raw":
        blobs["fd_raw"] = np.ascontiguousarray(fields["fd"], np.float32)
    elif spec["codec"] != "off":
        raise ValueError(f"unknown fd codec {spec['codec']}")

    # --- hd ----------------------------------------------------------------
    spec = coder("hd")
    layout["coders"]["hd"] = spec
    if spec["codec"] == "qint":
        blob, scale, _ = qint_encode(fields["hd"], spec["bits"],
                                       spec.get("global", False))
        blobs["hd_q"], blobs["hd_scale"] = blob, scale
    elif spec["codec"] == "f16":
        blobs["hd_f16"] = np.asarray(fields["hd"], np.float32).astype(np.float16)
    elif spec["codec"] == "raw":
        blobs["hd_raw"] = np.ascontiguousarray(fields["hd"], np.float32)
    elif spec["codec"] != "off":
        raise ValueError(f"unknown hd codec {spec['codec']}")

    layout["coders"]["B"] = b_spec
    return blobs, layout


def decode_fields(blobs, layout):
    """(blobs, layout) -> the five fields.  No external reference, by design."""
    S, H, N = layout["shape"]
    shape3 = (S, H, N)
    coders = layout["coders"]

    if "bitmap" in blobs:
        fired = np.unpackbits(np.asarray(blobs["bitmap"], np.uint8))[: S * H * N]
        fired = fired.astype(bool).reshape(shape3)
    else:
        fired = np.asarray(blobs["B_raw"]).reshape(shape3) > 0.5

    spec = coders["T"]
    if spec["codec"] == "qint":
        T = qint_decode(blobs["T_q"], blobs["T_scale"], spec["bits"], shape3)
    elif spec["codec"] == "f16":
        T = blobs["T_f16"].astype(np.float32).reshape(shape3)
    elif spec["codec"] == "raw":
        T = blobs["T_raw"].astype(np.float32).reshape(shape3)
    elif spec["codec"] == "r8":
        recon = np.einsum("snr,nhr->shn", blobs["T_code"].astype(np.float32),
                          blobs["T_basis"])
        T = (recon + blobs["T_mean"][None]).astype(np.float32)
    else:
        T = np.zeros(shape3, dtype=np.float32)

    spec = coders["E"]
    if spec["codec"] == "sparse":
        E = np.zeros(S * H * N, dtype=np.float32)
        E[fired.reshape(-1)] = blobs["E_val"].astype(np.float32)
        E = E.reshape(shape3)
    elif spec["codec"] == "raw":
        E = blobs["E_raw"].astype(np.float32).reshape(shape3)
    elif spec["codec"] == "qint":
        E = qint_decode(blobs["E_q"], blobs["E_scale"], spec["bits"], shape3)
    else:
        E = np.zeros(shape3, dtype=np.float32)

    spec = coders["fd"]
    if spec["codec"] == "log":
        fd = fd_log_decode(blobs["fd_q"], blobs["fd_top"], spec["bits"], shape3)
    elif spec["codec"] == "qint":
        fd = qint_decode(blobs["fd_qz"], blobs["fd_scale"], spec["bits"], shape3)
    elif spec["codec"] == "f16":
        fd = blobs["fd_f16"].astype(np.float32).reshape(shape3)
    elif spec["codec"] == "raw":
        fd = blobs["fd_raw"].astype(np.float32).reshape(shape3)
    else:
        fd = np.zeros(shape3, dtype=np.float32)

    spec = coders["hd"]
    if spec["codec"] == "qint":
        hd = qint_decode(blobs["hd_q"], blobs["hd_scale"], spec["bits"], (S, N))
    elif spec["codec"] == "f16":
        hd = blobs["hd_f16"].astype(np.float32).reshape(S, N)
    elif spec["codec"] == "raw":
        hd = blobs["hd_raw"].astype(np.float32).reshape(S, N)
    else:
        hd = np.zeros((S, N), dtype=np.float32)

    if "B_raw" in blobs:
        B = np.asarray(blobs["B_raw"], np.uint8).reshape(shape3 + (1,))
    else:
        B = fired.astype(np.float32)[..., None]
    return dict(T=T, E=E, fd=fd, hd=hd, B=B)


# ------------------------------------------------------------------- package --
def write_package(path, blobs, meta, preset=6):
    """Write one xz-compressed, self-describing file.  Returns its size."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = sorted(blobs)
    header = dict(meta)
    header.update(format="FCLPKG", version=FORMAT_VERSION,
                  compression=f"xz:preset={preset}",
                  built_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  blobs={})
    payload = bytearray()
    for name in order:
        a = np.ascontiguousarray(blobs[name])
        header["blobs"][name] = dict(dtype=a.dtype.str, shape=list(a.shape),
                                     offset=len(payload), nbytes=a.nbytes)
        payload += a.tobytes()
    header["payload_bytes"] = len(payload)
    header["payload_sha256"] = hashlib.sha256(bytes(payload)).hexdigest()
    head = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body = struct.pack("<Q", len(head)) + head + bytes(payload)
    path.write_bytes(MAGIC + lzma.compress(body, format=lzma.FORMAT_XZ, preset=preset))
    return path.stat().st_size


def read_package(path, verify=True):
    raw = Path(path).read_bytes()
    if raw[: len(MAGIC)] != MAGIC:
        raise ValueError(f"{path}: not an FCLPKG file")
    body = lzma.decompress(raw[len(MAGIC):])
    (head_len,) = struct.unpack("<Q", body[:8])
    header = json.loads(body[8: 8 + head_len].decode("utf-8"))
    if header.get("version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported version {header.get('version')}")
    payload = body[8 + head_len:]
    if verify:
        if len(payload) != header["payload_bytes"]:
            raise ValueError(f"{path}: payload length mismatch "
                             f"({len(payload)} != {header['payload_bytes']})")
        if hashlib.sha256(payload).hexdigest() != header["payload_sha256"]:
            raise ValueError(f"{path}: payload checksum mismatch")
    blobs = {}
    for name, d in header["blobs"].items():
        a = np.frombuffer(payload, dtype=np.dtype(d["dtype"]),
                          count=int(np.prod(d["shape"])) if d["shape"] else 1,
                          offset=d["offset"])
        blobs[name] = a.reshape(d["shape"]) if d["shape"] else a
    return blobs, header


def package_fields(path, verify=True):
    """Convenience: package file -> decoded five fields + header."""
    blobs, header = read_package(path, verify=verify)
    return decode_fields(blobs, header["layout"]), header


# --------------------------------------------------------------- index codec --
def encode_index(arr, groups, bits, sentinel, query_tod):
    """(Q,N,K,1) int32 endpoints -> packed TOD-offset codes (self-contained).

    ``groups`` maps tod -> sorted array of legal training endpoints.  The group
    table is written into the package (tens of KB) so that decoding never needs
    ``idx_train`` or the training split -- otherwise the index "codec" would
    again be a codec plus its own input.
    """
    a = np.asarray(arr)
    arr2 = a[..., 0] if a.ndim == 4 else a
    position_table = np.full(int(arr2.max()) + 2, -1, dtype=np.int64)
    for endpoints in groups.values():
        position_table[endpoints] = np.arange(endpoints.size, dtype=np.int64)
    positions = np.where(arr2 < 0, sentinel, position_table[np.maximum(arr2, 0)])
    if np.any(positions > sentinel):
        raise AssertionError("group position exceeds the sentinel range")
    code = positions.reshape(arr2.shape[0], -1)
    tods = np.array(sorted(groups), dtype=np.int64)
    offsets = np.array([0] + list(np.cumsum([groups[int(t)].size for t in tods])),
                       dtype=np.int64)
    endpoints_all = np.concatenate([groups[int(t)] for t in tods]).astype(np.int32)
    blobs = dict(codes=pack_bits(code, bits),
                 group_tods=tods.astype(np.int32),
                 group_offsets=offsets.astype(np.int32),
                 group_endpoints=endpoints_all,
                 query_tod=np.asarray(query_tod, dtype=np.int16))
    return blobs, dict(kind="candidate_index", bits=int(bits),
                       sentinel=int(sentinel), shape=list(arr2.shape))


def decode_index(blobs, layout):
    """Packed codes -> absolute endpoints (Q,N,K,1).  File-in, array-out."""
    bits, sentinel = int(layout["bits"]), int(layout["sentinel"])
    shape = layout["shape"]                                # [Q, N, K]
    flat = unpack_bits(blobs["codes"], int(np.prod(shape)), bits).reshape(shape)
    tods = np.asarray(blobs["group_tods"], dtype=np.int64)
    offs = np.asarray(blobs["group_offsets"], dtype=np.int64)
    ends = np.asarray(blobs["group_endpoints"], dtype=np.int64)
    query_tod = np.asarray(blobs["query_tod"], dtype=np.int64)
    out = np.full(shape, -1, dtype=np.int32)
    for j, tod in enumerate(tods):
        rows = np.flatnonzero(query_tod == tod)
        if rows.size == 0:
            continue
        members = ends[offs[j]: offs[j + 1]]
        block = flat[rows]
        valid = block != sentinel
        if np.any(block[valid] >= members.size):
            raise AssertionError(f"TOD={tod}: offset outside its group")
        out[rows] = np.where(valid, members[np.where(valid, block, 0)], -1)
    return out[..., None]
