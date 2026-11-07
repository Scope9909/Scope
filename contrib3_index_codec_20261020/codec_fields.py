"""Field codecs for the retrieval cache (FCL), in persisted form.

Every codec here reproduces *exactly* the quantiser used in the in-memory
accounting runs (``run_fieldwise_global_cross.py`` /
``run_fieldwise_global_accounting.py``), so a decoded artifact replays the same
array the MAE was measured on.  No new approximation is introduced here.

Fields (per split):
    T    retrieved mean over the K candidates      (S, H, N, 1) float32
    E    exact-match payload (single history)      (S, H, N, 1) float32
    fd   dispersion of candidate futures around T  (S, H, N, 1) float32
    hd   mean history distance of selected cands   (S, N)      float32
    B    exact-available mask                      (S, H, N, 1) uint8 {0,1}

Codecs:
    T   -> per-node rank-R PCA code, float16            (LRC)
    E   -> shared bitmap + compacted fired values fp16  (SPC)
    fd  -> float16                                      (QSC)
    hd  -> per-node int16 + one float32 scale per node  (PNS-INT)
    B   -> bit-packed, and it is the *same* bitmap SPC indexes (no double cost)
"""

import numpy as np

HORIZON = 12
RANK = 8


# ---------------------------------------------------------------- T: LRC -----
def fit_t_basis(train_t, rank=RANK, min_valid=6, block=64):
    """Per-node PCA over the horizon axis; zero entries are treated as missing.

    ``train_t``: (S, H, N).  Returns mean (H, N) float32 and basis
    (N, H, rank) float32.  Node-blocked so PEMS07 never materialises a full
    centred copy.
    """
    n_samp, n_time, n_node = train_t.shape
    mean_f = np.empty((n_time, n_node), dtype=np.float32)
    basis = np.empty((n_node, n_time, rank), dtype=np.float32)
    for start in range(0, n_node, block):
        end = min(start + block, n_node)
        sub = np.ascontiguousarray(train_t[:, :, start:end])
        observed = sub != 0
        counts = observed.sum(axis=0).astype(np.float64)
        mean_b = np.zeros((n_time, end - start), dtype=np.float64)
        np.divide(np.where(observed, sub, 0.0).sum(axis=0), counts,
                  out=mean_b, where=counts > 0)
        mean_b = mean_b.astype(np.float32)
        centered = (np.where(observed, sub,
                             np.broadcast_to(mean_b, sub.shape)) - mean_b)
        ok = (observed.sum(axis=1) >= min_valid)
        centered *= ok[:, None, :].astype(np.float32)
        x = np.ascontiguousarray(np.transpose(centered, (2, 0, 1)))  # (b,S,H)
        cov = np.matmul(x.transpose(0, 2, 1), x).astype(np.float64)
        del x
        mean_f[:, start:end] = mean_b
        for j in range(end - start):
            vals, vecs = np.linalg.eigh(cov[j])
            order = np.argsort(vals)[::-1]
            basis[start + j] = vecs[:, order][:, :rank].astype(np.float32)
        del sub, observed, centered, cov
    return mean_f, basis


def encode_t(t, mean, basis):
    """(S, H, N) -> float16 codes (S, N, R)."""
    centered = np.asarray(t, dtype=np.float32) - mean[None]
    return np.einsum("shn,nhr->snr", centered, basis).astype(np.float16)


def decode_t(code, mean, basis):
    """(S, N, R) codes -> (S, H, N) float32."""
    recon = np.einsum("snr,nhr->shn", code.astype(np.float32), basis)
    return (recon + mean[None]).astype(np.float32)


# ---------------------------------------------------------------- B: bitmap --
def encode_bitmap(fired):
    return np.packbits(np.ascontiguousarray(fired.reshape(-1)))


def decode_bitmap(bits, count):
    return np.unpackbits(bits)[:count].astype(bool)


# ----------------------------------------------------------- E: SPC --------
def encode_e_sparse(e, fired):
    """Shared bitmap + compacted fired values (float16)."""
    values = np.asarray(e, dtype=np.float32)[fired].astype(np.float16)
    return values


def decode_e_sparse(values, fired, shape):
    out = np.zeros(int(np.prod(shape)), dtype=np.float32)
    out[fired.reshape(-1)] = values.astype(np.float32)
    return out.reshape(shape)


# ----------------------------------------------------------- fd: QSC --------
def encode_fd(fd):
    return np.asarray(fd, dtype=np.float32).astype(np.float16)


def decode_fd(code):
    return code.astype(np.float32)


# ------------------------------------------------------ fd: log-domain ------
# fd is a dispersion (std-like) quantity: non-negative, 0.33..451 on PEMS04/07,
# median ~18.  A uniform int8 grid therefore has a step of 1.77 and wipes out
# every small value -- precisely where the downstream squash() is most
# sensitive.  Log-domain coding makes the error *relative* and uniform instead,
# and preserves the exact zeros that the payload relies on.
_LOG2 = float(np.log(2.0))


def encode_fd_log(fd, bits=8):
    """Per-(h,node) log code.  Returns (uint8/uint16 code, float32 top)."""
    levels = (1 << bits) - 1
    dtype = np.uint8 if bits <= 8 else np.uint16
    x = np.asarray(fd, dtype=np.float32)
    top = np.maximum(x.max(axis=0), 1e-9)                      # (H, N)
    t = np.log1p(x / top[None]) / _LOG2
    c = np.clip(np.round(t * levels), 0, levels).astype(dtype)
    return c, top.astype(np.float32)


def decode_fd_log(code, top, bits=8):
    levels = (1 << bits) - 1
    t = code.astype(np.float32) / float(levels) * _LOG2
    return (top[None] * np.expm1(t)).astype(np.float32)


def encode_fd_int8(fd):
    """Per-(h,node) symmetric linear int8, for the fd ablation."""
    x = np.asarray(fd, dtype=np.float32)
    absmax = np.abs(x).max(axis=0)
    scale = np.where(absmax > 0, absmax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(x / scale[None]), -127, 127).astype(np.int8)
    return q, scale


def decode_fd_int8(q, scale):
    return (q.astype(np.float32) * scale[None]).astype(np.float32)


# ------------------------------------------------------- hd: PNS-INT --------
def encode_hd(hd, bits=16):
    """Symmetric per-node integer code + one float32 scale per node.

    fp16 is *not* usable here: raw values reach 1e5-6e5 and overflow 65504.
    Any bit width is accepted; the sensitivity ladder showed hd only needs 4
    bits on PEMS08 and can be dropped entirely on PEMS04, so the width is now
    a configuration knob rather than a constant.
    """
    levels = (1 << (bits - 1)) - 1
    dtype = np.int16 if bits > 8 else np.int8
    x = np.asarray(hd, dtype=np.float32)
    absmax = np.abs(x).max(axis=0)                       # (N,)
    scale = np.where(absmax > 0, absmax / float(levels), 1.0).astype(np.float32)
    q = np.clip(np.round(x / scale[None]), -levels, levels).astype(dtype)
    return q, scale


def decode_hd(q, scale):
    return (q.astype(np.float32) * scale[None]).astype(np.float32)
