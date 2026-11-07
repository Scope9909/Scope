"""build_candidate_relative_residual_cache.py

构建 "candidate-specific residual construction" 的检索缓存.

背景
----
部署流水线中低秩适配器的组合公式为

    out = P + g * (R - P)

其中 P = 冻结父模型(backbone)输出, R = 检索缓存 channel 0, g = 低秩门.

当前构造:  R = M      = agg_i( Y_i )              (候选未来轨迹的均值)
            out_cur    = P + g * (M - P)

新构造(本脚本): 候选各自先跑一次冻结 parent 得到 Ŷ_i, 迁移的是
            R_i = Y_i - Ŷ_i                       (候选相对自身预测的残差)
            agg_i(R_i) = M - M_hat,  M_hat = agg_i(Ŷ_i)
            out_new    = P + g * (M - M_hat)

由于运行时 R 只能来自缓存, 我们把缓存 channel 0 设为

    T = M + (P_hat - M_hat)

则 out = P + g * (T - P) = P + g * (M - M_hat)   (当 P_hat == P, 见下方校验)

即净效应相对当前构造为  + g * (P - M_hat):
把检索注入项的基线从 "query 自己的预测" 换成 "候选的平均预测",
注入量变为 level-free 的候选残差.

严格性保证
----------
* P_hat: 由 dump_backbone_predictions.py 在 joint3 ckpt 上 dump, 与运行时
  main_output 同源(同一份权重、同样关掉 4 个 analog adapter、eval 模式),
  且逐 split 对齐 idx_*.
* 候选索引是全局时间索引, 需映射回 idx_train 位置才能查 backbone_pred_train;
  本脚本用 pos_of_global 查表并断言全部命中.
* 候选有效性 mask 复现 mean 缓存 builder 的语义:
  (cand >= 0) & (候选未来 != 0) & (train split 排除自身端点).
  该 mask 会先用 "重建 M 并与全空间缓存逐位比对" 的方式验证.
* T 用与 canonical 缓存完全相同的冻结 PCA basis 编码; 编码管线会先用
  M_full 重编码并与原 retrieval_code_* 比对, 作为端到端校验.

输出
----
与 latc_pca_r8 缓存结构完全一致, 仅 retrieval_code_* 被替换,
其余数组(idx/exact/quality/basis/元数据)逐位拷贝.
"""
import argparse
import os

import numpy as np
import torch

INPUT_LENGTH = 12
PREDICT_LENGTH = 12
CHUNK = 256


def build_pos_lookup(idx_train):
    """global time index -> position inside idx_train (-1 if absent)."""
    size = int(idx_train.max()) + 2
    pos = np.full(size, -1, dtype=np.int64)
    pos[idx_train] = np.arange(len(idx_train), dtype=np.int64)
    return pos


def gather_candidate_future(data, cand, scaler_mean, scaler_std, chunk=CHUNK):
    """Y_i for every (query, slot, node, horizon), plus a raw-space mask.

    cand: (Q, K, N) global endpoints, -1 invalid.
    returns (values_z, nonzero_mask):
      values_z      -- (Q, K, N, H) float32 in global z-score space
      nonzero_mask  -- (Q, K, N, H) bool, True where the RAW value is not the
                       dataset's missing marker (0).  Must be judged before
                       normalization because raw 0 maps to a nonzero z value.
    """
    q, k, n = cand.shape
    out = np.empty((q, k, n, PREDICT_LENGTH), dtype=np.float32)
    nz = np.empty((q, k, n, PREDICT_LENGTH), dtype=bool)
    y_offsets = np.arange(1, PREDICT_LENGTH + 1, dtype=np.int64)
    node_grid = np.broadcast_to(
        np.arange(n, dtype=np.int64)[None, None, :], (1, k, n)
    )
    for s in range(0, q, chunk):
        e = min(s + chunk, q)
        c = cand[s:e].astype(np.int64)
        future_time = c[..., None] + y_offsets[None, None, None, :]
        nodes = np.broadcast_to(node_grid, (e - s, k, n))
        nodes = np.broadcast_to(
            nodes[..., None], (e - s, k, n, PREDICT_LENGTH)
        )
        block = data[future_time, nodes, 0].astype(np.float32, copy=False)
        nz[s:e] = block != 0
        out[s:e] = block
    # The mean-cache builder aggregates raw futures and only normalizes at the
    # end, so everything downstream works in global z-score space.
    out = (out - scaler_mean) / scaler_std
    return out.astype(np.float32, copy=False), nz


def gather_candidate_parent(pred_train, cand, pos_lookup, chunk=CHUNK):
    """Ŷ_i for every (query, slot, node, horizon) in z-space.

    pred_train: (T_train, H, N, 1) backbone predictions aligned to idx_train.
    cand: (Q, K, N) global endpoints, -1 invalid.
    returns (Q, K, N, H, 1) float32; invalid slots -> 0.
    """
    q, k, n = cand.shape
    out = np.empty((q, k, n, PREDICT_LENGTH, 1), dtype=np.float32)
    node_grid = np.broadcast_to(
        np.arange(n, dtype=np.int64)[None, None, :], (1, k, n)
    )
    for s in range(0, q, chunk):
        e = min(s + chunk, q)
        c = cand[s:e].astype(np.int64)
        safe = np.where(c >= 0, c, 0)
        pos = pos_lookup[safe]
        bad = (c < 0) | (pos < 0)
        pos = np.where(bad, 0, pos)
        nodes = np.broadcast_to(node_grid, (e - s, k, n))
        # advanced indexing: (q,k,n) index arrays -> (q,k,n,H,1)
        vals = pred_train[pos, :, nodes, :]
        vals = np.where(bad[..., None, None], 0.0, vals)
        out[s:e] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='PEMS04')
    ap.add_argument('--tag', default='12to12')
    ap.add_argument(
        '--mean-cache',
        default='datasets/PEMS04/PEMS04_12to12_analog_k8_exact_mean.npz',
    )
    ap.add_argument(
        '--pca-cache',
        default='datasets/PEMS04/'
                'PEMS04_12to12_analog_k8_exact_mean_latc_pca_r8_fp16.npz',
    )
    ap.add_argument(
        '--candidate-cache',
        default='datasets/PEMS04/PEMS04_12to12_analog_k8_candidate_indices.npz',
    )
    ap.add_argument(
        '--backbone-dump',
        default='datasets/PEMS04/PEMS04_12to12_backbone_predictions_joint3.npz',
    )
    ap.add_argument(
        '--output',
        default='datasets/PEMS04/'
                'PEMS04_12to12_analog_k8_exact_mean_candrel_pca_r8_fp16.npz',
    )
    ap.add_argument(
        '--construction',
        choices=('candrel', 'damp'),
        default='candrel',
        help='candrel: T = M + (P_hat - M_hat)  ->  out = P + g*(M - M_hat). '
             'damp:    T = P_hat + alpha*(M - P_hat)  ->  out = P + alpha*g*(M - P); '
             'this is the pure-magnitude control that isolates shrinkage from '
             'residual-transfer information.',
    )
    ap.add_argument('--alpha', type=float, default=0.5215)
    ap.add_argument(
        '--beta', type=float, default=1.0,
        help='candrel only: scales the transferred residual.',
    )
    args = ap.parse_args()

    ds = args.dataset
    index = np.load(f'datasets/{ds}/{ds}_{args.tag}_index_scaler.npz')
    idx = {s: np.asarray(index[f'idx_{s}']) for s in ('train', 'val', 'test')}
    scaler_mean = float(index['scaler_mean'])
    scaler_std = float(index['scaler_std'])

    from utils.datasets_utils import file_path_dict, load_raw_data
    raw, _ = load_raw_data(file_path_dict[ds])
    raw = raw[..., :1].astype(np.float32, copy=False)
    print(f'raw data: {raw.shape}')

    mean_cache = np.load(args.mean_cache)
    pca = np.load(args.pca_cache)
    cand_cache = np.load(args.candidate_cache)
    dump = np.load(args.backbone_dump)

    temporal_mean = np.asarray(pca['retrieval_temporal_mean'], dtype=np.float32)
    temporal_basis = np.asarray(
        pca['retrieval_temporal_basis'], dtype=np.float32
    )
    print(f'basis {temporal_basis.shape}, mean {temporal_mean.shape}')

    pos_lookup = build_pos_lookup(idx['train'])

    # ---------- sanity: backbone dump alignment ----------
    for s in ('train', 'val', 'test'):
        assert np.array_equal(
            np.asarray(dump[f'backbone_pred_{s}']).shape[:1],
            (len(idx[s]),),
        ), f'backbone dump {s} length mismatch'
        assert np.array_equal(
            np.asarray(mean_cache[f'idx_{s}']), idx[s]
        ), f'mean cache idx_{s} mismatch'
        assert np.array_equal(
            np.asarray(pca[f'idx_{s}']), idx[s]
        ), f'pca cache idx_{s} mismatch'
        assert np.array_equal(
            np.asarray(cand_cache[f'idx_{s}']), idx[s]
        ), f'candidate cache idx_{s} mismatch'

    out_arrays = {}
    for k in pca.files:
        out_arrays[k] = pca[k]

    report = {}
    for split in ('train', 'val', 'test'):
        print(f'===== {split} =====')
        cand = np.asarray(cand_cache[f'candidate_index_{split}'])
        q, kk, n = cand.shape
        m_full = np.asarray(mean_cache[f'retrieval_{split}'], dtype=np.float32)
        p_hat = np.asarray(dump[f'backbone_pred_{split}'], dtype=np.float32)

        # ---- candidate futures (z-space) and raw nonzero mask ----
        y_future, y_nonzero = gather_candidate_future(
            raw, cand, scaler_mean, scaler_std
        )
        if split == 'train':
            self_hit = (cand == idx['train'][:, None, None])
        else:
            self_hit = np.zeros_like(cand, dtype=bool)

        # ---- candidate parent predictions ----
        y_hat = gather_candidate_parent(
            np.asarray(dump['backbone_pred_train'], dtype=np.float32),
            cand,
            pos_lookup,
        )  # (Q,K,N,H,1)

        # ---- validity mask, mirroring compute_analog_retrieval_cache ----
        # builder: selected_valid = isfinite(nearest_distance)[...,None]
        #                           & (selected_future != 0)     -> (Q,K,N,H)
        #          selected_count = selected_valid.sum(axis=1)    -> (Q,N,H)
        #          count==0 -> tod fallback;  stored as transpose(0,2,1)
        # train split additionally excludes the query's own endpoint (LOO).
        valid = (cand >= 0)[..., None]                    # (Q,K,N,1)
        valid = np.broadcast_to(valid, y_future.shape) & y_nonzero
        if split == 'train':
            valid = valid & (~self_hit[..., None])
        count = valid.sum(axis=1, dtype=np.float64)       # (Q,N,H)

        # ---- AUDIT 1: rebuild M with this mask and compare to the cache ----
        m_rebuilt_nh = np.divide(
            np.where(valid, y_future, 0.0).sum(axis=1, dtype=np.float64),
            np.maximum(count, 1.0),
        )
        m_rebuilt = m_rebuilt_nh.transpose(0, 2, 1).astype(np.float32)
        err = np.abs(m_rebuilt - m_full[..., 0])
        # count==0 entries legitimately differ: the builder substitutes a
        # time-of-day fallback there, which this rebuild does not model.
        has_evidence = (count > 0).transpose(0, 2, 1)
        err_ev = np.abs(m_rebuilt - m_full[..., 0])[has_evidence]
        mask_audit = float(err.max())
        mask_audit_ev = float(err_ev.max()) if err_ev.size else 0.0
        frac_close = float((err < 0.02).mean())
        print(f'  audit1 rebuild-M max|err| = {mask_audit:.6f} '
              f'(frac < 0.02: {frac_close:.6f})')
        print(f'  audit1 (count>0 only) max|err| = {mask_audit_ev:.3e}')

        # ---- M_hat: mean of candidate parent predictions, same mask ----
        y_hat_nh = y_hat[..., 0]                          # (Q,K,N,H)
        m_hat_nh = np.divide(
            np.where(valid, y_hat_nh, 0.0).sum(axis=1, dtype=np.float64),
            np.maximum(count, 1.0),
        )
        # no evidence -> keep the construction inert (T = M) instead of
        # inheriting the cache's tod fallback, which is unrelated to P.
        p_hat_nh = p_hat[..., 0].transpose(0, 2, 1)       # (Q,N,H)
        no_evidence = (count <= 0)
        m_hat_nh = np.where(no_evidence, p_hat_nh, m_hat_nh)
        m_hat = m_hat_nh.transpose(0, 2, 1)[..., None].astype(np.float32)
        print(f'  no-evidence (Q,N,H) entries: {int(no_evidence.sum())} '
              f'of {no_evidence.size}')

        # ---- new target ----
        if args.construction == 'candrel':
            # out = P + beta * g * (M - M_hat);  beta=1 is the plain formula.
            t_full = p_hat + args.beta * (m_full - m_hat)
        else:
            # Pure damping control: identical semantics, no candidate-specific
            # parent prediction involved.  alpha is chosen so the injected
            # magnitude matches the candrel term.
            t_full = p_hat + args.alpha * (m_full - p_hat)
        delta = t_full - m_full
        print(f'  |T - M|: mean={float(np.abs(delta).mean()):.4f} '
              f'max={float(np.abs(delta).max()):.4f} '
              f'(z units; raw x{scaler_std:.1f})')

        # ---- AUDIT 2: encoding pipeline reproduces original codes on M ----
        resid = m_full - temporal_mean[None]
        code_m = np.einsum(
            'bhnc,nhr->brnc',
            resid.astype(np.float32),
            temporal_basis.astype(np.float32),
        ).astype(np.float16)
        code_ref = np.asarray(pca[f'retrieval_code_{split}'])
        code_err = float(
            np.abs(code_m.astype(np.float32) - code_ref.astype(np.float32)).max()
        )
        print(f'  audit2 re-encode(M) vs original code max|err| = {code_err:.6f}')

        # ---- encode T with the same frozen basis ----
        resid_t = t_full - temporal_mean[None]
        code_t = np.einsum(
            'bhnc,nhr->brnc',
            resid_t.astype(np.float32),
            temporal_basis.astype(np.float32),
        ).astype(np.float16)
        out_arrays[f'retrieval_code_{split}'] = code_t

        report[split] = {
            'audit_rebuild_M_max_err': mask_audit,
            'audit_rebuild_M_frac_lt_0p02': frac_close,
            'audit_rebuild_M_max_err_count_gt_0': mask_audit_ev,
            'audit_reencode_M_code_max_err': code_err,
            'no_evidence_entries': int(no_evidence.sum()),
            'abs_T_minus_M_mean_z': float(np.abs(delta).mean()),
            'abs_T_minus_M_max_z': float(np.abs(delta).max()),
        }
        np.save(
            f'_candrel_diag_{split}.npy',
            np.stack([m_full[..., 0], m_hat[..., 0], p_hat[..., 0]], axis=0),
        )

    out_arrays['residual_construction'] = np.array(
        'candidate_specific_parent_residual'
    )
    out_arrays['residual_construction_note'] = np.array(
        'channel0 = M + (P_hat - M_hat); lowrank yields P + g*(M - M_hat)'
    )

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + '.tmp.npz'
    np.savez_compressed(tmp, **out_arrays)
    os.replace(tmp, out_path)
    print(f'\nSaved: {out_path}')
    for s, r in report.items():
        print(f'  {s}: {r}')


if __name__ == '__main__':
    main()
