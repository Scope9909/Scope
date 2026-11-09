"""Summarize completed Test confirmation without rerunning inference."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    cli = ap.parse_args()
    result = json.loads((cli.directory/'results.json').read_text(encoding='utf-8'))
    if not result['complete'] or len(result['results']) != 4:
        raise ValueError('All four completed evaluations are required')
    rows = result['results']
    lines = [
        '# 新无损编码：Test 实测复核', '',
        '本次按用户要求重新进行 Test 推理。它是已有 Test 评估后的无损编码复核，不是新的独立盲测。', '',
        'Q8 节点、模型 checkpoint 和校准参数保持原设置；每个字段的预测器、布局和压缩器由 Validation 固定。'
        'Test 不参与节点、阈值、压缩器或精度方案选择。校准只在 Validation 上拟合，Test 仅应用。', '',
        '| 数据集 | Test raw MAE | Test 校准 MAE | 校准 MAE 相对旧结果变化 | 新包 B | 相对原 NPZ 基线节省 | 相对同容器通用基线节省 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    totals = dict(selected=0, exact=0, generic=0, legacy=0)
    for r in rows:
        size = r['package_bytes']['selected']
        path = cli.directory / f"{r['dataset'].lower()}_test_selected.alsc"
        data = path.read_bytes()
        assert len(data) == size and hashlib.sha256(data).hexdigest() == r['package_sha256']
        assert r['runtime_state_bitwise_equal']
        old_saving = 100*(1-size/r['legacy_baseline_bytes'])
        matched_saving = 100*(1-size/r['package_bytes']['generic'])
        lines.append(f"| {r['dataset']} | {r['test_raw_metrics']['mae']:.6f} | "
                     f"{r['test_calibrated_metrics']['mae']:.6f} | {r['calibrated_mae_change_vs_previous']:+.10f} | "
                     f"{size:,} | {old_saving:.2f}% | {matched_saving:.2f}% |")
        for k in ('selected', 'exact', 'generic'):
            totals[k] += r['package_bytes'][k]
        totals['legacy'] += r['legacy_baseline_bytes']
    old_pct = 100*(1-totals['selected']/totals['legacy'])
    generic_pct = 100*(1-totals['selected']/totals['generic'])
    lines.extend([
        '', f"总字节：新方案 {totals['selected']:,} B，原 NPZ 基线 {totals['legacy']:,} B，"
        f"同容器直接通用压缩基线 {totals['generic']:,} B。两种节省率分别为 **{old_pct:.2f}%** 和 **{generic_pct:.2f}%**。", '',
        f"新编码的全精确对照为 {totals['exact']:,} B；冻结 Q8 状态相对它额外节省 "
        f"{totals['exact']-totals['selected']:,} B。总节省不能全部归因于 Q8。", '',
        '同容器通用基线：精确状态直接使用在 Validation 上选择的 C/F 布局和 raw/zlib/XZ/bz2 压缩器，'
        '不进行时间差分或字节/位平面重排。旧 NPZ 基线不应再称为最强通用压缩基线。', '',
        '每个新包都与原冻结 Test 包比较了完整解码状态，字段集合、dtype、shape 和数组字节完全相同。'
        '本次还重新执行了 Validation/Test 推理；样本数以真实时间索引为准，不计 batch 补齐重复样本。', '',
        '结果文件：`results.json`；冻结编码规则和旧 checkpoint/校准哈希：`frozen_recipes.json`；'
        '文件大小及状态校验：`packages.json`。原有 Test 记录保留。',
    ])
    max_delta = max(abs(r[k]) for r in rows for k in ('raw_mae_change_vs_previous', 'calibrated_mae_change_vs_previous'))
    lines.extend(['', f'相对旧 Test 记录，raw/校准 MAE 的最大绝对变化为 {max_delta:.12g}。'])
    (cli.directory/'REVIEW.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
