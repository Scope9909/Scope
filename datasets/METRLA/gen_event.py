"""
METRLA event 生成：支持两种设计
- use_traffic_event=True：基于流量的 event（滚动波动/偏离），更贴近预测难度
- use_traffic_event=False：基于时间的 event（高峰/非高峰 + 节点噪声），原逻辑
"""
import numpy as np
import os
from datetime import datetime

# 改为 True 使用「流量相关」event，更利于动态门控降低 MAE
USE_TRAFFIC_EVENT = True

data_dir = os.path.dirname(__file__) or "."
data = np.load(os.path.join(data_dir, "data.npz"))["data"]   # (T, N, C)
B, N, _ = data.shape
traffic = data[..., 0]   # (B, N)

if USE_TRAFFIC_EVENT:
    # 改进 event：用「滚动标准差」表示当前时刻波动，波动大时 event 高，更利于门控
    window = 12
    pad = np.pad(traffic, ((window // 2, window // 2), (0, 0)), mode="edge")
    rolling_std = np.zeros_like(traffic)
    for i in range(B):
        win = pad[i : i + window, :]
        rolling_std[i] = np.std(win, axis=0)
    s_min, s_max = rolling_std.min(), rolling_std.max()
    if s_max > s_min:
        event = (rolling_std - s_min) / (s_max - s_min)
    else:
        event = np.zeros_like(traffic)
    event = np.clip(event, 0, 1).astype(np.float32)[..., np.newaxis]   # (B, N, 1)
else:
    # 原设计：时间高峰 + 节点噪声
    event = np.zeros((B, N, 1))
    start_date = np.datetime64("2012-03-01T00:00")
    delta_5min = np.timedelta64(5, "m")
    dates = start_date + delta_5min * np.arange(B)
    rng = np.random.default_rng(42)
    for b in range(B):
        dow = dates[b].astype(object).weekday()
        hour = dates[b].astype(object).hour
        if (dow == 0 and 7 <= hour <= 9) or (dow == 4 and 17 <= hour <= 19):
            base = 1.0
        else:
            base = 0.0
        noise = rng.normal(loc=base, scale=0.15, size=N)
        event[b, :, 0] = np.clip(noise, 0, 1)

# 保存 event 到 event.npz（供 data_prepare 单独加载）
np.savez(os.path.join(data_dir, "event.npz"), event=event.astype(np.float32))
print("OK: saved event.npz (use_traffic_event=%s)" % USE_TRAFFIC_EVENT)

# 4. 可选：若 data 尚未含事件维，可升级为 4 维（此处不覆盖，保留原始 data.npz）
if data.shape[-1] == 4:
    print("OK: data already 4-dim")
else:
    std_t = event.std(axis=(1, 2))
    print("事件标准差（时间轴）前 10 样本：", std_t[:10])
    print("全局平均 std：", std_t.mean())