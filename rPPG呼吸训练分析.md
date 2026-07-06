# rPPG 三次深呼吸训练评估方案

适用场景：严肃游戏中引导老年人进行 3 次深呼吸（语音提示吸气/呼气）。

目标：

- 判断是否完成 3 次“有效深呼吸”。
- 给出呼吸节奏、呼吸质量与完成状态。

---

## 1. 输入与范围

- 输入：rPPG 推理区间内的时序信号（使用 detrended 信号）。
- 区间：以用户选定的起止时间为分析窗口。

---

## 2. 呼吸信号提取

对 rPPG 进行低频带通：

```
0.08 ~ 0.5 Hz （约 5~30 次/分钟）
```

得到 `resp_signal`，用于后续呼吸周期检测。

---

## 3. 呼吸周期识别

从 `resp_signal` 里检测峰/谷：

- 相邻峰距限制：约 1.5 秒以上，避免过密噪声。
- 以“峰到峰”为一个呼吸周期。

有效周期范围：

```
3 ~ 10 秒
```

---

## 4. 深呼吸判定

对每个周期计算振幅（峰值 − 谷值）：

```
amplitude = |peak - trough|
```

阈值：

```
threshold = median(amplitude) + 0.5 * std(amplitude)
```

满足条件：

- 周期在 3~10 秒
- 振幅 ≥ threshold

即计为一次“有效深呼吸”。

---

## 5. 合规判定

```
deep_breath_count >= 3 -> complete
deep_breath_count >= 1 -> partial
否则 -> insufficient
```

如果 `resp_quality < 0.35`，直接标记为 `low_quality`。

---

## 6. 输出指标

### summary

- `resp_bpm`：呼吸频率（BPM）
- `resp_quality`：呼吸峰值质量（0~1）
- `breath_count`：有效周期数（3~10s）
- `deep_breath_count`：有效深呼吸次数
- `deep_breath_target`：目标次数（3）
- `cycle_mean_sec` / `cycle_std_sec`：周期均值/标准差
- `depth_mean` / `depth_std`：振幅均值/标准差
- `depth_threshold`：深呼吸阈值
- `compliance`：complete / partial / insufficient / low_quality
- `deep_breath_times`：深呼吸发生时间点（秒）

### series

- `time_sec`：时间轴
- `signal`：呼吸波形

---

## 7. 展示建议

- 面板显示：
  - 深呼吸完成次数 / 目标次数
  - 呼吸率、节奏（周期均值）与质量
- 曲线显示：呼吸波形，支持查看关键段

---

## 8. 解释与注意事项

- 该方案不依赖语音提示内容，仅基于生理信号判断完成度。
- 若区间内说话/运动较多，会导致 `resp_quality` 下降。
- 适合老年人场景：节奏范围宽、短时波动不会直接否定。
