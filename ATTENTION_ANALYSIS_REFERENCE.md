# 注意力分析方案参考（OpenFace 视线 + 闭眼鲁棒处理）

本文档用于沉淀当前项目的注意力分析思路，重点解决：**闭眼或跟踪不稳定时，OpenFace 仍输出 gaze 导致角度异常** 的问题。

## 1. 问题与目标

- 现象：闭眼、遮挡、快速转头时，`gaze_angle_x/y` 可能出现跳变或极端值。
- 风险：直接用 `sqrt(gaze_angle_x^2 + gaze_angle_y^2)` 会把“不可用帧”误判为严重走神。
- 目标：先做有效性判定与清洗，再进行专注力指标统计与评分。

## 2. 可用字段（来自 OpenFace CSV）

建议优先使用以下字段：

- `success`：人脸跟踪是否成功。
- `confidence`：跟踪置信度。
- `gaze_angle_x`, `gaze_angle_y`：视线角（弧度）。
- `AU45_c` 或 `AU45_r`：眨眼/闭眼相关 AU。
- `timestamp`：统一时间轴（本项目主时间轴）。

可选增强字段：

- `x_*`, `y_*`：可估计眼部开合（EAR）或做额外质量检查。

## 3. 状态判定（逐帧）

将每帧判定为四类状态：

- `valid`：可用于注意力角度统计。
- `blink`：短时眨眼，不应当直接扣专注分。
- `eye_closed`：闭眼但不单独计分，主要用于无效帧门控。
- `track_lost`：跟踪失败或低置信度，不参与角度统计。

推荐默认阈值（首版可直接落地）：

- `track_ok = (success == 1) and (confidence >= 0.85)`
- `eye_closed = (AU45_c == 1) or (AU45_r >= 1.0)`
- `raw_angle = sqrt(gaze_angle_x^2 + gaze_angle_y^2)`
- `outlier = (raw_angle > 1.2) or (abs(delta_angle) > 0.35)`

最终有效帧：

- `valid = track_ok and (not eye_closed) and (not outlier)`

## 4. 闭眼分段与事件识别

对连续 `eye_closed == True` 的片段做时长判定：

- `80ms ~ 400ms`：记为 `blink`（生理眨眼）。
- 其余更长闭眼片段当前不单独统计，只保留 `eye_closed` 状态。

说明：

- 时长使用 `timestamp` 计算，不依赖固定 FPS。
- `blink` 片段不参与“走神时间”累计，避免误判。

## 5. 角度清洗与平滑

建议输出两条序列：

- `attention_angle_raw`：原始角度（保留用于排查）。
- `attention_angle_clean`：门控 + 平滑后的有效角度（用于评估与评分）。

处理步骤：

1. 非 `valid` 帧置为 `NaN`。
2. 仅对短缺口插值（如 `<= 0.2s`），长缺口保持 `NaN`。
3. 对有效段先做局部尖峰抑制，再做 Savitzky-Golay 平滑。

这样可去掉闭眼尖峰和偶发跳变，同时保留真实视线变化趋势。

当前项目实现已改为：

- 尖峰窗口：`10`（左右各 5 帧）
- Savitzky-Golay：`window=7`，`polyorder=2`

## 6. 注意力指标（建议）

以下指标都基于 `attention_angle_clean` 与状态标签：

- `valid_ratio`：有效帧时间占比。
- `focus_ratio`：`angle <= 0.25rad` 的时间占比（专注）。
- `acceptable_focus_ratio`：`0.25 ~ 0.45rad` 时间占比（可接受专注）。
- `off_focus_ratio`：`> 0.45rad` 时间占比（偏离）。
- `stability_std`：角度标准差（越小越稳定）。
- `reengage_time_median`：偏离后恢复到专注阈值的中位时长。
- `screen_focus_duration_median_sec`：非眨眼连续屏幕内注视片段的中位时长。
- `screen_focus_duration_max_sec`：非眨眼连续屏幕内注视片段的最大时长。
- `screen_focus_segment_count`：非眨眼连续屏幕内注视片段数量。
- `blink_count_per_min`：每分钟眨眼次数。
- `blink_excess_ratio`：超出正常上界后的额外眨眼时长占比。

## 7. 评分建议（0-100）

建议采用可解释的组合分：

- `FocusScore = 100 * (0.75 * focus_ratio + 0.25 * acceptable_focus_ratio)`
- `FixationScore = 100 * clip(screen_focus_duration_median_sec / 2.0, 0, 1)`
- `BlinkScore = 100 * (1 - clip((blink_ratio - 0.08) / 0.12, 0, 1))`
- `StabilityScore = 100 * clip((0.30 - stability_std) / 0.30, 0, 1)`
- `QualityScore = 100 * clip((valid_ratio - 0.35) / 0.65, 0, 1)`

总分：

- `AttentionScore = 0.40 * FocusScore + 0.15 * FixationScore + 0.10 * BlinkScore + 0.15 * StabilityScore + 0.20 * QualityScore`

数据质量门槛：

- 当 `valid_ratio < 0.35` 时，建议标记为“数据质量不足”，只展示趋势，不给强结论。

## 8. 与当前项目的对接建议

当前实现位置：`app/services/frame_service.py`。

- 现有 `compute_attention_series(...)` 仅做角度模长计算。
- 建议新增：
  - `compute_attention_series_clean(...)`：返回 `raw/clean/state`。
  - `compute_attention_metrics(...)`：返回上述指标与 `AttentionScore`。
- 在汇总中同时写入：
  - `attention_angle_raw`
  - `attention_angle_clean`
  - `attention_quality`（含 `valid_ratio`、闭眼统计、质量标记）

前端建议：

- 默认显示 `clean` 曲线；提供开关查看 `raw`。
- 可额外展示眨眼频率、眨眼分与非眨眼连续注视时长，增强可解释性。

说明：当前版本已不再使用长闭眼计分；若要和旧报告严格横向比较，建议重新标定 `focus/screen` 阈值，以及 `FixationScore`、`BlinkScore` 的映射参数。

## 9. 阈值标定建议（实践）

首版阈值可先用本文默认值，随后按人群与任务类型做标定：

1. 先收集 20~50 段样本，人工标记“专注/偏离/闭眼”。
2. 网格搜索阈值（如 `0.3/0.5/0.8/1.45` 等）。
3. 用 F1 或加权准确率选最优组合。
4. 逐步引入个体化阈值（基于个人基线分位数）。

## 10. 最小可行落地（MVP）

如果只做最小改动，建议按以下顺序：

1. 加 `track_ok + eye_closed + outlier` 门控。
2. 输出 `attention_angle_clean`（无效帧置空）。
3. 用 `clean` 重算专注率与稳定性。
4. 报告中新增 `valid_ratio`、`blink_count_per_min` 与 `screen_focus_duration_median_sec`。

完成这 4 步后，闭眼误判问题会明显下降，注意力评估可信度会显著提升。
