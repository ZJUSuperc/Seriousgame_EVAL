# DeepFace 全过程情绪分析与评分方案（当前实现）

本文档描述当前项目中 DeepFace 全过程分析的落地方案，重点是：

- 降低单帧抖动，避免只看 `dominant_emotion` 带来的误判。
- 给出可解释的全过程指标与总分。
- 与当前后端输出字段保持一致（已移除“恢复/调节”分项）。

## 1. 输入与预处理

数据来源：`frame_analysis.json` 中每个宏采样帧的 DeepFace 结果。

- `timestamp`
- `deepface.scores`（各情绪概率）
- `macro_inference`（是否执行了宏表情推理）

标准情绪集合（默认 7 类）：

- `angry, disgust, fear, happy, sad, surprise, neutral`

预处理步骤：

1. 仅保留 `deepface.scores` 可解析且总和大于 0 的样本。
2. 仅统计“成功检测”点（DeepFace 严格人脸检测，未检出人脸的帧直接跳过）。
3. 对每帧概率向量做归一化，保证和为 1。
4. 时间轴使用 `timestamp`，并计算样本持续时长作为统计权重。
5. 对概率向量做 EMA 平滑（`alpha = 0.35`）。

## 2. 标签规则与切换率

### 2.1 标签规则

- 默认使用 `top1` 情绪作为标签。
- 若 `top1` 为 `surprise/surprised` 且 `top1 < 0.55`，则改用 `top2` 标签。
- 不再输出 `mixed` 标签。

### 2.2 切换率

- `SwitchRate = 切换次数 / (总时长分钟数)`。
- 总时长来自 OpenFace `timestamp`；当宏表情按帧采样时，时间间隔约等于 `采样帧间隔 / 30fps`。

## 3. 全过程指标（当前落地）

### 3.1 情绪分布

- `emotion_distribution`：全过程各情绪占比（百分比）。
- `top_emotions`：Top3 情绪及占比。

### 3.2 基调指标

```text
PA = happy + 0.5 * neutral
NA = sad + fear + angry + disgust
Tone = PA - NA
```

含义：`Tone > 0` 偏正向/中性，`Tone < 0` 偏负向。

### 3.3 波动、活跃度与情绪段指标

- `Volatility`：相邻平滑概率向量的加权 `L1` 变化均值。
- `SwitchRate` / `SwitchCount`：情绪标签切换速度与次数。
- `DominanceRatio`：主导情绪占比（参考指标）。
- `PhaseChangeRate`：阶段变化率（短段 < 2 秒不计入变化）。
- `PhaseMinDurationSec`：阶段最小时长阈值（固定 2 秒）。
- `PhasePurityRatio`：段内纯度（1 − 段内杂质率）。
  - 若“中性”与其他情绪混杂，扣分权重减半。
- `Coverage`：当前实现按“仅成功点参与统计”，有数据时为 `1.0`，无数据时为 `0.0`。
- `MarginMean` / `EntropyMean`：平均区分度与平均熵（诊断辅助）。

## 4. 评分映射（0~100，含 50 分基线）

先计算原始分（0~100）：

```text
ToneRaw       = 50 + 50 * tanh(Tone / 0.6)
ActivityRaw   = 100 * (0.6 * EventRateScore + 0.4 * VolatilityScore)
PhaseScore    = 0.6 * PurityScore + 0.4 * PhaseChangeScore
```

再映射到“有数据时从 50 起步”：

```text
Rebase(score) = 50 + 0.5 * score

ToneScore     = Rebase(ToneRaw)
ActivityScore = Rebase(ActivityRaw)
PhaseScore    = Rebase(PhaseRaw)
```

总分：

```text
DeepFaceProcessScore =
    0.50 * ToneScore
  + 0.30 * ActivityScore
  + 0.20 * PhaseScore
```

等级：

- `A`：`>= 85`
- `B`：`>= 70`
- `C`：`>= 55`
- `D`：`< 55`

质量状态：

- `status = ok`：有有效样本
- `status = no_data`：无可用样本

注：当前版本不再输出 `recovery_time_median`、`regulation_score`。

## 5. 输出结构（与当前 `summary.json` 一致）

```json
{
  "deepface_overall": {
    "frames_used": 128,
    "sampled_frames": 160,
    "emotion_distribution": {
      "happy": 38.2,
      "neutral": 27.4,
      "sad": 14.6,
      "fear": 7.9,
      "angry": 6.1,
      "surprise": 3.2,
      "disgust": 2.6
    },
    "top_emotions": [
      {"emotion": "happy", "percent": 38.2},
      {"emotion": "neutral", "percent": 27.4},
      {"emotion": "sad", "percent": 14.6}
    ]
  },
  "deepface_metrics": {
    "tone": 0.31,
    "pa": 0.54,
    "na": 0.23,
    "volatility": 0.12,
    "switch_rate": 1.8,
    "switch_count": 5,
    "dominance_ratio": 0.42,
    "phase_change_rate": 1.2,
    "phase_purity_ratio": 0.83,
    "coverage": 1.0,
    "margin_mean": 0.22,
    "entropy_mean": 0.56
  },
  "deepface_scores": {
    "tone_score": 74.5,
    "confidence_score": 70.2,
    "phase_change_score": 78.0,
    "phase_purity_score": 81.0,
    "deepface_process_score": 73.5
  },
  "deepface_rating": "B",
  "deepface_quality": {
    "status": "ok",
    "coverage_status": "ok",
    "ambiguity_status": "medium"
  }
}
```

## 6. 前端展示建议（当前版本）

- 主图：全过程情绪占比饼图（`emotion_distribution`）。
- 核心分数：`deepface_process_score` + 等级。
- 分项分：`tone_score`、`confidence_score`（活跃度）、`phase_change_score`（阶段变化：含段内纯度）。
- 可信度提示：`coverage`、`deepface_quality.status`。

这样可以优先保证“单次视频全过程结论”稳定、可解释、可复现。
