# AffectNet V/A 全过程评分方案（当前实现）

本方案用于**单次视频全过程**评估，目标是：

- 指标少、解释清晰、可直接落地到 `summary.json`。
- 仅围绕 V/A（Valence / Arousal）连续维度输出。
- 与当前后端实现保持一致（已移除“参与度”概念）。

## 1. 输入与预处理

数据来源：`frame_analysis.json` 中每个宏采样帧的 `emonet.valence`、`emonet.arousal`、`timestamp`。

预处理约定：

1. 仅使用“成功检测”的 V/A 点：
   - `emonet` 不为空；
   - `valence/arousal` 可解析为数值；
   - `probabilities` 存在且最大概率 `>= 0.20`（或至少有有效 `emotion` 字段）。
2. `Valence` 裁剪到 `[-1, 1]`。
3. `Arousal` 裁剪到 `[-1, 1]`。
4. 统计按时间加权（相邻 `timestamp` 差值作为权重）。

## 2. 指标定义（4 个核心 + 1 个质量）

### 指标1：情绪基调 `valence_mean`

- 定义：`valence` 的时间加权均值。

### 指标2：激活水平 `arousal_mean`

- 定义：`arousal` 的时间加权均值。

### 指标3：情绪稳定性 `va_volatility`

- 定义：

```text
va_volatility = mean( sqrt((ΔV)^2 + (ΔA)^2) )
```

### 指标4：激活强度占比 `activation_ratio`

- 定义：`arousal > 0.25` 的时长占比。

### 质量指标：覆盖率 `coverage`

- 定义：有效样本数 ÷ 宏采样帧数。

## 3. 评分（0~100，含 50 分基线）

归一化函数：

```text
S_up_pow(x,l,u,p) = clip((x-l)/(u-l), 0, 1) ^ p * 100
S_down(x,l,u)     = clip((u-x)/(u-l), 0, 1) * 100
```

基线映射：

```text
Rebase(score) = 50 + 0.5 * score
```

原始分：

```text
ValenceRaw    = S_up_pow(valence_mean,   -0.20, 0.60, 0.6)
ArousalRaw    = S_up_pow(arousal_mean,   -0.10, 0.50, 0.6)
StabilityRaw  = S_down(va_volatility, 0.20, 0.08)
ActivationRaw = S_up_pow(activation_ratio, 0.08, 0.35, 1.3)
```

最终分：

```text
ValenceScore    = Rebase(ValenceRaw)
ArousalScore    = Rebase(ArousalRaw)
StabilityScore  = Rebase(StabilityRaw)
ActivationScore = Rebase(ActivationRaw)

VA_score = 0.40*ValenceScore + 0.35*ArousalScore + 0.10*StabilityScore + 0.15*ActivationScore
```

等级：

- `A`：`>= 85`
- `B`：`>= 70`
- `C`：`>= 55`
- `D`：`< 55`

## 4. 质量状态

- 有有效点且覆盖率足够：`emonet_quality.status = ok`。
- 覆盖率不足：`status = insufficient`。
- 无有效点：`status = no_data`。

## 5. 当前推荐输出结构（与代码一致）

```json
{
  "emonet_metrics": {
    "valence_mean": 0.12,
    "arousal_mean": 0.18,
    "va_volatility": 0.11,
    "activation_ratio": 0.22,
    "coverage": 0.92
  },
  "emonet_scores": {
    "valence_score": 68.0,
    "arousal_score": 72.4,
    "stability_score": 84.1,
    "activation_score": 66.0,
    "va_score": 72.3
  },
  "emonet_rating": "B",
  "emonet_quality": {
    "status": "ok",
    "coverage_status": "ok"
  }
}
```

## 6. 结果解读（简版）

- `VA_score >= 85`：整体状态稳定，基调与激活水平较好。
- `70~85`：整体可用，存在轻度波动。
- `55~70`：中等，建议结合 AU/注意力一起看。
- `<55`：结果偏弱，优先检查采样覆盖与视频质量。
