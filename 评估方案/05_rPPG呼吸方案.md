# rPPG 呼吸方案

## 1. 方案定位

rPPG 方案用于从普通视频中恢复脉搏相关光容积变化信号，再进一步估计心率与呼吸节律。当前项目中，它还承担了一个具体业务任务：评估被试是否完成了“三次深呼吸”训练。

核心代码：

- `app/services/rppg_service.py`
- `app/blueprints/api_tasks.py`

与主视频分析链不同，rPPG 是手动触发的独立模块。

## 2. 触发方式与输入

API 提供：

- `POST /api/tasks/<task_id>/rppg`
- `GET /api/tasks/<task_id>/rppg`

分析输入包括：

- `video_path`
- `start_sec`
- `duration_sec`
- `chunk_len`
- `device`

因此它支持只分析一个指定时间窗口，而不是必须处理整段视频。

## 3. 总体流程

`RppgService.analyze()` 的实际处理顺序为：

1. 加载视频片段与人脸区域
2. 帧标准化
3. 分块送入 RhythmMamba 模型
4. 得到 `rppg_raw`
5. 后处理：去趋势 + 心率带通
6. 从处理后信号估计心率
7. 分别对 `raw` 和 `detrended` 做呼吸分析
8. 选择呼吸质量更高的一条作为最终输出

## 4. 视频与人脸处理

### 4.1 时间窗口

若用户指定：

- 起点 `start_sec`
- 时长 `duration_sec`

则代码会根据视频 FPS 转换为帧区间：

$$
start\_frame = round(start\_sec \cdot fps)
$$

$$
end\_frame = start\_frame + round(duration\_sec \cdot fps)
$$

### 4.2 人脸裁剪

底层使用 Haar cascade 进行人脸检测，并将人脸区域统一缩放到：

$$
128 \times 128
$$

以满足后续 RhythmMamba 模型输入要求。

## 5. 帧标准化与分块

### 5.1 标准化

代码对输入帧序列做零均值、单位方差标准化：

$$
X' = \frac{X - \mu}{\sigma}
$$

若标准差极小，则只做去均值。

### 5.2 分块

模型不是逐帧独立推理，而是按固定长度切块。默认：

$$
chunk\_len = 160
$$

当视频帧数不足一个 chunk 时，会补零到一个完整块。

## 6. RhythmMamba 推理输出

模型输出一个一维 rPPG 序列。代码随后对每个 batch 内输出再做一次标准化：

$$
y' = \frac{y - mean(y)}{std(y) + 10^{-7}}
$$

最终得到：

- `rppg_raw`

## 7. rPPG 后处理

后处理函数 `_postprocess()` 返回两条信号：

- `detrended`
- `filtered`

### 7.1 去趋势

代码使用 Tarvainen 风格去趋势方法：

$$
\hat{x} = \left(I - (I + \lambda^2 D^T D)^{-1}\right)x
$$

其中：

$$
\lambda = 100
$$

### 7.2 心率带通滤波

随后对 `detrended` 信号做一阶 Butterworth 带通：

$$
f \in [0.75, 2.5] \text{ Hz}
$$

对应心率范围：

$$
[45, 150] \text{ BPM}
$$

## 8. 心率估计

心率使用 Welch 功率谱估计：

$$
PSD(f) = Welch(filtered\_signal)
$$

只在以下频段中找峰值：

$$
\frac{45}{60} < f < \frac{150}{60}
$$

主峰频率记为 $f_{peak}$，则：

$$
hr\_bpm = 60 \cdot f_{peak}
$$

最终输出：

- `summary.hr_bpm`

## 9. 呼吸信号提取

呼吸分析不是直接用心率带通信号，而是重新在低频带内滤波。

代码呼吸带通范围：

$$
f \in [0.08, 0.5] \text{ Hz}
$$

对应呼吸频率约为：

$$
[4.8, 30] \text{ 次/分钟}
$$

代码中经常近似描述为 5 到 30 次/分钟。

## 10. 呼吸率与质量评估

### 10.1 呼吸率

同样通过 Welch PSD，在呼吸带内找峰值频率：

$$
resp\_bpm = 60 \cdot f_{resp\_peak}
$$

### 10.2 呼吸质量

代码定义呼吸质量为“呼吸主峰功率占整个呼吸带总功率的比例”：

$$
resp\_quality = \frac{P_{peak}}{\sum_{f \in [0.08,0.5]} P(f)}
$$

这个值越高，说明呼吸频带内越存在清晰的主导节律。

## 11. 为什么同时分析 raw 和 detrended

代码会对两条信号各做一次呼吸分析：

- `rppg_raw`
- `detrended`

然后比较：

$$
resp\_quality_{det} > resp\_quality_{raw}
$$

若成立，则采用 detrended 结果；否则采用 raw 结果。

因此最终输出包含：

- `resp_source = raw` 或 `detrended`

这点非常重要，因为它说明项目并不是固定用 detrended 呼吸信号，而是做了信号质量选择。

## 12. 呼吸周期检测

### 12.1 峰检测

代码通过：

```text
find_peaks(signal, distance=1.5 * fps)
```

寻找呼吸峰值。

即两个峰之间最少间隔：

$$
distance \ge 1.5s
$$

### 12.2 周期定义

相邻峰之间构成一个呼吸周期：

$$
cycle = [peak_i, peak_{i+1}]
$$

周期时长要求：

$$
3.0s \le duration \le 10.0s
$$

超出该范围的周期将被丢弃。

### 12.3 谷值选择

在两个峰之间寻找 trough，若存在多个 trough，则选最低点；若没有，则在区间内直接找最小值位置。

### 12.4 周期振幅

代码定义周期振幅为：

$$
amplitude = |signal[end\_peak] - signal[trough]|
$$

即峰到谷的差值绝对值。

## 13. 深呼吸判定

对所有周期振幅组成集合：

$$
\{a_1, a_2, ..., a_n\}
$$

深呼吸阈值定义为：

$$
depth\_threshold = median(a) + 0.5 \cdot std(a)
$$

若某个周期满足：

$$
a_i \ge depth\_threshold
$$

则记为一次深呼吸。

最终得到：

- `breath_count`
- `deep_breath_count`
- `depth_threshold`

## 14. 合规性判定

代码定义呼吸训练完成度 `compliance`：

### 14.1 低质量优先拦截

若：

$$
resp\_quality < 0.35
$$

则：

$$
compliance = low\_quality
$$

### 14.2 深呼吸数量判定

否则：

$$
\begin{cases}
complete, & deep\_breath\_count \ge 3 \\
partial, & 1 \le deep\_breath\_count < 3 \\
insufficient, & deep\_breath\_count = 0
\end{cases}
$$

也就是说，项目当前的业务目标非常明确：

- 识别是否完成 3 次深呼吸。

## 15. 输出字段

主返回结果包括三部分：

### 15.1 summary

- `hr_bpm`
- `fps`
- `frames_used`
- `start_sec`
- `end_sec`
- `duration_sec`
- `chunk_len`

### 15.2 series

- `time_sec`
- `raw`
- `detrended`
- `filtered`

### 15.3 resp

`resp.summary` 中有：

- `resp_bpm`
- `resp_quality`
- `breath_count`
- `deep_breath_count`
- `deep_breath_target`
- `cycle_mean_sec`
- `cycle_std_sec`
- `depth_mean`
- `depth_std`
- `depth_threshold`
- `cycle_min_sec`
- `cycle_max_sec`
- `compliance`
- `deep_breath_times`
- `resp_source`

`resp.series` 中有：

- `time_sec`
- `signal`

## 16. 与现有文档的差异

与 `rPPG呼吸训练分析.md` 相比，代码层面要特别注意两点：

1. 呼吸分析并不是固定用 `detrended`，而是比较 `raw` 与 `detrended` 的 `resp_quality` 后择优。
2. `deep_breath_times` 当前记录的是相对分析窗口起点的周期结束秒：

$$
end\_time = \frac{end\_idx}{fps}
$$

它没有再加上 `start_sec`，而 `resp.series.time_sec` 是加了 `start_sec` 的。

这意味着两者时间基准当前并不完全一致，是实现细节中需要注意的问题。

## 17. 方案解读建议

从应用角度，这个模块可以被理解为：

- 用视频恢复低频生理节律；
- 利用频谱峰值估计心率与呼吸率；
- 再通过呼吸周期振幅检测“深呼吸动作是否足够明显”；
- 最终用 `low_quality / insufficient / partial / complete` 给出训练完成度判断。

因此，它既是一个生理信号估计模块，也是一个具体任务导向的呼吸训练评估模块。
