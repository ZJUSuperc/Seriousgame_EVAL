# 游戏OCR屏幕定位方案

## 1. 方案定位

本方案通过 **PaddleOCR** 技术对游戏画面视频进行定时抽帧，识别屏幕文本内容，进而实现：

1. **游戏模块定位** - 根据识别的文本标记（"思维苗圃"、"静心湖畔"等）确定当前位于哪个游戏模块
2. **游戏流程阶段识别** - 识别游戏内容线索文本，确定用户在该模块下的交互阶段
3. **宏表情采样关联** - 将宏表情采样帧与游戏流程对应，实现情感分析的游戏上下文关联

## 2. 核心服务模块

### 2.1 GameOcrService (`app/services/game_ocr_service.py`)

**主要职责：**
- 对游戏视频进行定时抽帧
- 调用 PaddleOCR 引擎进行文本识别
- 管理 OCR 识别结果和性能指标

**关键方法：**

```python
def analyze_video(
    video_path: Path,
    output_dir: Path,
    interval_sec: float | None = None,  # 采样间隔，默认2.0秒
    reference_timeline: list[dict] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict
```

**工作流程：**

```
游戏视频输入
    ↓
[视频元信息读取] - FPS、时长、总帧数
    ↓
[时间采样构建] - 根据间隔秒数生成抽帧时间点
    ↓
[逐帧处理循环]
    ├─ 按时间戳定位视频帧
    ├─ 帧图像预处理与保存
    ├─ PaddleOCR 识别文本
    ├─ 提取文本行、计算识别置信度
    └─ 保存样本结果 JSON
    ↓
[汇总与分析]
    ├─ 计算文本覆盖率（有文本的样本数 / 总样本数）
    ├─ 计算 OCR 延迟（P95百分位）
    ├─ 提取高频出现的文本
    ├─ 与参考时间线关联
    └─ 生成汇总报告
```

### 2.2 GameplayTimelineService (`app/services/gameplay_timeline_service.py`)

**主要职责：**
- 解析 OCR 识别的文本内容
- 将文本内容映射到游戏流程阶段
- 生成游戏播放时间轴

**游戏模块定义：**

| 模块代号 | 中文名称 | 文本标记 |
|---------|--------|--------|
| thinking | 思维苗圃 | "思维苗圃" |
| lake | 静心湖畔 | "静心湖畔" |
| attic | 回忆阁楼 | "回忆阁楼"、"回忆楼阁" |
| studio | 共创画室 | "共创画室" |

**内容识别关键词：**

- `thought` - 思维内容（例："我老了没用了"）
- `tool` - 工具操作（例："浇水"、"施肥"）
- `photo` - 照片内容
- `question` - 问题文本
- `breath_phase` - 呼吸训练阶段

## 3. OCR 识别流程

### 3.1 帧采样策略

**采样方式：** 等间隔时间采样

$$
t_i = i \times \Delta t, \quad i = 0, 1, 2, \ldots, n
$$

其中 $\Delta t$ 为采样间隔（默认2.0秒）

**采样优化：**

- 若视频总时长不能被采样间隔整除，会在视频末尾自动补充一帧
- 采样时间点去重，避免浮点精度问题导致重复采样

### 3.2 帧预处理

**分辨率调整：**

```python
def _prepare_frame_for_ocr(self, frame):
    # 当图像缩小后最长边超过阈值时，按比例缩放
    # 默认最大边长：1280px
    
    if longest_side > max_side:
        scale = max_side / longest_side
        resized_frame = cv2.resize(frame, (new_w, new_h), cv2.INTER_AREA)
    return resized_frame
```

**作用：** 加快 OCR 推理速度，降低内存占用，同时保持识别精度

### 3.3 PaddleOCR 引擎配置

**默认参数：**

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `lang` | `ch` | 识别语言（中文） |
| `ocr_version` | `PP-OCRv5` | PaddleOCR版本 |
| `device` | `cpu` | 计算设备（CPU或GPU） |
| `min_score` | `0.0` | 识别置信度下限（0.0~1.0） |
| `max_image_side` | `1280` | 图像最大边长 |
| `use_doc_orientation_classify` | `False` | 关闭文档方向分类 |
| `use_doc_unwarping` | `False` | 关闭文档变形纠正 |
| `use_textline_orientation` | `False` | 关闭文本行方向识别 |

**引擎实例化：**

```python
self._ocr = PaddleOCR(
    lang=self._lang,
    ocr_version=self._ocr_version,
    device=self._device,
    text_detection_model_dir=self._det_model_dir,  # 可选：自定义文本检测模型
    text_recognition_model_dir=self._rec_model_dir,  # 可选：自定义文本识别模型
    ...
)
```

### 3.4 文本识别与提取

**PaddleOCR 输出格式：**

```python
result = ocr_engine.ocr(image, cls=False)
# 结果结构：
# [
#   [
#     [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], "识别文本", confidence_score
#   ],
#   ...
# ]
```

**文本项提取逻辑：**

```python
def _extract_items(self, raw_result) -> list[dict]:
    for page in pages:
        for text, score in page:
            clean_text = text.strip()
            if clean_text and score >= min_score:
                items.append({
                    "page": page_idx,
                    "text": clean_text,
                    "score": score
                })
```

### 3.5 样本级结果数据结构

```json
{
  "sample_index": 10,
  "timestamp_sec": 20.0,
  "frame_index": 600,
  "image_path": "/path/to/frames/sample_00010_020000ms.jpg",
  "line_count": 3,
  "char_count": 45,
  "mean_score": 0.92,
  "text": "思维苗圃 浇水 经验丰富",
  "texts": ["思维苗圃", "浇水", "经验丰富"],
  "items": [
    {"page": 0, "text": "思维苗圃", "score": 0.95},
    {"page": 0, "text": "浇水", "score": 0.93},
    {"page": 0, "text": "经验丰富", "score": 0.87}
  ],
  "ocr_status": "ok",
  "ocr_error": null,
  "ocr_elapsed_sec": 0.234
}
```

## 4. 游戏流程定位策略

### 4.1 主屏幕识别

**主屏幕标记词：**

```python
HOME_SCREEN_MARKERS = (
    "思维苗圃",
    "静心湖畔",
    "回忆阁楼",
    "回忆楼阁",  # 识别容错变体
    "共创画室",
)
```

**判断逻辑：**

- 若 OCR 识别文本包含上述任一标记词，则判定为进入对应游戏模块
- 通过 Longest Common Subsequence (LCS) 模糊匹配，容纳用户界面文本变化

### 4.2 内容阶段识别

**关键字网络：**

```python
THINKING_DAILY_THOUGHTS = {
    1: ["我老了没用了", "孩子不来看我", "我什么都做不好"],
    2: ["我记忆力变差了", "我害怕孤独", "我是家人的负担"],
    ...
}

THINKING_THOUGHT_ALIASES = {
    "孩子不来看我": ["孩子们都不来看我", "孩子们不来看我", ...],
    ...
}
```

**文本匹配流程：**

```
OCR 识别文本
    ↓
[全文本清理] - 移除特殊字符、标准化空格
    ↓
[噪声过滤] - 移除浏览器地址栏、URL等干扰信息
    ↓
[关键词提取] - 在预定义的关键字库中查询匹配
    ↓
[阶段映射] - 根据匹配的关键词确定游戏阶段
    ↓
[时间段合并] - 合并持续时间过短(<0.4秒)的片段
```

### 4.3 噪声过滤规则

**URL 噪声规则：**

```python
OCR_NOISE_URL_REGEX = re.compile(
    r"(https?://|www\.|localhost|127(?:\.\d{1,3}){3}|...)",
    re.IGNORECASE
)
```

**标记词噪声：**

```python
OCR_NOISE_RAW_MARKERS = (
    "search or type url",
    "搜索或输入网址",
    "新标签页",
    "地址栏",
    "about:blank",
    "microsoft edge",
    ...
)
```

**作用：** 防止浏览器 UI 元素被误认为游戏内容

## 5. 性能指标

### 5.1 汇总级输出

```json
{
  "summary": {
    "video_path": "/path/to/game_video.webm",
    "interval_sec": 2.0,
    "fps": 30.0,
    "duration_sec": 600.0,
    "samples_total": 301,
    "samples_with_text": 298,
    "samples_timeout": 0,
    "line_total": 1254,
    "avg_line_per_sample": 4.16,
    "text_coverage_ratio": 0.9900,
    "ocr_avg_latency_sec": 0.235,
    "ocr_p95_latency_sec": 0.312,
    "top_texts": {
      "思维苗圃": 135,
      "浇水": 89,
      "回忆阁楼": 67,
      ...
    },
    "ocr_config": {
      "lang": "ch",
      "ocr_version": "PP-OCRv5",
      "device": "cpu",
      "min_score": 0.0,
      ...
    }
  },
  "timeline": [ /* 300+ 样本 */ ]
}
```

### 5.2 关键指标说明

| 指标 | 公式 | 含义 |
|-----|-----|------|
| `text_coverage_ratio` | $\frac{样本有文本数}{总样本数}$ | OCR 识别有效率 |
| `avg_line_per_sample` | $\frac{总文本行数}{总样本数}$ | 平均每帧检出的文本行数 |
| `ocr_avg_latency_sec` | $\frac{\sum ocr\_elapsed\_sec}{样本数}$ | OCR 平均推理耗时 |
| `ocr_p95_latency_sec` | 第 95 百分位延迟 | OCR 慢查询阈值 |

### 5.3 超时处理

**超时机制：**

```python
OCR_SAMPLE_TIMEOUT_SEC = 600.0  # 单个样本超时阈值
OCR_HEARTBEAT_SEC = 15.0        # 心跳日志间隔
OCR_MAX_CONSECUTIVE_TIMEOUTS = 0  # 连续超时上限（0=无限）
```

- 若 OCR 推理耗时超过 `timeout_sec`，标记为 `"timeout"` 状态
- 若连续超时达到上限，中止视频分析

## 6. 与宏表情采样的集成

### 6.1 集成入口

**Pipeline Service 中的调用：**

```python
# pipeline_service.py
game_ocr_result = self.game_ocr_service.analyze_video(
    video_path=game_video_path,
    output_dir=ocr_output_dir,
    interval_sec=game_ocr_interval_sec,
    reference_timeline=macro_samples,  # 参考时间线
)
```

### 6.2 时间线关联

**参考时间线：** 宏表情采样的时间戳列表

```python
reference_timeline = [
    {"timestamp_sec": 5.0, "emotion": "happy", ...},
    {"timestamp_sec": 15.0, "emotion": "sad", ...},
    ...
]
```

**关联过程：**

$$
\text{ocr\_sample} \leftarrow \arg\min_{\text{ref}} |t_{ocr} - t_{ref}|, \quad |t_{ocr} - t_{ref}| < threshold
$$

- 对每个 OCR 样本时间戳，找最接近的宏表情采样
- 若时间距离在阈值内（通常 2~3 秒），建立关联

### 6.3 报告集成

**综合分析报告中包含：**

1. **游戏流程分模块评估**
   - 各模块下的平均情感指标
   - 各阶段下的专注力变化

2. **上下文关联分析**
   - 某思维内容下的情绪分布
   - 完成特定工具操作时的表情特征

## 7. 配置参数说明

### 7.1 环境变量

```bash
# OCR 工具链路
SG_OCR_TOOLKIT_ROOT=/path/to/PaddleOCR-main/ocr_toolkit
SG_OCR_REPO_ROOT=/path/to/PaddleOCR-main

# 采样参数
SG_GAME_OCR_INTERVAL_SEC_DEFAULT=2.0  # 采样间隔（秒）

# 模型参数
SG_GAME_OCR_LANG=ch                    # 识别语言
SG_GAME_OCR_VERSION=PP-OCRv5           # PaddleOCR 版本
SG_GAME_OCR_DEVICE=cpu                 # 计算设备
SG_GAME_OCR_MIN_SCORE=0.0              # 置信度下限

# 性能参数
SG_GAME_OCR_MAX_IMAGE_SIDE=1280        # 最大图像边长
SG_GAME_OCR_SAMPLE_TIMEOUT_SEC=600.0   # 单样本超时
SG_GAME_OCR_HEARTBEAT_SEC=15.0         # 心跳间隔
SG_GAME_OCR_MAX_CONSECUTIVE_TIMEOUTS=0 # 超时上限

# 自定义模型路径（可选）
SG_GAME_OCR_DET_MODEL_DIR=/path/to/det/model
SG_GAME_OCR_REC_MODEL_DIR=/path/to/rec/model
```

### 7.2 默认配置（无环境变量时）

详见 `app/config.py` 中的 `load_config()` 函数

## 8. 局限与优化方向

### 8.1 当前局限

1. **固定采样间隔** - 可能遗漏短暂内容，或冗余采样稳定画面
2. **文本仅仅识别** - 未进行 NLP 语义理解和情感倾向分析
3. **噪声过滤规则硬编码** - 对不同游戏版本适配性有限
4. **CPU 推理慢** - GPU 加速仍需环境配置

### 8.2 优化方向

1. **自适应采样**
   - 基于帧图像变化度进行动态采样
   - 内容变化快时增加采样率

2. **语义级识别**
   - 集成思维阶段的 NLP 分类器
   - 进行诗词、感受词的情感倾向量化

3. **多模态融合**
   - 将 OCR 文本与注意力（眼动）数据融合
   - 分析用户读完文本时的视线滞留与情感变化关联

4. **样本库学习**
   - 针对游戏特定 UI 元素的微调模型
   - 提高识别准确率和速度

## 9. 代码引用

- **核心服务**: `app/services/game_ocr_service.py` (500+ 行)
- **游戏流程**: `app/services/gameplay_timeline_service.py` 
- **流水线集成**: `app/services/pipeline_service.py` 
- **配置管理**: `app/config.py`

---

**最后更新**: 2026-04-30  
**方案版本**: v1.0
