# seriousgame_eval 项目详尽说明文档

生成日期：2026-07-06

## 1. 项目一句话概括

`seriousgame_eval` 是一个面向严肃游戏实验视频的本地多模态评估项目。它以普通摄像头视频和可选的游戏屏幕录制为输入，围绕老年被试在游戏过程中的面部表情、视线注意力、情绪效价/唤醒、游戏流程、语音转写、rPPG 心率/呼吸等信息，生成帧级结果、过程指标、评分、可视化前端和综合报告。

从工程形态上看，它不是单纯的网页项目，也不是单纯的数据分析脚本集合，而是三块能力叠加：

1. Flask 本地评估平台：上传视频、创建任务、轮询进度、查看帧级分析、图表和辅助分析。
2. 视频批处理与研究分析脚本：批量跑视频、汇总报告、导出论文图表和相关性表。
3. AU 时序建模基础包：为后续用面部动作单元预测认知分与情绪分提供数据管线和 LOSO 评估框架。

## 2. 适用场景

本项目适用于以下任务：

- 对严肃游戏实验录像进行本地、离线、可追溯的多模态分析。
- 从被试正面视频中提取 OpenFace AU、视线、主体脸轨迹。
- 评估游戏过程中的宏表情分布、V/A 情绪轨迹、异常 AU 表情事件、注视屏幕稳定性。
- 对屏幕录制做 OCR，进一步推断游戏模块时间段，例如“思维苗圃”“静心湖畔”“回忆阁楼”“共创画室”。
- 在指定时间窗内运行 rPPG，估计心率、呼吸率和三次深呼吸完成度。
- 对视频语音做 SenseVoice 转写，得到文本、分句与时间戳。
- 汇总多被试、多视频的统计表、相关性热图和论文写作材料。
- 构建 AU 时序监督建模的基础数据集和留一被试交叉验证框架。

## 3. 当前代码状态摘要

项目当前不是 git 仓库，`git status` 无法使用。工作区中包含大量已生成数据、缓存、实验视频和报告文件。

核心应用入口是：

```text
run.py -> app.create_app() -> Flask blueprints + services
```

Web 服务默认端口：

```text
http://127.0.0.1:5050
```

当前 README、`评估方案` 目录和代码之间整体一致，但部分命名有历史遗留：

- `deepface_*` 字段名不代表一定使用 DeepFace。默认宏表情分类后端是 DAN，也可切换 AffectNet。
- `emonet_*` 字段名不代表当前一定使用旧版 EmoNet。当前 V/A 优先使用 AffectNet 回归，失败时回退到 DAN 情绪先验估计。
- “微表情”在当前主流程里更准确地说是“基于 OpenFace AU 时序的异常表情事件检测”。

## 4. 顶层目录结构

```text
seriousgame_eval/
  app/                         Flask 应用主体
    blueprints/                Web/API 路由
    services/                  分析服务、模型封装、流水线
    templates/index.html       单页前端
    static/images/             前端静态图片
    config.py                  环境变量与路径配置
  data/                        运行数据、任务、结果、报告、建模标签
    uploads/                   上传视频
    tasks/                     JSON 任务仓库
    results/                   单任务中间产物
    reports/                   综合报告与批处理结果
    modeling/labels.csv        12 名有标签被试标签表
  modeling/                    AU 建模基础包
  tests/modeling/              建模包单元测试
  docs/superpowers/            AU 建模设计与实现计划
  评估方案/                    评估算法说明文档
  new/                         视频样例/实验录像
  *.py                         批处理、导出、绘图、相关性分析脚本
  README.md                    基础运行说明
  requirements.txt             Python 依赖
  pytest.ini                   pytest 配置
```

## 5. 运行环境与依赖

`requirements.txt` 中列出的主要依赖包括：

```text
flask
pandas
numpy
opencv-python
tensorflow
pillow
torch
scipy
matplotlib
timm
einops
torchaudio
modelscope
huggingface_hub
funasr
```

外部工具依赖主要来自本机目录，默认路径写在 `app/config.py`：

- OpenFace：`C:/Users/Administrator/Desktop/openfacetest/3dparty/OpenFace_2.2.0_win_x64`
- AffectNet：`C:/Users/Administrator/Desktop/AffectNet-master`
- DAN 模型包：`C:/Users/Administrator/Desktop/download_packages/best_affectnet7_dan_exp1_baseline_ls_20260330_152032`
- RhythmMamba rPPG：`C:/Users/Administrator/Desktop/RhythmMamba-main/portable_rppg_infer`
- SenseVoice：`C:/Users/Administrator/Desktop/SenseVoice-main/video_transcription_tool`
- PaddleOCR 工具：`C:/Users/Administrator/Desktop/PaddleOCR-main/ocr_toolkit`
- ffmpeg：默认从 PATH 查找 `ffmpeg`

推荐运行方式：

```powershell
conda activate seriousgame
cd C:\Users\Administrator\Desktop\seriousgame_eval
python run.py
```

## 6. 配置系统

配置入口是 `app/config.py` 的 `load_config()`。它把项目内部目录、模型路径、开关和阈值整合成 Flask `app.config`。

### 6.1 内部数据目录

默认按项目根目录生成：

- `DATA_DIR`: `data`
- `UPLOAD_DIR`: `data/uploads`
- `RESULT_DIR`: `data/results`
- `TASK_DIR`: `data/tasks`
- `REPORT_DIR`: `data/reports`

`app/__init__.py` 会在启动时确保这些目录存在。

### 6.2 关键环境变量

常用环境变量：

| 变量 | 含义 | 默认值/说明 |
|---|---|---|
| `SG_SECRET_KEY` | Flask secret key | `seriousgame-local-secret` |
| `SG_TOOLKIT_ROOT` | 工具根目录 | `C:/Users/Administrator/Desktop/openfacetest` |
| `SG_OPENFACE_DIR` | OpenFace 二进制目录 | 由 `SG_TOOLKIT_ROOT` 推导 |
| `SG_AFFECTNET_ROOT` | AffectNet 根目录 | `C:/Users/Administrator/Desktop/AffectNet-master` |
| `SG_DEEPFACE_BACKEND` | 宏表情分类后端 | `dan`，可为 `dan` 或 `affectnet` |
| `SG_DAN_DEVICE` | DAN 推理设备 | `auto` |
| `SG_DAN_MODEL_PATH` | DAN 权重路径 | 默认包内 `model/best.pth` |
| `SG_EMONET_MODEL_PATH` | 旧 EmoNet 模型路径 | 当前不是主路径 |
| `SG_ENABLE_QUALITY_FILTER` | 是否启用 OpenFace 质量过滤 | 默认 `0` |
| `SG_CONF_THRESHOLD` | OpenFace confidence 阈值 | 默认 `0.8` |
| `SG_GAME_OCR_INTERVAL_SEC_DEFAULT` | 游戏 OCR 默认采样间隔 | 默认 `2.0` 秒 |
| `SG_GAME_OCR_DEVICE` | PaddleOCR 推理设备 | 默认 `cpu` |
| `SG_GAME_OCR_SAMPLE_TIMEOUT_SEC` | OCR 单样本超时 | 默认 `600` 秒 |
| `SG_REPORT_DIR` | 综合报告输出目录 | 默认 `data/reports` |
| `SG_RPPG_ROOT` | RhythmMamba 推理代码目录 | 本机默认路径 |
| `SG_RPPG_MODEL_PATH` | rPPG 模型权重 | 默认 UBFC cross 权重 |
| `SG_SENSEVOICE_ROOT` | SenseVoice 工具目录 | 本机默认路径 |
| `SG_FFMPEG_BIN` | ffmpeg 命令 | 默认 `ffmpeg` |

## 7. Flask 应用结构

### 7.1 应用创建

`app/__init__.py` 中 `create_app()` 完成：

1. 创建 Flask app。
2. 加载配置。
3. 创建必要目录。
4. 初始化任务仓库 `TaskStore`。
5. 恢复/标记上次异常中断的任务。
6. 初始化主流水线 `PipelineService`。
7. 初始化 `RppgService` 与 `AsrService`。
8. 注册蓝图：`web_bp`、`health_bp`、`tasks_bp`。

### 7.2 蓝图

| 文件 | 职责 |
|---|---|
| `app/blueprints/web.py` | 返回首页模板 `/` |
| `app/blueprints/api_health.py` | 健康检查 `/api/health` |
| `app/blueprints/api_tasks.py` | 上传、任务、结果、帧图、AU、OCR、rPPG、ASR 等 API |

## 8. API 总览

### 8.1 健康检查

```text
GET /api/health
```

返回 OpenFace、AffectNet、DAN、OCR、EmoNet、rPPG、SenseVoice、ffmpeg 等依赖路径是否可用。

### 8.2 上传与任务

```text
POST /api/uploads/video
GET  /api/tasks
POST /api/tasks
GET  /api/tasks/<task_id>
```

`POST /api/uploads/video` 接收 `video` 文件字段，保存到 `data/uploads`，返回 `video_path`。

`POST /api/tasks` 支持请求体：

```json
{
  "video_path": "C:/path/to/camera_video.webm",
  "video_name": "原始文件名.webm",
  "game_video_path": "C:/path/to/screen_video.webm",
  "game_video_name": "屏幕录制.webm",
  "macro_stride": 5,
  "macro_interval_sec": 0.8,
  "game_ocr_interval_sec": 2.0
}
```

其中：

- `video_path` 必填。
- `game_video_path` 可选，传入后会启用游戏画面 OCR 与游戏流程分析。
- `macro_stride` 控制按帧采样宏表情，默认 5。
- `macro_interval_sec` 控制按秒采样宏表情，设置后优先级高于 `macro_stride`。
- `game_ocr_interval_sec` 控制屏幕录制 OCR 采样间隔。

### 8.3 主分析结果

```text
GET /api/tasks/<task_id>/summary
GET /api/tasks/<task_id>/summary?allow_partial=1
GET /api/tasks/<task_id>/frame/<idx>
GET /api/tasks/<task_id>/frame/<idx>?allow_partial=1
GET /api/tasks/<task_id>/frame/<idx>/image
GET /api/tasks/<task_id>/video
GET /api/tasks/<task_id>/au?name=AU12
```

`allow_partial=1` 用于任务运行中读取实时预览产物。

### 8.4 游戏 OCR 与流程分析

```text
GET /api/tasks/<task_id>/game-ocr
GET /api/tasks/<task_id>/game-ocr/frame/<sample_idx>/image
GET /api/tasks/<task_id>/gameplay-analysis
```

只有任务创建时传了 `game_video_path`，这部分才会运行。

### 8.5 rPPG

```text
POST /api/tasks/<task_id>/rppg
GET  /api/tasks/<task_id>/rppg
```

请求体示例：

```json
{
  "start_sec": 0,
  "duration_sec": 80,
  "chunk_len": 160,
  "device": "cpu"
}
```

### 8.6 语音转写

```text
POST /api/tasks/<task_id>/transcription
GET  /api/tasks/<task_id>/transcription
```

请求体示例：

```json
{
  "segment_length_sec": 60,
  "language": "zh",
  "use_itn": true,
  "device": "cpu"
}
```

## 9. 任务仓库与状态管理

任务仓库由 `app/services/task_store.py` 实现，用本地 JSON 文件保存任务状态。

每个任务保存在：

```text
data/tasks/<task_id>.json
```

任务基本字段包括：

- `task_id`
- `status`: `queued` / `running` / `done` / `failed`
- `stage`: 当前阶段，例如 `openface`、`timeline`、`emotion_inference`、`summary`
- `progress`: 0 到 1
- `progress_detail`: 当前阶段 done/total
- `input`: 输入参数
- `artifacts`: 各类输出文件路径
- `error`: 主任务错误
- `rppg_status` / `rppg_error` / `rppg_summary`
- `transcription_status` / `transcription_error` / `transcription_summary`
- `game_ocr_status` / `game_ocr_error` / `game_ocr_summary`
- `gameplay_analysis_status` / `gameplay_analysis_error` / `gameplay_analysis_summary`
- `logs`: 最多保留 500 条日志
- `created_at` / `updated_at`

应用启动时 `recover_interrupted_aux_jobs()` 会把上次服务重启中断的 running 任务标记为 failed，避免前端永久显示运行中。

JSON 读写由 `app/utils/json_io.py` 提供：

- 写入使用临时文件再 replace，降低半写入风险。
- 读取遇到 JSONDecodeError 会短暂重试，适配并发读写。

## 10. 主流水线详解

主流水线位于 `app/services/pipeline_service.py`。

### 10.1 总体流程

`PipelineService._run_task()` 的主要阶段：

1. 读取任务输入。
2. 初始化结果目录与 live 预览文件。
3. 运行 OpenFace。
4. 读取 OpenFace CSV，构建时间轴。
5. 多人脸场景下自动锁定主体脸。
6. 按时间轴抽取视频帧。
7. 校准时间戳到视频解码时间。
8. 按帧间隔或秒间隔执行宏表情采样。
9. 对采样帧调用宏表情分类和 V/A 分析。
10. 周期性写入 live frame analysis 与 live summary。
11. 计算注意力、异常 AU 事件、宏表情过程指标、V/A 过程指标。
12. 如有屏幕录制，运行游戏 OCR。
13. 基于 OCR 结果做游戏流程分段与模块情绪聚合。
14. 生成最终 JSON/CSV/Markdown 综合报告。
15. 更新任务状态为 done 或 failed。

### 10.2 OpenFace 阶段

`OpenFaceService.run()` 支持三种模式：

- `single`: 只跑 `FeatureExtraction.exe`
- `multi`: 只跑 `FaceLandmarkVidMulti.exe`
- `auto_main`: 先尝试 multi，失败则回退 single

命令参数包括：

```text
-f <video>
-out_dir <output>
-2Dfp
-3Dfp
-pose
-aus
-gaze
```

输出核心是 OpenFace CSV，包含 frame、timestamp、face_id、AU、gaze、landmark 等字段。

### 10.3 时间轴与主体脸选择

`FrameService.load_timeline_dataframe()` 负责读取 CSV 并规范字段：

- 去除列名前后空格。
- 缺失 `frame` 时用递增帧号补齐。
- 缺失或部分缺失 `timestamp` 时插值/回退。
- 按 timestamp、frame、openface_row 排序。
- 多人脸时调用 `_select_main_face_rows()` 选择主体脸。
- 可选按 `success` 与 `confidence` 过滤质量。
- 根据 `frame_stride` 与 `max_frames` 裁剪，目前主流程固定为全帧。

主体脸选择策略综合：

- 离画面中心距离。
- 脸框面积。
- OpenFace confidence/success。
- 与上一帧中心位置的连续性。

最终策略标记为 `center_continuity_area`，并写入 `main_face_info`。

### 10.4 抽帧与时间戳校准

`FrameService.extract_frames()` 用 OpenCV 按 OpenFace frame 抽取 JPEG：

```text
data/results/<task_id>/frames/frame_000000.jpg
```

抽帧时会读取 `cv2.CAP_PROP_POS_MSEC`，如果可用，则把 item 的 `timestamp` 替换为视频解码时间，并在流水线中同步回写到 `timeline_df`。这可以缓解 OpenFace timestamp 与真实视频时间不一致的问题。

### 10.5 宏表情采样

宏表情采样两种模式：

1. `frame`：`idx % macro_stride == 0`
2. `time`：当 `macro_interval_sec > 0` 时，每隔固定秒数采样一次

采样帧会保存：

- `deepface`: 宏表情分类结果
- `emonet`: V/A 结果
- `macro_inference`: 是否执行过宏表情推理
- `errors`: 单帧推理错误

未采样帧仍保留基础 frame/timestamp/image 信息，但 `deepface` 与 `emonet` 为空。

## 11. 核心评估模块

### 11.1 宏表情分类模块

入口：

```text
DeepFaceService.analyze(image_path)
```

实际后端：

- 默认：`DanBackend`
- 可选：`AffectNetBackend`

输出：

```json
{
  "dominant_emotion": "happy",
  "scores": {
    "angry": 0.01,
    "disgust": 0.02,
    "fear": 0.03,
    "happy": 0.70,
    "sad": 0.04,
    "surprise": 0.05,
    "neutral": 0.15
  }
}
```

过程汇总由 `PipelineService._compute_deepface_process_outputs()` 完成，核心逻辑包括：

- 对概率分布归一化。
- 根据相邻时间戳估计样本持续时间。
- 使用 EMA 平滑情绪概率，`alpha = 0.30`。
- 对低置信 `surprise` 做 top2 回退。
- 合并连续同标签段。
- 将过短情绪段合并为 phase。
- 计算情绪占比、PA、NA、tone、volatility、事件频率、阶段变化率、阶段纯度、主导度。
- 生成 `deepface_process_score` 与 A/B/C/D 等级。

宏表情总分近似由三部分组成：

```text
0.50 * tone_score + 0.30 * activity_score + 0.20 * phase_quality_score
```

输出字段主要包括：

- `deepface_overall`
- `deepface_metrics`
- `deepface_scores`
- `deepface_process_score`
- `deepface_rating`
- `deepface_quality`

### 11.2 V/A 情绪轨迹模块

入口：

```text
EmoNetService.analyze(image_path)
```

真实逻辑：

1. 优先使用 AffectNet 回归，直接得到 `valence` 与 `arousal`。
2. AffectNet 失败后，回退到 DAN 分类概率，并用情绪先验估计 V/A。

DAN 回退先验示例：

| 情绪 | Valence | Arousal |
|---|---:|---:|
| neutral | 0.0 | 0.0 |
| happy | 0.75 | 0.35 |
| sad | -0.65 | -0.2 |
| surprise | 0.15 | 0.6 |
| fear | -0.65 | 0.75 |
| disgust | -0.6 | 0.3 |
| angry | -0.7 | 0.65 |
| contempt | -0.3 | 0.1 |

过程汇总由 `PipelineService._compute_emonet_process_outputs()` 完成，核心指标：

- `valence_mean`
- `arousal_mean`
- `va_volatility`
- `activation_ratio`
- `coverage`

总分：

```text
va_score = 0.40 * valence_score
         + 0.35 * arousal_score
         + 0.10 * stability_score
         + 0.15 * activation_score
```

输出字段主要包括：

- `emonet_metrics`
- `emonet_scores`
- `emonet_score`
- `emonet_rating`
- `emonet_quality`

### 11.3 AU 异常事件模块

入口：

```text
FrameService.detect_abnormal_events(df)
```

它基于 OpenFace AU 强度规则检测持续事件，而不是严格的亚秒级微表情分类器。

事件类型：

| 类型 | 含义 | 核心依据 |
|---|---|---|
| `long_eye_closure_event` | 长闭眼 | `AU45_r > 0.5` 且持续至少 1 秒 |
| `positive_event` | 积极/微笑 | 强 `AU12` 或 `AU12 + AU06` |
| `sad_event` | 悲伤 | `AU15/AU01/AU04/AU17` 多项组合 |
| `aversion_event` | 厌恶/回避 | `AU09/AU10` 核心 + `AU14/AU15` 等组合 |
| `stress_event` | 压力/紧张 | `AU04 + AU07`，必要时嘴部紧张证据 |
| `surprise_event` | 惊讶 | `AU01/AU02/AU05` 幅值与变化率 |
| `confusion_event` | 困惑 | `AU04 + AU01` |

每个事件输出：

- `type`
- `index`
- `frame`
- `timestamp`
- `start`
- `end`
- `duration`
- `intensity`
- `au_list`
- `au_peak`

代码中还保留 `detect_micro_expression_events()`，它按 AU 增量突增找点事件，但当前主汇总流程不使用它。

### 11.4 视线注意力模块

入口：

```text
FrameService.compute_attention_analysis(df)
```

输入字段：

- `gaze_angle_x`
- `gaze_angle_y`
- `success`
- `confidence`
- `AU45_c`
- `AU45_r`
- `timestamp`

处理逻辑：

1. 合成视线偏离角：`sqrt(gaze_angle_x^2 + gaze_angle_y^2)`。
2. 根据 success/confidence 判定追踪质量。
3. 根据 AU45 判定闭眼。
4. 过滤极端角度和相邻突变。
5. 对短缺口做线性插值。
6. 做局部尖峰抑制。
7. 做 Savitzky-Golay 平滑。
8. 计算聚焦、屏幕范围、偏离、眨眼、稳定性、连续注视等指标。

关键阈值：

| 常量 | 值 | 含义 |
|---|---:|---|
| `ATTENTION_CONFIDENCE_THRESHOLD` | 0.8 | OpenFace confidence 阈值 |
| `ATTENTION_MAX_ANGLE_RAD` | 1.45 | 绝对异常角 |
| `ATTENTION_DELTA_OUTLIER_RAD` | 0.55 | 相邻帧突变阈值 |
| `ATTENTION_INTERPOLATE_MAX_GAP_SEC` | 0.3 | 短缺口插值最大时长 |
| `ATTENTION_FOCUS_THRESHOLD` | 0.3 | 高聚焦角度阈值 |
| `ATTENTION_ACCEPTABLE_THRESHOLD` | 0.5 | 屏幕范围角度阈值 |
| `ATTENTION_MIN_VALID_RATIO` | 0.2 | 数据质量最低有效占比 |

总分：

```text
attention_score = 0.40 * focus_score
                + 0.15 * fixation_score
                + 0.10 * blink_score
                + 0.15 * stability_score
                + 0.20 * quality_score
```

输出字段：

- `attention_angle_raw`
- `attention_angle_clean`
- `attention_state`
- `attention_metrics`
- `attention_scores`
- `attention_events`

## 12. 游戏屏幕 OCR 与流程分段

### 12.1 OCR 模块

入口：

```text
GameOcrService.analyze_video(video_path, output_dir, interval_sec, reference_timeline)
```

功能：

- 按固定时间间隔采样屏幕录制。
- 保存采样帧图片。
- 调用 PaddleOCR 识别文本。
- 兼容新版 `predict` 和旧版 `ocr` 接口。
- 过滤低分文本。
- 将 OCR 样本与主视频分析时间轴做最近时间匹配。
- 统计 top 文本、覆盖率、延迟、超时数量等。

输出：

- `summary`
- `timeline`

OCR 采样帧保存在任务结果目录下的 OCR frames 子目录。

### 12.2 游戏流程分段模块

入口：

```text
GameplayTimelineService.analyze(ocr_result, analyzed_frames)
```

它根据 OCR 文本识别游戏模块和内容关键词。当前内置模块：

| key | 中文名 |
|---|---|
| `thinking` | 思维苗圃 |
| `lake` | 静心湖畔 |
| `attic` | 回忆阁楼 |
| `studio` | 共创画室 |

该模块包含大量领域词库，例如：

- 思维苗圃每日负面念头、积极重构句、工具关键词：浇水、施肥、除虫。
- 回忆阁楼年代、历史照片、互动问题关键词。
- 共创画室动作关键词：画笔、橡皮擦、完成绘画、下载绘画等。

处理结果包括：

- `segments`: 游戏流程时间片段。
- `modules`: 按模块聚合的持续时长、片段数、情绪统计、内容标签。
- `sample_labels`: 每个 OCR 样本的模块预测与关键词命中。
- `keyword_timeline`: 关键词时间定位。
- `summary`: 模块覆盖、噪声过滤、主导模块等指标。

流水线还会把游戏模块时间片段和主视频情绪/注意力结果结合，生成分模块分析和综合报告。

## 13. rPPG 与呼吸分析

入口：

```text
RppgService.analyze(video_path, start_sec, duration_sec, chunk_len, device)
```

rPPG 是手动触发的辅助分析，不在主任务中自动运行。

流程：

1. 按指定时间窗加载视频帧。
2. 使用 Haar cascade 检测人脸，裁剪并 resize 到 `128x128`。
3. 对帧序列标准化。
4. 按 `chunk_len` 切块，默认 160 帧。
5. 调用 RhythmMamba 模型输出 rPPG 原始序列。
6. 对输出做去趋势和心率带通滤波。
7. 用 Welch PSD 估计心率。
8. 对 raw 和 detrended 两条信号分别做低频呼吸分析。
9. 选择 `resp_quality` 更高的呼吸结果。
10. 检测呼吸周期与深呼吸次数。

输出结构：

```text
summary: hr_bpm、fps、frames_used、start_sec、end_sec、duration_sec、chunk_len
series: time_sec、raw、detrended、filtered
psd: freq_bpm、power
resp.summary: resp_bpm、resp_quality、breath_count、deep_breath_count、compliance 等
resp.series: time_sec、signal
```

深呼吸完成度：

- `complete`: 深呼吸次数 >= 3
- `partial`: 深呼吸次数 1 到 2
- `insufficient`: 没有检测到深呼吸
- `low_quality`: 呼吸信号质量低于 0.35
- `no_data`: 无有效呼吸数据

需要注意：当前代码中 `deep_breath_times` 是相对分析窗口起点的周期结束秒，而 `resp.series.time_sec` 是加上 `start_sec` 的绝对视频时间。两者时间基准不完全一致。

## 14. ASR 语音转写

入口：

```text
AsrService.transcribe(video_path, segment_length_sec, language, use_itn, device)
```

流程：

1. 用 ffmpeg 从视频提取 16kHz 单声道 WAV。
2. 按 `segment_length_sec` 切分音频，默认 60 秒，最小 10 秒。
3. 加载 SenseVoiceSmall 模型。
4. 对每个音频片段做转写并请求 char-level timestamp。
5. 移除 SenseVoice 标签，例如 `<|...|>`。
6. 将字符时间戳按标点聚合成句子。
7. 返回完整文本、片段和句子列表。

输出：

- `summary`: language、segment_length_sec、duration_sec、segments_total、sentences_total、device
- `full_text`
- `segments`
- `sentences`

## 15. 前端页面

前端是 `app/templates/index.html` 单页应用，不使用 Vue/React。

主要功能区：

- 视频上传与任务创建。
- 可选游戏画面上传与 OCR 采样间隔设置。
- 任务状态、进度条、日志。
- 帧浏览器与按宏表情帧播放。
- 当前帧宏表情情绪占比。
- 当前帧 V/A 点图。
- 宏表情全过程情绪占比与评分。
- V/A 时序图。
- 注意力视线角时序图。
- 情感状态 circumplex 映射。
- AU 曲线与异常事件列表。
- ASR 手动转写面板。
- 游戏 OCR 时间线与样本帧。
- 游戏流程分段、模块汇总、关键词定位、分模块复算分析。
- rPPG 手动区间分析、波形图、PSD 图、呼吸评估和区间视频预览。

前端通过 `fetch()` 调用 Flask API，任务运行中使用轮询，并优先读取 `allow_partial=1` 的实时预览结果。

## 16. 数据产物

### 16.1 单任务中间产物

通常位于：

```text
data/results/<task_id>/
```

可能包含：

- `openface/attempt_*/*.csv`
- `frames/frame_*.jpg`
- `frame_analysis_live.json`
- `summary_live.json`
- `frame_analysis.json`
- `summary.json`
- `rppg_result.json`
- `transcription_result.json`
- OCR 结果与样本帧

### 16.2 综合报告

通常位于：

```text
data/reports/<视频名或被试目录>/
```

常见产物：

- `<name>.json`: 综合报告结构化数据
- `<name>.md`: 综合报告 Markdown
- `<name>.csv`: OpenFace 或汇总 CSV
- `batch_backend_summary.json/csv`
- `association_analysis/` 相关性分析输出

### 16.3 关联分析产物

`data/reports/association_analysis/` 下包含：

- 表情指标逐视频/受试者级 CSV
- AU 特征逐视频/受试者级 CSV
- 表情与认知/情绪相关性 CSV
- AU 与认知/情绪相关性 CSV
- 单 AU、多 AU、宏表情 VA、注意力等热图 PNG
- `analysis_report.md`

## 17. 批处理脚本

### 17.1 `batch_process_videos.py`

用于批量扫描视频并串行运行完整分析。

默认输入目录：

```text
C:\Users\Administrator\Desktop\video_analyse\new
```

关键参数：

| 参数 | 含义 |
|---|---|
| `--source-dir` | 视频样例根目录 |
| `--output-dir` | 批处理输出目录，默认 `data/batch_reports/<时间戳>` |
| `--macro-stride` | 宏表情按帧采样步长 |
| `--macro-interval-sec` | 宏表情按秒采样 |
| `--game-ocr-interval-sec` | 游戏 OCR 采样间隔 |
| `--mode` | OpenFace 模式：auto_main/multi/single |
| `--limit` | 仅处理前 N 个主视频 |
| `--dry-run` | 只输出匹配清单，不执行分析 |
| `--include-fix` | 包含文件名含 fix 的视频 |
| `--skip-processed` | 根据已有 reports 自动跳过 |
| `--no-skip-processed` | 不跳过已处理视频 |
| `--no-ocr` / `--video-only` | 跳过游戏 OCR 与流程分析 |

它会自动匹配同名 `_screen` 视频作为游戏屏幕录制。

### 17.2 `analyze_video_analyse_new_backend.py`

用于后台批量分析普通视频，并按人员编号分类保存 OpenFace CSV 与评估报表。

关键参数：

| 参数 | 含义 |
|---|---|
| `--source-dir` | 待分析视频根目录 |
| `--output-dir` | 最终输出目录，默认 `data/reports` |
| `--runtime-dir` | 运行期临时目录 |
| `--mode` | OpenFace 模式 |
| `--macro-stride` | 宏表情帧采样步长 |
| `--macro-interval-sec` | 宏表情按秒采样 |
| `--limit` | 仅处理前 N 个普通视频 |
| `--person-id` | 仅处理指定人员编号 |
| `--overwrite` | 覆盖已有结果 |

### 17.3 统计与绘图脚本

项目根目录下还有一组面向论文/分析的脚本：

| 脚本 | 用途概括 |
|---|---|
| `explore_cognition_emotion_associations.py` | 从 reports 汇总表情/AU 特征并计算与认知、情绪标签的相关性 |
| `explore_video_level_duplicate_label_correlations.py` | 逐视频重复标签相关性探索 |
| `export_macro_expression_tables.py` | 导出宏表情指标表 |
| `export_person_video_full_summary.py` | 导出每人每次视频完整摘要 |
| `compare_first_last_by_time.py` | 比较首末时间/视频指标变化 |
| `draw_correlation_heatmaps.py` | 绘制相关性热图 |
| `draw_macro_va_correlation_figures.py` | 绘制宏表情 VA 相关图 |
| `draw_macro_va_selected_figures.py` | 绘制选定 VA 指标图 |
| `draw_attention_core_figures.py` | 绘制注意力核心指标图 |
| `draw_attention_selected_figures.py` | 绘制选定注意力指标图 |
| `draw_single_au_category_heatmaps.py` | 绘制单 AU 分类热图 |
| `draw_single_au_dimension_rows.py` | 绘制单 AU 各维度行图 |
| `draw_multi_au_selected_figures.py` | 绘制多 AU 组合相关图 |

这些脚本大多直接读写 `data/reports/association_analysis`，属于研究分析层，不是 Flask 主服务必须路径。

## 18. AU 时序建模基础包

`modeling/` 是 2026-06-29 后新增的监督建模基础设施，目标是用 AU 时序预测老年被试的认知分与情绪分。

### 18.1 标签

`modeling/labels.py` 内置 12 名有标签被试：

```text
1901-1912
```

每人有：

- `cognitive_score`
- `emotion_score`

同一被试的所有视频继承该被试标签。

`data/modeling/labels.csv` 是标签导出文件。

### 18.2 数据加载

`modeling/dataset.py` 提供：

- `AU_COLUMNS`: 17 个 OpenFace AU `_r` 强度列
- `VideoRecord`: 视频记录 dataclass
- `discover_videos(reports_root, subjects)`
- `load_video_au(csv_path, confidence_threshold=0.8, enable_quality_filter=True)`
- `build_dataset(reports_root, subjects, labeled, ...)`

17 个 AU：

```text
AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r,
AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r
```

### 18.3 LOSO 评估

`modeling/splits.py` 提供 `loso_folds()`。

`modeling/loso.py` 提供 `run_loso()`，约定模型函数接口：

```python
PredictFn = Callable[
    [Sequence[VideoRecord], dict[str, SubjectLabel], Sequence[VideoRecord]],
    dict[str, tuple[float, float]],
]
```

评估特点：

- 按被试留一，不让同一被试视频跨 train/test。
- 每个测试被试的多视频预测先求均值，再与被试真实标签比较。
- 输出 cognitive 和 emotion 两套回归指标。

### 18.4 指标

`modeling/eval.py` 提供：

- MAE
- RMSE
- R2
- Pearson r/p
- normalized MAE
- mean baseline

`modeling/targets.py` 提供按训练折拟合的 z-score 目标标准化。

### 18.5 测试

测试目录：

```text
tests/modeling/
```

覆盖：

- 标签读写
- AU CSV 加载与质量过滤
- LOSO 折划分
- 目标标准化
- 回归指标
- LOSO 端到端冒烟

可运行：

```powershell
python -m pytest tests/modeling -v
```

## 19. 研究设计背景

`docs/superpowers/specs/2026-06-29-au-cognition-emotion-modeling-design.md` 说明了 AU 建模路线。

核心目标：

- 仅用视频 AU 时序，同时回归预测 `cognitive_score` 和 `emotion_score`。
- 使用 LOSO，避免被试内数据泄露。

规划路线：

1. 特征工程 + 浅层模型：Ridge/SVR/RandomForest 等。
2. 时序深度学习：缩小版 FFNN、1D-CNN、GRU/LSTM。
3. 可选自监督预训练：masked AU reconstruction。
4. 后期融合：手工特征 + embedding。

当前已实现的是基础数据管线和 LOSO 评估框架，尚未实现具体路线 A/B/C 模型。

## 20. 文档体系

已有重要说明文档：

| 文件 | 内容 |
|---|---|
| `README.md` | 基础运行、API、环境变量、当前功能范围 |
| `评估方案/00_总览.md` | 评估方案总览，强调以代码为准 |
| `评估方案/01_宏表情方案.md` | 宏表情过程统计和评分说明 |
| `评估方案/02_宏表情VA方案.md` | V/A 模块说明 |
| `评估方案/03_微表情AU方案.md` | AU 异常事件规则说明 |
| `评估方案/04_视线注意力方案.md` | 注视稳定性分析说明 |
| `评估方案/05_rPPG呼吸方案.md` | rPPG、呼吸和深呼吸评估说明 |
| `评估方案/06_游戏OCR屏幕定位方案.md` | 游戏 OCR 与屏幕定位方案 |
| `docs/superpowers/specs/...` | AU 建模设计文档 |
| `docs/superpowers/plans/...` | AU 建模基础实现计划 |
| `论文写作参考_方法与分析说明.md` | 论文方法写作参考 |
| `论文写作参考_结果与表格版.md` | 论文结果和表格写作参考 |

## 21. 主要命名历史包袱

### 21.1 `deepface_*`

当前 `deepface_*` 是历史兼容字段，真实后端由 `SG_DEEPFACE_BACKEND` 控制，默认是 DAN。

推荐在论文或正式报告中表述为：

```text
宏表情分类模块（DAN/AffectNet 后端，字段沿用 deepface_*）
```

### 21.2 `emonet_*`

当前 `emonet_*` 字段表示 V/A 分析结果，但优先后端是 AffectNet 回归，不是旧 EmoNet。

推荐表述：

```text
V/A 情绪轨迹模块（AffectNet 回归 + DAN 回退先验，字段沿用 emonet_*）
```

### 21.3 “微表情”

当前主流程输出的是 AU 异常事件，而不是严格心理学定义的微表情识别。

推荐表述：

```text
基于 OpenFace AU 时序的异常表情事件检测
```

### 21.4 `fatigue_score`

注意力模块里保留了旧字段 `fatigue_score`，当前语义更接近 `fixation_score`，即非眨眼连续屏幕内注视能力。

## 22. 已知风险与维护注意事项

1. 项目不是 git 仓库，改动追踪困难。建议初始化 git 或至少定期备份关键代码与文档。
2. 大量外部模型路径写死为本机路径，迁移机器时需要集中配置环境变量。
3. Web 主流程用本地 JSON 任务仓库和线程执行，适合本地单机，不适合多人并发或生产部署。
4. `PipelineService` 文件体量很大，职责覆盖主流程、评分、OCR、游戏分析、报告生成，后续维护可考虑拆分。
5. `data/` 与 `new/` 中包含大量视频和报告，备份与同步时要注意体积。
6. 部分输出字段沿用旧名，外部使用方应以本说明中的真实语义为准。
7. OCR 依赖 PaddleOCR 版本差异，代码已做接口兼容，但模型初始化和设备参数仍可能随版本变化。
8. rPPG 对光照、人脸裁剪、视频压缩和运动很敏感，结果需要结合 `resp_quality` 解读。
9. 认知/情绪监督建模当前只有 12 个独立被试标签，模型评估必须坚持 LOSO，不能按视频随机划分。
10. `SG_ENABLE_QUALITY_FILTER` 默认关闭，主分析会尽量保留全视频全帧；建模包默认启用 confidence 过滤，两者数据口径不同。

## 23. 推荐后续改进

### 23.1 工程层

- 初始化 git 仓库，加入 `.gitignore`，明确不提交视频、大型模型和缓存。
- 将 `PipelineService` 拆分为：主编排、宏表情评分、V/A 评分、报告生成、游戏分析适配。
- 使用 SQLite 替代 JSON 任务仓库。
- 使用 Celery/RQ + Redis 替代 Flask 进程内线程。
- 为 `api_tasks.py` 的参数校验提取独立 schema。
- 给核心服务增加更多单元测试和小样本集成测试。

### 23.2 算法层

- 明确宏表情 DAN/AffectNet 后端在论文中的命名和版本。
- 对 `deepface_*` 与 `emonet_*` 输出字段做新旧兼容映射文档。
- 为 AU 异常事件增加可配置阈值或实验版本号。
- 对注意力分数做样本级稳定性验证。
- 修正或说明 rPPG `deep_breath_times` 与 `resp.series.time_sec` 时间基准差异。

### 23.3 研究层

- 补充 `cognitive_score` 和 `emotion_score` 的量表来源与解释。
- 继续实现 AU 建模路线 A：手工特征 + 浅层多输出回归。
- 在每次建模实验中记录 LOSO 每折预测值，避免只看平均指标。
- 对所有相关性热图和统计表保留脚本版本、输入数据版本和生成时间。

## 24. 快速上手路径

### 24.1 只看网页分析

```powershell
conda activate seriousgame
cd C:\Users\Administrator\Desktop\seriousgame_eval
python run.py
```

打开：

```text
http://127.0.0.1:5050
```

上传被试正面视频，可选再上传同名屏幕录制，点击开始分析。

### 24.2 跑批处理

```powershell
conda activate seriousgame
cd C:\Users\Administrator\Desktop\seriousgame_eval
python batch_process_videos.py --source-dir "C:\Users\Administrator\Desktop\video_analyse\new"
```

如只做视频分析、不跑游戏 OCR：

```powershell
python batch_process_videos.py --source-dir "C:\Users\Administrator\Desktop\video_analyse\new" --no-ocr
```

### 24.3 跑建模测试

```powershell
conda activate seriousgame
cd C:\Users\Administrator\Desktop\seriousgame_eval
python -m pytest tests/modeling -v
```

### 24.4 读取真实 reports 做 AU 数据集冒烟

```powershell
python -c "from pathlib import Path; from modeling.dataset import build_dataset; from modeling.labels import labeled_subject_ids; recs = build_dataset(Path('data/reports'), labeled_subject_ids(), set(labeled_subject_ids())); print('videos:', len(recs)); print('subjects:', sorted({r.subject_id for r in recs})); print('sample:', recs[0].au.shape, recs[0].valid_ratio)"
```

## 25. 最终理解

这个项目的核心价值不是某一个单独模型，而是把严肃游戏实验录像转成一组可研究、可回看、可汇总、可建模的多模态过程数据。

主链路可以概括为：

```text
被试视频
  -> OpenFace
  -> 主体脸时间轴
  -> 抽帧
  -> 宏表情分类 / V-A 回归
  -> AU 异常事件 / 视线注意力
  -> 可选游戏 OCR / 游戏模块分段
  -> summary + frame analysis + 综合报告
```

辅链路包括：

```text
视频时间窗 -> RhythmMamba rPPG -> 心率 / 呼吸 / 深呼吸完成度
视频音频 -> SenseVoice -> 语音文本 / 分句时间戳
OpenFace AU reports -> modeling 包 -> LOSO 认知/情绪预测实验
```

因此，维护和扩展时应始终区分三层：

1. 工程层：任务、API、前端、文件产物。
2. 算法层：OpenFace、DAN/AffectNet、OCR、rPPG、ASR、评分规则。
3. 研究层：标签、统计、相关性、LOSO 建模和论文解释。

只要这三层边界保持清晰，项目后续可以比较稳地继续扩展。
