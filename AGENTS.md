# AGENTS.md

## 项目概览

`seriousgame_eval` 是一个面向严肃游戏实验视频的本地多模态评估项目。它以被试正面视频和可选的游戏屏幕录制为输入，生成 OpenFace AU/视线时间轴、宏表情分类、Valence/Arousal 情绪轨迹、AU 异常事件、注意力评分、游戏 OCR 流程分段、语音转写、rPPG 心率/呼吸结果，以及面向研究汇总的报告和图表。

这个项目由三类能力组成：

1. Flask 本地评估平台：上传视频、创建任务、轮询进度、查看帧级分析、图表和辅助分析结果。
2. 批处理与研究分析脚本：批量处理视频、生成综合报告、导出相关性表和论文图表。
3. AU 时序建模包：用 OpenFace AU 时序构建认知分与情绪分预测实验，使用留一被试交叉验证。

## 重要入口

- Web 服务入口：`run.py`
- Flask app 创建：`app/__init__.py`
- 配置与外部工具路径：`app/config.py`
- API 路由：`app/blueprints/api_tasks.py`、`app/blueprints/api_health.py`
- 单页前端：`app/templates/index.html`
- 主分析流水线：`app/services/pipeline_service.py`
- OpenFace 调用：`app/services/openface_service.py`
- 帧、AU、注意力分析：`app/services/frame_service.py`
- 宏表情分类封装：`app/services/deepface_service.py`
- V/A 分析封装：`app/services/emonet_service.py`
- 游戏 OCR：`app/services/game_ocr_service.py`
- 游戏流程分段：`app/services/gameplay_timeline_service.py`
- rPPG/呼吸：`app/services/rppg_service.py`
- ASR 转写：`app/services/asr_service.py`
- AU 建模包：`modeling/`
- 建模测试：`tests/modeling/`
- 详细项目说明：`docs/PROJECT_DETAILED_GUIDE.md`
- 评估算法说明：`评估方案/`

## 本地运行

常规 Web 启动：

```powershell
conda activate seriousgame
cd C:\Users\Administrator\Desktop\seriousgame_eval
python run.py
```

浏览器打开：

```text
http://127.0.0.1:5050
```

建模测试：

```powershell
$tmp = Join-Path (Get-Location) '.pytest_tmp'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$env:TEMP = $tmp
$env:TMP = $tmp
python -m pytest tests\modeling -v
```

说明：在当前 Windows 环境中，默认系统临时目录可能出现权限问题；测试时优先把 `TEMP`/`TMP` 指到项目内 `.pytest_tmp`。

## 核心数据流

主链路：

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

辅助链路：

```text
视频时间窗 -> RhythmMamba rPPG -> 心率 / 呼吸 / 深呼吸完成度
视频音频 -> SenseVoice -> 语音文本 / 分句时间戳
OpenFace AU reports -> modeling 包 -> LOSO 认知/情绪预测实验
```

## 命名历史包袱

这些字段名不能按字面理解：

- `deepface_*`：历史兼容字段。当前默认宏表情分类后端是 DAN，可通过配置切换 AffectNet，并不等于一定使用 DeepFace。
- `emonet_*`：历史兼容字段。当前 V/A 优先使用 AffectNet 回归，失败时回退到 DAN 情绪先验估计，并不等于旧 EmoNet 主推理。
- “微表情”：当前主流程更准确地说是基于 OpenFace AU 时序的异常表情事件检测，不是严格心理学定义的亚秒级微表情分类。
- `fatigue_score`：注意力模块保留的旧字段，当前更接近 `fixation_score`，即非眨眼连续屏幕内注视能力。

正式写论文或汇报时，优先使用更准确的表述：

- 宏表情分类模块（DAN/AffectNet 后端，字段沿用 `deepface_*`）
- V/A 情绪轨迹模块（AffectNet 回归 + DAN 回退先验，字段沿用 `emonet_*`）
- 基于 OpenFace AU 时序的异常表情事件检测
- 基于视线角度、闭眼与追踪质量的屏幕注视稳定性分析

## 目录与提交边界

应提交到 GitHub 的内容：

- `app/` 源码与前端模板/静态图片
- `modeling/` 建模包
- `tests/` 测试
- `scripts/run_route_a.py` 等正式脚本
- 根目录正式分析、绘图、批处理脚本
- `README.md`、`AGENTS.md`、`docs/`、`评估方案/` 等文档
- `data/modeling/labels.csv`
- `.gitignore`、`.gitattributes`、`requirements.txt`、`pytest.ini`

不要提交：

- `data/reports/`
- `data/results/`
- `data/tasks/`
- `data/uploads/`
- `data/batch_reports/`
- `new/`
- 视频文件：`*.mp4`、`*.webm`、`*.avi`、`*.mov`、`*.mkv` 等
- 模型权重：`*.pth`、`*.pt`、`*.ckpt`、`*.onnx`、`*.pb`
- 缓存：`__pycache__/`、`.pytest_cache/`、`.pytest_tmp/`
- 本地助手/工具状态：`.agents/`、`.claude/`、`.opencode/`、`.superpowers/`
- 一次性实验脚本和本地提示文件，例如 `scripts/_run_route_a_progressive.py`、`*AI不必看.md`

## 外部依赖与路径

`app/config.py` 中有许多本机路径默认值，迁移机器时需要通过环境变量覆盖。重要依赖包括：

- OpenFace
- AffectNet
- DAN 模型包
- RhythmMamba rPPG
- SenseVoice
- PaddleOCR
- ffmpeg

不要把外部模型目录、权重或工具包复制进仓库。需要共享时，在文档中说明获取方式或环境变量配置。

## 代码工作建议

- 修改前先查看相关实现，尤其是 `app/services/pipeline_service.py` 和 `app/services/frame_service.py`，不要只根据旧文档推断行为。
- `PipelineService` 很大，职责包括主流程、评分、OCR、游戏分析和报告生成。改动时尽量局部、小步验证。
- 对结构化数据优先用 JSON/CSV 解析，不要靠字符串硬切。
- 涉及任务产物时，注意 live 文件和 final 文件的区别：`summary_live.json` / `frame_analysis_live.json` 支持运行中预览，最终产物用于任务完成后读取。
- API 参数校验主要在 `app/blueprints/api_tasks.py`，新增接口时保持中文错误信息风格一致。
- 评分逻辑与阈值要同步更新 `评估方案/` 或 `docs/PROJECT_DETAILED_GUIDE.md`，避免代码和说明漂移。
- 中文文档用 UTF-8。

## 测试与验证

当前自动化测试主要覆盖 `modeling/`，不是完整 Flask 主链路。

推荐验证顺序：

1. 修改建模包后运行：

   ```powershell
   $tmp = Join-Path (Get-Location) '.pytest_tmp'
   New-Item -ItemType Directory -Force -Path $tmp | Out-Null
   $env:TEMP = $tmp
   $env:TMP = $tmp
   python -m pytest tests\modeling -v
   ```

2. 修改 Flask/API/前端后至少启动本地服务并手动检查 `/api/health` 和首页。
3. 修改 OpenFace、OCR、rPPG、ASR 相关代码时，需要确认本机外部工具路径存在；这些模块依赖本地模型和可执行文件，普通 CI 很难完整覆盖。

## Git/GitHub 状态

当前仓库已初始化并推送到：

```text
https://github.com/ZJUSuperc/Seriousgame_EVAL.git
```

默认分支：

```text
main
```

常规提交流程：

```powershell
git status
git add .
git commit -m "说明本次改动"
git push
```

提交前务必确认 `git status --ignored --short` 中视频、报告、缓存和本地配置仍处于 ignored 状态。

## 关键注意事项

- 该项目包含研究数据和视频产物，本地目录很大，但 GitHub 仓库应只保存源码、文档、测试和必要小型标签数据。
- `data/modeling/labels.csv` 可以提交；其他 `data/` 内容通常不提交。
- `README.md` 和 `docs/PROJECT_DETAILED_GUIDE.md` 中若出现“项目不是 git 仓库”等旧描述，后续可顺手更新。
- 认知/情绪建模只有 12 个独立被试标签，实验必须按被试做 LOSO，不要按视频随机划分。
- `SG_ENABLE_QUALITY_FILTER` 默认关闭；主分析和建模包的质量过滤口径不同，比较结果时要说明数据口径。
- rPPG 对光照、人脸裁剪、运动和视频压缩敏感，解读时一定结合 `resp_quality`。
