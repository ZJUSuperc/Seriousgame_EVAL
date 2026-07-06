# 严肃游戏本地评估平台（Flask）

这是一个面向本地部署的多模态评估平台脚手架，目标是支持：

- OpenFace：微表情（AU 事件）与注意力视线角分析（重点 AU12/AU15/AU14/AU10）
- 注意力鲁棒分析：闭眼/低置信/异常角度门控 + 清洗序列 + 评分输出
- AffectNet：Valence / Arousal 时序分析（替代 EmoNet）
- AffectNet：Valence / Arousal 全过程精简量化评分（中文指标）
- AffectNet：逐帧情绪占比 + 全过程指标与评分分析（替代 DeepFace）
- 帧滑条联动：拖动帧时同步查看该帧图像与情绪结果
- AffectNet 全过程情绪占比图：圆形扇区显示各情绪总体占比
- 唤醒-效价圆：基于 V/A 模型显示情感状态点（X=Valence -1~1，Y=Arousal -1~1）
- V/A 时序图：Valence(-1~1) 与 Arousal(-1~1) 双轴展示
- 帧播放预览：按分析帧顺序播放，观察多模态指标的连续变化
- 网页指标中文化：AffectNet / 注意力评分均以中文指标呈现
- 手动语音转写：SenseVoice 视频转文本（分段可调）
- 手动 rPPG：RhythmMamba 心率估算（可选择时间区间）
- 深呼吸评估：基于 rPPG 低频呼吸波形判断三次深呼吸完成度
- 后续扩展：ASR（语音转文字）与 rPPG（心率）

## 1. 当前实现范围

本项目已提供第一版可运行骨架：

1. Flask 工程结构（蓝图 + 服务层）
2. 任务创建与任务状态跟踪（本地 JSON 任务仓库）
3. OpenFace 调用服务（最终输出固定单主体脸轨迹）
4. 按 OpenFace 时间轴对全视频做微表情分析，并支持宏表情按帧间隔或按秒间隔采样（AffectNet 情绪分类 + V/A）
5. 多人脸场景自动锁定主体脸（中心偏好 + 连续性 + 脸框面积）
6. 结果接口与基础前端页面（任务创建、轮询、帧滑动查看、曲线展示）
7. 运行中实时预览（分析进行时可查看已完成部分帧与时序曲线）
8. 曲线点选跳帧（点击时序图上的点可跳转到对应视频帧）
9. 时序图支持全过程与时间窗口两种视图，可配置窗口时长并支持跟随当前帧
10. 情感状态映射（唤醒-效价圆 + 当前帧高亮 + 点选跳帧）
11. 微表情与异常事件模块：可筛选异常事件、查看 AU 全程曲线，并支持点击跳帧
12. AffectNet 当前帧卡片内增加 V/A 预测点图
13. 注意力指标区分眨眼与长闭眼，新增屏幕范围占比（screen_focus_ratio）

## 2. 本地运行

在 `seriousgame` 环境中启动：

```powershell
conda activate seriousgame
cd C:\Users\Administrator\Desktop\seriousgame_eval
python run.py
```

浏览器打开：

```text
http://127.0.0.1:5050
```

如果只需要离线批量跑报表（不走前端），可直接运行：

```powershell
conda activate seriousgame
cd C:\Users\Administrator\Desktop\seriousgame_eval
python batch_process_videos.py --source-dir "C:\Users\Administrator\Desktop\video_analyse\new"
```

说明：

- 脚本按视频逐个串行执行（不会并发），降低显存压力。
- 自动匹配同名 `_screen` 视频作为 `game_video_path`；没有则仅跑主视频。
- 默认跳过文件名含 `fix` 的视频。
- 每个任务仍会生成系统标准结果；同时会汇总：
  - 综合报表集合（JSON/MD）
  - `openface_au_gaze_all.jsonl`（AU + 视线）
  - `batch_summary.json` / `batch_summary.md` / `report_index.csv`

## 3. 环境变量（可选）

默认会读取你现有工具目录 `C:/Users/Administrator/Desktop/openfacetest`。

- `SG_TOOLKIT_ROOT`：工具根目录（默认 `C:/Users/Administrator/Desktop/openfacetest`）
- `SG_OPENFACE_DIR`：OpenFace 二进制目录
- `SG_AFFECTNET_ROOT`：AffectNet 模型根目录（默认 `C:/Users/Administrator/Desktop/AffectNet-master`）
- `SG_EMOTION_ANALYZER_PARENT`：旧版 EmoNet 依赖（已不再使用，可忽略）
- `SG_EMONET_DEVICE`：旧版 EmoNet 设备参数（已不再使用，可忽略）
- `SG_ENABLE_QUALITY_FILTER`：是否启用 OpenFace 质量过滤（默认 `0`，即不过滤，全视频全帧分析）
- `SG_CONF_THRESHOLD`：质量过滤阈值（仅当 `SG_ENABLE_QUALITY_FILTER=1` 时生效，默认 `0.8`）
- `SG_GAME_OCR_DEVICE`：游戏 OCR 推理设备（默认 `cpu`，可设为 `gpu:0`）
- `SG_REPORT_DIR`：综合报表输出目录（默认 `data/reports`，任务完成后自动写入同名子目录）
- `SG_RPPG_ROOT`：RhythmMamba 推理脚本目录（默认 `C:/Users/Administrator/Desktop/RhythmMamba-main/portable_rppg_infer`）
- `SG_RPPG_MODEL_PATH`：rPPG 预训练模型路径（默认 `PreTrainedModels/UBFC_cross_RhythmMamba.pth`）
- `SG_RPPG_CASCADE_PATH`：Haar Cascade 人脸检测文件路径
- `SG_SENSEVOICE_ROOT`：SenseVoice 转写工具目录（默认 `C:/Users/Administrator/Desktop/SenseVoice-main/video_transcription_tool`）
- `SG_FFMPEG_BIN`：ffmpeg 可执行文件（默认 `ffmpeg`）

## 4. API 总览

- `GET /api/health`：环境与路径健康检查
- `POST /api/uploads/video`：上传视频文件并返回本地 `video_path`
- `POST /api/tasks`：创建分析任务
- `GET /api/tasks/<task_id>`：查询任务状态
- `GET /api/tasks/<task_id>/summary`：获取时序汇总
- `GET /api/tasks/<task_id>/frame/<idx>`：获取单帧分析结果
- `GET /api/tasks/<task_id>/frame/<idx>/image`：获取单帧图像
- `GET /api/tasks/<task_id>/au?name=AU12`：获取指定 AU 全程曲线
- `POST /api/tasks/<task_id>/rppg`：手动触发 rPPG 区间分析
- `GET /api/tasks/<task_id>/rppg`：获取 rPPG 结果
- `GET /api/tasks/<task_id>/video`：获取原视频（rPPG 区间预览）
- `POST /api/tasks/<task_id>/transcription`：手动触发语音转写
- `GET /api/tasks/<task_id>/transcription`：获取语音转写结果

前端默认流程为：先调用 `POST /api/uploads/video` 上传文件，再将返回的 `video_path` 传给 `POST /api/tasks`。

`POST /api/tasks` 请求体只需要：

```json
{
  "video_path": "C:/path/to/video.mp4",
  "macro_stride": 5,
  "macro_interval_sec": 0.8
}
```

说明：

- 当前版本最终只输出一个主体脸轨迹，微表情（AU/视线）默认走全视频全帧。
- 当前版本不开放人脸模式切换：内部默认优先执行 OpenFace 多脸检测并自动锁定主体脸；若多脸执行不可用会回退到单脸模式。
- `macro_stride` 用于控制宏表情采样间隔（每 N 帧执行一次 AffectNet 情绪/VA），默认 `5`，最小 `1`。
- `macro_interval_sec` 为可选参数；当设置为 `>0` 时，宏表情按 OpenFace 时间轴每 N 秒采样，优先级高于 `macro_stride`。
- 不再暴露旧的 `mode/frame_stride/max_frames` 参数。

`GET /api/tasks/<task_id>/summary` 会额外返回：

- `macro_mode`：`frame` 或 `time`
- `macro_interval_sec`：时间采样间隔（未启用时为 `null`）
- `main_face_info`：主体脸锁定信息（face_id、候选脸数量、筛选策略等）
- `deepface_overall`：全过程情绪占比（字段名沿用 `deepface_*`，数据来自 AffectNet 情绪分类）
- `deepface_metrics` / `deepface_scores`：全过程指标与评分（字段名沿用 `deepface_*`，数据来自 AffectNet 情绪分类）
- `deepface_rating` / `deepface_quality`：等级与可信度状态（字段名沿用 `deepface_*`）
- `emonet_metrics` / `emonet_scores`：Valence/Arousal 指标与评分（`valence_mean/arousal_mean/va_volatility/activation_ratio/coverage` 与 `valence_score/arousal_score/stability_score/activation_score/va_score`）
- `emonet_rating` / `emonet_quality`：等级与可信度状态（字段名沿用 `emonet_*`）
- `attention_angle_raw` / `attention_angle_clean`：注意力原始与清洗后的视线角序列
- `attention_metrics` / `attention_scores`：注意力指标与评分（含 `attention_score`、`screen_focus_ratio`）
- `attention_quality`：注意力质量摘要（`valid_ratio`、`long_eye_close_ratio` 等）
- `attention_events`：眨眼与长闭眼分段事件（支持定位时间片段，含 `blink_segments` 与 `long_eye_close_segments`）

实时预览参数：

- 在任务运行期间可追加 `allow_partial=1`，例如：
  - `GET /api/tasks/<task_id>/summary?allow_partial=1`
  - `GET /api/tasks/<task_id>/frame/<idx>?allow_partial=1`
  - `GET /api/tasks/<task_id>/frame/<idx>/image?allow_partial=1`

## 5. 下一步建议

1. 增加 SQLite（任务、会话、报告）替代 JSON 任务仓库
2. 引入 Celery + Redis 替代线程任务
3. 前端改为组件化（例如 Vue/React）提升交互与可维护性
4. 接入 ASR（faster-whisper）并输出文本时间戳
5. 接入 rPPG（POS/CHROM）并统一对齐 OpenFace 时间轴
