import math
import re
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

from ..utils.json_io import read_json, write_json
from .deepface_service import DeepFaceService
from .emonet_service import EmoNetService
from .frame_service import FrameService
from .game_ocr_service import GameOcrService
from .gameplay_timeline_service import GameplayTimelineService
from .openface_service import OpenFaceService
from .task_store import TaskStore


class PipelineService:
    DEEPFACE_EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
    DEEPFACE_EMA_ALPHA = 0.30
    DEEPFACE_AMBIGUITY_MARGIN_THRESHOLD = 0.10
    DEEPFACE_LABEL_MIN_TOP1 = 0.40
    DEEPFACE_SURPRISE_MIN_TOP1 = 0.55
    DEEPFACE_EVENT_MIN_FRAMES = 2
    DEEPFACE_DOMINANCE_SCORE_LOW = 0.35
    DEEPFACE_DOMINANCE_SCORE_HIGH = 0.85
    DEEPFACE_PHASE_MIN_DURATION_SEC = 2.0
    DEEPFACE_PHASE_RATE_LOW = 0.4
    DEEPFACE_PHASE_RATE_HIGH = 2.5
    DEEPFACE_PHASE_RATE_MAX = 5.0
    DEEPFACE_PHASE_PURITY_NEUTRAL_WEIGHT = 0.5
    DEEPFACE_PHASE_SCORE_PURITY_WEIGHT = 0.6
    DEEPFACE_ACTIVITY_EVENT_LOW = 0.5
    DEEPFACE_ACTIVITY_EVENT_HIGH = 6.0
    DEEPFACE_ACTIVITY_VARIANCE_LOW = 0.05
    DEEPFACE_ACTIVITY_VARIANCE_HIGH = 0.25
    DEEPFACE_MIN_COVERAGE = 0.40

    EMONET_VALENCE_SCORE_LOW = -0.20
    EMONET_VALENCE_SCORE_HIGH = 0.60
    EMONET_AROUSAL_SCORE_LOW = -0.10
    EMONET_AROUSAL_SCORE_HIGH = 0.50
    EMONET_STABILITY_VOLATILITY_HIGH = 0.20
    EMONET_STABILITY_VOLATILITY_LOW = 0.08
    EMONET_HIGH_AROUSAL_THRESHOLD = 0.25
    EMONET_ACTIVATION_SCORE_LOW = 0.08
    EMONET_ACTIVATION_SCORE_HIGH = 0.35
    EMONET_ACTIVATION_SCORE_POWER = 1.3
    EMONET_SCORE_POWER = 0.6
    EMONET_MIN_COVERAGE = 0.40
    EMONET_MIN_SUCCESS_PROB = 0.20
    SCORE_FLOOR_WITH_DATA = 50.0

    def __init__(self, config: dict, task_store: TaskStore):
        self.config = config
        self.task_store = task_store
        self.openface_service = OpenFaceService(config)
        self.frame_service = FrameService(config)
        self.deepface_service = DeepFaceService(config)
        self.emonet_service = EmoNetService(config)
        self.game_ocr_service = GameOcrService(config)
        self.gameplay_timeline_service = GameplayTimelineService(config)

    def run_async(self, task_id: str) -> None:
        thread = threading.Thread(target=self._run_task, args=(task_id,), daemon=True)
        thread.start()

    def _run_task(self, task_id: str) -> None:
        task = self.task_store.get_task(task_id)
        if not task:
            return

        input_data = task.get("input", {})
        video_path = Path(input_data.get("video_path", ""))
        mode = str(input_data.get("mode", "auto_main"))
        if mode not in {"auto_main", "multi", "single"}:
            mode = "auto_main"
        frame_stride = 1
        max_frames = 0
        macro_stride = max(1, int(input_data.get("macro_stride", 5)))
        macro_interval_sec = self._normalize_macro_interval_sec(input_data.get("macro_interval_sec"))
        macro_mode = "time" if macro_interval_sec is not None else "frame"
        game_video_path = self._normalize_optional_video_path(input_data.get("game_video_path"))
        game_ocr_interval_sec = self._normalize_game_ocr_interval_sec(input_data.get("game_ocr_interval_sec"))
        game_ocr_overview = self._build_game_ocr_overview(
            summary={},
            enabled=bool(game_video_path),
            status="pending" if game_video_path else "skipped",
            interval_sec=game_ocr_interval_sec,
            error=None,
        )
        gameplay_analysis_overview = self._build_gameplay_analysis_overview(
            summary={},
            enabled=bool(game_video_path),
            status="pending" if game_video_path else "skipped",
            error=None,
        )
        game_ocr_result = None
        gameplay_analysis_result = None

        result_root = Path(self.config["RESULT_DIR"]) / task_id
        result_root.mkdir(parents=True, exist_ok=True)
        live_frame_analysis_path = result_root / "frame_analysis_live.json"
        live_summary_path = result_root / "summary_live.json"

        try:
            self.task_store.set_artifact(task_id, "frame_analysis_live_json", str(live_frame_analysis_path))
            self.task_store.set_artifact(task_id, "summary_live_json", str(live_summary_path))
            write_json(live_frame_analysis_path, [])

            self.task_store.update_task(
                task_id,
                status="running",
                stage="openface",
                progress=0.05,
                error=None,
                progress_detail={"done": 0, "total": 0},
                game_ocr_status=game_ocr_overview.get("status"),
                game_ocr_error=game_ocr_overview.get("error"),
                game_ocr_summary=game_ocr_overview,
                gameplay_analysis_status=gameplay_analysis_overview.get("status"),
                gameplay_analysis_error=gameplay_analysis_overview.get("error"),
                gameplay_analysis_summary=gameplay_analysis_overview,
            )
            self._record_progress(task_id, 0.05, "openface", "开始执行 OpenFace（多人脸检测 + 主体脸锁定）")

            openface_output = self.openface_service.run(
                video_path=video_path,
                mode=mode,
                output_dir=result_root / "openface",
            )
            csv_path = Path(openface_output["csv_path"])
            openface_mode_used = str(openface_output.get("mode_used", "single"))
            self.task_store.set_artifact(task_id, "openface_csv", str(csv_path))
            self._record_progress(
                task_id,
                0.2,
                "timeline",
                f"OpenFace 完成（{openface_mode_used}），CSV: {csv_path.name}",
            )

            timeline_df = self.frame_service.load_timeline_dataframe(
                csv_path,
                frame_stride,
                max_frames,
                video_path=video_path,
                select_main_face=openface_mode_used == "multi",
            )
            main_face_info = dict(timeline_df.attrs.get("main_face_info") or {})
            frame_items = self.frame_service.build_frame_items(timeline_df)
            time_start = float(timeline_df["timestamp"].iloc[0])
            time_end = float(timeline_df["timestamp"].iloc[-1])
            macro_config_text = self._format_macro_config_text(macro_mode, macro_stride, macro_interval_sec)
            self._record_progress(
                task_id,
                0.24,
                "timeline",
                f"初始时间轴（OpenFace timestamp）: {time_start:.3f}s -> {time_end:.3f}s（{macro_config_text}）",
            )

            if main_face_info.get("applied"):
                selected_face_id = main_face_info.get("dominant_face_id")
                face_count = int(main_face_info.get("face_count", 0))
                self._record_progress(
                    task_id,
                    0.26,
                    "timeline",
                    f"多人脸场景已锁定主体脸（face_id={selected_face_id}，候选 {face_count}）",
                )

            total = len(frame_items)
            if macro_mode == "time":
                if macro_interval_sec is None:
                    raise RuntimeError("macro_mode=time 但 macro_interval_sec 为空")
                macro_total = self._estimate_macro_total_by_time(frame_items, float(macro_interval_sec))
            else:
                macro_total = (total + macro_stride - 1) // macro_stride
            write_json(
                live_summary_path,
                self._build_summary(
                    task_id,
                    timeline_df.iloc[:0].copy(),
                    [],
                    is_partial=True,
                    expected_total=total,
                    macro_stride=macro_stride,
                    macro_mode=macro_mode,
                    macro_interval_sec=macro_interval_sec,
                    main_face_info=main_face_info,
                    game_ocr_overview=game_ocr_overview,
                    gameplay_analysis_overview=gameplay_analysis_overview,
                ),
            )
            self._record_progress(
                task_id,
                0.3,
                "frame_extract",
                f"准备抽帧，数量: {total}",
                done=0,
                total=total,
            )
            frame_items = self.frame_service.extract_frames(
                video_path=video_path,
                items=frame_items,
                frame_dir=result_root / "frames",
            )

            timeline_df, aligned_count = self._apply_decoded_timestamps_to_timeline(timeline_df, frame_items)
            if aligned_count > 0:
                aligned_start = float(timeline_df["timestamp"].iloc[0])
                aligned_end = float(timeline_df["timestamp"].iloc[-1])
                self._record_progress(
                    task_id,
                    0.405,
                    "timeline",
                    f"时间轴已校准为视频秒级时间: {aligned_start:.3f}s -> {aligned_end:.3f}s",
                )

            if macro_mode == "time":
                if macro_interval_sec is None:
                    raise RuntimeError("macro_mode=time 但 macro_interval_sec 为空")
                macro_total = self._estimate_macro_total_by_time(frame_items, float(macro_interval_sec))
            else:
                macro_total = (total + macro_stride - 1) // macro_stride

            self._record_progress(
                task_id,
                0.4,
                "emotion_inference",
                f"抽帧完成，开始宏表情采样分析（{macro_config_text}）",
                done=0,
                total=macro_total,
            )

            analyzed_frames = []
            macro_done = 0
            notify_step = max(1, total // 40)
            next_macro_timestamp = None

            for idx, item in enumerate(frame_items):
                position = idx + 1
                if macro_mode == "time":
                    if macro_interval_sec is None:
                        raise RuntimeError("macro_mode=time 但 macro_interval_sec 为空")
                    should_run_macro, next_macro_timestamp = self._should_run_macro_by_time(
                        timestamp=item.get("timestamp", 0.0),
                        interval_sec=float(macro_interval_sec),
                        next_timestamp=next_macro_timestamp,
                    )
                else:
                    should_run_macro = idx % macro_stride == 0
                row = {
                    "index": item["index"],
                    "frame": item["frame"],
                    "openface_row": item.get("openface_row"),
                    "face_id": item.get("face_id"),
                    "frame_index": item.get("frame_index"),
                    "timestamp": item["timestamp"],
                    "timestamp_source": str(item.get("timestamp_source") or "openface_csv"),
                    "image_path": item["image_path"],
                    "macro_inference": should_run_macro,
                    "macro_mode": macro_mode,
                    "macro_interval_sec": macro_interval_sec,
                    "deepface": None,
                    "emonet": None,
                    "errors": [],
                }

                if should_run_macro:
                    try:
                        row["deepface"] = self.deepface_service.analyze(item["image_path"])
                    except Exception as e:
                        row["errors"].append(f"宏表情识别失败: {e}")

                    try:
                        row["emonet"] = self.emonet_service.analyze(item["image_path"])
                    except Exception as e:
                        row["errors"].append(f"AffectNet V/A 失败: {e}")

                    macro_done += 1

                analyzed_frames.append(row)

                if position == total or position % notify_step == 0:
                    self._write_live_outputs(
                        task_id=task_id,
                        timeline_df=timeline_df,
                        analyzed_frames=analyzed_frames,
                        frame_analysis_path=live_frame_analysis_path,
                        summary_path=live_summary_path,
                        expected_total=total,
                        macro_stride=macro_stride,
                        macro_mode=macro_mode,
                        macro_interval_sec=macro_interval_sec,
                        main_face_info=main_face_info,
                        game_ocr_overview=game_ocr_overview,
                        gameplay_analysis_overview=gameplay_analysis_overview,
                    )
                    progress = 0.4 + 0.5 * (position / total)
                    self._record_progress(
                        task_id,
                        progress,
                        "emotion_inference",
                        f"宏表情分析进度 {macro_done}/{macro_total}（时间轴遍历 {position}/{total}）",
                        done=macro_done,
                        total=macro_total,
                    )

            self._record_progress(task_id, 0.92, "summary", "主分析完成，准备汇总", done=macro_done, total=macro_total)

            if game_video_path:
                self._record_progress(
                    task_id,
                    0.935,
                    "game_ocr",
                    f"开始游戏画面 OCR（间隔 {game_ocr_interval_sec:.2f}s）",
                    done=macro_done,
                    total=macro_total,
                )
                game_ocr_overview = self._build_game_ocr_overview(
                    summary={},
                    enabled=True,
                    status="running",
                    interval_sec=game_ocr_interval_sec,
                    error=None,
                )
                self.task_store.update_task(
                    task_id,
                    game_ocr_status="running",
                    game_ocr_error=None,
                    game_ocr_summary=game_ocr_overview,
                )
                self._write_live_outputs(
                    task_id=task_id,
                    timeline_df=timeline_df,
                    analyzed_frames=analyzed_frames,
                    frame_analysis_path=live_frame_analysis_path,
                    summary_path=live_summary_path,
                    expected_total=total,
                    macro_stride=macro_stride,
                    macro_mode=macro_mode,
                    macro_interval_sec=macro_interval_sec,
                    main_face_info=main_face_info,
                    game_ocr_overview=game_ocr_overview,
                    gameplay_analysis_overview=gameplay_analysis_overview,
                )

                reference_timeline = [
                    {
                        "index": int(row.get("index", 0)),
                        "frame": int(row.get("frame", 0)),
                        "timestamp": float(row.get("timestamp", 0.0)),
                    }
                    for row in analyzed_frames
                ]

                ocr_progress_state = {
                    "last_done": -1,
                    "last_wait_bucket": None,
                    "timeout_count": 0,
                }

                def _on_game_ocr_progress(payload: dict) -> None:
                    if not isinstance(payload, dict):
                        return

                    event_type = str(payload.get("event") or "").strip()
                    if event_type == "sample_wait":
                        sample_index = self._safe_int(payload.get("sample_index"))
                        total = self._safe_int(payload.get("total"))
                        ts = self._safe_float(payload.get("timestamp_sec"))
                        wait_elapsed = self._safe_float(payload.get("wait_elapsed_sec"))

                        if sample_index is None or wait_elapsed is None or wait_elapsed < 0.0:
                            return

                        wait_bucket = (int(sample_index), int(wait_elapsed // 10))
                        if wait_bucket == ocr_progress_state.get("last_wait_bucket"):
                            return
                        ocr_progress_state["last_wait_bucket"] = wait_bucket

                        if total is not None and total > 0:
                            ratio = min(1.0, max(0.0, float(sample_index) / float(total)))
                            progress_value = 0.935 + 0.025 * ratio
                            sample_text = f"样本 {sample_index + 1}/{total}"
                        else:
                            progress_value = min(0.96, 0.935 + 0.00025 * float(sample_index + 1))
                            sample_text = f"样本 {sample_index + 1}"

                        if ts is not None:
                            sample_text = f"{sample_text} · t {ts:.2f}s"
                        sample_text = f"{sample_text} · 已等待 {wait_elapsed:.1f}s"

                        self._record_progress(
                            task_id,
                            progress_value,
                            "game_ocr",
                            f"游戏画面 OCR 等待中（{sample_text}）",
                        )
                        return

                    done = self._safe_int(payload.get("done"))
                    if done is None or done <= 0:
                        return
                    if done == ocr_progress_state["last_done"]:
                        return
                    ocr_progress_state["last_done"] = int(done)
                    ocr_progress_state["last_wait_bucket"] = None

                    total = self._safe_int(payload.get("total"))
                    ts = self._safe_float(payload.get("timestamp_sec"))
                    line_count = self._safe_int(payload.get("line_count"))
                    ocr_status = str(payload.get("ocr_status") or "ok").strip().lower()
                    ocr_elapsed_sec = self._safe_float(payload.get("ocr_elapsed_sec"))
                    if ocr_status == "timeout":
                        ocr_progress_state["timeout_count"] = int(ocr_progress_state.get("timeout_count", 0)) + 1

                    running_summary = {
                        "samples_done": int(done),
                        "samples_total": int(total) if total is not None and total > 0 else None,
                        "current_timestamp_sec": float(ts) if ts is not None else None,
                        "current_line_count": int(line_count) if line_count is not None and line_count >= 0 else None,
                        "timeout_count": int(ocr_progress_state.get("timeout_count", 0)),
                    }
                    game_ocr_running_overview = self._build_game_ocr_overview(
                        summary=running_summary,
                        enabled=True,
                        status="running",
                        interval_sec=game_ocr_interval_sec,
                        error=None,
                    )
                    self.task_store.update_task(
                        task_id,
                        game_ocr_status="running",
                        game_ocr_error=None,
                        game_ocr_summary=game_ocr_running_overview,
                    )

                    if total is not None and total > 0:
                        ratio = min(1.0, max(0.0, float(done) / float(total)))
                        progress_value = 0.935 + 0.025 * ratio
                        sample_text = f"样本 {done}/{total}"
                    else:
                        progress_value = min(0.96, 0.935 + 0.00025 * float(done))
                        sample_text = f"样本 {done}"

                    if ts is not None:
                        sample_text = f"{sample_text} · t {ts:.2f}s"
                    if line_count is not None and line_count >= 0:
                        sample_text = f"{sample_text} · 文本行 {line_count}"
                    if ocr_elapsed_sec is not None and ocr_elapsed_sec >= 0.0:
                        sample_text = f"{sample_text} · OCR {ocr_elapsed_sec:.2f}s"
                    if ocr_status == "timeout":
                        sample_text = f"{sample_text} · 超时已跳过"
                    elif ocr_status and ocr_status != "ok":
                        sample_text = f"{sample_text} · 状态 {ocr_status}"

                    self._record_progress(
                        task_id,
                        progress_value,
                        "game_ocr",
                        f"游戏画面 OCR 处理中（{sample_text}）",
                    )

                try:
                    game_ocr_result = self.game_ocr_service.analyze_video(
                        video_path=game_video_path,
                        output_dir=result_root / "game_ocr",
                        interval_sec=game_ocr_interval_sec,
                        reference_timeline=reference_timeline,
                        progress_callback=_on_game_ocr_progress,
                    )

                    game_ocr_path = result_root / "game_ocr_result.json"
                    write_json(game_ocr_path, game_ocr_result)
                    self.task_store.set_artifact(task_id, "game_ocr_json", str(game_ocr_path))

                    ocr_summary = game_ocr_result.get("summary") or {}
                    game_ocr_overview = self._build_game_ocr_overview(
                        summary=ocr_summary,
                        enabled=True,
                        status="done",
                        interval_sec=game_ocr_interval_sec,
                        error=None,
                    )
                    self.task_store.update_task(
                        task_id,
                        game_ocr_status="done",
                        game_ocr_error=None,
                        game_ocr_summary=game_ocr_overview,
                    )

                    samples_total = int(ocr_summary.get("samples_total", 0) or 0)
                    samples_with_text = int(ocr_summary.get("samples_with_text", 0) or 0)
                    samples_timeout = int(ocr_summary.get("samples_timeout", 0) or 0)
                    completion_text = f"游戏画面 OCR 完成（样本 {samples_total}，含文本 {samples_with_text}）"
                    if samples_timeout > 0:
                        completion_text = f"{completion_text}，超时跳过 {samples_timeout}"
                    self._record_progress(
                        task_id,
                        0.965,
                        "game_ocr",
                        completion_text,
                        done=macro_done,
                        total=macro_total,
                    )

                    self._record_progress(
                        task_id,
                        0.972,
                        "gameplay_analysis",
                        "开始游戏流程划分与模块情绪分析",
                        done=macro_done,
                        total=macro_total,
                    )
                    gameplay_analysis_overview = self._build_gameplay_analysis_overview(
                        summary={},
                        enabled=True,
                        status="running",
                        error=None,
                    )
                    self.task_store.update_task(
                        task_id,
                        gameplay_analysis_status="running",
                        gameplay_analysis_error=None,
                        gameplay_analysis_summary=gameplay_analysis_overview,
                    )

                    try:
                        gameplay_analysis_result = self.gameplay_timeline_service.analyze(
                            ocr_result=game_ocr_result,
                            analyzed_frames=analyzed_frames,
                        )
                        gameplay_analysis_result = self._enrich_gameplay_analysis_result(
                            payload=gameplay_analysis_result,
                            timeline_df=timeline_df,
                            analyzed_frames=analyzed_frames,
                        )
                        gameplay_analysis_path = result_root / "gameplay_analysis.json"
                        write_json(gameplay_analysis_path, gameplay_analysis_result)
                        self.task_store.set_artifact(task_id, "gameplay_analysis_json", str(gameplay_analysis_path))

                        gameplay_summary = gameplay_analysis_result.get("summary") or {}
                        gameplay_analysis_overview = self._build_gameplay_analysis_overview(
                            summary=gameplay_summary,
                            enabled=True,
                            status="done",
                            error=None,
                        )
                        self.task_store.update_task(
                            task_id,
                            gameplay_analysis_status="done",
                            gameplay_analysis_error=None,
                            gameplay_analysis_summary=gameplay_analysis_overview,
                        )

                        segments_total = int(gameplay_summary.get("segments_total", 0) or 0)
                        modules_detected = gameplay_summary.get("modules_detected") or []
                        modules_count = len(modules_detected) if isinstance(modules_detected, list) else 0
                        self._record_progress(
                            task_id,
                            0.978,
                            "gameplay_analysis",
                            f"游戏流程分析完成（分段 {segments_total}，模块 {modules_count}）",
                            done=macro_done,
                            total=macro_total,
                        )
                    except Exception as gameplay_error:
                        gameplay_message = str(gameplay_error)
                        gameplay_analysis_overview = self._build_gameplay_analysis_overview(
                            summary={},
                            enabled=True,
                            status="failed",
                            error=gameplay_message,
                        )
                        self.task_store.update_task(
                            task_id,
                            gameplay_analysis_status="failed",
                            gameplay_analysis_error=gameplay_message,
                            gameplay_analysis_summary=gameplay_analysis_overview,
                        )
                        self.task_store.add_log(task_id, f"游戏流程分析失败（已跳过，不影响主分析）: {gameplay_message}")
                except Exception as game_ocr_error:
                    game_ocr_message = str(game_ocr_error)
                    game_ocr_overview = self._build_game_ocr_overview(
                        summary={},
                        enabled=True,
                        status="failed",
                        interval_sec=game_ocr_interval_sec,
                        error=game_ocr_message,
                    )
                    self.task_store.update_task(
                        task_id,
                        game_ocr_status="failed",
                        game_ocr_error=game_ocr_message,
                        game_ocr_summary=game_ocr_overview,
                    )
                    self.task_store.add_log(task_id, f"游戏画面 OCR 失败（已跳过，不影响主分析）: {game_ocr_message}")
                    gameplay_analysis_overview = self._build_gameplay_analysis_overview(
                        summary={},
                        enabled=True,
                        status="skipped",
                        error="依赖游戏画面 OCR 失败",
                    )
                    self.task_store.update_task(
                        task_id,
                        gameplay_analysis_status="skipped",
                        gameplay_analysis_error="依赖游戏画面 OCR 失败",
                        gameplay_analysis_summary=gameplay_analysis_overview,
                    )
            else:
                game_ocr_overview = self._build_game_ocr_overview(
                    summary={},
                    enabled=False,
                    status="skipped",
                    interval_sec=game_ocr_interval_sec,
                    error=None,
                )
                self.task_store.update_task(
                    task_id,
                    game_ocr_status="skipped",
                    game_ocr_error=None,
                    game_ocr_summary=game_ocr_overview,
                    gameplay_analysis_status="skipped",
                    gameplay_analysis_error=None,
                    gameplay_analysis_summary=gameplay_analysis_overview,
                )

            self._record_progress(task_id, 0.985, "summary", "正在汇总时序指标", done=macro_done, total=macro_total)
            summary = self._build_summary(
                task_id,
                timeline_df,
                analyzed_frames,
                is_partial=False,
                expected_total=total,
                macro_stride=macro_stride,
                macro_mode=macro_mode,
                macro_interval_sec=macro_interval_sec,
                main_face_info=main_face_info,
                game_ocr_overview=game_ocr_overview,
                gameplay_analysis_overview=gameplay_analysis_overview,
            )

            frame_analysis_path = result_root / "frame_analysis.json"
            summary_path = result_root / "summary.json"
            write_json(frame_analysis_path, analyzed_frames)
            write_json(summary_path, summary)

            self.task_store.set_artifact(task_id, "frame_analysis_json", str(frame_analysis_path))
            self.task_store.set_artifact(task_id, "summary_json", str(summary_path))
            self.task_store.set_artifact(task_id, "result_root", str(result_root))

            self._record_progress(task_id, 0.992, "report", "正在生成综合报表", done=macro_done, total=macro_total)
            try:
                report_artifacts = self._generate_analysis_report_artifacts(
                    task_id=task_id,
                    input_data=input_data,
                    summary=summary,
                    game_ocr_result=game_ocr_result,
                    gameplay_analysis_result=gameplay_analysis_result,
                )
                for artifact_key, artifact_path in report_artifacts.items():
                    if not artifact_path:
                        continue
                    self.task_store.set_artifact(task_id, artifact_key, artifact_path)
                report_json_path = report_artifacts.get("analysis_report_json")
                if report_json_path:
                    report_name = Path(report_json_path).name
                    self._record_progress(
                        task_id,
                        0.997,
                        "report",
                        f"综合报表已生成: {report_name}",
                        done=macro_done,
                        total=macro_total,
                    )
            except Exception as report_error:
                self.task_store.add_log(task_id, f"综合报表生成失败（已跳过，不影响主分析）: {report_error}")

            self.task_store.update_task(
                task_id,
                status="done",
                stage="done",
                progress=1.0,
                progress_detail={"done": macro_done, "total": macro_total},
                game_ocr_status=game_ocr_overview.get("status"),
                game_ocr_error=game_ocr_overview.get("error"),
                game_ocr_summary=game_ocr_overview,
                gameplay_analysis_status=gameplay_analysis_overview.get("status"),
                gameplay_analysis_error=gameplay_analysis_overview.get("error"),
                gameplay_analysis_summary=gameplay_analysis_overview,
            )
            self._record_progress(task_id, 1.0, "done", "任务执行完成", done=macro_done, total=macro_total)

        except Exception as e:
            self.task_store.update_task(task_id, status="failed", stage="failed", error=str(e))
            self.task_store.add_log(task_id, f"任务失败: {e}")

    def _write_live_outputs(
        self,
        task_id: str,
        timeline_df,
        analyzed_frames: list[dict],
        frame_analysis_path: Path,
        summary_path: Path,
        expected_total: int,
        macro_stride: int,
        macro_mode: str,
        macro_interval_sec: float | None,
        main_face_info: dict | None,
        game_ocr_overview: dict | None,
        gameplay_analysis_overview: dict | None,
    ) -> None:
        processed = len(analyzed_frames)
        partial_df = timeline_df.iloc[:processed].copy()
        summary = self._build_summary(
            task_id,
            partial_df,
            analyzed_frames,
            is_partial=True,
            expected_total=expected_total,
            macro_stride=macro_stride,
            macro_mode=macro_mode,
            macro_interval_sec=macro_interval_sec,
            main_face_info=main_face_info,
            game_ocr_overview=game_ocr_overview,
            gameplay_analysis_overview=gameplay_analysis_overview,
        )
        write_json(frame_analysis_path, analyzed_frames)
        write_json(summary_path, summary)

    def _record_progress(
        self,
        task_id: str,
        progress: float,
        stage: str,
        message: str,
        done: int | None = None,
        total: int | None = None,
    ) -> None:
        clamped = max(0.0, min(1.0, float(progress)))
        payload = {
            "stage": stage,
            "progress": round(clamped, 4),
        }
        if done is not None and total is not None:
            payload["progress_detail"] = {"done": int(done), "total": int(total)}

        self.task_store.update_task(task_id, **payload)

        bar = self._format_bar(clamped)
        percent = int(round(clamped * 100))
        detail = ""
        if done is not None and total is not None and total > 0:
            detail = f" ({done}/{total})"
        content = f"{message} {bar} {percent}%{detail}"
        self.task_store.add_log(task_id, content)
        print(f"[{task_id[:8]}] {content}")

    @staticmethod
    def _format_bar(progress: float, width: int = 24) -> str:
        filled = int(round(progress * width))
        filled = max(0, min(width, filled))
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def _build_summary(
        self,
        task_id: str,
        timeline_df,
        analyzed_frames: list[dict],
        is_partial: bool,
        expected_total: int | None = None,
        macro_stride: int = 5,
        macro_mode: str = "frame",
        macro_interval_sec: float | None = None,
        main_face_info: dict | None = None,
        game_ocr_overview: dict | None = None,
        gameplay_analysis_overview: dict | None = None,
    ) -> dict:
        timestamps = [float(row.get("timestamp", 0.0)) for row in analyzed_frames]
        frames = [int(row.get("frame", 0)) for row in analyzed_frames]
        timestamp_source_counter = Counter()
        for row in analyzed_frames:
            source_text = str(row.get("timestamp_source") or "").strip()
            if source_text:
                timestamp_source_counter[source_text] += 1
        timestamp_source = "openface_csv"
        if timestamp_source_counter:
            timestamp_source = timestamp_source_counter.most_common(1)[0][0]

        attention_analysis = self.frame_service.compute_attention_analysis(timeline_df)
        attention_raw = attention_analysis.get("raw") or []
        attention_clean = attention_analysis.get("clean") or []
        attention_state = attention_analysis.get("state") or []
        attention_metrics = attention_analysis.get("metrics") or {}
        attention_scores = attention_analysis.get("scores") or {}
        attention_events = attention_analysis.get("events") or {}

        valence = []
        arousal = []
        for row in analyzed_frames:
            emonet = row.get("emonet")
            if not isinstance(emonet, dict) or not self._is_valid_emonet_result(emonet):
                valence.append(None)
                arousal.append(None)
                continue

            valence.append(self._safe_float(emonet.get("valence")))
            arousal.append(self._safe_float(emonet.get("arousal")))

        abnormal_events = self.frame_service.detect_abnormal_events(timeline_df)

        deepface_outputs = self._compute_deepface_process_outputs(analyzed_frames)
        deepface_overall = deepface_outputs.get("overall") or {}
        deepface_metrics = deepface_outputs.get("metrics") or {}
        deepface_scores = deepface_outputs.get("scores") or {}
        deepface_rating = deepface_outputs.get("rating")
        deepface_quality = deepface_outputs.get("quality") or {}

        emonet_outputs = self._compute_emonet_process_outputs(analyzed_frames)
        emonet_metrics = emonet_outputs.get("metrics") or {}
        emonet_scores = emonet_outputs.get("scores") or {}
        emonet_rating = emonet_outputs.get("rating")
        emonet_quality = emonet_outputs.get("quality") or {}

        processed_frames = len(analyzed_frames)
        total_frames = int(expected_total) if expected_total is not None else processed_frames
        macro_flags = [bool(row.get("macro_inference")) for row in analyzed_frames]

        return {
            "task_id": task_id,
            "frames_total": processed_frames,
            "processed_frames": processed_frames,
            "expected_frames": total_frames,
            "is_partial": bool(is_partial),
            "macro_stride": int(macro_stride),
            "macro_mode": str(macro_mode),
            "macro_interval_sec": float(macro_interval_sec) if macro_interval_sec is not None else None,
            "macro_flags": macro_flags,
            "main_face_info": main_face_info or {},
            "game_ocr": game_ocr_overview or {},
            "gameplay_analysis": gameplay_analysis_overview or {},
            "timestamp_source": timestamp_source,
            "frames": frames,
            "timestamps": timestamps,
            "valence": valence,
            "arousal": arousal,
            "deepface_overall": deepface_overall,
            "deepface_metrics": deepface_metrics,
            "deepface_scores": deepface_scores,
            "deepface_process_score": deepface_scores.get("deepface_process_score"),
            "deepface_rating": deepface_rating,
            "deepface_quality": deepface_quality,
            "emonet_metrics": emonet_metrics,
            "emonet_scores": emonet_scores,
            "emonet_score": emonet_scores.get("va_score") or emonet_scores.get("emonet_score"),
            "emonet_rating": emonet_rating,
            "emonet_quality": emonet_quality,
            "attention_angle": attention_clean,
            "attention_angle_raw": attention_raw,
            "attention_angle_clean": attention_clean,
            "attention_state": attention_state,
            "attention_metrics": attention_metrics,
            "attention_scores": attention_scores,
            "attention_events": attention_events,
            "attention_score": attention_scores.get("attention_score"),
            "attention_quality": {
                "quality_status": attention_metrics.get("quality_status"),
                "valid_ratio": attention_metrics.get("valid_ratio"),
                "long_eye_close_ratio": attention_metrics.get("long_eye_close_ratio"),
                "long_eye_close_count": attention_metrics.get("long_eye_close_count"),
            },
            "abnormal_events": abnormal_events,
            "timeline": [
                {
                    "index": int(row.get("index", 0)),
                    "frame": int(row.get("frame", 0)),
                    "openface_row": int(row.get("openface_row", 0)) if row.get("openface_row") is not None else None,
                    "face_id": int(row.get("face_id", 0)) if row.get("face_id") is not None else None,
                    "timestamp": float(row.get("timestamp", 0.0)),
                    "timestamp_source": str(row.get("timestamp_source") or "openface_csv"),
                }
                for row in analyzed_frames
            ],
        }

    @staticmethod
    def _compute_deepface_distribution(analyzed_frames: list[dict]) -> tuple[dict[str, float], int]:
        outputs = PipelineService._compute_deepface_process_outputs(analyzed_frames)
        overall = outputs.get("overall") or {}
        distribution = overall.get("emotion_distribution") or {}
        frames_used = int(overall.get("frames_used") or 0)
        return distribution, frames_used

    @classmethod
    def _build_phase_segments(cls, segments: list[dict], min_duration_sec: float) -> list[dict]:
        if not segments:
            return []

        valid_flags = [float(seg.get("duration_sec", 0.0)) >= min_duration_sec for seg in segments]
        if not any(valid_flags):
            dominant = max(segments, key=lambda seg: float(seg.get("duration_sec", 0.0)))
            return [
                {
                    "label": dominant.get("label"),
                    "segments": list(segments),
                    "start_index": int(segments[0].get("start_index", 0)),
                    "end_index": int(segments[-1].get("end_index", 0)),
                }
            ]

        next_valid_label: list[str | None] = [None] * len(segments)
        upcoming = None
        for idx in range(len(segments) - 1, -1, -1):
            if valid_flags[idx]:
                upcoming = segments[idx].get("label")
            next_valid_label[idx] = upcoming

        phase_segments: list[dict] = []
        current = None
        prev_valid_label = None
        for idx, seg in enumerate(segments):
            label = seg.get("label")
            if valid_flags[idx]:
                assigned_label = label
                prev_valid_label = label
            else:
                assigned_label = prev_valid_label or next_valid_label[idx] or label

            if current is None or current.get("label") != assigned_label:
                if current is not None:
                    phase_segments.append(current)
                current = {
                    "label": assigned_label,
                    "segments": [],
                    "start_index": int(seg.get("start_index", 0)),
                    "end_index": int(seg.get("end_index", 0)),
                }

            current["segments"].append(seg)
            current["end_index"] = int(seg.get("end_index", current["end_index"]))

        if current is not None:
            phase_segments.append(current)
        return phase_segments

    @classmethod
    def _compute_phase_impurity(cls, phase_segments: list[dict], neutral_weight: float) -> float:
        if not phase_segments:
            return 0.0

        total_duration = 0.0
        mismatch_duration = 0.0
        for phase in phase_segments:
            phase_label = phase.get("label")
            for seg in phase.get("segments", []):
                duration = float(seg.get("duration_sec", 0.0))
                if duration <= 0:
                    continue
                total_duration += duration
                seg_label = seg.get("label")
                if seg_label == phase_label:
                    continue
                weight = 1.0
                if seg_label == "neutral" or phase_label == "neutral":
                    weight = neutral_weight
                mismatch_duration += duration * weight

        if total_duration <= 1e-9:
            return 0.0
        return max(0.0, min(1.0, mismatch_duration / total_duration))

    @classmethod
    def _compute_deepface_process_outputs(cls, analyzed_frames: list[dict]) -> dict:
        macro_sampled_frames = sum(1 for row in analyzed_frames if bool(row.get("macro_inference")))
        sampled_frames_total = macro_sampled_frames if macro_sampled_frames > 0 else len(analyzed_frames)
        samples = cls._collect_deepface_samples(analyzed_frames)
        frames_used = len(samples)

        if frames_used == 0:
            coverage_denom = sampled_frames_total
            coverage = 0.0 if coverage_denom <= 0 else 0.0
            return {
                "overall": {
                    "frames_used": 0,
                    "sampled_frames": int(sampled_frames_total),
                    "emotion_distribution": {},
                    "top_emotions": [],
                },
                "metrics": {
                    "tone": 0.0,
                    "pa": 0.0,
                    "na": 0.0,
                    "volatility": 0.0,
                    "switch_rate": 0.0,
                    "switch_count": 0,
                    "events_per_minute": 0.0,
                    "ambiguity_ratio": 1.0,
                    "coverage": float(coverage),
                    "margin_mean": None,
                    "entropy_mean": None,
                    "emotion_event_count": {},
                    "emotion_mean_duration": {},
                    "emotion_max_duration": {},
                },
                "scores": {
                    "tone_score": 0.0,
                    "stability_score": 0.0,
                    "confidence_score": 0.0,
                    "deepface_process_score": 0.0,
                },
                "rating": "D",
                "quality": {
                    "status": "no_data",
                    "coverage_status": "insufficient",
                    "ambiguity_status": "high",
                },
            }

        extra_emotions = sorted(
            {
                emotion
                for sample in samples
                for emotion in sample["probs"].keys()
                if emotion not in cls.DEEPFACE_EMOTIONS
            }
        )
        emotions = list(cls.DEEPFACE_EMOTIONS) + extra_emotions
        emotion_index = {emotion: idx for idx, emotion in enumerate(emotions)}

        timestamps = cls._sanitize_time_series([sample["timestamp"] for sample in samples])
        durations = cls._compute_sample_durations(timestamps)
        total_duration = sum(durations)
        if total_duration <= 1e-9:
            durations = [1.0 for _ in samples]
            total_duration = float(len(samples))

        weights = [duration / total_duration for duration in durations]

        raw_vectors: list[list[float]] = []
        for sample in samples:
            vector = [float(sample["probs"].get(emotion, 0.0)) for emotion in emotions]
            vec_sum = sum(vector)
            if vec_sum <= 1e-9:
                vector = [0.0 for _ in emotions]
            else:
                vector = [value / vec_sum for value in vector]
            raw_vectors.append(vector)

        smooth_vectors = cls._ema_smooth_vectors(raw_vectors, cls.DEEPFACE_EMA_ALPHA)

        overall_distribution_raw = cls._weighted_average_vector(smooth_vectors, weights)

        labels: list[str] = []
        margins: list[float] = []
        entropies: list[float] = []
        mixed_duration = 0.0
        for idx, vector in enumerate(smooth_vectors):
            pairs = sorted(((value, pos) for pos, value in enumerate(vector)), reverse=True)
            top1 = pairs[0][0] if pairs else 0.0
            top_idx = pairs[0][1] if pairs else 0
            top2 = pairs[1][0] if len(pairs) > 1 else 0.0
            top2_idx = pairs[1][1] if len(pairs) > 1 else top_idx
            top3 = pairs[2][0] if len(pairs) > 2 else 0.0
            raw_margin = top1 - top2
            margins.append(float(raw_margin))
            entropies.append(float(cls._normalized_entropy(vector)))

            top_emotion = emotions[top_idx] if top_idx < len(emotions) else ""
            label_emotion = top_emotion

            if top_emotion in {"surprise", "surprised"} and top1 < cls.DEEPFACE_SURPRISE_MIN_TOP1:
                if len(pairs) > 1:
                    label_emotion = emotions[top2_idx] if top2_idx < len(emotions) else ""
                else:
                    label_emotion = ""

            if not label_emotion:
                label_emotion = top_emotion or "neutral"

            labels.append(label_emotion)

        segments = cls._build_label_segments(labels, durations, timestamps)
        events: list[dict] = []
        for segment in segments:
            if segment.get("label") == "mixed":
                continue
            start_index = int(segment.get("start_index", 0))
            end_index = int(segment.get("end_index", 0))
            if (end_index - start_index + 1) < cls.DEEPFACE_EVENT_MIN_FRAMES:
                continue
            events.append(segment)

        label_duration = {emotion: 0.0 for emotion in emotions}

        for segment in segments:
            label = segment.get("label")
            if not isinstance(label, str) or not label:
                continue
            label_text = label
            if label_text not in label_duration:
                continue
            duration = float(segment.get("duration_sec", 0.0))
            label_duration[label_text] += duration

        emotion_duration = {emotion: 0.0 for emotion in emotions}
        emotion_event_count = {emotion: 0 for emotion in emotions}
        emotion_max_duration = {emotion: 0.0 for emotion in emotions}

        event_total_duration = 0.0
        for event in events:
            label = event.get("label")
            if label not in emotion_duration:
                continue
            duration = float(event.get("duration_sec", 0.0))
            emotion_duration[label] += duration
            emotion_event_count[label] += 1
            emotion_max_duration[label] = max(emotion_max_duration[label], duration)
            event_total_duration += duration

        distribution_total_duration = sum(label_duration.values())
        if distribution_total_duration > 1e-9:
            distribution_fraction = {
                emotion: label_duration.get(emotion, 0.0) / distribution_total_duration
                for emotion in label_duration.keys()
            }
        else:
            distribution_fraction = {
                emotion: float(overall_distribution_raw[idx]) if idx < len(overall_distribution_raw) else 0.0
                for idx, emotion in enumerate(emotions)
            }

        ordered_distribution = sorted(
            ((emotion, value * 100.0) for emotion, value in distribution_fraction.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        emotion_distribution = {
            emotion: float(round(percent, 4))
            for emotion, percent in ordered_distribution
            if percent > 1e-8
        }
        top_emotions = [
            {"emotion": emotion, "percent": float(round(percent, 3))}
            for emotion, percent in ordered_distribution[:3]
            if percent > 1e-8
        ]

        emotion_mean_duration = {
            emotion: float(emotion_duration[emotion] / emotion_event_count[emotion])
            for emotion in emotions
            if emotion_event_count[emotion] > 0
        }

        total_events = int(sum(emotion_event_count.values()))
        events_per_minute = 0.0 if total_duration <= 1e-9 else total_events / (total_duration / 60.0)

        phase_min_duration_sec = float(cls.DEEPFACE_PHASE_MIN_DURATION_SEC)
        phase_segments = PipelineService._build_phase_segments(segments, phase_min_duration_sec)
        phase_change_count = max(0, len(phase_segments) - 1)
        phase_change_rate = 0.0 if total_duration <= 1e-9 else phase_change_count / (total_duration / 60.0)

        phase_impurity = PipelineService._compute_phase_impurity(
            phase_segments,
            neutral_weight=cls.DEEPFACE_PHASE_PURITY_NEUTRAL_WEIGHT,
        )
        phase_purity_ratio = max(0.0, min(1.0, 1.0 - phase_impurity))

        switch_count = 0
        prev_label = None
        for event in events:
            label = event.get("label")
            if prev_label is None:
                prev_label = label
                continue
            if label != prev_label:
                switch_count += 1
            prev_label = label
        switch_rate = 0.0 if total_duration <= 1e-9 else switch_count / (total_duration / 60.0)

        if len(smooth_vectors) < 2:
            volatility = 0.0
        else:
            diff_total = 0.0
            weight_total = 0.0
            for idx in range(1, len(smooth_vectors)):
                l1 = sum(abs(smooth_vectors[idx][k] - smooth_vectors[idx - 1][k]) for k in range(len(emotions)))
                diff_weight = durations[idx]
                diff_total += l1 * diff_weight
                weight_total += diff_weight
            volatility = diff_total / weight_total if weight_total > 1e-9 else diff_total / max(len(smooth_vectors) - 1, 1)

        def dist_value(*keys: str) -> float:
            return float(sum(distribution_fraction.get(key, 0.0) for key in keys))

        pa = dist_value("happy", "surprise", "surprised") + 0.5 * dist_value("neutral")
        na = dist_value("sad", "fear", "angry", "disgust", "contempt", "contemptuous")
        tone = pa - na

        tone_score_raw = 50.0 + 50.0 * math.tanh(tone / 0.6)
        tone_score = cls._apply_score_floor(tone_score_raw)

        dominance_ratio = 0.0
        if distribution_fraction:
            dominance_ratio = max(distribution_fraction.values())
        dominance_score = cls._score_up(
            dominance_ratio,
            lower=cls.DEEPFACE_DOMINANCE_SCORE_LOW,
            upper=cls.DEEPFACE_DOMINANCE_SCORE_HIGH,
        )

        phase_score_raw = 0.0
        if phase_change_rate > 0.0:
            if phase_change_rate < cls.DEEPFACE_PHASE_RATE_LOW:
                phase_score_raw = 100.0 * (phase_change_rate / cls.DEEPFACE_PHASE_RATE_LOW)
            elif phase_change_rate <= cls.DEEPFACE_PHASE_RATE_HIGH:
                phase_score_raw = 100.0
            elif phase_change_rate <= cls.DEEPFACE_PHASE_RATE_MAX:
                phase_score_raw = 100.0 * (
                    1.0
                    - (phase_change_rate - cls.DEEPFACE_PHASE_RATE_HIGH)
                    / (cls.DEEPFACE_PHASE_RATE_MAX - cls.DEEPFACE_PHASE_RATE_HIGH)
                )
            else:
                phase_score_raw = 0.0
        phase_change_score = cls._apply_score_floor(phase_score_raw)
        phase_purity_score = cls._apply_score_floor(100.0 * phase_purity_ratio)
        phase_quality_score = (
            cls.DEEPFACE_PHASE_SCORE_PURITY_WEIGHT * phase_purity_score
            + (1.0 - cls.DEEPFACE_PHASE_SCORE_PURITY_WEIGHT) * phase_change_score
        )

        event_ratio = 0.0
        if cls.DEEPFACE_ACTIVITY_EVENT_HIGH > cls.DEEPFACE_ACTIVITY_EVENT_LOW:
            event_ratio = cls._clip01(
                (events_per_minute - cls.DEEPFACE_ACTIVITY_EVENT_LOW)
                / (cls.DEEPFACE_ACTIVITY_EVENT_HIGH - cls.DEEPFACE_ACTIVITY_EVENT_LOW)
            )
        variance_ratio = 0.0
        if cls.DEEPFACE_ACTIVITY_VARIANCE_HIGH > cls.DEEPFACE_ACTIVITY_VARIANCE_LOW:
            variance_ratio = cls._clip01(
                (volatility - cls.DEEPFACE_ACTIVITY_VARIANCE_LOW)
                / (cls.DEEPFACE_ACTIVITY_VARIANCE_HIGH - cls.DEEPFACE_ACTIVITY_VARIANCE_LOW)
            )
        activity_score_raw = 100.0 * (0.6 * event_ratio + 0.4 * variance_ratio)
        activity_score = cls._apply_score_floor(activity_score_raw)

        deepface_process_score = (
            0.50 * tone_score
            + 0.30 * activity_score
            + 0.20 * phase_quality_score
        )
        deepface_process_score = float(min(100.0, max(0.0, deepface_process_score)))

        if deepface_process_score >= 85.0:
            rating = "A"
        elif deepface_process_score >= 70.0:
            rating = "B"
        elif deepface_process_score >= 55.0:
            rating = "C"
        else:
            rating = "D"

        ambiguity_ratio = 0.0
        coverage = 0.0 if sampled_frames_total <= 0 else float(frames_used) / float(sampled_frames_total)

        coverage_status = "ok" if coverage >= cls.DEEPFACE_MIN_COVERAGE else "insufficient"
        quality_status = "ok" if coverage_status == "ok" else "insufficient"

        ambiguity_status = "ok"
        if ambiguity_ratio >= 0.4:
            ambiguity_status = "high"
        elif ambiguity_ratio >= 0.25:
            ambiguity_status = "medium"

        return {
            "overall": {
                "frames_used": int(frames_used),
                "sampled_frames": int(sampled_frames_total),
                "emotion_distribution": emotion_distribution,
                "top_emotions": top_emotions,
            },
            "metrics": {
                "tone": float(tone),
                "pa": float(pa),
                "na": float(na),
                "volatility": float(volatility),
                "switch_rate": float(switch_rate),
                "switch_count": int(switch_count),
                "ambiguity_ratio": float(ambiguity_ratio),
                "dominance_ratio": float(dominance_ratio),
                "phase_change_rate": float(phase_change_rate),
                "phase_change_count": int(phase_change_count),
                "phase_purity_ratio": float(phase_purity_ratio),
                "phase_min_duration_sec": float(phase_min_duration_sec),
                "coverage": float(coverage),
                "margin_mean": float(sum(margins) / len(margins)) if margins else None,
                "entropy_mean": float(sum(entropies) / len(entropies)) if entropies else None,
                "events_per_minute": float(events_per_minute),
                "emotion_event_count": emotion_event_count,
                "emotion_mean_duration": emotion_mean_duration,
                "emotion_max_duration": emotion_max_duration,
            },
            "scores": {
                "tone_score": float(tone_score),
                "confidence_score": float(activity_score),
                "dominance_score": float(dominance_score),
                "phase_change_score": float(phase_quality_score),
                "phase_purity_score": float(phase_purity_score),
                "phase_change_raw_score": float(phase_change_score),
                "deepface_process_score": float(deepface_process_score),
            },
            "rating": rating,
            "quality": {
                "status": quality_status,
                "coverage_status": coverage_status,
                "ambiguity_status": ambiguity_status,
            },
        }

    @staticmethod
    def _clip01(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    @classmethod
    def _collect_deepface_samples(cls, analyzed_frames: list[dict]) -> list[dict]:
        samples: list[dict] = []
        for row in analyzed_frames:
            deepface = row.get("deepface") or {}
            scores = deepface.get("scores") or {}
            normalized = cls._normalize_probability_scores(scores)
            if normalized is None:
                continue

            timestamp = row.get("timestamp")
            ts = float(len(samples))
            if isinstance(timestamp, (int, float)):
                ts = float(timestamp)
            elif isinstance(timestamp, str):
                try:
                    ts = float(timestamp)
                except (TypeError, ValueError):
                    ts = float(len(samples))

            if not math.isfinite(ts):
                frame_value = row.get("frame")
                if isinstance(frame_value, (int, float)) and math.isfinite(float(frame_value)):
                    ts = float(frame_value) / 30.0
                else:
                    ts = float(len(samples))

            samples.append({"timestamp": ts, "probs": normalized})
        return samples

    @staticmethod
    def _normalize_probability_scores(scores: dict) -> dict[str, float] | None:
        if not isinstance(scores, dict) or not scores:
            return None

        cleaned: dict[str, float] = {}
        total = 0.0
        for emotion, raw_value in scores.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or value < 0:
                continue

            cleaned[str(emotion)] = value
            total += value

        if total <= 1e-9:
            return None

        return {emotion: value / total for emotion, value in cleaned.items()}

    @staticmethod
    def _sanitize_time_series(values: list[float]) -> list[float]:
        if not values:
            return []

        timestamps: list[float] = []
        prev = None
        for idx, value in enumerate(values):
            try:
                ts = float(value)
            except (TypeError, ValueError):
                ts = float(idx)
            if not math.isfinite(ts):
                ts = float(idx)

            if prev is not None and ts < prev:
                ts = prev

            timestamps.append(ts)
            prev = ts

        return timestamps

    @staticmethod
    def _compute_sample_durations(timestamps: list[float]) -> list[float]:
        count = len(timestamps)
        if count == 0:
            return []
        if count == 1:
            return [1.0]

        deltas = [timestamps[idx + 1] - timestamps[idx] for idx in range(count - 1)]
        valid_deltas = [delta for delta in deltas if delta > 1e-6 and math.isfinite(delta)]
        fallback = 1.0 / 30.0
        if valid_deltas:
            fallback = max(PipelineService._median(valid_deltas), 1.0 / 120.0)

        durations = []
        for delta in deltas:
            if not math.isfinite(delta) or delta <= 1e-6:
                durations.append(fallback)
            else:
                durations.append(max(delta, 1.0 / 120.0))
        durations.append(durations[-1] if durations else fallback)
        return durations

    @staticmethod
    def _ema_smooth_vectors(vectors: list[list[float]], alpha: float) -> list[list[float]]:
        if not vectors:
            return []

        smoothed: list[list[float]] = []
        previous = None
        for vector in vectors:
            if previous is None:
                current = [float(value) for value in vector]
            else:
                current = [
                    alpha * float(vector[idx]) + (1.0 - alpha) * previous[idx]
                    for idx in range(len(vector))
                ]

            total = sum(current)
            if total > 1e-9:
                current = [value / total for value in current]

            smoothed.append(current)
            previous = current

        return smoothed

    @staticmethod
    def _weighted_average_vector(vectors: list[list[float]], weights: list[float]) -> list[float]:
        if not vectors:
            return []

        size = len(vectors[0])
        output = [0.0 for _ in range(size)]
        for idx, vector in enumerate(vectors):
            weight = weights[idx] if idx < len(weights) else 0.0
            for pos in range(size):
                output[pos] += weight * vector[pos]
        return output

    @staticmethod
    def _normalized_entropy(vector: list[float]) -> float:
        if not vector:
            return 1.0
        size = len(vector)
        if size <= 1:
            return 0.0

        entropy = 0.0
        for value in vector:
            if value <= 0.0:
                continue
            entropy -= value * math.log(value + 1e-12)

        denom = math.log(size)
        if denom <= 1e-12:
            return 0.0
        return min(1.0, max(0.0, entropy / denom))

    @staticmethod
    def _build_label_segments(labels: list[str], durations: list[float], timestamps: list[float]) -> list[dict]:
        if not labels:
            return []

        segments: list[dict] = []
        start = 0
        for idx in range(1, len(labels) + 1):
            is_end = idx == len(labels)
            if not is_end and labels[idx] == labels[idx - 1]:
                continue

            end = idx - 1
            duration = sum(durations[start : end + 1]) if durations else 0.0
            segments.append(
                {
                    "label": labels[start],
                    "start_index": int(start),
                    "end_index": int(end),
                    "start_timestamp": float(timestamps[start]),
                    "end_timestamp": float(timestamps[end]),
                    "duration_sec": float(duration),
                }
            )
            start = idx

        return segments

    @staticmethod
    def _merge_short_segments(segments: list[dict], min_duration_sec: float) -> list[dict]:
        if len(segments) < 3:
            return segments

        merged = [dict(segment) for segment in segments]
        changed = True
        while changed and len(merged) >= 3:
            changed = False
            idx = 1
            while idx < len(merged) - 1:
                left = merged[idx - 1]
                mid = merged[idx]
                right = merged[idx + 1]

                if (
                    mid.get("duration_sec", 0.0) < min_duration_sec
                    and left.get("label") == right.get("label")
                ):
                    combined = {
                        "label": left.get("label"),
                        "start_index": int(left.get("start_index", 0)),
                        "end_index": int(right.get("end_index", 0)),
                        "start_timestamp": float(left.get("start_timestamp", 0.0)),
                        "end_timestamp": float(right.get("end_timestamp", 0.0)),
                        "duration_sec": float(
                            left.get("duration_sec", 0.0)
                            + mid.get("duration_sec", 0.0)
                            + right.get("duration_sec", 0.0)
                        ),
                    }
                    merged[idx - 1 : idx + 2] = [combined]
                    changed = True
                    idx = max(1, idx - 1)
                    continue

                idx += 1

        return merged

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        data = sorted(float(v) for v in values)
        size = len(data)
        half = size // 2
        if size % 2 == 1:
            return data[half]
        return (data[half - 1] + data[half]) / 2.0

    @classmethod
    def _compute_emonet_process_outputs(cls, analyzed_frames: list[dict]) -> dict:
        macro_sampled_frames = sum(1 for row in analyzed_frames if bool(row.get("macro_inference")))
        sampled_frames_total = macro_sampled_frames if macro_sampled_frames > 0 else len(analyzed_frames)

        samples: list[dict] = []
        for row in analyzed_frames:
            emonet = row.get("emonet")
            if not isinstance(emonet, dict) or not cls._is_valid_emonet_result(emonet):
                continue

            valence = cls._safe_float(emonet.get("valence"))
            arousal_raw = cls._safe_float(emonet.get("arousal"))
            if valence is None or arousal_raw is None:
                continue

            valence = max(-1.0, min(1.0, valence))
            arousal = max(-1.0, min(1.0, arousal_raw))

            timestamp = row.get("timestamp")
            ts = float(len(samples))
            if isinstance(timestamp, (int, float)):
                ts = float(timestamp)
            elif isinstance(timestamp, str):
                try:
                    ts = float(timestamp)
                except (TypeError, ValueError):
                    ts = float(len(samples))
            if not math.isfinite(ts):
                frame_value = row.get("frame")
                if isinstance(frame_value, (int, float)) and math.isfinite(float(frame_value)):
                    ts = float(frame_value) / 30.0
                else:
                    ts = float(len(samples))

            samples.append(
                {
                    "timestamp": ts,
                    "valence": valence,
                    "arousal": arousal,
                }
            )

        frames_used = len(samples)
        if frames_used == 0:
            return {
                "metrics": {
                    "valence_mean": 0.0,
                    "arousal_mean": 0.0,
                    "va_volatility": 0.0,
                    "activation_ratio": 0.0,
                    "coverage": 0.0,
                },
                "scores": {
                    "valence_score": 0.0,
                    "arousal_score": 0.0,
                    "stability_score": 0.0,
                    "activation_score": 0.0,
                    "va_score": 0.0,
                    "emonet_score": 0.0,
                },
                "rating": "D",
                "quality": {
                    "status": "no_data",
                    "coverage_status": "insufficient",
                },
            }

        timestamps = cls._sanitize_time_series([sample["timestamp"] for sample in samples])
        durations = cls._compute_sample_durations(timestamps)
        total_duration = sum(durations)
        if total_duration <= 1e-9:
            durations = [1.0 for _ in samples]
            total_duration = float(len(samples))

        valence_weighted_sum = 0.0
        arousal_weighted_sum = 0.0
        activation_duration = 0.0
        for idx, sample in enumerate(samples):
            duration = durations[idx]
            valence_weighted_sum += duration * float(sample["valence"])
            arousal_value = float(sample["arousal"])
            arousal_weighted_sum += duration * arousal_value
            if arousal_value > cls.EMONET_HIGH_AROUSAL_THRESHOLD:
                activation_duration += duration

        valence_mean = valence_weighted_sum / total_duration if total_duration > 1e-9 else 0.0
        arousal_mean = arousal_weighted_sum / total_duration if total_duration > 1e-9 else 0.0
        activation_ratio = activation_duration / total_duration if total_duration > 1e-9 else 0.0

        if len(samples) < 2:
            va_volatility = 0.0
        else:
            change_total = 0.0
            change_weight = 0.0
            for idx in range(1, len(samples)):
                dv = float(samples[idx]["valence"] - samples[idx - 1]["valence"])
                da = float(samples[idx]["arousal"] - samples[idx - 1]["arousal"])
                change = math.sqrt(dv * dv + da * da)
                weight = durations[idx]
                change_total += change * weight
                change_weight += weight
            va_volatility = change_total / change_weight if change_weight > 1e-9 else 0.0

        coverage = 0.0 if sampled_frames_total <= 0 else float(frames_used) / float(sampled_frames_total)

        valence_score = PipelineService._score_up_power(
            valence_mean,
            lower=cls.EMONET_VALENCE_SCORE_LOW,
            upper=cls.EMONET_VALENCE_SCORE_HIGH,
            power=cls.EMONET_SCORE_POWER,
        )
        arousal_score = PipelineService._score_up_power(
            arousal_mean,
            lower=cls.EMONET_AROUSAL_SCORE_LOW,
            upper=cls.EMONET_AROUSAL_SCORE_HIGH,
            power=cls.EMONET_SCORE_POWER,
        )
        stability_score = cls._score_down(
            va_volatility,
            upper=cls.EMONET_STABILITY_VOLATILITY_HIGH,
            lower=cls.EMONET_STABILITY_VOLATILITY_LOW,
        )
        activation_score = PipelineService._score_up_power(
            activation_ratio,
            lower=cls.EMONET_ACTIVATION_SCORE_LOW,
            upper=cls.EMONET_ACTIVATION_SCORE_HIGH,
            power=cls.EMONET_ACTIVATION_SCORE_POWER,
        )

        va_score = (
            0.40 * valence_score
            + 0.35 * arousal_score
            + 0.10 * stability_score
            + 0.15 * activation_score
        )
        va_score = float(min(100.0, max(0.0, va_score)))

        if va_score >= 85.0:
            rating = "A"
        elif va_score >= 70.0:
            rating = "B"
        elif va_score >= 55.0:
            rating = "C"
        else:
            rating = "D"

        coverage_status = "ok" if coverage >= cls.EMONET_MIN_COVERAGE else "insufficient"
        quality_status = "ok" if coverage_status == "ok" else "insufficient"

        return {
            "metrics": {
                "valence_mean": float(valence_mean),
                "arousal_mean": float(arousal_mean),
                "va_volatility": float(va_volatility),
                "activation_ratio": float(activation_ratio),
                "coverage": float(coverage),
            },
            "scores": {
                "valence_score": float(valence_score),
                "arousal_score": float(arousal_score),
                "stability_score": float(stability_score),
                "activation_score": float(activation_score),
                "va_score": float(va_score),
                "emonet_score": float(va_score),
            },
            "rating": rating,
            "quality": {
                "status": quality_status,
                "coverage_status": coverage_status,
            },
        }

    @staticmethod
    def _safe_float(value) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            number = float(value)
            return number if math.isfinite(number) else None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                number = float(text)
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) else None
        return None

    @staticmethod
    def _safe_int(value) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            pass

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(number):
            return None
        return int(number)

    @classmethod
    def _is_valid_emonet_result(cls, emonet) -> bool:
        if not isinstance(emonet, dict) or not emonet:
            return False

        valence = cls._safe_float(emonet.get("valence"))
        arousal = cls._safe_float(emonet.get("arousal"))
        if valence is None or arousal is None:
            return False

        probabilities = emonet.get("probabilities")
        if isinstance(probabilities, dict) and probabilities:
            top_probability = 0.0
            for value in probabilities.values():
                number = cls._safe_float(value)
                if number is None:
                    continue
                top_probability = max(top_probability, float(number))
            if top_probability >= cls.EMONET_MIN_SUCCESS_PROB:
                return True

        emotion = emonet.get("emotion")
        return isinstance(emotion, str) and bool(emotion.strip())

    @classmethod
    def _score_up(cls, value: float, lower: float, upper: float) -> float:
        if not math.isfinite(value) or upper <= lower:
            return 0.0
        ratio = cls._clip01((value - lower) / (upper - lower))
        return cls._apply_score_floor(100.0 * ratio)

    @classmethod
    def _score_up_power(cls, value: float, lower: float, upper: float, power: float) -> float:
        if not math.isfinite(value) or upper <= lower:
            return 0.0
        ratio = cls._clip01((value - lower) / (upper - lower))
        powered = ratio ** max(0.05, float(power))
        return cls._apply_score_floor(100.0 * powered)

    @classmethod
    def _score_saturation(cls, ratio: float, scale: float) -> float:
        if not math.isfinite(ratio) or scale <= 1e-9:
            return 0.0
        normalized = max(0.0, float(ratio))
        raw = 100.0 * (1.0 - math.exp(-normalized / float(scale)))
        return cls._apply_score_floor(raw)

    @classmethod
    def _score_down(cls, value: float, upper: float, lower: float) -> float:
        if not math.isfinite(value) or upper <= lower:
            return 0.0
        ratio = cls._clip01((upper - value) / (upper - lower))
        return cls._apply_score_floor(100.0 * ratio)

    @classmethod
    def _apply_score_floor(cls, score: float, floor: float | None = None) -> float:
        if not math.isfinite(score):
            return 0.0

        floor_value = cls.SCORE_FLOOR_WITH_DATA if floor is None else float(floor)
        floor_value = min(99.0, max(0.0, floor_value))

        raw = min(100.0, max(0.0, float(score)))
        scaled = floor_value + (100.0 - floor_value) * (raw / 100.0)
        return float(min(100.0, max(0.0, scaled)))

    @staticmethod
    def _normalize_macro_interval_sec(raw_value) -> float | None:
        if raw_value is None:
            return None
        try:
            interval = float(raw_value)
        except (TypeError, ValueError):
            return None
        if interval <= 0:
            return None
        return interval

    @staticmethod
    def _format_macro_config_text(macro_mode: str, macro_stride: int, macro_interval_sec: float | None) -> str:
        if macro_mode == "time" and macro_interval_sec is not None:
            return f"宏表情按 {macro_interval_sec:.2f}s 采样"
        return f"宏表情每 {macro_stride} 帧采样"

    @staticmethod
    def _should_run_macro_by_time(
        timestamp: float,
        interval_sec: float,
        next_timestamp: float | None,
    ) -> tuple[bool, float]:
        ts = float(timestamp)
        if next_timestamp is None:
            return True, ts + interval_sec

        if ts + 1e-9 < next_timestamp:
            return False, next_timestamp

        while next_timestamp <= ts + 1e-9:
            next_timestamp += interval_sec
        return True, next_timestamp

    def _estimate_macro_total_by_time(self, frame_items: list[dict], interval_sec: float) -> int:
        total = 0
        next_timestamp = None
        for item in frame_items:
            should_run, next_timestamp = self._should_run_macro_by_time(
                timestamp=item.get("timestamp", 0.0),
                interval_sec=interval_sec,
                next_timestamp=next_timestamp,
            )
            if should_run:
                total += 1
        return total

    @classmethod
    def _apply_decoded_timestamps_to_timeline(cls, timeline_df, frame_items: list[dict]):
        if timeline_df is None or timeline_df.empty:
            return timeline_df, 0
        if not isinstance(frame_items, list) or not frame_items:
            return timeline_df, 0
        if "timestamp" not in timeline_df.columns:
            return timeline_df, 0

        timestamp_by_index: dict[int, float] = {}
        for item in frame_items:
            if not isinstance(item, dict):
                continue
            source_text = str(item.get("timestamp_source") or "").strip()
            if source_text != "video_decode_pos_msec":
                continue

            idx = cls._safe_int(item.get("index"))
            ts = cls._safe_float(item.get("timestamp"))
            if idx is None or ts is None:
                continue
            timestamp_by_index[int(idx)] = float(ts)

        if not timestamp_by_index:
            return timeline_df, 0

        aligned_df = timeline_df.copy()
        aligned_values = aligned_df["timestamp"].tolist()
        aligned_count = 0
        total_rows = len(aligned_values)

        for idx, ts in timestamp_by_index.items():
            if 0 <= idx < total_rows:
                aligned_values[idx] = float(ts)
                aligned_count += 1

        if aligned_count <= 0:
            return timeline_df, 0

        aligned_df["timestamp"] = aligned_values
        attrs = dict(getattr(timeline_df, "attrs", {}) or {})
        attrs["timestamp_source"] = "video_decode_pos_msec"
        aligned_df.attrs = attrs
        return aligned_df, int(aligned_count)

    def _enrich_gameplay_analysis_result(self, payload: dict, timeline_df, analyzed_frames: list[dict]) -> dict:
        if not isinstance(payload, dict):
            return {}

        raw_segments = payload.get("segments")
        raw_modules = payload.get("modules")
        segments = [item for item in raw_segments if isinstance(item, dict)] if isinstance(raw_segments, list) else []
        modules = [item for item in raw_modules if isinstance(item, dict)] if isinstance(raw_modules, list) else []

        for segment in segments:
            module_key = str(segment.get("module_key") or "")
            time_ranges = self._normalize_time_ranges(
                [
                    (
                        segment.get("start_sec"),
                        segment.get("end_sec"),
                    )
                ]
            )
            index_ranges = self._normalize_index_ranges(
                [
                    (
                        segment.get("start_linked_analysis_index"),
                        segment.get("end_linked_analysis_index"),
                    )
                ]
            )
            segment["analysis"] = self._build_gameplay_slice_analysis(
                module_key=module_key,
                time_ranges=time_ranges,
                index_ranges=index_ranges,
                timeline_df=timeline_df,
                analyzed_frames=analyzed_frames,
            )
            segment["content_excerpt"] = self._build_gameplay_content_excerpt(
                module_key=module_key,
                content_labels=segment.get("content_labels"),
            )

        for module in modules:
            module_key = str(module.get("module_key") or "")
            module_segments = [item for item in segments if str(item.get("module_key") or "") == module_key]
            time_candidates = []
            index_candidates = []
            segment_ranges = []

            for segment in module_segments:
                time_candidates.append((segment.get("start_sec"), segment.get("end_sec")))
                index_candidates.append(
                    (
                        segment.get("start_linked_analysis_index"),
                        segment.get("end_linked_analysis_index"),
                    )
                )
                segment_ranges.append(
                    {
                        "segment_index": int(segment.get("segment_index") or 0),
                        "start_sec": self._safe_float(segment.get("start_sec")),
                        "end_sec": self._safe_float(segment.get("end_sec")),
                        "duration_sec": self._safe_float(segment.get("duration_sec")),
                    }
                )

            time_ranges = self._normalize_time_ranges(time_candidates)
            index_ranges = self._normalize_index_ranges(index_candidates)

            module["segment_ranges"] = segment_ranges
            module["analysis"] = self._build_gameplay_slice_analysis(
                module_key=module_key,
                time_ranges=time_ranges,
                index_ranges=index_ranges,
                timeline_df=timeline_df,
                analyzed_frames=analyzed_frames,
            )
            module["content_excerpt"] = self._build_gameplay_content_excerpt(
                module_key=module_key,
                content_labels=module.get("content_labels"),
            )

        payload["segments"] = segments
        payload["modules"] = modules

        summary = payload.get("summary")
        if isinstance(summary, dict):
            summary["segment_analysis_ready"] = bool(segments)
            summary["module_analysis_ready"] = bool(modules)

        return payload

    def _build_gameplay_slice_analysis(
        self,
        module_key: str,
        time_ranges: list[tuple[float, float]],
        index_ranges: list[tuple[int, int]],
        timeline_df,
        analyzed_frames: list[dict],
    ) -> dict:
        normalized_module_key = str(module_key or "")

        selected_frames = self._collect_frames_in_ranges(
            analyzed_frames=analyzed_frames,
            time_ranges=time_ranges,
            index_ranges=index_ranges,
            strict_time=True,
        )
        selected_timeline_df = self._slice_timeline_in_ranges(
            timeline_df=timeline_df,
            time_ranges=time_ranges,
            index_ranges=index_ranges,
            strict_time=True,
        )

        deepface_outputs = self._compute_deepface_process_outputs(selected_frames)
        emonet_outputs = self._compute_emonet_process_outputs(selected_frames)

        attention_analysis = self.frame_service.compute_attention_analysis(selected_timeline_df)
        abnormal_events = self.frame_service.detect_abnormal_events(selected_timeline_df)

        attention_events = attention_analysis.get("events") or {}
        blink_segments = attention_events.get("blink_segments") or []
        long_eye_close_segments = attention_events.get("long_eye_close_segments") or []

        state_counts = Counter()
        for state in attention_analysis.get("state") or []:
            text = str(state or "").strip()
            if not text:
                continue
            state_counts[text] += 1

        abnormal_type_counter = Counter()
        for event in abnormal_events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "").strip()
            if not event_type:
                continue
            abnormal_type_counter[event_type] += 1

        return {
            "enabled": True,
            "status": "done",
            "skip_reason": None,
            "module_key": normalized_module_key,
            "alignment_mode": "strict_seconds",
            "time_ranges": [
                {
                    "start_sec": float(start),
                    "end_sec": float(end),
                }
                for start, end in time_ranges
            ],
            "index_ranges": [
                {
                    "start_index": int(start),
                    "end_index": int(end),
                }
                for start, end in index_ranges
            ],
            "frames_total": int(len(selected_frames)),
            "macro_frames_total": int(sum(1 for row in selected_frames if bool(row.get("macro_inference")))),
            "timeline_rows": int(len(selected_timeline_df)),
            "deepface_overall": deepface_outputs.get("overall") or {},
            "deepface_metrics": deepface_outputs.get("metrics") or {},
            "deepface_scores": deepface_outputs.get("scores") or {},
            "deepface_rating": deepface_outputs.get("rating"),
            "deepface_quality": deepface_outputs.get("quality") or {},
            "emonet_metrics": emonet_outputs.get("metrics") or {},
            "emonet_scores": emonet_outputs.get("scores") or {},
            "emonet_rating": emonet_outputs.get("rating"),
            "emonet_quality": emonet_outputs.get("quality") or {},
            "attention_metrics": attention_analysis.get("metrics") or {},
            "attention_scores": attention_analysis.get("scores") or {},
            "attention_event_counts": {
                "blink_count": int(len(blink_segments) if isinstance(blink_segments, list) else 0),
                "long_eye_close_count": int(
                    len(long_eye_close_segments) if isinstance(long_eye_close_segments, list) else 0
                ),
            },
            "attention_state_counts": {
                str(key): int(value)
                for key, value in state_counts.items()
            },
            "abnormal_event_count": int(len(abnormal_events) if isinstance(abnormal_events, list) else 0),
            "abnormal_event_types": {
                str(key): int(value)
                for key, value in abnormal_type_counter.items()
            },
        }

    @classmethod
    def _collect_frames_in_ranges(
        cls,
        analyzed_frames: list[dict],
        time_ranges: list[tuple[float, float]],
        index_ranges: list[tuple[int, int]],
        strict_time: bool = True,
    ) -> list[dict]:
        selected: list[dict] = []
        seen = set()

        for row in analyzed_frames:
            if not isinstance(row, dict):
                continue

            ts = cls._safe_float(row.get("timestamp"))
            idx = cls._safe_int(row.get("index"))
            matched = False

            if ts is not None and time_ranges:
                for start, end in time_ranges:
                    if start - 1e-9 <= ts <= end + 1e-9:
                        matched = True
                        break

            use_index_fallback = (not strict_time) or (not time_ranges)
            if (not matched) and use_index_fallback and idx is not None and index_ranges:
                for start, end in index_ranges:
                    if start <= idx <= end:
                        matched = True
                        break

            if not matched:
                continue

            dedup_key = idx if idx is not None else id(row)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            selected.append(row)

        return selected

    @classmethod
    def _slice_timeline_in_ranges(
        cls,
        timeline_df,
        time_ranges: list[tuple[float, float]],
        index_ranges: list[tuple[int, int]],
        strict_time: bool = True,
    ):
        if timeline_df is None or timeline_df.empty:
            return timeline_df.iloc[0:0].copy()

        size = int(len(timeline_df))
        if size <= 0:
            return timeline_df.iloc[0:0].copy()

        mask = [False for _ in range(size)]
        matched = False

        if time_ranges and "timestamp" in timeline_df.columns:
            values = timeline_df["timestamp"].tolist()
            for pos, raw in enumerate(values):
                ts = cls._safe_float(raw)
                if ts is None:
                    continue
                for start, end in time_ranges:
                    if start - 1e-9 <= ts <= end + 1e-9:
                        mask[pos] = True
                        matched = True
                        break

        use_index_fallback = (not strict_time) or (not time_ranges)
        if (not matched) and use_index_fallback and index_ranges:
            for pos in range(size):
                for start, end in index_ranges:
                    if start <= pos <= end:
                        mask[pos] = True
                        matched = True
                        break

        if not matched:
            return timeline_df.iloc[0:0].copy()

        return timeline_df.loc[mask].copy()

    @classmethod
    def _normalize_time_ranges(cls, ranges: list[tuple[object, object]]) -> list[tuple[float, float]]:
        output: list[tuple[float, float]] = []
        min_span_sec = 0.5
        for start_raw, end_raw in ranges:
            start = cls._safe_float(start_raw)
            end = cls._safe_float(end_raw)
            if start is None and end is None:
                continue
            if start is None:
                start = end
            if end is None:
                end = start
            if start is None or end is None:
                continue

            lower = min(float(start), float(end))
            upper = max(float(start), float(end))
            if upper - lower < 1e-6:
                lower = max(0.0, lower - min_span_sec)
                upper = upper + min_span_sec
            output.append((lower, upper))
        return output

    @classmethod
    def _normalize_index_ranges(cls, ranges: list[tuple[object, object]]) -> list[tuple[int, int]]:
        output: list[tuple[int, int]] = []
        for start_raw, end_raw in ranges:
            start = cls._safe_int(start_raw)
            end = cls._safe_int(end_raw)
            if start is None and end is None:
                continue
            if start is None:
                start = end
            if end is None:
                end = start
            if start is None or end is None:
                continue

            lower = min(int(start), int(end))
            upper = max(int(start), int(end))
            output.append((lower, upper))
        return output

    @staticmethod
    def _build_gameplay_content_excerpt(module_key: str, content_labels) -> dict:
        labels = [item for item in content_labels if isinstance(item, dict)] if isinstance(content_labels, list) else []

        def _normalize_source(value) -> str:
            source = str(value or "").strip()
            if source in {"ocr", "lexicon_fill"}:
                return source
            return "ocr"

        def _collect(content_type: str, limit: int) -> list[dict]:
            output: list[dict] = []
            seen = set()
            max_size = max(1, int(limit))

            for item in labels:
                if str(item.get("content_type") or "") != content_type:
                    continue

                text = str(item.get("label") or "").strip()
                if not text:
                    continue

                day = PipelineService._safe_int(item.get("day"))
                period_index = PipelineService._safe_int(item.get("period_index"))
                tool = str(item.get("tool") or "").strip() or None
                thought = str(item.get("thought") or "").strip() or None
                dedup_key = (
                    GameplayTimelineService._normalize_text(text),
                    day,
                    period_index,
                    tool,
                    thought,
                )
                if dedup_key in seen:
                    continue

                seen.add(dedup_key)
                output.append(
                    {
                        "text": text,
                        "hit_count": int(item.get("hit_count") or 0),
                        "day": day,
                        "period_index": period_index,
                        "tool": tool,
                        "thought": thought,
                        "source": _normalize_source(item.get("source")),
                    }
                )
                if len(output) >= max_size:
                    break

            return output

        def _append_lexicon_row(
            rows: list[dict],
            *,
            text: str,
            day: int | None = None,
            period_index: int | None = None,
            tool: str | None = None,
            thought: str | None = None,
        ) -> None:
            content = str(text or "").strip()
            if not content:
                return

            norm = GameplayTimelineService._normalize_text(content)
            if not norm:
                return

            row_day = PipelineService._safe_int(day)
            row_period = PipelineService._safe_int(period_index)
            row_tool = str(tool or "").strip() or None
            row_thought = str(thought or "").strip() or None
            dedup_key = (norm, row_day, row_period, row_tool, row_thought)

            for row in rows:
                exists_key = (
                    GameplayTimelineService._normalize_text(str(row.get("text") or "")),
                    PipelineService._safe_int(row.get("day")),
                    PipelineService._safe_int(row.get("period_index")),
                    str(row.get("tool") or "").strip() or None,
                    str(row.get("thought") or "").strip() or None,
                )
                if exists_key == dedup_key:
                    return

            rows.append(
                {
                    "text": content,
                    "hit_count": 0,
                    "day": row_day,
                    "period_index": row_period,
                    "tool": row_tool,
                    "thought": row_thought,
                    "source": "lexicon_fill",
                }
            )

        normalized_module_key = str(module_key or "")
        if normalized_module_key == "thinking":
            thoughts = _collect("thought", 24)
            tools = _collect("tool", 24)
            reframes = _collect("reframe", 36)

            day_counter = Counter()
            for row in thoughts:
                day = PipelineService._safe_int(row.get("day"))
                if day and day > 0:
                    day_counter[day] += max(1, int(row.get("hit_count") or 0))
            for row in reframes:
                day = PipelineService._safe_int(row.get("day"))
                if day and day > 0:
                    day_counter[day] += max(1, int(row.get("hit_count") or 0))

            selected_day = day_counter.most_common(1)[0][0] if day_counter else None
            day_thoughts = GameplayTimelineService.THINKING_DAILY_THOUGHTS.get(selected_day) if selected_day else None
            if isinstance(day_thoughts, list):
                for thought in day_thoughts:
                    _append_lexicon_row(thoughts, text=thought, day=selected_day)

            for tool_name in GameplayTimelineService.THINKING_TOOL_KEYWORDS:
                _append_lexicon_row(tools, text=tool_name, day=selected_day)

            if selected_day:
                positive_map = GameplayTimelineService.THINKING_POSITIVE_THOUGHTS.get(selected_day) or {}
                if isinstance(positive_map, dict) and positive_map:
                    selected_tools = []
                    for row in tools:
                        tool_name = str(row.get("text") or "").strip()
                        if not tool_name:
                            continue
                        if tool_name not in GameplayTimelineService.THINKING_TOOL_KEYWORDS:
                            continue
                        if tool_name in selected_tools:
                            continue
                        selected_tools.append(tool_name)
                    if not selected_tools:
                        selected_tools = list(GameplayTimelineService.THINKING_TOOL_KEYWORDS)

                    selected_thoughts = []
                    for row in thoughts:
                        thought_text = str(row.get("text") or "").strip()
                        if not thought_text:
                            continue
                        if thought_text in selected_thoughts:
                            continue
                        selected_thoughts.append(thought_text)
                    if not selected_thoughts:
                        selected_thoughts = list(positive_map.keys())

                    for thought in selected_thoughts:
                        mapping = positive_map.get(thought)
                        if not isinstance(mapping, dict):
                            continue
                        for tool_name in selected_tools:
                            positive_text = str(mapping.get(tool_name) or "").strip()
                            if not positive_text:
                                continue
                            _append_lexicon_row(
                                reframes,
                                text=positive_text,
                                day=selected_day,
                                tool=tool_name,
                                thought=thought,
                            )

            return {
                "day": selected_day,
                "thoughts": thoughts[:12],
                "tools": tools[:6],
                "reframes": reframes[:12],
                "supplement_mode": "ocr_plus_lexicon",
            }

        if normalized_module_key == "attic":
            photos = _collect("photo", 24)
            questions = _collect("question", 24)
            periods = _collect("period", 24)

            period_counter = Counter()
            for row in photos:
                period_index = PipelineService._safe_int(row.get("period_index"))
                if period_index and period_index > 0:
                    period_counter[period_index] += max(1, int(row.get("hit_count") or 0))
            for row in questions:
                period_index = PipelineService._safe_int(row.get("period_index"))
                if period_index and period_index > 0:
                    period_counter[period_index] += max(1, int(row.get("hit_count") or 0))
            for row in periods:
                period_index = PipelineService._safe_int(row.get("period_index"))
                if period_index and period_index > 0:
                    period_counter[period_index] += max(1, int(row.get("hit_count") or 0))

            selected_period = period_counter.most_common(1)[0][0] if period_counter else None

            if selected_period:
                for title in GameplayTimelineService.MEMORY_PHOTOS.get(selected_period) or []:
                    _append_lexicon_row(photos, text=title, period_index=selected_period)

                question_text = str(GameplayTimelineService.MEMORY_QUESTIONS.get(selected_period) or "").strip()
                if question_text:
                    _append_lexicon_row(questions, text=question_text, period_index=selected_period)

            period_name = None
            if selected_period:
                for period in GameplayTimelineService.MEMORY_PERIODS:
                    if PipelineService._safe_int(period.get("period_index")) == selected_period:
                        period_name = str(period.get("period_name") or "").strip() or None
                        break

            return {
                "period_index": selected_period,
                "period_name": period_name,
                "photos": photos[:6],
                "questions": questions[:3],
                "supplement_mode": "ocr_plus_lexicon",
            }

        return {}

    def _generate_analysis_report_artifacts(
        self,
        task_id: str,
        input_data: dict,
        summary: dict,
        game_ocr_result: dict | None,
        gameplay_analysis_result: dict | None,
    ) -> dict[str, str]:
        payload_input = dict(input_data) if isinstance(input_data, dict) else {}
        report_name = self._build_report_output_name(payload_input, task_id)

        data_dir_value = self.config.get("DATA_DIR") or "."
        report_root_value = self.config.get("REPORT_DIR") or str(Path(data_dir_value) / "reports")
        report_root = Path(str(report_root_value))
        report_root.mkdir(parents=True, exist_ok=True)

        report_dir = report_root / f"{report_name}_{task_id[:8]}"
        report_dir.mkdir(parents=True, exist_ok=True)

        task_snapshot = self.task_store.get_task(task_id) or {}
        transcription_status = str(task_snapshot.get("transcription_status") or "idle")
        transcription_summary = task_snapshot.get("transcription_summary")
        transcription_result = None

        artifacts = task_snapshot.get("artifacts") if isinstance(task_snapshot.get("artifacts"), dict) else {}
        transcription_path_value = artifacts.get("transcription_json")
        if transcription_path_value and transcription_status == "done":
            transcription_path = Path(str(transcription_path_value))
            if transcription_path.exists() and transcription_path.is_file():
                try:
                    transcription_result = read_json(transcription_path)
                except Exception:
                    transcription_result = None

        report_payload = self._build_analysis_report_payload(
            task_id=task_id,
            input_data=payload_input,
            summary=summary,
            game_ocr_result=game_ocr_result,
            gameplay_analysis_result=gameplay_analysis_result,
            transcription_status=transcription_status,
            transcription_summary=transcription_summary,
            transcription_result=transcription_result,
            report_dir=report_dir,
            report_name=report_name,
        )

        report_json_path = report_dir / f"{report_name}.json"
        report_md_path = report_dir / f"{report_name}.md"
        write_json(report_json_path, report_payload)
        report_md_path.write_text(self._build_analysis_report_markdown(report_payload), encoding="utf-8")

        return {
            "analysis_report_json": str(report_json_path),
            "analysis_report_md": str(report_md_path),
            "analysis_report_dir": str(report_dir),
        }

    def _build_analysis_report_payload(
        self,
        task_id: str,
        input_data: dict,
        summary: dict,
        game_ocr_result: dict | None,
        gameplay_analysis_result: dict | None,
        transcription_status: str,
        transcription_summary,
        transcription_result,
        report_dir: Path,
        report_name: str,
    ) -> dict:
        summary_payload = dict(summary) if isinstance(summary, dict) else {}

        deepface_overall = summary_payload.get("deepface_overall") if isinstance(summary_payload.get("deepface_overall"), dict) else {}
        deepface_scores = summary_payload.get("deepface_scores") if isinstance(summary_payload.get("deepface_scores"), dict) else {}
        deepface_metrics = summary_payload.get("deepface_metrics") if isinstance(summary_payload.get("deepface_metrics"), dict) else {}
        deepface_quality = summary_payload.get("deepface_quality") if isinstance(summary_payload.get("deepface_quality"), dict) else {}

        emonet_scores = summary_payload.get("emonet_scores") if isinstance(summary_payload.get("emonet_scores"), dict) else {}
        emonet_metrics = summary_payload.get("emonet_metrics") if isinstance(summary_payload.get("emonet_metrics"), dict) else {}
        emonet_quality = summary_payload.get("emonet_quality") if isinstance(summary_payload.get("emonet_quality"), dict) else {}

        attention_scores = summary_payload.get("attention_scores") if isinstance(summary_payload.get("attention_scores"), dict) else {}
        attention_metrics = summary_payload.get("attention_metrics") if isinstance(summary_payload.get("attention_metrics"), dict) else {}
        attention_quality = summary_payload.get("attention_quality") if isinstance(summary_payload.get("attention_quality"), dict) else {}

        emotion_distribution_percent = {}
        raw_emotion_distribution = deepface_overall.get("emotion_distribution") if isinstance(deepface_overall, dict) else {}
        if isinstance(raw_emotion_distribution, dict):
            normalized_distribution = []
            for emotion, value in raw_emotion_distribution.items():
                number = self._safe_float(value)
                if number is None:
                    continue
                normalized_distribution.append((str(emotion), float(round(number, 4))))
            normalized_distribution.sort(key=lambda item: item[1], reverse=True)
            emotion_distribution_percent = {emotion: percent for emotion, percent in normalized_distribution}

        input_video_path = str(input_data.get("video_path") or "").strip()
        input_video_name = str(input_data.get("video_name") or "").strip()
        if not input_video_name and input_video_path:
            input_video_name = Path(input_video_path).name

        game_video_path = str(input_data.get("game_video_path") or "").strip()
        game_video_name = str(input_data.get("game_video_name") or "").strip()
        if not game_video_name and game_video_path:
            game_video_name = Path(game_video_path).name

        game_ocr_summary = {}
        if isinstance(game_ocr_result, dict):
            maybe_summary = game_ocr_result.get("summary")
            if isinstance(maybe_summary, dict):
                game_ocr_summary = maybe_summary

        transcription_payload = {
            "status": str(transcription_status or "idle"),
            "summary": dict(transcription_summary) if isinstance(transcription_summary, dict) else {},
            "available": isinstance(transcription_result, dict),
        }
        if isinstance(transcription_result, dict):
            preview = []
            for sentence in (transcription_result.get("sentences") or [])[:12]:
                if not isinstance(sentence, dict):
                    continue
                text = str(sentence.get("text") or "").strip()
                if not text:
                    continue
                preview.append(
                    {
                        "text": text,
                        "start_sec": self._safe_round(sentence.get("start_sec"), 3),
                        "end_sec": self._safe_round(sentence.get("end_sec"), 3),
                    }
                )
            transcription_payload["sentence_preview"] = preview

        gameplay_payload = self._build_gameplay_report_payload(gameplay_analysis_result)

        return {
            "report_version": "v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "task_id": task_id,
            "report_name": report_name,
            "report_dir": str(report_dir),
            "input": {
                "video_name": input_video_name or None,
                "video_path": input_video_path or None,
                "game_video_name": game_video_name or None,
                "game_video_path": game_video_path or None,
                "macro_stride": self._safe_int(input_data.get("macro_stride")),
                "macro_interval_sec": self._safe_round(input_data.get("macro_interval_sec"), 3),
                "game_ocr_interval_sec": self._safe_round(input_data.get("game_ocr_interval_sec"), 3),
            },
            "overall": {
                "affectnet_emotions": {
                    "score": self._safe_round(deepface_scores.get("deepface_process_score"), 3),
                    "rating": summary_payload.get("deepface_rating"),
                    "frames_used": self._safe_int(deepface_overall.get("frames_used")),
                    "sampled_frames": self._safe_int(deepface_overall.get("sampled_frames")),
                    "emotion_distribution_percent": emotion_distribution_percent,
                    "top_emotions": deepface_overall.get("top_emotions") if isinstance(deepface_overall.get("top_emotions"), list) else [],
                    "metrics": deepface_metrics,
                    "scores": deepface_scores,
                    "quality": deepface_quality,
                },
                "affectnet_va": {
                    "score": self._safe_round(emonet_scores.get("va_score") or emonet_scores.get("emonet_score"), 3),
                    "rating": summary_payload.get("emonet_rating"),
                    "metrics": emonet_metrics,
                    "scores": emonet_scores,
                    "quality": emonet_quality,
                },
                "attention": {
                    "score": self._safe_round(attention_scores.get("attention_score"), 3),
                    "metrics": attention_metrics,
                    "scores": attention_scores,
                    "quality": attention_quality,
                },
            },
            "game_ocr": {
                "enabled": bool(game_video_path),
                "summary": game_ocr_summary,
            },
            "transcription": transcription_payload,
            "gameplay": gameplay_payload,
        }

    def _build_gameplay_report_payload(self, gameplay_analysis_result: dict | None) -> dict:
        if not isinstance(gameplay_analysis_result, dict):
            return {"has_screen_info": False}

        raw_segments = gameplay_analysis_result.get("segments")
        raw_modules = gameplay_analysis_result.get("modules")
        segments = [item for item in raw_segments if isinstance(item, dict)] if isinstance(raw_segments, list) else []
        modules = [item for item in raw_modules if isinstance(item, dict)] if isinstance(raw_modules, list) else []

        has_screen_info = bool(segments or modules)
        if not has_screen_info:
            return {"has_screen_info": False}

        payload = {
            "has_screen_info": True,
            "time_segments": self._build_report_segment_timeline(segments),
        }

        raw_summary = gameplay_analysis_result.get("summary")
        summary = raw_summary if isinstance(raw_summary, dict) else {}
        payload["summary"] = {
            "analysis_version": summary.get("analysis_version"),
            "samples_total": self._safe_int(summary.get("samples_total")),
            "samples_with_text": self._safe_int(summary.get("samples_with_text")),
            "segments_total": self._safe_int(summary.get("segments_total")),
            "modules_detected": summary.get("modules_detected") if isinstance(summary.get("modules_detected"), list) else [],
            "module_duration_sec": summary.get("module_duration_sec") if isinstance(summary.get("module_duration_sec"), dict) else {},
            "module_segment_count": summary.get("module_segment_count") if isinstance(summary.get("module_segment_count"), dict) else {},
            "dominant_module": summary.get("dominant_module"),
            "dominant_module_name": summary.get("dominant_module_name"),
        }

        module_lookup = {}
        for module in modules:
            module_key = str(module.get("module_key") or "").strip()
            if not module_key:
                continue
            module_lookup[module_key] = module

        ordered_keys = []
        for module_key in GameplayTimelineService.MODULE_ORDER:
            if module_key in module_lookup:
                ordered_keys.append(module_key)
        for module_key in module_lookup.keys():
            if module_key not in ordered_keys:
                ordered_keys.append(module_key)

        module_assessment = {}
        for module_key in ordered_keys:
            module_entry = self._build_report_module_assessment(module_lookup.get(module_key) or {})
            if not module_entry:
                continue
            module_assessment[module_key] = module_entry

        if module_assessment:
            payload["module_assessment"] = module_assessment

        excerpt_payload = {}
        thinking_entry = module_assessment.get("thinking") if isinstance(module_assessment, dict) else None
        attic_entry = module_assessment.get("attic") if isinstance(module_assessment, dict) else None

        if isinstance(thinking_entry, dict):
            thinking_excerpt = thinking_entry.get("content_excerpt")
            if isinstance(thinking_excerpt, dict) and thinking_excerpt:
                excerpt_payload["thinking"] = thinking_excerpt

        if isinstance(attic_entry, dict):
            attic_excerpt = attic_entry.get("content_excerpt")
            if isinstance(attic_excerpt, dict) and attic_excerpt:
                excerpt_payload["attic"] = attic_excerpt

        if excerpt_payload:
            payload["content_excerpt"] = excerpt_payload

        return payload

    def _build_report_module_assessment(self, module: dict) -> dict:
        if not isinstance(module, dict):
            return {}

        module_key = str(module.get("module_key") or "").strip()
        if not module_key:
            return {}
        module_name = str(module.get("module_name") or GameplayTimelineService.MODULE_NAMES.get(module_key, module_key)).strip()

        analysis = module.get("analysis") if isinstance(module.get("analysis"), dict) else {}
        deepface_scores = analysis.get("deepface_scores") if isinstance(analysis.get("deepface_scores"), dict) else {}
        deepface_overall = analysis.get("deepface_overall") if isinstance(analysis.get("deepface_overall"), dict) else {}
        emonet_scores = analysis.get("emonet_scores") if isinstance(analysis.get("emonet_scores"), dict) else {}
        emonet_metrics = analysis.get("emonet_metrics") if isinstance(analysis.get("emonet_metrics"), dict) else {}
        attention_scores = analysis.get("attention_scores") if isinstance(analysis.get("attention_scores"), dict) else {}
        attention_metrics = analysis.get("attention_metrics") if isinstance(analysis.get("attention_metrics"), dict) else {}

        time_ranges = []
        raw_ranges = module.get("segment_ranges")
        if isinstance(raw_ranges, list):
            for item in raw_ranges:
                if not isinstance(item, dict):
                    continue
                start_sec = self._safe_float(item.get("start_sec"))
                end_sec = self._safe_float(item.get("end_sec"))
                duration_sec = self._safe_float(item.get("duration_sec"))
                if duration_sec is None and start_sec is not None and end_sec is not None:
                    duration_sec = max(0.0, end_sec - start_sec)
                time_ranges.append(
                    {
                        "segment_index": self._safe_int(item.get("segment_index")),
                        "start_sec": self._safe_round(start_sec, 3),
                        "end_sec": self._safe_round(end_sec, 3),
                        "duration_sec": self._safe_round(duration_sec, 3),
                        "time_span": (
                            f"{self._format_report_time_label(start_sec)} -> {self._format_report_time_label(end_sec)}"
                            f"（{self._format_report_value(duration_sec, digits=3)}s）"
                        ),
                    }
                )

        emotion_distribution_percent = {}
        raw_distribution = deepface_overall.get("emotion_distribution") if isinstance(deepface_overall, dict) else {}
        if isinstance(raw_distribution, dict):
            normalized_distribution = []
            for emotion, value in raw_distribution.items():
                number = self._safe_float(value)
                if number is None:
                    continue
                normalized_distribution.append((str(emotion), float(round(number, 4))))
            normalized_distribution.sort(key=lambda item: item[1], reverse=True)
            emotion_distribution_percent = {emotion: percent for emotion, percent in normalized_distribution}

        payload = {
            "module_key": module_key,
            "module_name": module_name,
            "segment_count": self._safe_int(module.get("segment_count")) if self._safe_int(module.get("segment_count")) is not None else len(time_ranges),
            "duration_sec": self._safe_round(module.get("duration_sec"), 3),
            "time_ranges": time_ranges,
            "scores": {
                "affectnet_emotion_score": self._safe_round(deepface_scores.get("deepface_process_score"), 3),
                "affectnet_va_score": self._safe_round(emonet_scores.get("va_score") or emonet_scores.get("emonet_score"), 3),
                "attention_score": self._safe_round(attention_scores.get("attention_score"), 3),
            },
            "ratings": {
                "affectnet_emotion_rating": analysis.get("deepface_rating"),
                "affectnet_va_rating": analysis.get("emonet_rating"),
            },
            "analysis": {
                "affectnet_emotion_distribution_percent": emotion_distribution_percent,
                "affectnet_va_metrics": emonet_metrics,
                "attention_metrics": attention_metrics,
                "abnormal_event_count": self._safe_int(analysis.get("abnormal_event_count")),
                "abnormal_event_types": analysis.get("abnormal_event_types") if isinstance(analysis.get("abnormal_event_types"), dict) else {},
            },
        }

        content_excerpt = module.get("content_excerpt") if isinstance(module.get("content_excerpt"), dict) else {}
        if module_key == "thinking":
            thinking_excerpt = self._build_report_thinking_excerpt(content_excerpt)
            if thinking_excerpt:
                payload["content_excerpt"] = thinking_excerpt
        elif module_key == "attic":
            attic_excerpt = self._build_report_attic_excerpt(content_excerpt)
            if attic_excerpt:
                payload["content_excerpt"] = attic_excerpt

        return payload

    def _build_report_segment_timeline(self, segments: list[dict]) -> list[dict]:
        output = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue

            module_key = str(segment.get("module_key") or "").strip() or None
            module_name = str(segment.get("module_name") or GameplayTimelineService.MODULE_NAMES.get(module_key or "", module_key or "未知模块")).strip()
            start_sec = self._safe_float(segment.get("start_sec"))
            end_sec = self._safe_float(segment.get("end_sec"))
            duration_sec = self._safe_float(segment.get("duration_sec"))
            if duration_sec is None and start_sec is not None and end_sec is not None:
                duration_sec = max(0.0, end_sec - start_sec)

            output.append(
                {
                    "segment_index": self._safe_int(segment.get("segment_index")),
                    "module_key": module_key,
                    "module_name": module_name,
                    "start_sec": self._safe_round(start_sec, 3),
                    "end_sec": self._safe_round(end_sec, 3),
                    "duration_sec": self._safe_round(duration_sec, 3),
                    "time_span": (
                        f"{self._format_report_time_label(start_sec)} -> {self._format_report_time_label(end_sec)}"
                        f"（{self._format_report_value(duration_sec, digits=3)}s）"
                    ),
                }
            )

        return output

    def _build_report_thinking_excerpt(self, content_excerpt: dict) -> dict:
        if not isinstance(content_excerpt, dict):
            return {}

        thoughts = self._build_report_content_rows(content_excerpt.get("thoughts"), row_type="thought")
        tools = self._build_report_content_rows(content_excerpt.get("tools"), row_type="tool")
        reframes = self._build_report_content_rows(content_excerpt.get("reframes"), row_type="reframe")

        payload = {
            "day": self._safe_int(content_excerpt.get("day")),
            "negative_thoughts": thoughts,
            "coping_tools": tools,
            "reframed_statements": reframes,
        }
        return payload

    def _build_report_attic_excerpt(self, content_excerpt: dict) -> dict:
        if not isinstance(content_excerpt, dict):
            return {}

        period_name = str(content_excerpt.get("period_name") or "").strip() or None
        period_index = self._safe_int(content_excerpt.get("period_index"))
        photos = self._build_report_content_rows(content_excerpt.get("photos"), row_type="photo")
        questions = self._build_report_content_rows(content_excerpt.get("questions"), row_type="question")

        payload = {
            "period_index": period_index,
            "period_name": period_name,
            "identified_period": f"识别年代: {period_name}" if period_name else None,
            "historical_photos": photos,
            "interactive_questions": questions,
        }
        return payload

    def _build_report_content_rows(self, rows, row_type: str) -> list[dict]:
        source_rows = [item for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []
        output = []

        for item in source_rows:
            text = str(item.get("text") or "").strip()
            if not text:
                continue

            hit_count = self._safe_int(item.get("hit_count"))
            if hit_count is None or hit_count < 0:
                hit_count = 0

            source, source_label = self._normalize_report_source(item.get("source"))
            tool = str(item.get("tool") or "").strip() or None
            thought = str(item.get("thought") or "").strip() or None

            display_text = text
            if row_type == "reframe" and tool:
                display_text = f"{display_text}（工具:{tool}）"

            if hit_count > 0:
                display_text = f"{display_text}（x{hit_count}） {source_label}"
            else:
                display_text = f"{display_text} {source_label}"

            output.append(
                {
                    "text": text,
                    "hit_count": int(hit_count),
                    "source": source,
                    "source_label": source_label,
                    "day": self._safe_int(item.get("day")),
                    "period_index": self._safe_int(item.get("period_index")),
                    "tool": tool,
                    "thought": thought,
                    "display": display_text,
                }
            )

        return output

    @staticmethod
    def _normalize_report_source(source_value) -> tuple[str, str]:
        source = str(source_value or "").strip()
        if source == "lexicon_fill":
            return "lexicon_fill", "词库补齐"
        return "ocr", "OCR命中"

    def _build_analysis_report_markdown(self, report_payload: dict) -> str:
        payload = dict(report_payload) if isinstance(report_payload, dict) else {}
        overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
        gameplay = payload.get("gameplay") if isinstance(payload.get("gameplay"), dict) else {}
        input_payload = payload.get("input") if isinstance(payload.get("input"), dict) else {}

        affectnet_emotions = overall.get("affectnet_emotions") if isinstance(overall.get("affectnet_emotions"), dict) else {}
        affectnet_va = overall.get("affectnet_va") if isinstance(overall.get("affectnet_va"), dict) else {}
        attention = overall.get("attention") if isinstance(overall.get("attention"), dict) else {}

        lines = [
            "# 综合分析报表",
            "",
            f"- 任务ID: {payload.get('task_id') or '-'}",
            f"- 生成时间: {payload.get('generated_at') or '-'}",
            f"- 输入视频: {input_payload.get('video_name') or '-'}",
            f"- 输入视频路径: {input_payload.get('video_path') or '-'}",
            "",
            "## 全过程评分总览",
            (
                f"- AffectNet 情绪评分: {self._format_report_value(affectnet_emotions.get('score'), digits=3)}"
                f"（等级: {affectnet_emotions.get('rating') or '-'}）"
            ),
            (
                f"- AffectNet V/A 评分: {self._format_report_value(affectnet_va.get('score'), digits=3)}"
                f"（等级: {affectnet_va.get('rating') or '-'}）"
            ),
            f"- 注意力评分: {self._format_report_value(attention.get('score'), digits=3)}",
            "",
            "## AffectNet 全过程情绪占比（%）",
        ]

        distribution = (
            affectnet_emotions.get("emotion_distribution_percent")
            if isinstance(affectnet_emotions.get("emotion_distribution_percent"), dict)
            else {}
        )
        if distribution:
            for emotion, percent in distribution.items():
                lines.append(f"- {emotion}: {self._format_report_value(percent, digits=4)}")
        else:
            lines.append("- 无有效情绪占比数据")

        lines.extend(
            [
                "",
                "## AffectNet V/A 全过程简版指标",
            ]
        )
        va_metrics = affectnet_va.get("metrics") if isinstance(affectnet_va.get("metrics"), dict) else {}
        if va_metrics:
            for key, value in va_metrics.items():
                lines.append(f"- {key}: {self._format_report_value(value, digits=4)}")
        else:
            lines.append("- 无有效 V/A 指标")

        lines.extend(
            [
                "",
                "## 注意力结果",
            ]
        )
        attention_metrics = attention.get("metrics") if isinstance(attention.get("metrics"), dict) else {}
        if attention_metrics:
            for key, value in attention_metrics.items():
                lines.append(f"- {key}: {self._format_report_value(value, digits=4)}")
        else:
            lines.append("- 无有效注意力指标")

        if gameplay.get("has_screen_info"):
            lines.extend(
                [
                    "",
                    "## 游戏流程时间片段",
                ]
            )
            segments = gameplay.get("time_segments") if isinstance(gameplay.get("time_segments"), list) else []
            if segments:
                for segment in segments:
                    if not isinstance(segment, dict):
                        continue
                    module_name = segment.get("module_name") or segment.get("module_key") or "未知模块"
                    lines.append(f"- [{module_name}] {segment.get('time_span') or '-'}")
            else:
                lines.append("- 无可用时间片段")

            module_assessment = (
                gameplay.get("module_assessment") if isinstance(gameplay.get("module_assessment"), dict) else {}
            )
            if module_assessment:
                lines.extend(
                    [
                        "",
                        "## 分模块评估",
                    ]
                )

                ordered_keys = [
                    key for key in GameplayTimelineService.MODULE_ORDER if key in module_assessment
                ]
                for key in module_assessment.keys():
                    if key not in ordered_keys:
                        ordered_keys.append(key)

                for module_key in ordered_keys:
                    module = module_assessment.get(module_key)
                    if not isinstance(module, dict):
                        continue

                    module_name = module.get("module_name") or module_key
                    scores = module.get("scores") if isinstance(module.get("scores"), dict) else {}
                    ratings = module.get("ratings") if isinstance(module.get("ratings"), dict) else {}

                    lines.extend(
                        [
                            "",
                            f"### {module_name}",
                            f"- 分段数量: {self._format_report_value(module.get('segment_count'), digits=0)}",
                            f"- 持续时长: {self._format_report_value(module.get('duration_sec'), digits=3)}s",
                            (
                                f"- AffectNet 情绪评分: {self._format_report_value(scores.get('affectnet_emotion_score'), digits=3)}"
                                f"（等级: {ratings.get('affectnet_emotion_rating') or '-'}）"
                            ),
                            (
                                f"- AffectNet V/A 评分: {self._format_report_value(scores.get('affectnet_va_score'), digits=3)}"
                                f"（等级: {ratings.get('affectnet_va_rating') or '-'}）"
                            ),
                            f"- 注意力评分: {self._format_report_value(scores.get('attention_score'), digits=3)}",
                        ]
                    )

                    content_excerpt = module.get("content_excerpt") if isinstance(module.get("content_excerpt"), dict) else {}
                    if module_key == "thinking" and content_excerpt:
                        for row in content_excerpt.get("negative_thoughts") or []:
                            if isinstance(row, dict):
                                lines.append(f"- 负面念头: {row.get('display') or '-'}")
                        for row in content_excerpt.get("reframed_statements") or []:
                            if isinstance(row, dict):
                                lines.append(f"- 重构句: {row.get('display') or '-'}")

                    if module_key == "attic" and content_excerpt:
                        if content_excerpt.get("identified_period"):
                            lines.append(f"- {content_excerpt.get('identified_period')}")
                        for row in content_excerpt.get("historical_photos") or []:
                            if isinstance(row, dict):
                                lines.append(f"- 历史照片关键词: {row.get('display') or '-'}")
                        for row in content_excerpt.get("interactive_questions") or []:
                            if isinstance(row, dict):
                                lines.append(f"- 当天互动问题: {row.get('display') or '-'}")
        else:
            lines.extend(
                [
                    "",
                    "## 游戏流程与分模块评估",
                    "- 未检测到可用屏幕信息，已跳过分模块评估输出",
                ]
            )

        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _format_report_time_label(seconds_value) -> str:
        if seconds_value is None:
            return "-"
        try:
            seconds = float(seconds_value)
        except (TypeError, ValueError):
            return "-"
        if not math.isfinite(seconds):
            return "-"

        sign = "-" if seconds < 0 else ""
        seconds = abs(seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        rest = seconds - hours * 3600 - minutes * 60
        if hours > 0:
            return f"{sign}{hours:02d}:{minutes:02d}:{rest:06.3f}"
        return f"{sign}{minutes:02d}:{rest:06.3f}"

    def _build_report_output_name(self, input_data: dict, task_id: str) -> str:
        payload = dict(input_data) if isinstance(input_data, dict) else {}
        raw_name = str(payload.get("video_name") or "").strip()
        if not raw_name:
            raw_path = str(payload.get("video_path") or "").strip()
            if raw_path:
                raw_name = Path(raw_path).name

        stem = Path(raw_name).stem if raw_name else ""
        return self._sanitize_report_name(stem, default_value=f"task_{task_id[:8]}")

    @staticmethod
    def _sanitize_report_name(raw_name: str, default_value: str) -> str:
        text = str(raw_name or "").strip()
        if not text:
            text = str(default_value)

        cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", text)
        cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = cleaned.strip("._ ")
        if not cleaned:
            cleaned = str(default_value).strip() or "report"
        return cleaned

    @classmethod
    def _safe_round(cls, value, digits: int = 3):
        number = cls._safe_float(value)
        if number is None:
            return None
        return float(round(number, max(0, int(digits))))

    @staticmethod
    def _format_report_value(value, digits: int = 3) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if not math.isfinite(value):
                return "-"
            text = f"{value:.{max(0, int(digits))}f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return text
        return str(value)

    def _normalize_game_ocr_interval_sec(self, raw_value) -> float:
        default_interval = self._safe_float(self.config.get("GAME_OCR_INTERVAL_SEC_DEFAULT"))
        if default_interval is None or default_interval <= 0:
            default_interval = 2.0

        interval = self._safe_float(raw_value)
        if interval is None or interval <= 0:
            interval = default_interval
        return float(interval)

    @staticmethod
    def _normalize_optional_video_path(raw_value) -> Path | None:
        if raw_value is None:
            return None
        text = str(raw_value).strip()
        if not text:
            return None

        path = Path(text)
        if not path.exists() or not path.is_file():
            return None
        return path

    @staticmethod
    def _build_game_ocr_overview(
        summary: dict,
        enabled: bool,
        status: str,
        interval_sec: float,
        error: str | None,
    ) -> dict:
        payload_summary = dict(summary) if isinstance(summary, dict) else {}

        interval_value = None
        try:
            number = float(interval_sec)
            if math.isfinite(number) and number > 0:
                interval_value = number
        except (TypeError, ValueError):
            interval_value = None

        result = {
            "enabled": bool(enabled),
            "status": str(status or "idle"),
            "interval_sec": interval_value,
            "error": str(error) if error else None,
            "summary": payload_summary,
        }

        for key in (
            "samples_total",
            "samples_with_text",
            "samples_timeout",
            "samples_done",
            "timeout_count",
            "line_total",
            "text_coverage_ratio",
            "ocr_avg_latency_sec",
            "ocr_p95_latency_sec",
            "reference_linked_count",
            "reference_max_delta_sec",
            "current_timestamp_sec",
            "current_line_count",
        ):
            if key in payload_summary:
                result[key] = payload_summary.get(key)

        return result

    @staticmethod
    def _build_gameplay_analysis_overview(
        summary: dict,
        enabled: bool,
        status: str,
        error: str | None,
    ) -> dict:
        payload_summary = dict(summary) if isinstance(summary, dict) else {}

        result = {
            "enabled": bool(enabled),
            "status": str(status or "idle"),
            "error": str(error) if error else None,
            "summary": payload_summary,
        }

        for key in (
            "analysis_version",
            "samples_total",
            "samples_with_text",
            "samples_with_raw_text",
            "keyword_samples_total",
            "keyword_hits_total",
            "noise_filtered_samples",
            "noise_filtered_ratio",
            "raw_line_total",
            "effective_line_total",
            "noise_line_total",
            "segments_total",
            "modules_detected",
            "module_duration_sec",
            "module_segment_count",
            "dominant_module",
            "dominant_module_name",
        ):
            if key in payload_summary:
                result[key] = payload_summary.get(key)

        return result
