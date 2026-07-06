from pathlib import Path
import math
import threading
import mimetypes
from uuid import uuid4

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.utils import secure_filename

import pandas as pd

from ..utils.json_io import read_json, write_json
from ..services.frame_service import FrameService


tasks_bp = Blueprint("tasks", __name__)


_RPPG_WORKERS: dict[str, threading.Thread] = {}
_TRANSCRIPTION_WORKERS: dict[str, threading.Thread] = {}
_WORKER_LOCK = threading.Lock()


ALLOWED_VIDEO_SUFFIXES = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".webm",
    ".m4v",
    ".flv",
    ".mpeg",
    ".mpg",
}


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _existing_file(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return None
    return path


def _parse_float(value, default=None) -> float | None:
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value, default: int, min_value: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = int(default)
    if min_value is not None:
        number = max(min_value, number)
    return number


def _register_worker(pool: dict[str, threading.Thread], task_id: str, worker: threading.Thread) -> None:
    with _WORKER_LOCK:
        pool[task_id] = worker


def _remove_worker(pool: dict[str, threading.Thread], task_id: str) -> None:
    with _WORKER_LOCK:
        pool.pop(task_id, None)


def _is_worker_alive(pool: dict[str, threading.Thread], task_id: str) -> bool:
    with _WORKER_LOCK:
        worker = pool.get(task_id)
        if worker is None:
            return False
        if worker.is_alive():
            return True
        pool.pop(task_id, None)
        return False


def _select_artifact_path(task: dict, final_key: str, live_key: str, allow_partial: bool) -> tuple[Path | None, bool]:
    artifacts = task.get("artifacts", {})
    final_path = _existing_file(artifacts.get(final_key))
    if final_path:
        return final_path, False

    if allow_partial:
        live_path = _existing_file(artifacts.get(live_key))
        if live_path:
            return live_path, True

    return None, False


@tasks_bp.post("/api/uploads/video")
def upload_video():
    file = request.files.get("video")
    if file is None:
        return jsonify({"success": False, "error": "缺少 video 文件字段"}), 400

    raw_name = str(file.filename or "").strip()
    if not raw_name:
        return jsonify({"success": False, "error": "视频文件名为空"}), 400

    safe_name = secure_filename(raw_name)
    suffix = Path(safe_name).suffix.lower() or Path(raw_name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        return jsonify({"success": False, "error": f"不支持的视频格式: {suffix or '未知'}"}), 400

    upload_dir = Path(current_app.config["UPLOAD_DIR"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / f"{uuid4().hex}{suffix}"

    file.save(save_path)
    return jsonify(
        {
            "success": True,
            "video_path": str(save_path),
            "filename": raw_name,
            "stored_name": save_path.name,
        }
    )


@tasks_bp.get("/api/tasks")
def list_tasks():
    task_store = current_app.extensions["task_store"]
    items = task_store.list_tasks(limit=30)
    return jsonify({"success": True, "tasks": items})


@tasks_bp.post("/api/tasks")
def create_task():
    data = request.get_json(silent=True) or {}
    video_path = data.get("video_path")
    video_name = str(data.get("video_name") or "").strip() or None
    game_video_path = data.get("game_video_path")
    game_video_name = str(data.get("game_video_name") or "").strip() or None
    default_game_ocr_interval_sec = _parse_float(current_app.config.get("GAME_OCR_INTERVAL_SEC_DEFAULT"), 2.0)
    if default_game_ocr_interval_sec is None or not math.isfinite(default_game_ocr_interval_sec) or default_game_ocr_interval_sec <= 0:
        default_game_ocr_interval_sec = 2.0
    raw_macro_stride = data.get("macro_stride", data.get("macro_frame_interval", 5))
    raw_macro_interval_sec = data.get("macro_interval_sec", data.get("macro_interval_seconds"))
    raw_game_ocr_interval_sec = data.get("game_ocr_interval_sec", default_game_ocr_interval_sec)
    mode = "auto_main"
    frame_stride = 1
    max_frames = 0

    try:
        macro_stride = int(raw_macro_stride)
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "macro_stride 必须是整数"}), 400

    if macro_stride < 1:
        return jsonify({"success": False, "error": "macro_stride 必须 >= 1"}), 400

    macro_interval_sec = None
    if raw_macro_interval_sec is not None:
        if isinstance(raw_macro_interval_sec, str):
            raw_macro_interval_sec = raw_macro_interval_sec.strip()
        if raw_macro_interval_sec != "":
            try:
                macro_interval_sec = float(raw_macro_interval_sec)
            except (TypeError, ValueError):
                return jsonify({"success": False, "error": "macro_interval_sec 必须是数字"}), 400
            if macro_interval_sec <= 0:
                return jsonify({"success": False, "error": "macro_interval_sec 必须 > 0"}), 400

    game_ocr_interval_sec = float(default_game_ocr_interval_sec)
    if raw_game_ocr_interval_sec is not None:
        if isinstance(raw_game_ocr_interval_sec, str):
            raw_game_ocr_interval_sec = raw_game_ocr_interval_sec.strip()
        if raw_game_ocr_interval_sec != "":
            try:
                game_ocr_interval_sec = float(raw_game_ocr_interval_sec)
            except (TypeError, ValueError):
                return jsonify({"success": False, "error": "game_ocr_interval_sec 必须是数字"}), 400
            if game_ocr_interval_sec <= 0:
                return jsonify({"success": False, "error": "game_ocr_interval_sec 必须 > 0"}), 400

    if not video_path:
        return jsonify({"success": False, "error": "video_path 不能为空"}), 400

    path_obj = Path(video_path)
    if not path_obj.exists():
        return jsonify({"success": False, "error": f"视频不存在: {video_path}"}), 400

    game_path_obj = None
    if game_video_path:
        game_path_obj = Path(str(game_video_path))
        if not game_path_obj.exists() or not game_path_obj.is_file():
            return jsonify({"success": False, "error": f"游戏画面视频不存在: {game_video_path}"}), 400

    task_store = current_app.extensions["task_store"]
    pipeline = current_app.extensions["pipeline_service"]

    task = task_store.create_task(
        {
            "video_path": str(path_obj),
            "video_name": video_name,
            "mode": mode,
            "frame_stride": frame_stride,
            "max_frames": max_frames,
            "macro_stride": macro_stride,
            "macro_interval_sec": macro_interval_sec,
            "game_video_path": str(game_path_obj) if game_path_obj else None,
            "game_video_name": game_video_name,
            "game_ocr_interval_sec": float(game_ocr_interval_sec),
        }
    )
    pipeline.run_async(task["task_id"])

    return jsonify(
        {
            "success": True,
            "task_id": task["task_id"],
            "status": task["status"],
            "macro_stride": macro_stride,
            "macro_interval_sec": macro_interval_sec,
            "video_name": video_name,
            "game_video_path": str(game_path_obj) if game_path_obj else None,
            "game_video_name": game_video_name,
            "game_ocr_interval_sec": float(game_ocr_interval_sec),
        }
    )


@tasks_bp.get("/api/tasks/<task_id>")
def get_task(task_id: str):
    task_store = current_app.extensions["task_store"]
    task = task_store.get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404
    return jsonify({"success": True, "task": task})


@tasks_bp.get("/api/tasks/<task_id>/summary")
def get_summary(task_id: str):
    task_store = current_app.extensions["task_store"]
    task = task_store.get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404

    allow_partial = _parse_bool(request.args.get("allow_partial"))
    summary_path, is_partial = _select_artifact_path(
        task,
        final_key="summary_json",
        live_key="summary_live_json",
        allow_partial=allow_partial,
    )
    if not summary_path:
        return jsonify({"success": False, "error": "汇总结果不存在，任务可能尚未进入可视化阶段"}), 404

    return jsonify({"success": True, "summary": read_json(summary_path), "partial": is_partial})


@tasks_bp.get("/api/tasks/<task_id>/frame/<int:frame_idx>")
def get_frame(task_id: str, frame_idx: int):
    task_store = current_app.extensions["task_store"]
    task = task_store.get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404

    allow_partial = _parse_bool(request.args.get("allow_partial"))
    analysis_path, is_partial = _select_artifact_path(
        task,
        final_key="frame_analysis_json",
        live_key="frame_analysis_live_json",
        allow_partial=allow_partial,
    )
    if not analysis_path:
        return jsonify({"success": False, "error": "帧结果不存在，任务可能尚未产出帧级结果"}), 404

    rows = read_json(analysis_path)
    if frame_idx < 0 or frame_idx >= len(rows):
        return jsonify({"success": False, "error": "帧索引越界"}), 400

    frame = rows[frame_idx]
    image_path = frame.get("image_path")
    response = dict(frame)
    response["has_image"] = bool(image_path and Path(image_path).exists())
    response["partial"] = is_partial
    response.pop("image_path", None)
    return jsonify({"success": True, "frame": response})


@tasks_bp.get("/api/tasks/<task_id>/frame/<int:frame_idx>/image")
def get_frame_image(task_id: str, frame_idx: int):
    task_store = current_app.extensions["task_store"]
    task = task_store.get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404

    allow_partial = _parse_bool(request.args.get("allow_partial"))
    analysis_path, _ = _select_artifact_path(
        task,
        final_key="frame_analysis_json",
        live_key="frame_analysis_live_json",
        allow_partial=allow_partial,
    )
    if not analysis_path:
        return jsonify({"success": False, "error": "帧结果不存在，任务可能尚未产出帧级结果"}), 404

    rows = read_json(analysis_path)
    if frame_idx < 0 or frame_idx >= len(rows):
        return jsonify({"success": False, "error": "帧索引越界"}), 400

    image_path = _existing_file(rows[frame_idx].get("image_path"))
    if not image_path:
        return jsonify({"success": False, "error": "帧图像不存在"}), 404

    return send_file(image_path)


@tasks_bp.get("/api/tasks/<task_id>/video")
def get_task_video(task_id: str):
    task_store = current_app.extensions["task_store"]
    task = task_store.get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404

    video_path = task.get("input", {}).get("video_path")
    video_file = _existing_file(video_path)
    if not video_file:
        return jsonify({"success": False, "error": "视频不存在"}), 404

    mime, _ = mimetypes.guess_type(str(video_file))
    return send_file(video_file, mimetype=mime or "video/mp4", conditional=True)


@tasks_bp.get("/api/tasks/<task_id>/game-ocr")
def get_game_ocr(task_id: str):
    task_store = current_app.extensions["task_store"]
    task = task_store.get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404

    status = str(task.get("game_ocr_status") or "idle")
    summary = task.get("game_ocr_summary")
    error = task.get("game_ocr_error")
    result = None

    if status == "done":
        result_path = _existing_file((task.get("artifacts") or {}).get("game_ocr_json"))
        if result_path:
            result = read_json(result_path)
        else:
            status = "failed"
            error = "游戏画面 OCR 结果文件不存在"

    return jsonify(
        {
            "success": True,
            "status": status,
            "summary": summary,
            "error": error,
            "result": result,
            "params": {
                "interval_sec": (task.get("input") or {}).get("game_ocr_interval_sec"),
                "video_path": (task.get("input") or {}).get("game_video_path"),
            },
        }
    )


@tasks_bp.get("/api/tasks/<task_id>/game-ocr/frame/<int:sample_idx>/image")
def get_game_ocr_frame_image(task_id: str, sample_idx: int):
    task_store = current_app.extensions["task_store"]
    task = task_store.get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404

    result_path = _existing_file((task.get("artifacts") or {}).get("game_ocr_json"))
    if not result_path:
        return jsonify({"success": False, "error": "游戏画面 OCR 结果不存在"}), 404

    payload = read_json(result_path)
    timeline = payload.get("timeline") if isinstance(payload, dict) else None
    if not isinstance(timeline, list):
        return jsonify({"success": False, "error": "游戏画面 OCR 时间线缺失"}), 404
    if sample_idx < 0 or sample_idx >= len(timeline):
        return jsonify({"success": False, "error": "样本索引越界"}), 400

    image_path = _existing_file((timeline[sample_idx] or {}).get("image_path"))
    if not image_path:
        return jsonify({"success": False, "error": "OCR 样本图像不存在"}), 404

    return send_file(image_path)


@tasks_bp.get("/api/tasks/<task_id>/gameplay-analysis")
def get_gameplay_analysis(task_id: str):
    task_store = current_app.extensions["task_store"]
    task = task_store.get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404

    status = str(task.get("gameplay_analysis_status") or "idle")
    summary = task.get("gameplay_analysis_summary")
    error = task.get("gameplay_analysis_error")
    result = None

    if status == "done":
        result_path = _existing_file((task.get("artifacts") or {}).get("gameplay_analysis_json"))
        if result_path:
            result = read_json(result_path)
        else:
            status = "failed"
            error = "游戏流程分析结果文件不存在"

    return jsonify(
        {
            "success": True,
            "status": status,
            "summary": summary,
            "error": error,
            "result": result,
            "params": {
                "video_path": (task.get("input") or {}).get("game_video_path"),
                "interval_sec": (task.get("input") or {}).get("game_ocr_interval_sec"),
                "game_ocr_status": task.get("game_ocr_status"),
            },
        }
    )


@tasks_bp.get("/api/tasks/<task_id>/au")
def get_au_series(task_id: str):
    task_store = current_app.extensions["task_store"]
    task = task_store.get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404

    name = str(request.args.get("name", "")).strip()
    if not name:
        return jsonify({"success": False, "error": "缺少 AU 名称"}), 400

    normalized = name.replace(" ", "").upper()
    if normalized.endswith("_R"):
        normalized = normalized[:-2]
    if normalized.startswith("AU"):
        au_name = f"{normalized}_r"
    else:
        return jsonify({"success": False, "error": "AU 名称格式不正确"}), 400

    if au_name not in FrameService.AU_COLUMNS:
        return jsonify({"success": False, "error": f"不支持的 AU: {au_name}"}), 400

    artifacts = task.get("artifacts", {})
    csv_path = _existing_file(artifacts.get("openface_csv"))
    if not csv_path:
        return jsonify({"success": False, "error": "OpenFace CSV 不存在"}), 404

    input_data = task.get("input", {})
    video_path = input_data.get("video_path")
    frame_stride = _parse_int(input_data.get("frame_stride", 1), 1, min_value=1)
    max_frames = _parse_int(input_data.get("max_frames", 0), 0, min_value=0)

    pipeline = current_app.extensions["pipeline_service"]
    frame_service = pipeline.frame_service
    timeline_df = frame_service.load_timeline_dataframe(
        csv_path,
        frame_stride=frame_stride,
        max_frames=max_frames,
        video_path=Path(video_path) if video_path else None,
        select_main_face=True,
    )

    timestamps = pd.to_numeric(timeline_df["timestamp"], errors="coerce").fillna(0.0).astype(float).to_list()
    frames = pd.to_numeric(timeline_df["frame"], errors="coerce").fillna(0).astype(int).to_list()
    raw_values = pd.to_numeric(timeline_df[au_name], errors="coerce").to_list()
    values = []
    for value in raw_values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            values.append(None)
            continue
        if not math.isfinite(number):
            values.append(None)
            continue
        values.append(number)

    return jsonify(
        {
            "success": True,
            "au": au_name,
            "timestamps": timestamps,
            "frames": frames,
            "values": values,
        }
    )


def _run_rppg_job(
    task_id: str,
    video_path: Path,
    params: dict,
    result_path: Path,
    task_store,
    rppg_service,
) -> None:
    task_store.add_log(task_id, "rPPG 后台任务开始执行")
    try:
        result = rppg_service.analyze(
            video_path=video_path,
            start_sec=params.get("start_sec"),
            duration_sec=params.get("duration_sec"),
            chunk_len=params.get("chunk_len", 160),
            device=params.get("device", "cpu"),
        )
        write_json(result_path, result)
        task_store.update_task(
            task_id,
            rppg_status="done",
            rppg_error=None,
            rppg_summary=result.get("summary"),
        )
        task_store.add_log(task_id, "rPPG 后台任务执行完成")
    except Exception as e:
        task_store.update_task(
            task_id,
            rppg_status="failed",
            rppg_error=str(e),
        )
        task_store.add_log(task_id, f"rPPG 后台任务失败: {e}")
    finally:
        _remove_worker(_RPPG_WORKERS, task_id)


def _run_transcription_job(
    task_id: str,
    video_path: Path,
    params: dict,
    result_path: Path,
    task_store,
    asr_service,
) -> None:
    task_store.add_log(task_id, "语音转写后台任务开始执行")
    try:
        result = asr_service.transcribe(
            video_path=video_path,
            segment_length_sec=params.get("segment_length_sec", 60),
            language=params.get("language", "zh"),
            use_itn=params.get("use_itn", True),
            device=params.get("device", "cpu"),
        )
        write_json(result_path, result)
        task_store.update_task(
            task_id,
            transcription_status="done",
            transcription_error=None,
            transcription_summary=result.get("summary"),
        )
        task_store.add_log(task_id, "语音转写后台任务执行完成")
    except Exception as e:
        task_store.update_task(
            task_id,
            transcription_status="failed",
            transcription_error=str(e),
        )
        task_store.add_log(task_id, f"语音转写后台任务失败: {e}")
    finally:
        _remove_worker(_TRANSCRIPTION_WORKERS, task_id)


def _reconcile_rppg_task(task_id: str, task: dict, task_store) -> dict:
    if task.get("rppg_status") != "running":
        return task
    if _is_worker_alive(_RPPG_WORKERS, task_id):
        return task

    artifacts = task.get("artifacts") or {}
    result_path = _existing_file(artifacts.get("rppg_json"))
    if result_path:
        try:
            result = read_json(result_path)
            updated = task_store.update_task(
                task_id,
                rppg_status="done",
                rppg_error=None,
                rppg_summary=(result or {}).get("summary"),
            )
            if isinstance(updated, dict):
                return updated
        except Exception:
            pass

    updated = task_store.update_task(
        task_id,
        rppg_status="failed",
        rppg_error="rPPG 后台任务已中断（服务重启或进程退出）",
    )
    if isinstance(updated, dict):
        return updated
    return task


def _reconcile_transcription_task(task_id: str, task: dict, task_store) -> dict:
    if task.get("transcription_status") != "running":
        return task
    if _is_worker_alive(_TRANSCRIPTION_WORKERS, task_id):
        return task

    artifacts = task.get("artifacts") or {}
    result_path = _existing_file(artifacts.get("transcription_json"))
    if result_path:
        try:
            result = read_json(result_path)
            updated = task_store.update_task(
                task_id,
                transcription_status="done",
                transcription_error=None,
                transcription_summary=(result or {}).get("summary"),
            )
            if isinstance(updated, dict):
                return updated
        except Exception:
            pass

    updated = task_store.update_task(
        task_id,
        transcription_status="failed",
        transcription_error="语音转写后台任务已中断（服务重启或进程退出）",
    )
    if isinstance(updated, dict):
        return updated
    return task


@tasks_bp.post("/api/tasks/<task_id>/rppg")
def run_rppg(task_id: str):
    task_store = current_app.extensions["task_store"]
    task = task_store.get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404

    if task.get("rppg_status") == "running":
        if _is_worker_alive(_RPPG_WORKERS, task_id):
            return jsonify({"success": False, "error": "rPPG 正在运行"}), 400
        task_store.update_task(
            task_id,
            rppg_status="failed",
            rppg_error="上一次 rPPG 任务已中断（服务重启或进程退出），请重新运行",
        )
        task = task_store.get_task(task_id) or task

    video_path_value = task.get("input", {}).get("video_path")
    video_path = _existing_file(video_path_value)
    if not video_path:
        return jsonify({"success": False, "error": "视频不存在"}), 400

    data = request.get_json(silent=True) or {}
    start_sec = _parse_float(data.get("start_sec"), 0.0)
    duration_sec = _parse_float(data.get("duration_sec"), None)
    if duration_sec is not None and duration_sec <= 0:
        duration_sec = None
    if start_sec is None or start_sec < 0:
        start_sec = 0.0

    chunk_len = _parse_int(data.get("chunk_len"), 160, min_value=8)
    device = str(data.get("device") or "cpu")

    params = {
        "start_sec": float(start_sec),
        "duration_sec": duration_sec,
        "chunk_len": int(chunk_len),
        "device": device,
    }

    result_root = Path(current_app.config["RESULT_DIR"]) / task_id
    result_root.mkdir(parents=True, exist_ok=True)
    result_path = result_root / "rppg_result.json"

    task_store.set_artifact(task_id, "rppg_json", str(result_path))
    task_store.update_task(
        task_id,
        rppg_status="running",
        rppg_error=None,
        rppg_summary=None,
        rppg_params=params,
    )

    rppg_service = current_app.extensions["rppg_service"]
    thread = threading.Thread(
        target=_run_rppg_job,
        args=(task_id, video_path, params, result_path, task_store, rppg_service),
    )
    _register_worker(_RPPG_WORKERS, task_id, thread)
    try:
        thread.start()
    except Exception as e:
        _remove_worker(_RPPG_WORKERS, task_id)
        task_store.update_task(
            task_id,
            rppg_status="failed",
            rppg_error=f"rPPG 启动失败: {e}",
        )
        return jsonify({"success": False, "error": f"rPPG 启动失败: {e}"}), 500

    return jsonify({"success": True, "status": "running", "params": params})


@tasks_bp.get("/api/tasks/<task_id>/rppg")
def get_rppg(task_id: str):
    task_store = current_app.extensions["task_store"]
    task = task_store.get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404

    task = _reconcile_rppg_task(task_id, task, task_store)

    status = task.get("rppg_status", "idle")
    summary = task.get("rppg_summary")
    error = task.get("rppg_error")
    result = None
    if status == "done":
        result_path = _existing_file((task.get("artifacts") or {}).get("rppg_json"))
        if result_path:
            result = read_json(result_path)
        else:
            status = "failed"
            error = "rPPG 结果文件不存在"

    return jsonify(
        {
            "success": True,
            "status": status,
            "summary": summary,
            "error": error,
            "result": result,
            "params": task.get("rppg_params"),
        }
    )


@tasks_bp.post("/api/tasks/<task_id>/transcription")
def run_transcription(task_id: str):
    task_store = current_app.extensions["task_store"]
    task = task_store.get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404

    if task.get("transcription_status") == "running":
        if _is_worker_alive(_TRANSCRIPTION_WORKERS, task_id):
            return jsonify({"success": False, "error": "语音转写正在运行"}), 400
        task_store.update_task(
            task_id,
            transcription_status="failed",
            transcription_error="上一次语音转写任务已中断（服务重启或进程退出），请重新运行",
        )
        task = task_store.get_task(task_id) or task

    video_path_value = task.get("input", {}).get("video_path")
    video_path = _existing_file(video_path_value)
    if not video_path:
        return jsonify({"success": False, "error": "视频不存在"}), 400

    data = request.get_json(silent=True) or {}
    segment_length_sec = _parse_int(data.get("segment_length_sec"), 60, min_value=10)
    language = str(data.get("language") or "zh")
    if "use_itn" in data:
        use_itn = _parse_bool(str(data.get("use_itn")))
    else:
        use_itn = True
    device = str(data.get("device") or "cpu")

    params = {
        "segment_length_sec": int(segment_length_sec),
        "language": language,
        "use_itn": bool(use_itn),
        "device": device,
    }

    result_root = Path(current_app.config["RESULT_DIR"]) / task_id
    result_root.mkdir(parents=True, exist_ok=True)
    result_path = result_root / "transcription_result.json"

    task_store.set_artifact(task_id, "transcription_json", str(result_path))
    task_store.update_task(
        task_id,
        transcription_status="running",
        transcription_error=None,
        transcription_summary=None,
        transcription_params=params,
    )

    asr_service = current_app.extensions["asr_service"]
    thread = threading.Thread(
        target=_run_transcription_job,
        args=(task_id, video_path, params, result_path, task_store, asr_service),
    )
    _register_worker(_TRANSCRIPTION_WORKERS, task_id, thread)
    try:
        thread.start()
    except Exception as e:
        _remove_worker(_TRANSCRIPTION_WORKERS, task_id)
        task_store.update_task(
            task_id,
            transcription_status="failed",
            transcription_error=f"语音转写启动失败: {e}",
        )
        return jsonify({"success": False, "error": f"语音转写启动失败: {e}"}), 500

    return jsonify({"success": True, "status": "running", "params": params})


@tasks_bp.get("/api/tasks/<task_id>/transcription")
def get_transcription(task_id: str):
    task_store = current_app.extensions["task_store"]
    task = task_store.get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404

    task = _reconcile_transcription_task(task_id, task, task_store)

    status = task.get("transcription_status", "idle")
    summary = task.get("transcription_summary")
    error = task.get("transcription_error")
    result = None
    if status == "done":
        result_path = _existing_file((task.get("artifacts") or {}).get("transcription_json"))
        if result_path:
            result = read_json(result_path)
        else:
            status = "failed"
            error = "语音转写结果文件不存在"

    return jsonify(
        {
            "success": True,
            "status": status,
            "summary": summary,
            "error": error,
            "result": result,
        }
    )
