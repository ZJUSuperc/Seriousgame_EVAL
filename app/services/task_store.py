import threading
import uuid
from datetime import datetime
from pathlib import Path

from ..utils.json_io import read_json, write_json


class TaskStore:
    def __init__(self, task_dir: Path):
        self.task_dir = task_dir
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def create_task(self, payload: dict) -> dict:
        task_id = str(uuid.uuid4())
        now = _now()
        task = {
            "task_id": task_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "input": payload,
            "artifacts": {},
            "error": None,
            "rppg_status": "idle",
            "rppg_error": None,
            "rppg_summary": None,
            "transcription_status": "idle",
            "transcription_error": None,
            "transcription_summary": None,
            "game_ocr_status": "idle",
            "game_ocr_error": None,
            "game_ocr_summary": None,
            "gameplay_analysis_status": "idle",
            "gameplay_analysis_error": None,
            "gameplay_analysis_summary": None,
            "logs": [{"time": now, "message": "任务已创建"}],
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            write_json(self._task_path(task_id), task)
        return task

    def get_task(self, task_id: str) -> dict | None:
        path = self._task_path(task_id)
        if not path.exists():
            return None
        with self._lock:
            return read_json(path)

    def list_tasks(self, limit: int = 30) -> list[dict]:
        files = sorted(self.task_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        items = []
        for file in files[:limit]:
            task = read_json(file)
            items.append(
                {
                    "task_id": task.get("task_id"),
                    "status": task.get("status"),
                    "stage": task.get("stage"),
                    "progress": task.get("progress"),
                    "created_at": task.get("created_at"),
                    "updated_at": task.get("updated_at"),
                }
            )
        return items

    def update_task(self, task_id: str, **updates) -> dict | None:
        with self._lock:
            path = self._task_path(task_id)
            if not path.exists():
                return None
            task = read_json(path)
            task.update(updates)
            task["updated_at"] = _now()
            write_json(path, task)
            return task

    def add_log(self, task_id: str, message: str) -> dict | None:
        with self._lock:
            path = self._task_path(task_id)
            if not path.exists():
                return None
            task = read_json(path)
            logs = task.get("logs", [])
            logs.append({"time": _now(), "message": message})
            if len(logs) > 500:
                logs = logs[-500:]
            task["logs"] = logs
            task["updated_at"] = _now()
            write_json(path, task)
            return task

    def set_artifact(self, task_id: str, key: str, value: str) -> dict | None:
        with self._lock:
            path = self._task_path(task_id)
            if not path.exists():
                return None
            task = read_json(path)
            artifacts = task.get("artifacts", {})
            artifacts[key] = value
            task["artifacts"] = artifacts
            task["updated_at"] = _now()
            write_json(path, task)
            return task

    def recover_interrupted_aux_jobs(self) -> dict[str, int]:
        recovered = {
            "pipeline": 0,
            "rppg": 0,
            "transcription": 0,
        }
        with self._lock:
            for path in self.task_dir.glob("*.json"):
                try:
                    task = read_json(path)
                except Exception:
                    continue

                changed = False
                logs = task.get("logs")
                if not isinstance(logs, list):
                    logs = []

                if task.get("status") == "running":
                    task["status"] = "failed"
                    task["stage"] = "failed"
                    if not task.get("error"):
                        task["error"] = "主分析任务因服务重启被中断，请重新运行"

                    if task.get("game_ocr_status") == "running":
                        task["game_ocr_status"] = "failed"
                        if not task.get("game_ocr_error"):
                            task["game_ocr_error"] = "游戏画面 OCR 因服务重启被中断，请重新运行"

                    if task.get("gameplay_analysis_status") == "running":
                        task["gameplay_analysis_status"] = "failed"
                        if not task.get("gameplay_analysis_error"):
                            task["gameplay_analysis_error"] = "游戏流程分析因服务重启被中断，请重新运行"

                    recovered["pipeline"] += 1
                    changed = True

                if task.get("rppg_status") == "running":
                    task["rppg_status"] = "failed"
                    if not task.get("rppg_error"):
                        task["rppg_error"] = "rPPG 后台任务因服务重启被中断，请重新运行"
                    recovered["rppg"] += 1
                    changed = True

                if task.get("transcription_status") == "running":
                    task["transcription_status"] = "failed"
                    if not task.get("transcription_error"):
                        task["transcription_error"] = "语音转写后台任务因服务重启被中断，请重新运行"
                    recovered["transcription"] += 1
                    changed = True

                if not changed:
                    continue

                logs.append({"time": _now(), "message": "检测到上次服务异常退出，已将中断的辅助任务标记为失败"})
                if len(logs) > 500:
                    logs = logs[-500:]
                task["logs"] = logs
                task["updated_at"] = _now()
                write_json(path, task)

        return recovered

    def _task_path(self, task_id: str) -> Path:
        return self.task_dir / f"{task_id}.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
