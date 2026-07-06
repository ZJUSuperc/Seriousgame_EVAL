from pathlib import Path

from flask import Flask

from .blueprints.api_health import health_bp
from .blueprints.api_tasks import tasks_bp
from .blueprints.web import web_bp
from .config import load_config
from .services.asr_service import AsrService
from .services.pipeline_service import PipelineService
from .services.rppg_service import RppgService
from .services.task_store import TaskStore


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(load_config())
    _ensure_directories(app.config)

    task_store = TaskStore(Path(app.config["TASK_DIR"]))
    recovered = task_store.recover_interrupted_aux_jobs()
    pipeline_service = PipelineService(app.config, task_store)
    rppg_service = RppgService(app.config)
    asr_service = AsrService(app.config)

    app.extensions["task_store"] = task_store
    app.extensions["pipeline_service"] = pipeline_service
    app.extensions["rppg_service"] = rppg_service
    app.extensions["asr_service"] = asr_service

    app.register_blueprint(web_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(tasks_bp)

    recovered_count = (
        int(recovered.get("pipeline", 0))
        + int(recovered.get("rppg", 0))
        + int(recovered.get("transcription", 0))
    )
    if recovered_count > 0:
        app.logger.warning(
            "已恢复中断任务: pipeline=%s, rPPG=%s, transcription=%s",
            recovered.get("pipeline", 0),
            recovered.get("rppg", 0),
            recovered.get("transcription", 0),
        )

    return app


def _ensure_directories(config: dict) -> None:
    for key in ("DATA_DIR", "UPLOAD_DIR", "RESULT_DIR", "TASK_DIR", "REPORT_DIR"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)
