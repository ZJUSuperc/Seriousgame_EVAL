import os
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    value = os.getenv(name)
    return Path(value) if value else Path(default)


def _env_bool(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_config() -> dict:
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    report_dir = _env_path("SG_REPORT_DIR", str(data_dir / "reports"))
    toolkit_root = _env_path("SG_TOOLKIT_ROOT", "C:/Users/Administrator/Desktop/openfacetest")
    affectnet_root = _env_path("SG_AFFECTNET_ROOT", "C:/Users/Administrator/Desktop/AffectNet-master")
    emonet_root = _env_path("SG_EMONET_ROOT", "C:/Users/Administrator/Desktop/emonet-master")
    dan_package_root = _env_path(
        "SG_DAN_PACKAGE_ROOT",
        "C:/Users/Administrator/Desktop/download_packages/best_affectnet7_dan_exp1_baseline_ls_20260330_152032",
    )
    rppg_root = _env_path("SG_RPPG_ROOT", "C:/Users/Administrator/Desktop/RhythmMamba-main/portable_rppg_infer")
    sensevoice_root = _env_path("SG_SENSEVOICE_ROOT", "C:/Users/Administrator/Desktop/SenseVoice-main/video_transcription_tool")
    ocr_toolkit_root = _env_path("SG_OCR_TOOLKIT_ROOT", "C:/Users/Administrator/Desktop/PaddleOCR-main/ocr_toolkit")
    ocr_repo_root = _env_path("SG_OCR_REPO_ROOT", str(ocr_toolkit_root.parent))
    openface_dir = _env_path(
        "SG_OPENFACE_DIR",
        str(toolkit_root / "3dparty" / "OpenFace_2.2.0_win_x64"),
    )
    emotion_analyzer_parent = _env_path(
        "SG_EMOTION_ANALYZER_PARENT",
        str(toolkit_root / "3dparty"),
    )

    emonet_n_classes = os.getenv("SG_EMONET_N_CLASSES", "8")
    try:
        emonet_n_classes = int(emonet_n_classes)
    except (TypeError, ValueError):
        emonet_n_classes = 8
    if emonet_n_classes not in {5, 8}:
        emonet_n_classes = 8

    emonet_model_path = os.getenv(
        "SG_EMONET_MODEL_PATH",
        str(emonet_root / "pretrained" / f"emonet_{emonet_n_classes}.pth"),
    )

    deepface_backend = str(os.getenv("SG_DEEPFACE_BACKEND", "dan")).strip().lower()
    if deepface_backend not in {"dan", "affectnet"}:
        deepface_backend = "dan"

    dan_num_head = _env_int("SG_DAN_NUM_HEAD", 0)
    if dan_num_head < 0:
        dan_num_head = 0

    dan_input_size = _env_int("SG_DAN_INPUT_SIZE", 224)
    if dan_input_size <= 0:
        dan_input_size = 224

    dan_model_path = os.getenv(
        "SG_DAN_MODEL_PATH",
        str(dan_package_root / "model" / "best.pth"),
    )
    dan_train_config_path = os.getenv(
        "SG_DAN_TRAIN_CONFIG_PATH",
        str(dan_package_root / "model" / "train_config.json"),
    )
    dan_face_cascade_path = os.getenv(
        "SG_DAN_FACE_CASCADE_PATH",
        str(rppg_root / "dataset" / "haarcascade_frontalface_default.xml"),
    )

    return {
        "SECRET_KEY": os.getenv("SG_SECRET_KEY", "seriousgame-local-secret"),
        "BASE_DIR": str(base_dir),
        "DATA_DIR": str(data_dir),
        "REPORT_DIR": str(report_dir),
        "UPLOAD_DIR": str(data_dir / "uploads"),
        "RESULT_DIR": str(data_dir / "results"),
        "TASK_DIR": str(data_dir / "tasks"),
        "TOOLKIT_ROOT": str(toolkit_root),
        "AFFECTNET_ROOT": str(affectnet_root),
        "AFFECTNET_FACE_MODEL": str(affectnet_root / "deployment" / "frozen_graphs" / "frozen_inference_graph_face.pb"),
        "AFFECTNET_CLASS_MODEL": str(affectnet_root / "deployment" / "frozen_graphs" / "classificator_full_model.pb"),
        "AFFECTNET_REGRESS_MODEL": str(affectnet_root / "deployment" / "frozen_graphs" / "regressor_full_model.pb"),
        "DEEPFACE_BACKEND": deepface_backend,
        "DAN_PACKAGE_ROOT": str(dan_package_root),
        "DAN_MODEL_PATH": str(dan_model_path),
        "DAN_TRAIN_CONFIG_PATH": str(dan_train_config_path),
        "DAN_NUM_HEAD": int(dan_num_head),
        "DAN_DEVICE": os.getenv("SG_DAN_DEVICE", "auto"),
        "DAN_INPUT_SIZE": int(dan_input_size),
        "DAN_FACE_CASCADE_PATH": str(dan_face_cascade_path),
        "EMONET_ROOT": str(emonet_root),
        "EMONET_N_CLASSES": int(emonet_n_classes),
        "EMONET_MODEL_PATH": str(emonet_model_path),
        "EMONET_IMAGE_SIZE": int(os.getenv("SG_EMONET_IMAGE_SIZE", "256")),
        "OPENFACE_DIR": str(openface_dir),
        "OPENFACE_SINGLE_EXE": str(openface_dir / "FeatureExtraction.exe"),
        "OPENFACE_MULTI_EXE": str(openface_dir / "FaceLandmarkVidMulti.exe"),
        "EMOTION_ANALYZER_PARENT": str(emotion_analyzer_parent),
        "EMONET_DEVICE": os.getenv("SG_EMONET_DEVICE", "cpu"),
        "CONFIDENCE_THRESHOLD": float(os.getenv("SG_CONF_THRESHOLD", "0.8")),
        "ENABLE_QUALITY_FILTER": _env_bool("SG_ENABLE_QUALITY_FILTER", "0"),
        "RPPG_ROOT": str(rppg_root),
        "RPPG_MODEL_PATH": os.getenv(
            "SG_RPPG_MODEL_PATH",
            str(rppg_root / "PreTrainedModels" / "UBFC_cross_RhythmMamba.pth"),
        ),
        "RPPG_CASCADE_PATH": os.getenv(
            "SG_RPPG_CASCADE_PATH",
            str(rppg_root / "dataset" / "haarcascade_frontalface_default.xml"),
        ),
        "OCR_TOOLKIT_ROOT": str(ocr_toolkit_root),
        "OCR_REPO_ROOT": str(ocr_repo_root),
        "GAME_OCR_INTERVAL_SEC_DEFAULT": _env_float("SG_GAME_OCR_INTERVAL_SEC_DEFAULT", 2.0),
        "GAME_OCR_LANG": os.getenv("SG_GAME_OCR_LANG", "ch"),
        "GAME_OCR_VERSION": os.getenv("SG_GAME_OCR_VERSION", "PP-OCRv5"),
        "GAME_OCR_DEVICE": os.getenv("SG_GAME_OCR_DEVICE", "cpu"),
        "GAME_OCR_MIN_SCORE": _env_float("SG_GAME_OCR_MIN_SCORE", 0.0),
        "GAME_OCR_DET_MODEL_DIR": os.getenv("SG_GAME_OCR_DET_MODEL_DIR", ""),
        "GAME_OCR_REC_MODEL_DIR": os.getenv("SG_GAME_OCR_REC_MODEL_DIR", ""),
        "GAME_OCR_MAX_IMAGE_SIDE": _env_int("SG_GAME_OCR_MAX_IMAGE_SIDE", 1280),
        "GAME_OCR_SAMPLE_TIMEOUT_SEC": _env_float("SG_GAME_OCR_SAMPLE_TIMEOUT_SEC", 600.0),
        "GAME_OCR_HEARTBEAT_SEC": _env_float("SG_GAME_OCR_HEARTBEAT_SEC", 15.0),
        "GAME_OCR_MAX_CONSECUTIVE_TIMEOUTS": _env_int("SG_GAME_OCR_MAX_CONSECUTIVE_TIMEOUTS", 0),
        "GAME_OCR_DISABLE_MODEL_SOURCE_CHECK": _env_bool("SG_GAME_OCR_DISABLE_MODEL_SOURCE_CHECK", "1"),
        "SENSEVOICE_ROOT": str(sensevoice_root),
        "FFMPEG_BIN": os.getenv("SG_FFMPEG_BIN", "ffmpeg"),
    }
