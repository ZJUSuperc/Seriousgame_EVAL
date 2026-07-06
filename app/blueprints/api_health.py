from pathlib import Path
import shutil

from flask import Blueprint, current_app, jsonify


health_bp = Blueprint("health", __name__)


@health_bp.get("/api/health")
def health_check():
    cfg = current_app.config
    openface_single = Path(cfg["OPENFACE_SINGLE_EXE"])
    openface_multi = Path(cfg["OPENFACE_MULTI_EXE"])
    emotion_parent = Path(cfg["EMOTION_ANALYZER_PARENT"])
    affectnet_root = Path(cfg.get("AFFECTNET_ROOT", ""))
    affectnet_face = Path(cfg.get("AFFECTNET_FACE_MODEL", ""))
    affectnet_class = Path(cfg.get("AFFECTNET_CLASS_MODEL", ""))
    affectnet_regress = Path(cfg.get("AFFECTNET_REGRESS_MODEL", ""))
    deepface_backend = str(cfg.get("DEEPFACE_BACKEND", "affectnet"))
    dan_package_root = Path(cfg.get("DAN_PACKAGE_ROOT", ""))
    dan_model = Path(cfg.get("DAN_MODEL_PATH", ""))
    dan_train_config = Path(cfg.get("DAN_TRAIN_CONFIG_PATH", ""))
    dan_face_cascade = Path(cfg.get("DAN_FACE_CASCADE_PATH", ""))
    ocr_toolkit_root = Path(cfg.get("OCR_TOOLKIT_ROOT", ""))
    ocr_repo_root = Path(cfg.get("OCR_REPO_ROOT", ""))
    ocr_script = ocr_toolkit_root / "ocr_image_zh.py"
    emonet_root = Path(cfg.get("EMONET_ROOT", ""))
    emonet_model = Path(cfg.get("EMONET_MODEL_PATH", ""))
    rppg_model = Path(cfg["RPPG_MODEL_PATH"])
    rppg_cascade = Path(cfg["RPPG_CASCADE_PATH"])
    sensevoice_root = Path(cfg["SENSEVOICE_ROOT"])
    ffmpeg_bin = str(cfg.get("FFMPEG_BIN", "ffmpeg"))

    return jsonify(
        {
            "success": True,
            "openface_single_exists": openface_single.exists(),
            "openface_multi_exists": openface_multi.exists(),
            "emotion_analyzer_parent_exists": emotion_parent.exists(),
            "affectnet_root_exists": affectnet_root.exists(),
            "affectnet_face_model_exists": affectnet_face.exists(),
            "affectnet_class_model_exists": affectnet_class.exists(),
            "affectnet_regress_model_exists": affectnet_regress.exists(),
            "deepface_backend": deepface_backend,
            "dan_package_root_exists": dan_package_root.exists(),
            "dan_model_exists": dan_model.exists(),
            "dan_train_config_exists": dan_train_config.exists(),
            "dan_face_cascade_exists": dan_face_cascade.exists(),
            "ocr_toolkit_root_exists": ocr_toolkit_root.exists(),
            "ocr_repo_root_exists": ocr_repo_root.exists(),
            "ocr_toolkit_script_exists": ocr_script.exists(),
            "emonet_root_exists": emonet_root.exists(),
            "emonet_model_exists": emonet_model.exists(),
            "openface_dir": cfg["OPENFACE_DIR"],
            "emonet_device": cfg["EMONET_DEVICE"],
            "dan_device": cfg.get("DAN_DEVICE", "auto"),
            "game_ocr_interval_sec_default": cfg.get("GAME_OCR_INTERVAL_SEC_DEFAULT", 2.0),
            "rppg_model_exists": rppg_model.exists(),
            "rppg_cascade_exists": rppg_cascade.exists(),
            "sensevoice_root_exists": sensevoice_root.exists(),
            "ffmpeg_available": bool(shutil.which(ffmpeg_bin)),
        }
    )
