from __future__ import annotations

import math
import os
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_AFFECTNET_ROOT = "C:/Users/Administrator/Desktop/AffectNet-master"


class AffectNetBackend:
    EMOTION_CLASSES = [
        "Neutral",
        "Happy",
        "Sad",
        "Surprised",
        "Afraid",
        "Disgusted",
        "Angry",
        "Contemptuous",
    ]
    EMOTION_MAP = {
        "Neutral": "neutral",
        "Happy": "happy",
        "Sad": "sad",
        "Surprised": "surprise",
        "Afraid": "fear",
        "Disgusted": "disgust",
        "Angry": "angry",
        "Contemptuous": "contempt",
    }

    def __init__(
        self,
        root: Path,
        face_model: Path | None = None,
        class_model: Path | None = None,
        regress_model: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.face_model = Path(face_model) if face_model else self.root / "deployment" / "frozen_graphs" / "frozen_inference_graph_face.pb"
        self.class_model = Path(class_model) if class_model else self.root / "deployment" / "frozen_graphs" / "classificator_full_model.pb"
        self.regress_model = Path(regress_model) if regress_model else self.root / "deployment" / "frozen_graphs" / "regressor_full_model.pb"
        self._lock = threading.Lock()
        self._loaded = False
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_limit = 4

        self._detection_graph = None
        self._classification_graph = None
        self._regression_graph = None
        self._sess_detection = None
        self._sess_classification = None
        self._sess_regression = None

        self._tensor_detection = {}
        self._tensor_classification = {}
        self._tensor_regression = {}

        self._preprocess_input = None
        self._img_to_array = None

    @classmethod
    def map_emotion(cls, value: str | None) -> str:
        if value is None:
            return ""
        raw = str(value).strip()
        if not raw:
            return ""
        return cls.EMOTION_MAP.get(raw, raw.lower())

    @classmethod
    def map_probabilities(cls, probs: dict) -> dict[str, float]:
        mapped: dict[str, float] = {}
        if not isinstance(probs, dict):
            return mapped
        for key, value in probs.items():
            mapped_key = cls.map_emotion(key)
            if not mapped_key:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            mapped[mapped_key] = mapped.get(mapped_key, 0.0) + number
        return mapped

    def analyze(self, image_path: str) -> dict:
        path_text = str(image_path)
        with self._lock:
            self._ensure_loaded()
            cached = self._cache_get(path_text)
            if cached is not None:
                return cached
            face = self._infer_primary_face(path_text)
            self._cache_set(path_text, face)
            return face

    def _cache_get(self, key: str) -> dict | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_set(self, key: str, value: dict) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        missing = [path for path in (self.face_model, self.class_model, self.regress_model) if not Path(path).exists()]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"AffectNet 模型文件不存在: {missing_text}")

        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet import preprocess_input
        from tensorflow.keras.preprocessing.image import img_to_array

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        self._detection_graph = tf.Graph()
        with self._detection_graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(str(self.face_model), "rb") as fid:
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name="")
        self._sess_detection = tf.compat.v1.Session(graph=self._detection_graph, config=config)

        self._classification_graph = tf.Graph()
        with self._classification_graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(str(self.class_model), "rb") as fid:
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name="")
        self._sess_classification = tf.compat.v1.Session(graph=self._classification_graph, config=config)

        self._regression_graph = tf.Graph()
        with self._regression_graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(str(self.regress_model), "rb") as fid:
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name="")
        self._sess_regression = tf.compat.v1.Session(graph=self._regression_graph, config=config)

        self._tensor_detection = {
            "image": self._detection_graph.get_tensor_by_name("image_tensor:0"),
            "boxes": self._detection_graph.get_tensor_by_name("detection_boxes:0"),
            "scores": self._detection_graph.get_tensor_by_name("detection_scores:0"),
            "classes": self._detection_graph.get_tensor_by_name("detection_classes:0"),
            "count": self._detection_graph.get_tensor_by_name("num_detections:0"),
        }
        self._tensor_classification = {
            "input": self._classification_graph.get_tensor_by_name("input_1:0"),
            "output": self._classification_graph.get_tensor_by_name("dense_2/Softmax:0"),
        }
        self._tensor_regression = {
            "input": self._regression_graph.get_tensor_by_name("input_1:0"),
            "output": self._regression_graph.get_tensor_by_name("dense_2/BiasAdd:0"),
        }

        self._preprocess_input = preprocess_input
        self._img_to_array = img_to_array
        self._loaded = True

    def _infer_primary_face(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像不存在: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"无法读取图像: {image_path}")

        faces = self._detect_and_analyze(image)
        primary = self._select_primary_face(faces)
        if not primary:
            raise RuntimeError("未检测到人脸")
        return primary

    def _detect_and_analyze(self, image_bgr: np.ndarray) -> list[dict]:
        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_rgb, axis=0)

        det = self._tensor_detection
        boxes, scores, classes, num_detections = self._sess_detection.run(
            [det["boxes"], det["scores"], det["classes"], det["count"]],
            feed_dict={det["image"]: image_np_expanded},
        )
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if scores.ndim == 0:
            scores = np.expand_dims(scores, axis=0)

        images_for_prediction = []
        face_info = []
        max_faces = min(20, boxes.shape[0]) if boxes.ndim > 1 else 0
        for i in range(max_faces):
            score = float(scores[i]) if scores is not None else 0.0
            if scores is not None and score <= 0.7:
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            y1 = max(int(height * ymin) - 20, 0)
            y2 = min(int(height * ymax) + 20, height)
            x1 = max(int(width * xmin) - 20, 0)
            x2 = min(int(width * xmax) + 20, width)
            face_img = image_rgb[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            face_img = Image.fromarray(face_img).resize((224, 224))
            face_arr = self._img_to_array(face_img)
            face_arr = self._preprocess_input(face_arr)

            images_for_prediction.append(face_arr)
            face_info.append(
                {
                    "box": [float(ymin), float(xmin), float(ymax), float(xmax)],
                    "score": score,
                }
            )

        if not images_for_prediction:
            return []

        cls = self._tensor_classification
        reg = self._tensor_regression

        batch = np.asarray(images_for_prediction, dtype=np.float32)
        class_predictions = self._sess_classification.run(
            cls["output"],
            feed_dict={cls["input"]: batch},
        )
        va_predictions = self._sess_regression.run(
            reg["output"],
            feed_dict={reg["input"]: batch},
        )

        results = []
        for idx, (class_pred, va_pred, info) in enumerate(zip(class_predictions, va_predictions, face_info)):
            emotion_idx = int(np.argmax(class_pred))
            emotion = self.EMOTION_CLASSES[emotion_idx]
            confidence = float(class_pred[emotion_idx])
            valence = float(va_pred[0]) if len(va_pred) > 0 else 0.0
            arousal = float(va_pred[1]) if len(va_pred) > 1 else 0.0

            probs = {self.EMOTION_CLASSES[j]: float(class_pred[j]) for j in range(len(self.EMOTION_CLASSES))}
            results.append(
                {
                    "face_id": idx + 1,
                    "emotion": emotion,
                    "emotion_idx": emotion_idx,
                    "confidence": confidence,
                    "all_probabilities": probs,
                    "valence": valence,
                    "arousal": arousal,
                    "bounding_box": info["box"],
                    "detection_score": info["score"],
                }
            )
        return results

    @staticmethod
    def _select_primary_face(faces: list[dict]) -> dict | None:
        if not faces:
            return None

        best_face = None
        best_score = -1.0
        for face in faces:
            det_score = float(face.get("detection_score") or 0.0)
            box = face.get("bounding_box") or face.get("box")
            area_score = 0.0
            center_score = 0.0
            if isinstance(box, (list, tuple)) and len(box) == 4:
                ymin, xmin, ymax, xmax = [float(v) for v in box]
                area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
                area_score = max(0.0, min(1.0, area))
                cx = (xmin + xmax) / 2.0
                cy = (ymin + ymax) / 2.0
                dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
                center_score = math.exp(-2.0 * dist)
            final_score = 0.6 * det_score + 0.3 * area_score + 0.1 * center_score
            if final_score > best_score:
                best_score = final_score
                best_face = face
        return best_face


_BACKEND: AffectNetBackend | None = None
_BACKEND_LOCK = threading.Lock()


def get_affectnet_backend(config: dict | None = None) -> AffectNetBackend:
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is not None:
            return _BACKEND

        root = None
        face_model = None
        class_model = None
        regress_model = None

        if isinstance(config, dict):
            root = config.get("AFFECTNET_ROOT")
            face_model = config.get("AFFECTNET_FACE_MODEL")
            class_model = config.get("AFFECTNET_CLASS_MODEL")
            regress_model = config.get("AFFECTNET_REGRESS_MODEL")

        if not root:
            root = os.getenv("SG_AFFECTNET_ROOT", DEFAULT_AFFECTNET_ROOT)

        _BACKEND = AffectNetBackend(
            Path(root),
            face_model=Path(face_model) if face_model else None,
            class_model=Path(class_model) if class_model else None,
            regress_model=Path(regress_model) if regress_model else None,
        )
        return _BACKEND
