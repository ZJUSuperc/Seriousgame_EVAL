from __future__ import annotations

import json
import math
import os
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.init as init
from PIL import Image
from torch import nn
from torch.nn import functional as F


DEFAULT_DAN_PACKAGE_ROOT = (
    "C:/Users/Administrator/Desktop/download_packages/"
    "best_affectnet7_dan_exp1_baseline_ls_20260330_152032"
)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def _build_resnet18_features() -> nn.Sequential:
    inplanes = 64

    def _make_layer(planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        nonlocal inplanes

        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                _conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = [_ResNetBasicBlock(inplanes, planes, stride, downsample)]
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(_ResNetBasicBlock(inplanes, planes))
        return nn.Sequential(*layers)

    features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        _make_layer(64, 2),
        _make_layer(128, 2, stride=2),
        _make_layer(256, 2, stride=2),
        _make_layer(512, 2, stride=2),
    )

    for module in features.modules():
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

    return features


class DAN(nn.Module):
    def __init__(self, num_class: int = 7, num_head: int = 4) -> None:
        super().__init__()

        self.features = _build_resnet18_features()
        self.num_head = num_head
        for i in range(num_head):
            setattr(self, f"cat_head{i}", CrossAttentionHead())
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self, f"cat_head{i}")(x))

        heads = torch.stack(heads).permute([1, 0, 2])
        if heads.size(1) > 1:
            heads = F.log_softmax(heads, dim=1)

        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)

        return out, x, heads


class CrossAttentionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)
        return ca


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1, keepdim=True)
        out = x * y
        return out


class ChannelAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid(),
        )

    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0), -1)
        y = self.attention(sa)
        out = sa * y
        return out


class DanBackend:
    EMOTION_CLASSES = [
        "neutral",
        "happiness",
        "sadness",
        "surprise",
        "fear",
        "disgust",
        "anger",
    ]
    EMOTION_MAP = {
        "neutral": "neutral",
        "happiness": "happy",
        "happy": "happy",
        "sadness": "sad",
        "sad": "sad",
        "surprise": "surprise",
        "fear": "fear",
        "disgust": "disgust",
        "anger": "angry",
        "angry": "angry",
    }

    def __init__(
        self,
        package_root: Path,
        model_path: Path | None = None,
        train_config_path: Path | None = None,
        device: str = "auto",
        num_head: int | None = None,
        input_size: int = 224,
        face_cascade_path: Path | None = None,
    ) -> None:
        self.package_root = Path(package_root)
        self.model_path = Path(model_path) if model_path else self.package_root / "model" / "best.pth"
        self.train_config_path = (
            Path(train_config_path) if train_config_path else self.package_root / "model" / "train_config.json"
        )
        self.device_text = str(device or "auto")
        self.num_head = int(num_head) if isinstance(num_head, int) else None
        self.input_size = max(64, int(input_size) if isinstance(input_size, int) else 224)
        self.face_cascade_path = Path(face_cascade_path) if face_cascade_path else None

        self._lock = threading.Lock()
        self._loaded = False
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_limit = 8

        self._device = torch.device("cpu")
        self._model: DAN | None = None
        self._transform = None
        self._face_detector = None

    @classmethod
    def map_emotion(cls, value: str | None) -> str:
        if value is None:
            return ""
        raw = str(value).strip()
        if not raw:
            return ""
        key = raw.lower()
        return cls.EMOTION_MAP.get(key, key)

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
            if not math.isfinite(number):
                continue
            mapped[mapped_key] = mapped.get(mapped_key, 0.0) + number
        return mapped

    def analyze(self, image_path: str) -> dict:
        path_text = str(image_path)
        with self._lock:
            self._ensure_loaded()
            cached = self._cache_get(path_text)
            if cached is not None:
                return dict(cached)
            face = self._infer_primary_face(path_text)
            self._cache_set(path_text, face)
            return dict(face)

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

        if not self.model_path.exists():
            raise FileNotFoundError(f"DAN 模型文件不存在: {self.model_path}")

        self._device = self._resolve_device(self.device_text)
        num_head = self._resolve_num_head()

        model = DAN(num_class=len(self.EMOTION_CLASSES), num_head=num_head)
        try:
            checkpoint = torch.load(self.model_path, map_location=self._device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location=self._device)
        state_dict = self._extract_state_dict(checkpoint)

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            cleaned = self._strip_module_prefix(state_dict)
            model.load_state_dict(cleaned, strict=True)

        model.to(self._device)
        model.eval()

        self._model = model
        self._transform = self._build_transform()
        self._face_detector = self._build_face_detector()
        self._loaded = True

    def _resolve_num_head(self) -> int:
        if self.num_head is not None and self.num_head > 0:
            return int(self.num_head)

        config_path = self.train_config_path
        if config_path.exists():
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                value = payload.get("num_head")
                value = int(value)
                if value > 0:
                    return value
            except (OSError, ValueError, TypeError):
                pass

        return 4

    @staticmethod
    def _resolve_device(value: str) -> torch.device:
        text = str(value or "auto").strip().lower()
        if text == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if text.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(text)
            return torch.device("cpu")
        return torch.device("cpu")

    def _build_transform(self):
        target_size = max(64, int(self.input_size))
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        if hasattr(Image, "Resampling"):
            resample = getattr(Image.Resampling, "BILINEAR", 2)
        else:
            resample = getattr(Image, "BILINEAR", 2)

        def _transform(image: Image.Image) -> torch.Tensor:
            if not isinstance(image, Image.Image):
                raise TypeError("DAN 预处理仅支持 PIL.Image")

            rgb = image.convert("RGB")
            resized = rgb.resize((target_size, target_size), resample)
            array = np.asarray(resized, dtype=np.float32)
            if array.ndim == 2:
                array = np.stack([array, array, array], axis=-1)
            if array.shape[-1] != 3:
                raise RuntimeError("DAN 输入图像通道数异常")

            array = array / 255.0
            array = np.transpose(array, (2, 0, 1))
            array = (array - mean) / std
            array = np.ascontiguousarray(array, dtype=np.float32)
            return torch.from_numpy(array)

        return _transform

    @staticmethod
    def _extract_state_dict(checkpoint) -> dict:
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError("DAN checkpoint 内容无效，未找到可用 state_dict")
        return state_dict

    @staticmethod
    def _strip_module_prefix(state_dict: dict) -> dict:
        cleaned: dict = {}
        for key, value in state_dict.items():
            key_text = str(key)
            if key_text.startswith("module."):
                key_text = key_text[7:]
            cleaned[key_text] = value
        return cleaned

    def _build_face_detector(self):
        candidates: list[Path] = []
        if self.face_cascade_path is not None:
            candidates.append(self.face_cascade_path)

        cv2_default = Path(getattr(cv2.data, "haarcascades", "")) / "haarcascade_frontalface_default.xml"
        candidates.append(cv2_default)

        for path in candidates:
            if not str(path):
                continue
            if not path.exists():
                continue
            detector = cv2.CascadeClassifier(str(path))
            if detector.empty():
                continue
            return detector

        return None

    def _infer_primary_face(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像不存在: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"无法读取图像: {image_path}")

        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            raise RuntimeError(f"图像尺寸异常: {image_path}")

        box = self._detect_primary_face(image)
        if box is None:
            x1, y1, x2, y2 = 0, 0, width, height
            detection_score = 0.0
        else:
            x, y, w, h = box
            margin_x = int(round(w * 0.18))
            margin_y = int(round(h * 0.18))
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(width, x + w + margin_x)
            y2 = min(height, y + h + margin_y)
            detection_score = 1.0

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image
            x1, y1, x2, y2 = 0, 0, width, height
            detection_score = 0.0

        if self._model is None or self._transform is None:
            raise RuntimeError("DAN 模型未初始化")

        face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_rgb)
        tensor = self._transform(face_image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits, _, _ = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        emotion_idx = int(np.argmax(probs))
        emotion = self.EMOTION_CLASSES[emotion_idx]
        confidence = float(probs[emotion_idx])
        all_probabilities = {
            self.EMOTION_CLASSES[idx]: float(probs[idx])
            for idx in range(min(len(self.EMOTION_CLASSES), len(probs)))
        }

        bounding_box = [
            float(y1 / height),
            float(x1 / width),
            float(y2 / height),
            float(x2 / width),
        ]

        return {
            "emotion": emotion,
            "emotion_idx": emotion_idx,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "bounding_box": bounding_box,
            "detection_score": detection_score,
        }

    def _detect_primary_face(self, image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        if self._face_detector is None:
            return None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        try:
            faces = self._face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48),
            )
        except cv2.error:
            return None

        if faces is None or len(faces) == 0:
            return None

        height, width = gray.shape[:2]
        best_face = None
        best_score = -1.0

        for face in faces:
            x, y, w, h = [int(v) for v in face]
            area = max(0, w) * max(0, h)
            if area <= 0:
                continue

            area_score = float(area) / float(max(1, width * height))
            cx = (x + w / 2.0) / float(max(1, width))
            cy = (y + h / 2.0) / float(max(1, height))
            dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
            center_score = math.exp(-2.0 * dist)

            score = 0.7 * area_score + 0.3 * center_score
            if score > best_score:
                best_score = score
                best_face = (x, y, w, h)

        return best_face


_BACKEND: DanBackend | None = None
_BACKEND_LOCK = threading.Lock()


def get_dan_backend(config: dict | None = None) -> DanBackend:
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is not None:
            return _BACKEND

        package_root = None
        model_path = None
        train_config_path = None
        device = None
        num_head = None
        input_size = None
        face_cascade_path = None

        if isinstance(config, dict):
            package_root = config.get("DAN_PACKAGE_ROOT")
            model_path = config.get("DAN_MODEL_PATH")
            train_config_path = config.get("DAN_TRAIN_CONFIG_PATH")
            device = config.get("DAN_DEVICE")
            num_head = config.get("DAN_NUM_HEAD")
            input_size = config.get("DAN_INPUT_SIZE")
            face_cascade_path = config.get("DAN_FACE_CASCADE_PATH")

        if not package_root:
            package_root = os.getenv("SG_DAN_PACKAGE_ROOT", DEFAULT_DAN_PACKAGE_ROOT)
        package_root = Path(str(package_root))

        if not model_path:
            model_path = os.getenv("SG_DAN_MODEL_PATH", str(package_root / "model" / "best.pth"))

        if not train_config_path:
            train_config_path = os.getenv("SG_DAN_TRAIN_CONFIG_PATH", str(package_root / "model" / "train_config.json"))

        if not device:
            device = os.getenv("SG_DAN_DEVICE", "auto")

        parsed_num_head: int | None = None
        if num_head is not None:
            try:
                candidate = int(num_head)
                if candidate > 0:
                    parsed_num_head = candidate
            except (TypeError, ValueError):
                parsed_num_head = None

        parsed_input_size = 224
        if input_size is not None:
            try:
                candidate_input = int(input_size)
                if candidate_input > 0:
                    parsed_input_size = candidate_input
            except (TypeError, ValueError):
                parsed_input_size = 224

        _BACKEND = DanBackend(
            package_root=package_root,
            model_path=Path(str(model_path)),
            train_config_path=Path(str(train_config_path)),
            device=str(device),
            num_head=parsed_num_head,
            input_size=parsed_input_size,
            face_cascade_path=Path(str(face_cascade_path)) if face_cascade_path else None,
        )
        return _BACKEND
