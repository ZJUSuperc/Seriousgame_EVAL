from __future__ import annotations

import math

from .affectnet_backend import AffectNetBackend, get_affectnet_backend
from .dan_backend import DanBackend, get_dan_backend


class EmoNetService:
    DAN_VALENCE_AROUSAL_PRIORS = {
        "neutral": (0.0, 0.0),
        "happy": (0.75, 0.35),
        "sad": (-0.65, -0.2),
        "surprise": (0.15, 0.6),
        "fear": (-0.65, 0.75),
        "disgust": (-0.6, 0.3),
        "angry": (-0.7, 0.65),
        "contempt": (-0.3, 0.1),
    }

    def __init__(self, config: dict | None = None) -> None:
        self._backend = get_affectnet_backend(config)
        self._fallback_backend = get_dan_backend(config)
        self._prefer_fallback = False

    def analyze(self, image_path: str) -> dict:
        primary_error = None

        if not self._prefer_fallback:
            try:
                face = self._backend.analyze(image_path)
                return self._build_affectnet_result(face)
            except Exception as exc:
                primary_error = exc
                self._prefer_fallback = True

        try:
            fallback_face = self._fallback_backend.analyze(image_path)
            return self._build_dan_fallback_result(fallback_face, primary_error)
        except Exception:
            if primary_error is not None:
                raise primary_error
            raise

    def _build_affectnet_result(self, face: dict) -> dict:
        valence = self._clip_value(face.get("valence"), -1.0, 1.0)
        arousal = self._clip_value(face.get("arousal"), -1.0, 1.0)
        scores = AffectNetBackend.map_probabilities(face.get("all_probabilities") or {})
        emotion = AffectNetBackend.map_emotion(face.get("emotion"))
        if not emotion:
            emotion = face.get("emotion") or "unknown"

        emotion_idx = face.get("emotion_idx")
        if isinstance(emotion_idx, bool):
            emotion_idx = None
        if emotion_idx is not None:
            try:
                emotion_idx = int(emotion_idx)
            except (TypeError, ValueError):
                emotion_idx = None

        return {
            "backend": "affectnet",
            "emotion": emotion,
            "emotion_idx": emotion_idx,
            "valence": valence,
            "arousal": arousal,
            "probabilities": scores,
        }

    def _build_dan_fallback_result(self, face: dict, primary_error: Exception | None) -> dict:
        scores = DanBackend.map_probabilities(face.get("all_probabilities") or {})
        emotion = DanBackend.map_emotion(face.get("emotion"))
        if not emotion:
            emotion = str(face.get("emotion") or "unknown")

        valence, arousal = self._estimate_valence_arousal(scores, emotion)

        result = {
            "backend": "dan_fallback",
            "emotion": emotion,
            "emotion_idx": None,
            "valence": valence,
            "arousal": arousal,
            "probabilities": scores,
        }
        if primary_error is not None:
            result["fallback_reason"] = str(primary_error)
        return result

    def _estimate_valence_arousal(self, probabilities: dict[str, float], dominant_emotion: str) -> tuple[float | None, float | None]:
        normalized = self._normalize_probabilities(probabilities)
        if normalized:
            valence = 0.0
            arousal = 0.0
            total = 0.0
            for emotion, weight in normalized.items():
                prior = self.DAN_VALENCE_AROUSAL_PRIORS.get(emotion)
                if prior is None:
                    continue
                valence += prior[0] * weight
                arousal += prior[1] * weight
                total += weight

            if total > 1e-9:
                return self._clip_value(valence / total, -1.0, 1.0), self._clip_value(arousal / total, -1.0, 1.0)

        prior = self.DAN_VALENCE_AROUSAL_PRIORS.get(str(dominant_emotion or "").strip().lower())
        if prior is None:
            return None, None
        return self._clip_value(prior[0], -1.0, 1.0), self._clip_value(prior[1], -1.0, 1.0)

    @staticmethod
    def _normalize_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
        if not isinstance(probabilities, dict):
            return {}

        output: dict[str, float] = {}
        total = 0.0
        for key, value in probabilities.items():
            emotion = str(key or "").strip().lower()
            if not emotion:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number) or number <= 0:
                continue
            output[emotion] = output.get(emotion, 0.0) + number
            total += number

        if total <= 1e-9:
            return {}
        return {
            key: value / total
            for key, value in output.items()
        }

    @staticmethod
    def _clip_value(value, min_value: float, max_value: float) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return max(min_value, min(max_value, number))
