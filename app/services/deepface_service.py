from __future__ import annotations

from .affectnet_backend import AffectNetBackend, get_affectnet_backend
from .dan_backend import DanBackend, get_dan_backend


class DeepFaceService:
    def __init__(self, config: dict | None = None) -> None:
        backend_name = "dan"
        if isinstance(config, dict):
            backend_name = str(config.get("DEEPFACE_BACKEND", "dan")).strip().lower()

        if backend_name == "affectnet":
            self._backend = get_affectnet_backend(config)
            self._map_probabilities = AffectNetBackend.map_probabilities
            self._map_emotion = AffectNetBackend.map_emotion
        else:
            self._backend = get_dan_backend(config)
            self._map_probabilities = DanBackend.map_probabilities
            self._map_emotion = DanBackend.map_emotion

    def analyze(self, image_path: str) -> dict:
        face = self._backend.analyze(image_path)
        scores = self._map_probabilities(face.get("all_probabilities") or {})
        dominant = self._map_emotion(face.get("emotion"))
        if not dominant:
            dominant = face.get("emotion") or "unknown"
        return {
            "dominant_emotion": dominant,
            "scores": scores,
        }
