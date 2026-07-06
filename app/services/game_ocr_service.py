from __future__ import annotations

import bisect
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import cv2


class OcrSampleTimeoutError(TimeoutError):
    def __init__(self, sample_index: int, timestamp_sec: float, elapsed_sec: float, timeout_sec: float) -> None:
        self.sample_index = int(sample_index)
        self.timestamp_sec = float(timestamp_sec)
        self.elapsed_sec = float(elapsed_sec)
        self.timeout_sec = float(timeout_sec)
        super().__init__(
            "游戏画面 OCR 样本超时: "
            f"sample={self.sample_index}, t={self.timestamp_sec:.3f}s, "
            f"elapsed={self.elapsed_sec:.3f}s, timeout={self.timeout_sec:.3f}s"
        )


class GameOcrService:
    def __init__(self, config: dict) -> None:
        self.config = config
        self._toolkit_root = Path(config.get("OCR_TOOLKIT_ROOT", ""))
        self._repo_root = Path(config.get("OCR_REPO_ROOT", ""))
        self._default_interval_sec = self._normalize_interval(config.get("GAME_OCR_INTERVAL_SEC_DEFAULT"), default=2.0)
        self._lang = str(config.get("GAME_OCR_LANG", "ch") or "ch")
        self._ocr_version = str(config.get("GAME_OCR_VERSION", "PP-OCRv5") or "PP-OCRv5")
        self._device = str(config.get("GAME_OCR_DEVICE", "cpu") or "cpu").strip()
        self._min_score = self._normalize_min_score(config.get("GAME_OCR_MIN_SCORE"), default=0.0)
        self._det_model_dir = self._normalize_path(config.get("GAME_OCR_DET_MODEL_DIR"))
        self._rec_model_dir = self._normalize_path(config.get("GAME_OCR_REC_MODEL_DIR"))
        self._max_image_side = self._normalize_max_image_side(config.get("GAME_OCR_MAX_IMAGE_SIDE"), default=1280)
        self._sample_timeout_sec = self._normalize_timeout(config.get("GAME_OCR_SAMPLE_TIMEOUT_SEC"), default=600.0)
        self._heartbeat_sec = self._normalize_timeout(config.get("GAME_OCR_HEARTBEAT_SEC"), default=15.0)
        self._max_consecutive_timeouts = self._normalize_max_consecutive_timeouts(
            config.get("GAME_OCR_MAX_CONSECUTIVE_TIMEOUTS"),
            default=0,
        )
        self._disable_model_source_check = self._normalize_bool(
            config.get("GAME_OCR_DISABLE_MODEL_SOURCE_CHECK"),
            default=True,
        )
        if self._disable_model_source_check:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        self._lock = threading.Lock()
        self._ocr = None

    def analyze_video(
        self,
        video_path: Path,
        output_dir: Path,
        interval_sec: float | None = None,
        reference_timeline: list[dict] | None = None,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        video_path = Path(video_path)
        if not video_path.exists() or not video_path.is_file():
            raise FileNotFoundError(f"游戏画面视频不存在: {video_path}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_dir = output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        interval_value = self._normalize_interval(interval_sec, default=self._default_interval_sec)
        self._get_ocr_engine()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开游戏画面视频: {video_path}")

        try:
            meta = self._read_video_meta(cap)
            duration = meta.get("duration_sec")
            fps = meta.get("fps")

            if duration is not None:
                sample_times = self._build_sample_times(duration, interval_value)
                samples = self._sample_video_by_timestamps(
                    cap=cap,
                    frame_dir=frame_dir,
                    sample_times=sample_times,
                    fps=fps,
                    progress_callback=progress_callback,
                )
            else:
                samples = self._sample_video_by_frame_scan(
                    cap=cap,
                    frame_dir=frame_dir,
                    interval_sec=interval_value,
                    fps=fps,
                    progress_callback=progress_callback,
                )
                if samples:
                    duration = max(0.0, float(samples[-1].get("timestamp_sec", 0.0)))

        finally:
            cap.release()

        reference_stats = self._attach_reference_links(samples, reference_timeline or [])
        text_analysis = self._build_text_analysis(samples)

        samples_total = len(samples)
        samples_with_text = sum(1 for item in samples if int(item.get("line_count", 0)) > 0)
        line_total = sum(int(item.get("line_count", 0)) for item in samples)
        timeout_total = sum(1 for item in samples if str(item.get("ocr_status") or "") == "timeout")
        fps_value = self._safe_float(meta.get("fps"))
        duration_value = self._safe_float(duration)
        ocr_elapsed_values = []
        for item in samples:
            elapsed_value = self._safe_float(item.get("ocr_elapsed_sec"))
            if elapsed_value is None:
                continue
            ocr_elapsed_values.append(float(elapsed_value))

        summary = {
            "video_path": str(video_path),
            "interval_sec": float(interval_value),
            "fps": float(fps_value) if fps_value is not None else None,
            "duration_sec": float(duration_value) if duration_value is not None else None,
            "samples_total": int(samples_total),
            "samples_with_text": int(samples_with_text),
            "samples_timeout": int(timeout_total),
            "line_total": int(line_total),
            "avg_line_per_sample": float(line_total / samples_total) if samples_total > 0 else 0.0,
            "text_coverage_ratio": float(samples_with_text / samples_total) if samples_total > 0 else 0.0,
            "ocr_avg_latency_sec": float(sum(ocr_elapsed_values) / len(ocr_elapsed_values)) if ocr_elapsed_values else None,
            "ocr_p95_latency_sec": self._compute_percentile(ocr_elapsed_values, 0.95),
            "reference_linked_count": int(reference_stats.get("linked_count", 0)),
            "reference_max_delta_sec": reference_stats.get("max_delta_sec"),
            "top_texts": text_analysis,
            "ocr_config": {
                "lang": self._lang,
                "ocr_version": self._ocr_version,
                "device": self._device,
                "min_score": float(self._min_score),
                "max_image_side": int(self._max_image_side),
                "sample_timeout_sec": float(self._sample_timeout_sec),
                "heartbeat_sec": float(self._heartbeat_sec),
                "max_consecutive_timeouts": int(self._max_consecutive_timeouts),
                "toolkit_root": str(self._toolkit_root),
                "repo_root": str(self._repo_root),
            },
        }

        return {
            "summary": summary,
            "timeline": samples,
        }

    def _get_ocr_engine(self):
        with self._lock:
            if self._ocr is not None:
                return self._ocr

            PaddleOCR = self._resolve_paddleocr_class()

            kwargs = {
                "lang": self._lang,
                "ocr_version": self._ocr_version,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            }
            if self._device:
                kwargs["device"] = self._device
            if self._det_model_dir is not None and self._det_model_dir.exists():
                kwargs["text_detection_model_dir"] = str(self._det_model_dir)
            if self._rec_model_dir is not None and self._rec_model_dir.exists():
                kwargs["text_recognition_model_dir"] = str(self._rec_model_dir)

            try:
                self._ocr = PaddleOCR(**kwargs)
            except TypeError as exc:
                if "device" not in str(exc):
                    raise

                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("device", None)
                device_text = self._device.lower()
                if device_text.startswith("gpu"):
                    fallback_kwargs["use_gpu"] = True
                elif device_text == "cpu":
                    fallback_kwargs["use_gpu"] = False
                self._ocr = PaddleOCR(**fallback_kwargs)
            return self._ocr

    def _resolve_paddleocr_class(self):
        try:
            from paddleocr import PaddleOCR  # type: ignore

            return PaddleOCR
        except ModuleNotFoundError:
            repo_root = self._repo_root
            if not repo_root.exists() and self._toolkit_root.exists():
                repo_root = self._toolkit_root.parent

            paddle_pkg = repo_root / "paddleocr"
            if repo_root.exists() and paddle_pkg.exists() and str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))

            from paddleocr import PaddleOCR  # type: ignore

            return PaddleOCR

    @staticmethod
    def _read_video_meta(cap: cv2.VideoCapture) -> dict:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if (
            (not GameOcrService._is_finite_number(fps))
            or float(fps) <= 1e-6
            or float(fps) > 240.0
        ):
            fps = None
        else:
            fps = float(fps)

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not GameOcrService._is_finite_number(frame_count) or float(frame_count) <= 0:
            frame_count = None
        else:
            frame_count = float(frame_count)

        duration = None
        if fps is not None and frame_count is not None and fps > 1e-6:
            duration = frame_count / fps

        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration,
        }

    @staticmethod
    def _build_sample_times(duration_sec: float, interval_sec: float) -> list[float]:
        duration = max(0.0, float(duration_sec))
        interval = max(0.1, float(interval_sec))
        if duration <= 1e-9:
            return [0.0]

        times: list[float] = []
        cursor = 0.0
        guard = 0
        while cursor <= duration + 1e-9:
            times.append(float(round(cursor, 3)))
            cursor += interval
            guard += 1
            if guard > 200000:
                break

        if not times:
            times = [0.0]

        if duration - times[-1] > max(0.25 * interval, 0.35):
            times.append(float(round(duration, 3)))

        deduped: list[float] = []
        for value in times:
            if not deduped or abs(value - deduped[-1]) > 1e-6:
                deduped.append(value)
        return deduped

    def _sample_video_by_timestamps(
        self,
        cap: cv2.VideoCapture,
        frame_dir: Path,
        sample_times: list[float],
        fps: float | None,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        samples: list[dict] = []
        total_samples = len(sample_times)
        consecutive_timeouts = 0

        for sample_index, timestamp in enumerate(sample_times):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            actual_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            if self._is_finite_number(actual_ts) and float(actual_ts) >= 0.0:
                sample_ts = float(actual_ts) / 1000.0
            else:
                sample_ts = float(timestamp)

            frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if self._is_finite_number(frame_pos):
                frame_index_override = max(0, int(round(float(frame_pos) - 1.0)))
            else:
                frame_index_override = None

            ocr_engine = self._get_ocr_engine()
            sample = self._run_sample_ocr(
                ocr_engine=ocr_engine,
                frame=frame,
                frame_dir=frame_dir,
                sample_index=sample_index,
                timestamp_sec=float(sample_ts),
                fps=fps,
                frame_index_override=frame_index_override,
                progress_callback=progress_callback,
                total_samples=total_samples,
            )

            if str(sample.get("ocr_status") or "") == "timeout":
                consecutive_timeouts += 1
                if self._max_consecutive_timeouts > 0 and consecutive_timeouts >= self._max_consecutive_timeouts:
                    raise RuntimeError(f"游戏画面 OCR 连续超时 {consecutive_timeouts} 次，已中止")
            else:
                consecutive_timeouts = 0

            samples.append(sample)
            self._emit_progress(
                progress_callback,
                done=sample_index + 1,
                total=total_samples,
                sample=sample,
            )

        return samples

    def _sample_video_by_frame_scan(
        self,
        cap: cv2.VideoCapture,
        frame_dir: Path,
        interval_sec: float,
        fps: float | None,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        samples: list[dict] = []
        interval = max(0.1, float(interval_sec))
        consecutive_timeouts = 0

        frame_index = 0
        sample_index = 0
        next_sample_sec = 0.0
        fps_value = self._safe_float(fps)
        fallback_fps = float(fps_value) if fps_value is not None and fps_value > 1e-6 else None
        if fallback_fps is not None and fallback_fps > 240.0:
            fallback_fps = None

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            if self._is_finite_number(timestamp) and float(timestamp) >= 0.0:
                current_ts = float(timestamp) / 1000.0
            elif fallback_fps is not None:
                current_ts = float(frame_index) / float(fallback_fps)
            else:
                current_ts = float(sample_index) * interval

            should_sample = sample_index == 0 or current_ts + 1e-9 >= next_sample_sec

            if should_sample:
                ocr_engine = self._get_ocr_engine()
                sample = self._run_sample_ocr(
                    ocr_engine=ocr_engine,
                    frame=frame,
                    frame_dir=frame_dir,
                    sample_index=sample_index,
                    timestamp_sec=float(current_ts),
                    fps=fps,
                    frame_index_override=frame_index,
                    progress_callback=progress_callback,
                    total_samples=None,
                )

                if str(sample.get("ocr_status") or "") == "timeout":
                    consecutive_timeouts += 1
                    if self._max_consecutive_timeouts > 0 and consecutive_timeouts >= self._max_consecutive_timeouts:
                        raise RuntimeError(f"游戏画面 OCR 连续超时 {consecutive_timeouts} 次，已中止")
                else:
                    consecutive_timeouts = 0

                samples.append(sample)
                self._emit_progress(
                    progress_callback,
                    done=sample_index + 1,
                    total=None,
                    sample=sample,
                )
                sample_index += 1
                while next_sample_sec <= current_ts + 1e-9:
                    next_sample_sec += interval

            frame_index += 1

        return samples

    def _run_sample_ocr(
        self,
        ocr_engine,
        frame,
        frame_dir: Path,
        sample_index: int,
        timestamp_sec: float,
        fps: float | None,
        frame_index_override: int | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        total_samples: int | None = None,
    ) -> dict:
        timestamp_value = max(0.0, float(timestamp_sec))
        ms = int(round(timestamp_value * 1000.0))
        image_name = f"sample_{sample_index:05d}_{ms:010d}ms.jpg"
        image_path = frame_dir / image_name
        frame_for_ocr = self._prepare_frame_for_ocr(frame)

        if not cv2.imwrite(str(image_path), frame_for_ocr):
            raise RuntimeError(f"写入游戏画面帧失败: {image_path}")

        self._emit_callback(
            progress_callback,
            {
                "event": "sample_start",
                "sample_index": int(sample_index),
                "timestamp_sec": float(timestamp_value),
                "total": int(total_samples) if total_samples is not None else None,
            },
        )

        try:
            result, ocr_elapsed_sec = self._predict_with_guard(
                ocr_engine=ocr_engine,
                image=frame_for_ocr,
                image_path=image_path,
                sample_index=sample_index,
                timestamp_sec=timestamp_value,
                total_samples=total_samples,
                progress_callback=progress_callback,
            )
            ocr_status = "ok"
            ocr_error = None
        except OcrSampleTimeoutError as exc:
            self._invalidate_ocr_engine()

            frame_index = frame_index_override
            if frame_index is None:
                if fps is not None and fps > 1e-6:
                    frame_index = int(round(timestamp_value * fps))
                else:
                    frame_index = sample_index

            return {
                "sample_index": int(sample_index),
                "timestamp_sec": float(timestamp_value),
                "frame_index": int(frame_index),
                "image_path": str(image_path),
                "line_count": 0,
                "char_count": 0,
                "mean_score": None,
                "text": "",
                "texts": [],
                "items": [],
                "ocr_status": "timeout",
                "ocr_error": str(exc),
                "ocr_elapsed_sec": float(round(exc.elapsed_sec, 6)),
            }

        items = self._extract_items(result)
        lines = [str(item.get("text", "")) for item in items if str(item.get("text", "")).strip()]
        joined_text = " ".join(lines).strip()
        char_count = len("".join(lines))
        score_values = [float(item.get("score", 0.0)) for item in items if self._is_finite_number(item.get("score"))]
        mean_score = float(sum(score_values) / len(score_values)) if score_values else None

        frame_index = frame_index_override
        if frame_index is None:
            if fps is not None and fps > 1e-6:
                frame_index = int(round(timestamp_value * fps))
            else:
                frame_index = sample_index

        return {
            "sample_index": int(sample_index),
            "timestamp_sec": float(timestamp_value),
            "frame_index": int(frame_index),
            "image_path": str(image_path),
            "line_count": int(len(lines)),
            "char_count": int(char_count),
            "mean_score": mean_score,
            "text": joined_text,
            "texts": lines,
            "items": items,
            "ocr_status": ocr_status,
            "ocr_error": ocr_error,
            "ocr_elapsed_sec": float(round(ocr_elapsed_sec, 6)),
        }

    def _predict_with_guard(
        self,
        ocr_engine,
        image,
        image_path: Path | None,
        sample_index: int,
        timestamp_sec: float,
        total_samples: int | None,
        progress_callback: Callable[[dict], None] | None,
    ) -> tuple[object, float]:
        timeout_sec = float(self._sample_timeout_sec)
        heartbeat_sec = float(self._heartbeat_sec)

        started = time.monotonic()
        result = self._predict_once(
            ocr_engine=ocr_engine,
            image=image,
            image_path=image_path,
        )
        elapsed = max(0.0, time.monotonic() - started)

        if heartbeat_sec > 0.0 and elapsed >= heartbeat_sec:
            self._emit_callback(
                progress_callback,
                {
                    "event": "sample_wait",
                    "sample_index": int(sample_index),
                    "timestamp_sec": float(timestamp_sec),
                    "total": int(total_samples) if total_samples is not None else None,
                    "wait_elapsed_sec": float(round(elapsed, 6)),
                },
            )

        if timeout_sec > 0.0 and elapsed >= timeout_sec:
            raise OcrSampleTimeoutError(sample_index, timestamp_sec, elapsed, timeout_sec)

        return result, elapsed

    @staticmethod
    def _predict_once(ocr_engine, image, image_path: Path | None = None):
        input_candidates = []
        if image_path is not None:
            input_candidates.append(str(image_path))
        input_candidates.append(image)

        if hasattr(ocr_engine, "predict"):
            last_error: Exception | None = None
            for input_item in input_candidates:
                try:
                    return ocr_engine.predict(input_item)
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error

        if hasattr(ocr_engine, "ocr"):
            last_error = None
            for input_item in input_candidates:
                try:
                    return ocr_engine.ocr(input_item, cls=False)
                except TypeError:
                    try:
                        return ocr_engine.ocr(input_item)
                    except Exception as exc:
                        last_error = exc
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error

        raise RuntimeError("OCR 引擎未提供可用推理接口（predict/ocr）")

    def _invalidate_ocr_engine(self) -> None:
        with self._lock:
            self._ocr = None

    def _prepare_frame_for_ocr(self, frame):
        if frame is None:
            return frame

        max_side = int(self._max_image_side or 0)
        if max_side <= 0:
            return frame

        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < 2:
            return frame

        height = int(shape[0])
        width = int(shape[1])
        longest = max(height, width)
        if longest <= max_side or longest <= 0:
            return frame

        scale = float(max_side) / float(longest)
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _emit_callback(callback: Callable[[dict], None] | None, payload: dict) -> None:
        if callback is None:
            return
        try:
            callback(payload)
        except Exception:
            return

    @staticmethod
    def _emit_progress(
        callback: Callable[[dict], None] | None,
        done: int,
        total: int | None,
        sample: dict,
    ) -> None:
        if callback is None:
            return

        elapsed_value = GameOcrService._safe_float(sample.get("ocr_elapsed_sec"))

        payload = {
            "event": "sample_done",
            "done": int(done),
            "total": int(total) if total is not None else None,
            "sample_index": int(sample.get("sample_index") or 0),
            "timestamp_sec": float(sample.get("timestamp_sec") or 0.0),
            "line_count": int(sample.get("line_count") or 0),
            "ocr_status": str(sample.get("ocr_status") or "ok"),
            "ocr_elapsed_sec": float(elapsed_value) if elapsed_value is not None else None,
        }
        GameOcrService._emit_callback(callback, payload)

    def _extract_items(self, raw_result) -> list[dict]:
        if raw_result is None:
            return []

        pages = self._to_pages(raw_result)
        if not pages:
            return []

        items: list[dict] = []
        for page_idx, page in enumerate(pages):
            for text, score in self._to_pairs(page):
                clean_text = str(text).strip()
                if not clean_text:
                    continue
                score_value = float(score)
                if score_value < self._min_score:
                    continue
                items.append(
                    {
                        "page": int(page_idx),
                        "text": clean_text,
                        "score": score_value,
                    }
                )
        return items

    @staticmethod
    def _to_dict(result_item: object) -> dict:
        if hasattr(result_item, "to_dict"):
            try:
                result_item = result_item.to_dict()  # type: ignore[assignment]
            except Exception:
                pass

        if isinstance(result_item, dict):
            return result_item
        try:
            return dict(result_item)  # type: ignore[arg-type]
        except Exception:
            return {}

    @classmethod
    def _to_pages(cls, raw_result) -> list[object]:
        pages = raw_result[0] if isinstance(raw_result, tuple) else raw_result
        if pages is None:
            return []
        if not isinstance(pages, (list, tuple)):
            return [pages]

        page_list = list(pages)
        if not page_list:
            return []

        if cls._looks_like_legacy_line(page_list[0]):
            return [page_list]

        return page_list

    @staticmethod
    def _looks_like_legacy_line(item: object) -> bool:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return False

        rec_info = item[1]
        if not isinstance(rec_info, (list, tuple)) or len(rec_info) < 2:
            return False

        return True

    @classmethod
    def _to_pairs(cls, result_item: object) -> list[tuple[str, float]]:
        pairs = cls._to_pairs_from_pipeline(result_item)
        if pairs:
            return pairs
        return cls._to_pairs_from_legacy(result_item)

    @classmethod
    def _to_pairs_from_pipeline(cls, result_item: object) -> list[tuple[str, float]]:
        payload = cls._to_dict(result_item)
        payload = payload.get("res", payload)
        if not isinstance(payload, dict):
            return []

        texts_raw = payload.get("rec_texts")
        scores_raw = payload.get("rec_scores")

        if texts_raw is None:
            single_text = payload.get("rec_text")
            if single_text is None:
                return []
            texts = [single_text]
            single_score = payload.get("rec_score")
            scores = [single_score] if single_score is not None else []
        else:
            if isinstance(texts_raw, (str, bytes)):
                texts = [texts_raw]
            elif isinstance(texts_raw, Sequence):
                texts = list(texts_raw)
            else:
                return []

            if isinstance(scores_raw, (str, bytes)):
                scores = []
            elif isinstance(scores_raw, Sequence):
                scores = list(scores_raw)
            else:
                scores = []

        normalized_scores: list[float] = []
        for score in scores:
            try:
                value = float(score)
            except Exception:
                value = 0.0
            if not math.isfinite(value):
                value = 0.0
            normalized_scores.append(value)
        if len(normalized_scores) < len(texts):
            normalized_scores.extend([1.0] * (len(texts) - len(normalized_scores)))

        pairs: list[tuple[str, float]] = []
        for text, score in zip(texts, normalized_scores):
            pairs.append((str(text), float(score)))
        return pairs

    @classmethod
    def _to_pairs_from_legacy(cls, result_item: object) -> list[tuple[str, float]]:
        if not isinstance(result_item, (list, tuple)):
            return []

        pairs: list[tuple[str, float]] = []
        for row in result_item:
            if not cls._looks_like_legacy_line(row):
                continue

            rec_info = row[1]
            text = str(rec_info[0] or "").strip()
            score_value = cls._safe_float(rec_info[1])
            if score_value is None:
                score_value = 0.0

            pairs.append((text, float(score_value)))

        return pairs

    @classmethod
    def _attach_reference_links(cls, samples: list[dict], reference_timeline: list[dict]) -> dict:
        reference = []
        for row in reference_timeline:
            if not isinstance(row, dict):
                continue
            ts = cls._safe_float(row.get("timestamp"))
            if ts is None:
                continue
            idx_value = cls._safe_int(row.get("index"))
            frame_value = cls._safe_int(row.get("frame"))
            reference.append((float(ts), idx_value, frame_value))

        if not reference:
            for sample in samples:
                sample["linked_analysis_index"] = None
                sample["linked_analysis_frame"] = None
                sample["linked_analysis_timestamp"] = None
                sample["linked_time_delta_sec"] = None
            return {
                "linked_count": 0,
                "max_delta_sec": None,
            }

        reference.sort(key=lambda item: item[0])
        ref_times = [item[0] for item in reference]

        linked_count = 0
        max_delta = 0.0
        for sample in samples:
            ts = cls._safe_float(sample.get("timestamp_sec"))
            if ts is None:
                sample["linked_analysis_index"] = None
                sample["linked_analysis_frame"] = None
                sample["linked_analysis_timestamp"] = None
                sample["linked_time_delta_sec"] = None
                continue

            insert_pos = bisect.bisect_left(ref_times, ts)
            candidates = []
            if 0 <= insert_pos < len(reference):
                candidates.append(reference[insert_pos])
            if insert_pos - 1 >= 0:
                candidates.append(reference[insert_pos - 1])
            if not candidates:
                sample["linked_analysis_index"] = None
                sample["linked_analysis_frame"] = None
                sample["linked_analysis_timestamp"] = None
                sample["linked_time_delta_sec"] = None
                continue

            nearest = min(candidates, key=lambda item: abs(item[0] - ts))
            delta = abs(float(nearest[0]) - float(ts))
            sample["linked_analysis_index"] = nearest[1]
            sample["linked_analysis_frame"] = nearest[2]
            sample["linked_analysis_timestamp"] = float(nearest[0])
            sample["linked_time_delta_sec"] = float(delta)

            linked_count += 1
            max_delta = max(max_delta, delta)

        return {
            "linked_count": int(linked_count),
            "max_delta_sec": float(max_delta) if linked_count > 0 else None,
        }

    @classmethod
    def _build_text_analysis(cls, samples: list[dict], topk: int = 20) -> list[dict]:
        line_counter: Counter[str] = Counter()
        coverage_counter: Counter[str] = Counter()
        display_text: dict[str, str] = {}

        for sample in samples:
            texts = sample.get("texts") or []
            if not isinstance(texts, list):
                continue

            seen: set[str] = set()
            for raw in texts:
                text = str(raw or "").strip()
                if not text:
                    continue
                key = cls._normalize_text_key(text)
                if len(key) < 2:
                    continue
                if key not in display_text:
                    display_text[key] = text
                line_counter[key] += 1
                seen.add(key)

            for key in seen:
                coverage_counter[key] += 1

        total_samples = len(samples)
        top_items: list[dict] = []
        for key, count in line_counter.most_common(max(1, int(topk))):
            top_items.append(
                {
                    "text": display_text.get(key, key),
                    "hit_count": int(count),
                    "sample_coverage": float(coverage_counter.get(key, 0) / total_samples) if total_samples > 0 else 0.0,
                }
            )

        return top_items

    @staticmethod
    def _compute_percentile(values: list[float], ratio: float) -> float | None:
        cleaned = []
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                continue
            cleaned.append(number)

        if not cleaned:
            return None

        ratio_value = max(0.0, min(1.0, float(ratio)))
        cleaned.sort()
        if len(cleaned) == 1:
            return float(cleaned[0])

        pos = ratio_value * float(len(cleaned) - 1)
        lower = int(math.floor(pos))
        upper = int(math.ceil(pos))
        if lower == upper:
            return float(cleaned[lower])

        lower_value = cleaned[lower]
        upper_value = cleaned[upper]
        weight = pos - float(lower)
        return float(lower_value * (1.0 - weight) + upper_value * weight)

    @staticmethod
    def _normalize_text_key(text: str) -> str:
        compact = re.sub(r"\s+", "", str(text or "")).strip()
        return compact.lower()

    @staticmethod
    def _normalize_interval(value, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float(default)
        if not math.isfinite(number) or number <= 0:
            number = float(default)
        return max(0.1, float(number))

    @staticmethod
    def _normalize_timeout(value, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float(default)
        if not math.isfinite(number):
            number = float(default)
        if number <= 0:
            return 0.0
        return max(0.0, float(number))

    @staticmethod
    def _normalize_max_consecutive_timeouts(value, default: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = int(default)
        if number <= 0:
            return 0
        return int(number)

    @staticmethod
    def _normalize_min_score(value, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float(default)
        if not math.isfinite(number):
            number = float(default)
        return max(0.0, min(1.0, float(number)))

    @staticmethod
    def _normalize_path(value) -> Path | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return Path(text)

    @staticmethod
    def _normalize_bool(value, default: bool) -> bool:
        if isinstance(value, bool):
            return bool(value)
        if value is None:
            return bool(default)

        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)

    @staticmethod
    def _normalize_max_image_side(value, default: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = int(default)
        if number <= 0:
            return 0
        return int(number)

    @staticmethod
    def _is_finite_number(value) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _safe_float(value) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return float(number)

    @staticmethod
    def _safe_int(value) -> int | None:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return int(number)
