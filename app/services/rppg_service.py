import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import butter, welch, find_peaks
from scipy.sparse import spdiags


class RppgService:
    def __init__(self, config: dict):
        self.config = config
        self._models: dict[str, torch.nn.Module] = {}
        self._root = Path(config.get("RPPG_ROOT", ""))
        self._model_path = Path(config.get("RPPG_MODEL_PATH", ""))
        self._cascade_path = Path(config.get("RPPG_CASCADE_PATH", ""))

    def analyze(
        self,
        video_path: Path,
        start_sec: float | None,
        duration_sec: float | None,
        chunk_len: int = 160,
        device: str = "cpu",
    ) -> dict:
        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")

        resolved_device = self._resolve_device(device)
        model = self._get_model(resolved_device)

        frames, fps, used_start, used_end = self._load_frames(
            video_path,
            start_sec=start_sec,
            duration_sec=duration_sec,
            target_hw=(128, 128),
            cascade_path=self._cascade_path,
        )
        if len(frames) == 0:
            raise RuntimeError("rPPG 未能读取到有效帧")

        chunk_len = max(8, int(chunk_len))
        frames_std = self._standardize(frames)
        chunks, valid_frames = self._chunk_frames(frames_std, chunk_len)

        rppg_raw = self._run_inference(chunks, model, resolved_device)
        rppg_raw = rppg_raw[:valid_frames]

        detrended, filtered = self._postprocess(rppg_raw, fps=fps)
        hr_bpm, freqs, psd = self._estimate_hr(filtered, fs=fps)

        resp_summary_raw, resp_series_raw = self._analyze_respiration(rppg_raw, fps=fps, start_sec=used_start)
        resp_summary_det, resp_series_det = self._analyze_respiration(detrended, fps=fps, start_sec=used_start)
        use_detrended = resp_summary_det.get("resp_quality", 0.0) > resp_summary_raw.get("resp_quality", 0.0)
        resp_summary = resp_summary_det if use_detrended else resp_summary_raw
        resp_series = resp_series_det if use_detrended else resp_series_raw
        resp_summary = dict(resp_summary)
        resp_summary["resp_source"] = "detrended" if use_detrended else "raw"

        time = np.arange(len(filtered), dtype=float) / fps + used_start

        series = {
            "time_sec": time.tolist(),
            "raw": rppg_raw.tolist(),
            "detrended": detrended.tolist(),
            "filtered": filtered.tolist(),
        }
        series = self._downsample_series(series, max_points=1500)

        psd_series = {
            "freq_bpm": (freqs * 60.0).tolist(),
            "power": psd.tolist(),
        }
        psd_series = self._downsample_psd(psd_series, max_points=900)

        return {
            "summary": {
                "hr_bpm": float(hr_bpm),
                "fps": float(fps),
                "frames_used": int(valid_frames),
                "start_sec": float(used_start),
                "end_sec": float(used_end),
                "duration_sec": float(max(0.0, used_end - used_start)),
                "chunk_len": int(chunk_len),
                "downsample_step": int(series.get("downsample_step", 1)),
            },
            "series": series,
            "psd": psd_series,
            "resp": {
                "summary": resp_summary,
                "series": resp_series,
            },
        }

    def _get_model(self, device: torch.device) -> torch.nn.Module:
        key = str(device)
        if key in self._models:
            return self._models[key]

        root = self._root
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from neural_methods.model.RhythmMamba import RhythmMamba

        if not self._model_path.exists():
            raise FileNotFoundError(f"rPPG 模型不存在: {self._model_path}")

        model = RhythmMamba()
        state_dict = torch.load(self._model_path, map_location=device)
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned)
        model.to(device)
        model.eval()

        self._models[key] = model
        return model

    @staticmethod
    def _resolve_device(value: str) -> torch.device:
        device_text = str(value or "cpu").lower()
        if device_text == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_text.startswith("cuda"):
            return torch.device(device_text)
        return torch.device("cpu")

    @staticmethod
    def _standardize(data: np.ndarray) -> np.ndarray:
        result = data - data.mean()
        std = data.std()
        if std > 1e-8:
            result = result / std
        result[np.isnan(result)] = 0
        return result.astype(np.float32)

    @staticmethod
    def _chunk_frames(frames: np.ndarray, chunk_length: int) -> tuple[np.ndarray, int]:
        n_chunks = len(frames) // chunk_length
        if n_chunks == 0:
            pad_len = chunk_length - len(frames)
            pad = np.zeros((pad_len, *frames.shape[1:]), dtype=np.float32)
            frames_padded = np.concatenate([frames, pad], axis=0)
            return frames_padded[np.newaxis], len(frames)

        valid_frames = n_chunks * chunk_length
        chunks = frames[:valid_frames].reshape(n_chunks, chunk_length, *frames.shape[1:])
        return chunks, valid_frames

    @staticmethod
    def _run_inference(chunks: np.ndarray, model: torch.nn.Module, device: torch.device) -> np.ndarray:
        all_preds = []
        batch_size = 4
        with torch.no_grad():
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                batch = np.transpose(batch, (0, 1, 4, 2, 3))
                tensor = torch.from_numpy(batch).float().to(device)

                pred = model(tensor)
                mean = pred.mean(dim=-1, keepdim=True)
                std = pred.std(dim=-1, keepdim=True) + 1e-7
                pred = (pred - mean) / std
                all_preds.append(pred.cpu().numpy())

        return np.concatenate(all_preds, axis=0).reshape(-1)

    @staticmethod
    def _detrend(signal: np.ndarray, lambda_value: float = 100) -> np.ndarray:
        n = len(signal)
        if n < 4:
            return signal
        h = np.identity(n)
        ones = np.ones(n)
        minus_twos = -2 * np.ones(n)
        diags_data = np.array([ones, minus_twos, ones])
        diags_index = np.array([0, 1, 2])
        d_mat = spdiags(diags_data, diags_index, n - 2, n).toarray()
        a_mat = h + (lambda_value**2) * d_mat.T @ d_mat
        return np.dot(h - np.linalg.solve(a_mat.T, h.T).T, signal)

    def _postprocess(self, signal_raw: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
        detrended = self._detrend(signal_raw, lambda_value=100)
        b, a = butter(1, [0.75 / (fps / 2), 2.5 / (fps / 2)], btype="bandpass")
        filtered = self._safe_filtfilt(b, a, detrended)
        return detrended, filtered

    @staticmethod
    def _safe_filtfilt(b, a, signal: np.ndarray) -> np.ndarray:
        if len(signal) < 4:
            return signal
        import scipy.signal

        return scipy.signal.filtfilt(b, a, np.double(signal))

    @staticmethod
    def _estimate_hr(filtered_signal: np.ndarray, fs: float) -> tuple[float, np.ndarray, np.ndarray]:
        if len(filtered_signal) < 4:
            return 0.0, np.array([]), np.array([])
        nfft = max(int(1e5 / fs), 512)
        nperseg = min(len(filtered_signal) - 1, 256)
        freqs, psd = welch(filtered_signal, fs=fs, nfft=nfft, nperseg=nperseg)
        valid = (freqs > 45 / 60) & (freqs < 150 / 60)
        if not np.any(valid):
            return 0.0, freqs, psd
        hr = freqs[valid][np.argmax(psd[valid])] * 60
        return float(hr), freqs, psd

    def _analyze_respiration(
        self,
        signal_raw: np.ndarray,
        fps: float,
        start_sec: float,
    ) -> tuple[dict, dict]:
        if len(signal_raw) < 10 or fps <= 1e-3:
            return self._empty_respiration_summary(), {"time_sec": [], "signal": []}

        nyquist = fps / 2.0
        low = 0.08 / nyquist
        high = 0.5 / nyquist
        if high >= 1.0 or low <= 0.0:
            return self._empty_respiration_summary(), {"time_sec": [], "signal": []}

        b, a = butter(2, [low, high], btype="bandpass")
        resp_signal = self._safe_filtfilt(b, a, signal_raw)
        resp_signal = np.nan_to_num(resp_signal).astype(float)

        resp_bpm, resp_quality = self._estimate_resp_rate(resp_signal, fps)
        cycles = self._detect_breath_cycles(resp_signal, fps)

        breath_count = len(cycles)
        amplitudes = [cycle["amplitude"] for cycle in cycles]
        durations = [cycle["duration_sec"] for cycle in cycles]

        depth_threshold = 0.0
        deep_breath_count = 0
        deep_breath_times: list[float] = []
        if amplitudes:
            depth_median = float(np.median(amplitudes))
            depth_std = float(np.std(amplitudes))
            depth_threshold = max(0.0, depth_median + 0.5 * depth_std)
            for cycle in cycles:
                if cycle["amplitude"] >= depth_threshold:
                    deep_breath_count += 1
                    deep_breath_times.append(float(cycle["end_time"]))

        cycle_mean = float(np.mean(durations)) if durations else None
        cycle_std = float(np.std(durations)) if len(durations) > 1 else None
        depth_mean = float(np.mean(amplitudes)) if amplitudes else None
        depth_std = float(np.std(amplitudes)) if len(amplitudes) > 1 else None

        compliance = "insufficient"
        if resp_quality < 0.35:
            compliance = "low_quality"
        elif deep_breath_count >= 3:
            compliance = "complete"
        elif deep_breath_count >= 1:
            compliance = "partial"

        time = np.arange(len(resp_signal), dtype=float) / fps + start_sec
        series = {
            "time_sec": time.tolist(),
            "signal": resp_signal.tolist(),
        }
        series = self._downsample_series(series, max_points=1500)

        summary = {
            "resp_bpm": float(resp_bpm),
            "resp_quality": float(resp_quality),
            "breath_count": int(breath_count),
            "deep_breath_count": int(deep_breath_count),
            "deep_breath_target": 3,
            "cycle_mean_sec": cycle_mean,
            "cycle_std_sec": cycle_std,
            "depth_mean": depth_mean,
            "depth_std": depth_std,
            "depth_threshold": float(depth_threshold),
            "cycle_min_sec": 3.0,
            "cycle_max_sec": 10.0,
            "compliance": compliance,
            "deep_breath_times": deep_breath_times,
        }
        return summary, series

    @staticmethod
    def _empty_respiration_summary() -> dict:
        return {
            "resp_bpm": 0.0,
            "resp_quality": 0.0,
            "breath_count": 0,
            "deep_breath_count": 0,
            "deep_breath_target": 3,
            "cycle_mean_sec": None,
            "cycle_std_sec": None,
            "depth_mean": None,
            "depth_std": None,
            "depth_threshold": 0.0,
            "cycle_min_sec": 3.0,
            "cycle_max_sec": 10.0,
            "compliance": "no_data",
            "deep_breath_times": [],
        }

    @staticmethod
    def _estimate_resp_rate(signal: np.ndarray, fs: float) -> tuple[float, float]:
        if len(signal) < 4:
            return 0.0, 0.0
        nfft = max(int(6e4 / fs), 512)
        nperseg = min(len(signal) - 1, 512)
        freqs, psd = welch(signal, fs=fs, nfft=nfft, nperseg=nperseg)
        valid = (freqs >= 0.08) & (freqs <= 0.5)
        if not np.any(valid):
            return 0.0, 0.0
        valid_psd = psd[valid]
        peak_idx = int(np.argmax(valid_psd))
        peak_freq = freqs[valid][peak_idx]
        peak_power = float(valid_psd[peak_idx])
        total_power = float(np.sum(valid_psd)) if valid_psd.size > 0 else 0.0
        quality = peak_power / total_power if total_power > 1e-9 else 0.0
        return float(peak_freq * 60.0), float(min(1.0, max(0.0, quality)))

    @staticmethod
    def _detect_breath_cycles(signal: np.ndarray, fps: float) -> list[dict]:
        if len(signal) < 4 or fps <= 1e-3:
            return []
        min_distance = max(1, int(round(1.5 * fps)))
        peaks, _ = find_peaks(signal, distance=min_distance)
        if len(peaks) < 2:
            return []
        troughs, _ = find_peaks(-signal, distance=min_distance)

        cycles: list[dict] = []
        for idx in range(1, len(peaks)):
            start = int(peaks[idx - 1])
            end = int(peaks[idx])
            duration = float(end - start) / fps
            if duration < 3.0 or duration > 10.0:
                continue

            trough_candidates = troughs[(troughs > start) & (troughs < end)]
            if len(trough_candidates) > 0:
                trough_idx = int(trough_candidates[np.argmin(signal[trough_candidates])])
            else:
                segment = signal[start : end + 1]
                if segment.size == 0:
                    continue
                trough_idx = int(start + np.argmin(segment))

            amplitude = float(abs(signal[end] - signal[trough_idx]))
            cycles.append(
                {
                    "start_idx": start,
                    "end_idx": end,
                    "duration_sec": duration,
                    "amplitude": amplitude,
                    "end_time": float(end) / fps,
                }
            )
        return cycles

    def _load_frames(
        self,
        video_path: Path,
        start_sec: float | None,
        duration_sec: float | None,
        target_hw: tuple[int, int],
        cascade_path: Path,
    ) -> tuple[np.ndarray, float, float, float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 1e-3:
            fps = 30.0

        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(total_frames) if total_frames and total_frames > 0 else None

        start_sec = max(0.0, float(start_sec)) if start_sec is not None else 0.0
        start_frame = int(round(start_sec * fps))
        if total_frames is not None:
            start_frame = min(start_frame, max(total_frames - 1, 0))

        end_frame = None
        if duration_sec is not None and duration_sec > 0:
            end_frame = start_frame + int(round(duration_sec * fps))
            if total_frames is not None:
                end_frame = min(end_frame, total_frames)

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        raw_frames: list[np.ndarray] = []
        index = start_frame
        while True:
            if end_frame is not None and index >= end_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
        cap.release()

        if not raw_frames:
            return np.zeros((0, *target_hw, 3), dtype=np.float32), fps, start_sec, start_sec

        face_box = None
        if cascade_path.exists():
            face_box = self._detect_face_box(raw_frames[0], cascade_path)

        if face_box is None:
            h, w = raw_frames[0].shape[:2]
            face_box = [0, 0, w, h]

        x, y, w, h = face_box
        target_h, target_w = target_hw
        processed = np.zeros((len(raw_frames), target_h, target_w, 3), dtype=np.float32)
        for i, frame in enumerate(raw_frames):
            y1 = max(y, 0)
            y2 = min(y + h, frame.shape[0])
            x1 = max(x, 0)
            x2 = min(x + w, frame.shape[1])
            cropped = frame[y1:y2, x1:x2]
            processed[i] = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)

        used_end = start_sec + len(raw_frames) / fps
        return processed, fps, start_sec, used_end

    @staticmethod
    def _detect_face_box(frame_rgb: np.ndarray, cascade_path: Path) -> list[int] | None:
        detector = cv2.CascadeClassifier(str(cascade_path))
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        if len(faces) > 1:
            areas = faces[:, 2] * faces[:, 3]
            face = faces[np.argmax(areas)]
        else:
            face = faces[0]
        x, y, w, h = face.astype(int)
        larger_box_coef = 1.5
        x = max(0, int(x - (larger_box_coef - 1.0) / 2 * w))
        y = max(0, int(y - (larger_box_coef - 1.0) / 2 * h))
        w = int(larger_box_coef * w)
        h = int(larger_box_coef * h)
        return [x, y, w, h]

    @staticmethod
    def _downsample_series(series: dict, max_points: int = 1500) -> dict:
        time = series.get("time_sec") or []
        if len(time) <= max_points:
            series["downsample_step"] = 1
            return series
        step = max(1, int(math.ceil(len(time) / max_points)))
        series = dict(series)
        for key, values in list(series.items()):
            if not isinstance(values, list):
                continue
            series[key] = values[::step]
        series["downsample_step"] = step
        return series

    @staticmethod
    def _downsample_psd(psd: dict, max_points: int = 900) -> dict:
        freq = psd.get("freq_bpm") or []
        if len(freq) <= max_points:
            return psd
        step = max(1, int(math.ceil(len(freq) / max_points)))
        return {
            "freq_bpm": freq[::step],
            "power": (psd.get("power") or [])[::step],
        }
