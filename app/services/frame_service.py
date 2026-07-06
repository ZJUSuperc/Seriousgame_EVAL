import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


class FrameService:
    AU_COLUMNS = [
        "AU01_r",
        "AU02_r",
        "AU04_r",
        "AU05_r",
        "AU06_r",
        "AU07_r",
        "AU09_r",
        "AU10_r",
        "AU12_r",
        "AU14_r",
        "AU15_r",
        "AU17_r",
        "AU20_r",
        "AU23_r",
        "AU25_r",
        "AU26_r",
        "AU45_r",
    ]

    CORE_FOCUS_AUS = {
        "AU12_r": "微笑核心",
        "AU15_r": "悲伤核心",
        "AU14_r": "轻蔑核心",
        "AU10_r": "厌恶核心",
    }

    RELATED_FOCUS_AUS = {
        "AU06_r": "微笑相关",
        "AU25_r": "微笑相关",
        "AU26_r": "微笑相关",
        "AU01_r": "悲伤相关",
        "AU04_r": "悲伤相关",
        "AU17_r": "悲伤相关",
        "AU09_r": "厌恶相关",
    }

    AU_MEANINGS = {
        "AU01_r": "内眉上提",
        "AU02_r": "外眉上提",
        "AU04_r": "眉下压",
        "AU05_r": "上眼睑上提",
        "AU06_r": "脸颊上提",
        "AU07_r": "眼睑收紧",
        "AU09_r": "鼻皱起",
        "AU10_r": "上唇上提",
        "AU12_r": "嘴角上提（微笑）",
        "AU14_r": "单侧嘴角收紧（轻蔑）",
        "AU15_r": "嘴角下拉（悲伤）",
        "AU17_r": "下巴上提",
        "AU20_r": "嘴角拉伸",
        "AU23_r": "嘴唇收紧",
        "AU25_r": "双唇分离",
        "AU26_r": "下颌下落",
        "AU45_r": "眨眼/闭眼",
    }

    AU_EMOTION_TAGS = {
        "AU06_r": ["微笑"],
        "AU09_r": ["厌恶"],
        "AU10_r": ["厌恶"],
        "AU12_r": ["微笑"],
        "AU14_r": ["轻蔑"],
        "AU15_r": ["悲伤"],
        "AU25_r": ["微笑", "惊讶"],
        "AU26_r": ["微笑", "惊讶"],
        "AU01_r": ["悲伤"],
        "AU04_r": ["悲伤", "愤怒"],
        "AU17_r": ["悲伤"],
    }

    ATTENTION_CONFIDENCE_THRESHOLD = 0.8
    ATTENTION_AU45_C_THRESHOLD = 0.5
    ATTENTION_AU45_R_THRESHOLD = 1.35
    ATTENTION_AU45_STRONG_R_THRESHOLD = 1.9
    ATTENTION_MAX_ANGLE_RAD = 1.45
    ATTENTION_DELTA_OUTLIER_RAD = 0.55
    ATTENTION_INTERPOLATE_MAX_GAP_SEC = 0.3
    ATTENTION_SPIKE_WINDOW = 10
    ATTENTION_SAVGOL_WINDOW = 7
    ATTENTION_SAVGOL_ORDER = 2
    ATTENTION_FOCUS_THRESHOLD = 0.3
    ATTENTION_ACCEPTABLE_THRESHOLD = 0.5
    ATTENTION_BLINK_MIN_SEC = 0.06
    ATTENTION_BLINK_MAX_SEC = 0.8
    ATTENTION_MIN_VALID_RATIO = 0.2
    ATTENTION_FIXATION_MEDIAN_FULL_SCORE_SEC = 2.0
    ATTENTION_BLINK_RATIO_OK = 0.08
    ATTENTION_BLINK_RATIO_HIGH = 0.20
    ABNORMAL_EVENT_MIN_DURATION_SEC = 0.5
    ABNORMAL_POSITIVE_MIN_DURATION_SEC = 0.35
    ABNORMAL_LONG_EYE_MIN_DURATION_SEC = 1.0
    ABNORMAL_AVERSION_MIN_DURATION_SEC = 0.6
    ABNORMAL_POSITIVE_AU12_STRONG = 0.85
    ABNORMAL_POSITIVE_AU12 = 0.55
    ABNORMAL_POSITIVE_AU6 = 0.4
    ABNORMAL_AVERSION_AU9_AU10 = 0.6
    ABNORMAL_AVERSION_STRONG = 0.75
    ABNORMAL_FALLBACK_FRAME_SEC = 1.0 / 30.0
    ABNORMAL_MOUTH_OPEN_THRESHOLD = 0.5
    ABNORMAL_MOUTH_OPEN_STRONG_THRESHOLD = 0.75
    ABNORMAL_LIP_TIGHT_THRESHOLD = 0.35
    ABNORMAL_STRESS_AU4 = 0.55
    ABNORMAL_STRESS_AU7 = 0.55

    def __init__(self, config: dict):
        self.config = config

    def load_timeline_dataframe(
        self,
        csv_path: Path,
        frame_stride: int,
        max_frames: int,
        video_path: Path | None = None,
        select_main_face: bool = True,
    ) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        df["openface_row"] = np.arange(len(df))

        if "frame" not in df.columns:
            df["frame"] = np.arange(1, len(df) + 1)
        if "timestamp" not in df.columns:
            df["timestamp"] = np.arange(0, len(df), dtype=float)

        df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

        if df["frame"].isna().any():
            fallback_frame = np.arange(1, len(df) + 1)
            df["frame"] = df["frame"].fillna(pd.Series(fallback_frame, index=df.index))

        if df["timestamp"].isna().all():
            df["timestamp"] = np.arange(0, len(df), dtype=float)
        elif df["timestamp"].isna().any():
            df["timestamp"] = df["timestamp"].interpolate(method="linear", limit_direction="both")

        df = df.dropna(subset=["frame", "timestamp"]).copy()
        df = df.sort_values(by=["timestamp", "frame", "openface_row"], kind="stable").copy()

        main_face_info = {
            "enabled": bool(select_main_face),
            "applied": False,
            "face_count": 1,
            "strategy": "none",
        }

        if "face_id" in df.columns:
            df["face_id"] = pd.to_numeric(df["face_id"], errors="coerce")
            valid_face_ids = [int(v) for v in df["face_id"].dropna().unique().tolist()]
            main_face_info["face_count"] = len(valid_face_ids)

            if select_main_face and len(valid_face_ids) > 1:
                df, selected_info = self._select_main_face_rows(df, video_path)
                main_face_info.update(selected_info)
            else:
                df = self._deduplicate_frame_rows(df)
                if valid_face_ids:
                    main_face_info["dominant_face_id"] = int(valid_face_ids[0])
                    main_face_info["strategy"] = "single_face"
        else:
            df = self._deduplicate_frame_rows(df)

        if bool(self.config.get("ENABLE_QUALITY_FILTER", False)):
            if "success" in df.columns:
                df = df[df["success"] == 1]
            if "confidence" in df.columns:
                threshold = float(self.config.get("CONFIDENCE_THRESHOLD", 0.8))
                df = df[df["confidence"] >= threshold]

        df = self._deduplicate_frame_rows(df)
        df = df.sort_values(by=["timestamp", "frame", "openface_row"], kind="stable").copy()

        frame_stride = max(1, int(frame_stride))
        df = df.iloc[::frame_stride].copy()

        effective_max = int(max_frames)
        if effective_max > 0:
            df = df.iloc[:effective_max].copy()

        if df.empty:
            raise RuntimeError("经过质量筛选后没有可用帧")

        df.reset_index(drop=True, inplace=True)
        main_face_info["frames_selected"] = int(len(df))
        df.attrs["main_face_info"] = main_face_info
        return df

    def build_frame_items(self, df: pd.DataFrame) -> list[dict]:
        items = []
        for i, row in df.iterrows():
            items.append(
                {
                    "index": int(i),
                    "frame": int(row["frame"]),
                    "timestamp": float(row["timestamp"]),
                    "openface_row": int(row["openface_row"]),
                    "face_id": int(row["face_id"]) if "face_id" in row and pd.notna(row["face_id"]) else None,
                }
            )
        return items

    def extract_frames(self, video_path: Path, items: list[dict], frame_dir: Path) -> list[dict]:
        frame_dir.mkdir(parents=True, exist_ok=True)

        targets = []
        for item in items:
            frame_index = max(item["frame"] - 1, 0)
            targets.append((frame_index, item))
        targets.sort(key=lambda x: x[0])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        current = -1
        target_idx = 0
        total = len(targets)

        while target_idx < total:
            ok, frame = cap.read()
            current += 1
            if not ok:
                break

            decoded_timestamp = None
            timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp_msec is not None:
                try:
                    ts_value = float(timestamp_msec)
                except (TypeError, ValueError):
                    ts_value = None
                if ts_value is not None and math.isfinite(ts_value) and ts_value >= 0:
                    decoded_timestamp = ts_value / 1000.0

            while target_idx < total and targets[target_idx][0] == current:
                _, item = targets[target_idx]
                image_path = frame_dir / f"frame_{item['index']:06d}.jpg"
                cv2.imwrite(str(image_path), frame)
                item["frame_index"] = current
                item["image_path"] = str(image_path)
                if decoded_timestamp is not None:
                    item["video_timestamp_sec"] = float(decoded_timestamp)
                    item["timestamp"] = float(decoded_timestamp)
                    item["timestamp_source"] = "video_decode_pos_msec"
                target_idx += 1

            if target_idx >= total:
                break

        cap.release()

        missing = [x for _, x in targets if "image_path" not in x]
        if missing:
            raise RuntimeError(f"有 {len(missing)} 帧未能成功抽取")

        return items

    @staticmethod
    def _deduplicate_frame_rows(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "frame" not in df.columns:
            return df

        work = df.copy()
        if "confidence" in work.columns:
            work["_conf_rank"] = pd.to_numeric(work["confidence"], errors="coerce").fillna(-1.0)
        else:
            work["_conf_rank"] = -1.0

        if "success" in work.columns:
            work["_succ_rank"] = pd.to_numeric(work["success"], errors="coerce").fillna(0.0)
        else:
            work["_succ_rank"] = 0.0

        sort_columns = [col for col in ["frame", "_succ_rank", "_conf_rank", "timestamp", "openface_row"] if col in work.columns]
        ascending = [col in {"frame", "timestamp", "openface_row"} for col in sort_columns]

        work = work.sort_values(by=sort_columns, ascending=ascending, kind="stable")
        work = work.drop_duplicates(subset=["frame"], keep="first").copy()
        work.drop(columns=["_conf_rank", "_succ_rank"], inplace=True, errors="ignore")
        return work

    def _select_main_face_rows(self, df: pd.DataFrame, video_path: Path | None) -> tuple[pd.DataFrame, dict]:
        work = df.copy()
        work["face_id"] = pd.to_numeric(work["face_id"], errors="coerce")
        work = work.dropna(subset=["face_id"]).copy()
        if work.empty:
            raise RuntimeError("多人脸筛选失败: OpenFace face_id 全部为空")

        if "confidence" in work.columns:
            work["confidence"] = pd.to_numeric(work["confidence"], errors="coerce").fillna(0.0)
        else:
            work["confidence"] = 0.0

        if "success" in work.columns:
            work["success"] = pd.to_numeric(work["success"], errors="coerce").fillna(0.0)
        else:
            work["success"] = 0.0

        x_cols = sorted([c for c in work.columns if c.startswith("x_")], key=lambda name: int(name.split("_")[1]))
        y_cols = sorted([c for c in work.columns if c.startswith("y_")], key=lambda name: int(name.split("_")[1]))

        face_count = int(work["face_id"].nunique(dropna=True))

        if not x_cols or not y_cols:
            fallback = self._deduplicate_frame_rows(work)
            dominant = fallback["face_id"].mode(dropna=True)
            dominant_face_id = int(dominant.iloc[0]) if not dominant.empty else None
            return fallback, {
                "applied": True,
                "face_count": face_count,
                "dominant_face_id": dominant_face_id,
                "strategy": "face_id_fallback",
            }

        try:
            x_values = work[x_cols].to_numpy(dtype=float)
            y_values = work[y_cols].to_numpy(dtype=float)
        except Exception:
            x_values = work[x_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            y_values = work[y_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        with np.errstate(invalid="ignore"):
            center_x = np.nanmean(x_values, axis=1)
            center_y = np.nanmean(y_values, axis=1)
            width = np.nanmax(x_values, axis=1) - np.nanmin(x_values, axis=1)
            height = np.nanmax(y_values, axis=1) - np.nanmin(y_values, axis=1)

        valid_center_x = center_x[np.isfinite(center_x)]
        valid_center_y = center_y[np.isfinite(center_y)]
        default_center_x = float(np.median(valid_center_x)) if valid_center_x.size > 0 else 0.0
        default_center_y = float(np.median(valid_center_y)) if valid_center_y.size > 0 else 0.0

        center_x = np.where(np.isfinite(center_x), center_x, default_center_x)
        center_y = np.where(np.isfinite(center_y), center_y, default_center_y)

        width = np.where(np.isfinite(width), width, 1.0)
        height = np.where(np.isfinite(height), height, 1.0)
        width = np.clip(width, 1.0, None)
        height = np.clip(height, 1.0, None)
        face_area = width * height

        frame_width, frame_height = self._probe_video_resolution(video_path)
        if frame_width <= 0 or frame_height <= 0:
            valid_x = x_values[np.isfinite(x_values)]
            valid_y = y_values[np.isfinite(y_values)]
            if valid_x.size > 0 and valid_y.size > 0:
                min_x, max_x = float(np.min(valid_x)), float(np.max(valid_x))
                min_y, max_y = float(np.min(valid_y)), float(np.max(valid_y))
                frame_width = max(max_x - min_x, 1.0)
                frame_height = max(max_y - min_y, 1.0)
                center_ref_x = min_x + frame_width / 2.0
                center_ref_y = min_y + frame_height / 2.0
            else:
                frame_width = 2.0
                frame_height = 2.0
                center_ref_x = 0.0
                center_ref_y = 0.0
        else:
            center_ref_x = frame_width / 2.0
            center_ref_y = frame_height / 2.0

        half_w = max(frame_width / 2.0, 1.0)
        half_h = max(frame_height / 2.0, 1.0)

        dx = (center_x - center_ref_x) / half_w
        dy = (center_y - center_ref_y) / half_h
        center_dist = np.sqrt(np.square(dx) + np.square(dy))
        center_score = np.exp(-1.3 * np.nan_to_num(center_dist, nan=4.0, posinf=4.0, neginf=4.0))

        valid_area = face_area[np.isfinite(face_area)]
        area_ref = float(np.percentile(valid_area, 80)) if valid_area.size > 0 else 1.0
        area_ref = max(area_ref, 1.0)
        area_score = np.clip(face_area / area_ref, 0.0, 1.5) / 1.5

        conf = np.clip(work["confidence"].to_numpy(dtype=float), 0.0, 1.0)
        succ = np.clip(work["success"].to_numpy(dtype=float), 0.0, 1.0)
        quality_score = 0.75 * conf + 0.25 * succ

        work["_center_x"] = center_x
        work["_center_y"] = center_y
        work["_main_face_score"] = 0.58 * center_score + 0.27 * area_score + 0.15 * quality_score
        work = work.sort_values(by=["frame", "timestamp", "openface_row"], kind="stable")

        selected_indices: list[int] = []
        prev_center: tuple[float, float] | None = None

        for _, group in work.groupby("frame", sort=True):
            scores = group["_main_face_score"].to_numpy(dtype=float)
            gx = group["_center_x"].to_numpy(dtype=float)
            gy = group["_center_y"].to_numpy(dtype=float)

            if prev_center is not None:
                motion = np.sqrt(np.square((gx - prev_center[0]) / half_w) + np.square((gy - prev_center[1]) / half_h))
                scores = scores - 0.22 * np.nan_to_num(motion, nan=3.0, posinf=3.0, neginf=3.0)

            if scores.size == 0:
                continue

            best_pos = int(np.argmax(scores))
            best_index = group.index[best_pos]
            selected_indices.append(int(best_index))

            pick_x = float(gx[best_pos])
            pick_y = float(gy[best_pos])
            if prev_center is None:
                prev_center = (pick_x, pick_y)
            else:
                prev_center = (
                    0.7 * prev_center[0] + 0.3 * pick_x,
                    0.7 * prev_center[1] + 0.3 * pick_y,
                )

        selected = work.loc[selected_indices].copy()
        selected = self._deduplicate_frame_rows(selected)
        selected = selected.sort_values(by=["timestamp", "frame", "openface_row"], kind="stable").copy()

        dominant = selected["face_id"].mode(dropna=True)
        dominant_face_id = int(dominant.iloc[0]) if not dominant.empty else None

        selected.drop(columns=["_center_x", "_center_y", "_main_face_score"], inplace=True, errors="ignore")

        return selected, {
            "applied": True,
            "face_count": face_count,
            "dominant_face_id": dominant_face_id,
            "strategy": "center_continuity_area",
            "rows_before": int(len(df)),
            "rows_after": int(len(selected)),
        }

    @staticmethod
    def _probe_video_resolution(video_path: Path | None) -> tuple[float, float]:
        if video_path is None:
            return 0.0, 0.0

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0, 0.0

        try:
            width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
            height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
            return width, height
        finally:
            cap.release()

    def compute_attention_series(self, df: pd.DataFrame) -> list[float | None]:
        analysis = self.compute_attention_analysis(df)
        return analysis["clean"]

    def compute_attention_analysis(self, df: pd.DataFrame) -> dict:
        count = len(df)
        if count == 0:
            return {
                "raw": [],
                "clean": [],
                "state": [],
                "metrics": {
                    "total_duration_sec": 0.0,
                    "analysis_duration_sec": 0.0,
                    "valid_ratio": 0.0,
                    "focus_ratio": 0.0,
                    "screen_focus_ratio": 0.0,
                    "acceptable_focus_ratio": 0.0,
                    "off_focus_ratio": 0.0,
                    "stability_std": None,
                    "reengage_time_median": None,
                    "screen_focus_duration_median_sec": None,
                    "screen_focus_duration_max_sec": 0.0,
                    "screen_focus_segment_count": 0,
                    "long_eye_close_ratio": 0.0,
                    "long_eye_close_count": 0,
                    "long_eye_close_count_per_min": 0.0,
                    "blink_ratio": 0.0,
                    "blink_count": 0,
                    "blink_count_per_min": 0.0,
                    "blink_excess_ratio": 0.0,
                    "quality_status": "insufficient",
                },
                "scores": {
                    "focus_score": 0.0,
                    "fatigue_score": 0.0,
                    "fixation_score": 0.0,
                    "blink_score": 0.0,
                    "stability_score": 0.0,
                    "quality_score": 0.0,
                    "attention_score": 0.0,
                },
                "events": {
                    "blink_segments": [],
                    "long_eye_close_segments": [],
                },
            }

        timestamps = self._sanitize_timestamps(self._read_numeric_column(df, "timestamp", default=0.0))
        durations = self._compute_frame_durations(timestamps)
        total_duration = float(np.sum(durations)) if durations.size > 0 else 0.0
        if total_duration <= 1e-9:
            durations = np.full(count, 1.0, dtype=float)
            total_duration = float(count)

        gaze_x = self._read_numeric_column(df, "gaze_angle_x", default=np.nan)
        gaze_y = self._read_numeric_column(df, "gaze_angle_y", default=np.nan)
        raw_angle = np.sqrt(np.square(gaze_x) + np.square(gaze_y))
        raw_finite = np.isfinite(raw_angle)

        success = self._read_numeric_column(df, "success", default=1.0)
        confidence = self._read_numeric_column(df, "confidence", default=1.0)
        track_ok = (
            np.isfinite(success)
            & np.isfinite(confidence)
            & (success >= 0.5)
            & (confidence >= self.ATTENTION_CONFIDENCE_THRESHOLD)
        )

        au45_c = self._read_numeric_column(df, "AU45_c", default=0.0)
        au45_r = self._read_numeric_column(df, "AU45_r", default=0.0)
        au45_c_closed = np.isfinite(au45_c) & (au45_c >= self.ATTENTION_AU45_C_THRESHOLD)
        au45_r_moderate = np.isfinite(au45_r) & (au45_r >= self.ATTENTION_AU45_R_THRESHOLD)
        au45_r_strong = np.isfinite(au45_r) & (au45_r >= self.ATTENTION_AU45_STRONG_R_THRESHOLD)
        eye_closed_raw = au45_r_strong | (au45_c_closed & au45_r_moderate)
        eye_closed = self._suppress_short_true_segments(
            eye_closed_raw,
            min_length=2,
        )

        delta = np.abs(np.diff(raw_angle, prepend=raw_angle[0]))
        delta = np.where(np.isfinite(delta), delta, np.inf)

        outlier = (
            (raw_finite & (raw_angle > self.ATTENTION_MAX_ANGLE_RAD))
            | (raw_finite & (delta > self.ATTENTION_DELTA_OUTLIER_RAD))
        )

        valid = track_ok & (~eye_closed) & (~outlier) & raw_finite

        blink_segments = self._classify_blink_segments(
            eye_closed=(eye_closed & track_ok),
            timestamps=timestamps,
            durations=durations,
        )
        blink_mask = self._mask_from_segments(count, blink_segments)

        states = np.full(count, "invalid", dtype=object)
        states[~track_ok] = "track_lost"
        states[track_ok & eye_closed] = "eye_closed"
        states[track_ok & (~eye_closed) & outlier] = "outlier"
        states[valid] = "valid"
        states[track_ok & blink_mask] = "blink"

        clean = np.where(valid, raw_angle, np.nan)
        clean = self._interpolate_short_gaps(
            values=clean,
            timestamps=timestamps,
            max_gap_sec=self.ATTENTION_INTERPOLATE_MAX_GAP_SEC,
        )
        clean = self._replace_attention_spikes(
            clean,
            window=self.ATTENTION_SPIKE_WINDOW,
        )
        clean = self._savgol_smooth_attention(
            clean,
            window=self.ATTENTION_SAVGOL_WINDOW,
            order=self.ATTENTION_SAVGOL_ORDER,
        )

        analysis_mask = np.isfinite(clean)
        analysis_duration = float(np.sum(durations[analysis_mask])) if np.any(analysis_mask) else 0.0

        blink_duration = float(sum(seg["duration_sec"] for seg in blink_segments))
        quality_total_duration = max(total_duration - blink_duration, 1e-9)
        valid_ratio = self._duration_ratio(durations, valid, quality_total_duration)

        focus_mask = analysis_mask & (clean <= self.ATTENTION_FOCUS_THRESHOLD)
        screen_focus_mask = analysis_mask & (clean <= self.ATTENTION_ACCEPTABLE_THRESHOLD)
        acceptable_mask = (
            analysis_mask
            & (clean > self.ATTENTION_FOCUS_THRESHOLD)
            & (clean <= self.ATTENTION_ACCEPTABLE_THRESHOLD)
        )
        off_focus_mask = analysis_mask & (clean > self.ATTENTION_ACCEPTABLE_THRESHOLD)

        focus_ratio = self._duration_ratio(durations, focus_mask, analysis_duration)
        screen_focus_ratio = self._duration_ratio(durations, screen_focus_mask, analysis_duration)
        acceptable_focus_ratio = self._duration_ratio(durations, acceptable_mask, analysis_duration)
        off_focus_ratio = self._duration_ratio(durations, off_focus_mask, analysis_duration)

        screen_focus_segments = self._find_true_segments(screen_focus_mask, timestamps, durations)
        screen_focus_durations = [float(seg["duration_sec"]) for seg in screen_focus_segments]
        screen_focus_duration_median = float(np.median(screen_focus_durations)) if screen_focus_durations else None
        screen_focus_duration_max = float(max(screen_focus_durations)) if screen_focus_durations else 0.0
        screen_focus_segment_count = len(screen_focus_segments)

        if np.count_nonzero(analysis_mask) >= 2:
            stability_std = float(np.nanstd(clean[analysis_mask]))
        else:
            stability_std = None

        reengage_times = self._compute_reengage_times(
            clean=clean,
            timestamps=timestamps,
            focus_threshold=self.ATTENTION_FOCUS_THRESHOLD,
            off_threshold=self.ATTENTION_ACCEPTABLE_THRESHOLD,
        )
        reengage_time_median = float(np.median(reengage_times)) if reengage_times else None

        long_eye_close_ratio = 0.0
        long_eye_close_count = 0
        long_eye_close_count_per_min = 0.0

        blink_ratio = 0.0 if total_duration <= 1e-9 else blink_duration / total_duration
        blink_count = len(blink_segments)
        blink_count_per_min = 0.0 if total_duration <= 1e-9 else blink_count / (total_duration / 60.0)
        blink_excess_ratio = max(0.0, blink_ratio - self.ATTENTION_BLINK_RATIO_OK)

        if analysis_duration <= 1e-9:
            focus_score = 0.0
            fixation_score = 0.0
            blink_score = 0.0
            stability_score = 0.0
        else:
            focus_score = 100.0 * (0.55 * focus_ratio + 0.45 * screen_focus_ratio)
            fixation_base = screen_focus_duration_median if screen_focus_duration_median is not None else 0.0
            fixation_score = 100.0 * self._clip01(fixation_base / self.ATTENTION_FIXATION_MEDIAN_FULL_SCORE_SEC)
            blink_score = 100.0 * (1.0 - self._clip01(
                (blink_ratio - self.ATTENTION_BLINK_RATIO_OK)
                / max(self.ATTENTION_BLINK_RATIO_HIGH - self.ATTENTION_BLINK_RATIO_OK, 1e-9)
            ))
            stability_base = stability_std if stability_std is not None else self.ATTENTION_ACCEPTABLE_THRESHOLD
            stability_score = 100.0 * self._clip01((0.35 - stability_base) / 0.35)

        quality_score = 100.0 * self._clip01((valid_ratio - self.ATTENTION_MIN_VALID_RATIO) / (1.0 - self.ATTENTION_MIN_VALID_RATIO))
        fatigue_score = fixation_score
        attention_score = 0.40 * focus_score + 0.15 * fixation_score + 0.10 * blink_score + 0.15 * stability_score + 0.20 * quality_score
        quality_status = "ok" if valid_ratio >= self.ATTENTION_MIN_VALID_RATIO else "insufficient"

        return {
            "raw": [float(v) if np.isfinite(v) else None for v in raw_angle],
            "clean": [float(v) if np.isfinite(v) else None for v in clean],
            "state": [str(v) for v in states.tolist()],
            "metrics": {
                "total_duration_sec": float(total_duration),
                "analysis_duration_sec": float(analysis_duration),
                "valid_ratio": float(valid_ratio),
                "focus_ratio": float(focus_ratio),
                "screen_focus_ratio": float(screen_focus_ratio),
                "acceptable_focus_ratio": float(acceptable_focus_ratio),
                "off_focus_ratio": float(off_focus_ratio),
                "stability_std": float(stability_std) if stability_std is not None else None,
                "reengage_time_median": float(reengage_time_median) if reengage_time_median is not None else None,
                "screen_focus_duration_median_sec": float(screen_focus_duration_median) if screen_focus_duration_median is not None else None,
                "screen_focus_duration_max_sec": float(screen_focus_duration_max),
                "screen_focus_segment_count": int(screen_focus_segment_count),
                "long_eye_close_ratio": float(long_eye_close_ratio),
                "long_eye_close_count": int(long_eye_close_count),
                "long_eye_close_count_per_min": float(long_eye_close_count_per_min),
                "blink_ratio": float(blink_ratio),
                "blink_count": int(blink_count),
                "blink_count_per_min": float(blink_count_per_min),
                "blink_excess_ratio": float(blink_excess_ratio),
                "quality_status": quality_status,
            },
            "scores": {
                "focus_score": float(focus_score),
                "fatigue_score": float(fatigue_score),
                "fixation_score": float(fixation_score),
                "blink_score": float(blink_score),
                "stability_score": float(stability_score),
                "quality_score": float(quality_score),
                "attention_score": float(attention_score),
            },
            "events": {
                "blink_segments": blink_segments,
                "long_eye_close_segments": [],
            },
        }

    @staticmethod
    def _clip01(value: float) -> float:
        return float(min(1.0, max(0.0, value)))

    @staticmethod
    def _duration_ratio(durations: np.ndarray, mask: np.ndarray, denominator: float) -> float:
        if denominator <= 1e-9 or durations.size == 0:
            return 0.0
        return float(np.sum(durations[mask]) / denominator)

    @staticmethod
    def _mask_from_segments(count: int, segments: list[dict]) -> np.ndarray:
        mask = np.zeros(count, dtype=bool)
        if count <= 0:
            return mask

        for segment in segments:
            start = int(segment.get("start_index", 0))
            end = int(segment.get("end_index", -1))
            start = max(0, min(count - 1, start))
            end = max(0, min(count - 1, end))
            if start <= end:
                mask[start : end + 1] = True
        return mask

    @staticmethod
    def _suppress_short_true_segments(mask: np.ndarray, min_length: int = 2) -> np.ndarray:
        result = np.array(mask, dtype=bool, copy=True)
        if min_length <= 1 or result.size == 0:
            return result

        count = len(result)
        index = 0
        while index < count:
            if not result[index]:
                index += 1
                continue

            start = index
            while index + 1 < count and result[index + 1]:
                index += 1
            end = index

            if (end - start + 1) < min_length:
                result[start : end + 1] = False

            index += 1

        return result

    @staticmethod
    def _read_numeric_column(df: pd.DataFrame, column: str, default: float) -> np.ndarray:
        count = len(df)
        if count == 0:
            return np.zeros(0, dtype=float)

        if column not in df.columns:
            return np.full(count, float(default), dtype=float)

        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if values.size != count:
            return np.full(count, float(default), dtype=float)
        return values

    @staticmethod
    def _sanitize_timestamps(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values

        timestamps = np.array(values, dtype=float, copy=True)
        finite_mask = np.isfinite(timestamps)
        if not np.any(finite_mask):
            return np.arange(len(timestamps), dtype=float)

        indices = np.arange(len(timestamps), dtype=float)
        if np.any(~finite_mask):
            finite_idx = indices[finite_mask]
            finite_values = timestamps[finite_mask]
            if finite_values.size == 1:
                timestamps[~finite_mask] = float(finite_values[0])
            else:
                timestamps[~finite_mask] = np.interp(indices[~finite_mask], finite_idx, finite_values)

        for idx in range(1, len(timestamps)):
            if timestamps[idx] < timestamps[idx - 1]:
                timestamps[idx] = timestamps[idx - 1]

        return timestamps

    @staticmethod
    def _compute_frame_durations(timestamps: np.ndarray) -> np.ndarray:
        count = len(timestamps)
        if count == 0:
            return np.zeros(0, dtype=float)
        if count == 1:
            return np.array([1.0], dtype=float)

        delta = np.diff(timestamps)
        valid_delta = delta[np.isfinite(delta) & (delta > 1e-6)]
        fallback = float(np.median(valid_delta)) if valid_delta.size > 0 else (1.0 / 30.0)
        fallback = max(fallback, 1.0 / 120.0)

        durations = np.empty(count, dtype=float)
        durations[:-1] = np.where(np.isfinite(delta) & (delta > 1e-6), delta, fallback)
        durations[-1] = durations[-2] if count > 1 else fallback
        durations = np.clip(durations, 1.0 / 120.0, None)
        return durations

    def _classify_blink_segments(
        self,
        eye_closed: np.ndarray,
        timestamps: np.ndarray,
        durations: np.ndarray,
    ) -> list[dict]:
        segments = self._find_true_segments(eye_closed, timestamps, durations)
        blink_segments: list[dict] = []

        for segment in segments:
            duration = float(segment["duration_sec"])
            if self.ATTENTION_BLINK_MIN_SEC <= duration <= self.ATTENTION_BLINK_MAX_SEC:
                blink_segments.append(segment)

        return blink_segments

    @staticmethod
    def _find_true_segments(mask: np.ndarray, timestamps: np.ndarray, durations: np.ndarray) -> list[dict]:
        segments: list[dict] = []
        count = len(mask)
        index = 0
        while index < count:
            if not mask[index]:
                index += 1
                continue

            start = index
            while index + 1 < count and mask[index + 1]:
                index += 1
            end = index

            duration_sec = float(np.sum(durations[start : end + 1]))
            segments.append(
                {
                    "start_index": int(start),
                    "end_index": int(end),
                    "start_timestamp": float(timestamps[start]),
                    "end_timestamp": float(timestamps[end]),
                    "duration_sec": duration_sec,
                }
            )
            index += 1

        return segments

    @staticmethod
    def _interpolate_short_gaps(values: np.ndarray, timestamps: np.ndarray, max_gap_sec: float) -> np.ndarray:
        result = np.array(values, dtype=float, copy=True)
        count = len(result)
        index = 0
        while index < count:
            if np.isfinite(result[index]):
                index += 1
                continue

            start = index
            while index < count and not np.isfinite(result[index]):
                index += 1
            end = index - 1

            left = start - 1
            right = index
            if left < 0 or right >= count:
                continue
            if not np.isfinite(result[left]) or not np.isfinite(result[right]):
                continue

            gap_sec = float(timestamps[right] - timestamps[left])
            if not np.isfinite(gap_sec) or gap_sec <= 0 or gap_sec > max_gap_sec:
                continue

            denom = timestamps[right] - timestamps[left]
            for fill_idx in range(start, end + 1):
                if denom > 1e-9:
                    ratio = float((timestamps[fill_idx] - timestamps[left]) / denom)
                else:
                    ratio = float((fill_idx - left) / max(right - left, 1))
                ratio = min(1.0, max(0.0, ratio))
                result[fill_idx] = (1.0 - ratio) * result[left] + ratio * result[right]

        return result

    @staticmethod
    def _median_filter(values: np.ndarray, window: int = 3) -> np.ndarray:
        if window <= 1 or len(values) == 0:
            return values

        half = window // 2
        result = np.array(values, dtype=float, copy=True)
        for idx in range(len(result)):
            if not np.isfinite(result[idx]):
                continue

            left = max(0, idx - half)
            right = min(len(result), idx + half + 1)
            segment = result[left:right]
            segment = segment[np.isfinite(segment)]
            if segment.size >= 2:
                result[idx] = float(np.median(segment))

        return result

    @staticmethod
    def _replace_attention_spikes(values: np.ndarray, window: int) -> np.ndarray:
        result = np.array(values, dtype=float, copy=True)
        if result.size == 0 or window <= 1:
            return result

        finite_mask = np.isfinite(result)
        index = 0
        count = len(result)
        while index < count:
            if not finite_mask[index]:
                index += 1
                continue

            start = index
            while index + 1 < count and finite_mask[index + 1]:
                index += 1
            end = index

            segment = result[start : end + 1]
            result[start : end + 1] = FrameService._replace_attention_spikes_in_segment(segment, window)
            index += 1

        return result

    @staticmethod
    def _replace_attention_spikes_in_segment(values: np.ndarray, window: int) -> np.ndarray:
        result = np.array(values, dtype=float, copy=True)
        if result.size == 0 or window <= 1:
            return result

        half = window // 2
        if len(result) < (2 * half + 1):
            return result

        source = np.array(result, dtype=float, copy=True)
        for idx in range(half, len(source) - half):
            center = source[idx]
            left = source[idx - half : idx]
            right = source[idx + 1 : idx + half + 1]
            if left.size == 0 or right.size == 0:
                continue
            if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)) or not np.isfinite(center):
                continue

            if center > float(np.max(left)) and center > float(np.max(right)):
                replacement = float((float(np.mean(left)) + float(np.mean(right))) / 2.0)
                result[idx - half : idx + half + 1] = replacement

        return result

    @staticmethod
    def _savgol_smooth_attention(values: np.ndarray, window: int, order: int) -> np.ndarray:
        result = np.array(values, dtype=float, copy=True)
        if result.size == 0 or window <= 2:
            return result

        finite_mask = np.isfinite(result)
        index = 0
        count = len(result)
        while index < count:
            if not finite_mask[index]:
                index += 1
                continue

            start = index
            while index + 1 < count and finite_mask[index + 1]:
                index += 1
            end = index

            segment = result[start : end + 1]
            result[start : end + 1] = FrameService._savgol_smooth_segment(segment, window, order)
            index += 1

        return result

    @staticmethod
    def _savgol_smooth_segment(values: np.ndarray, window: int, order: int) -> np.ndarray:
        segment = np.array(values, dtype=float, copy=True)
        size = len(segment)
        if size == 0:
            return segment

        effective_window = min(window, size)
        if effective_window % 2 == 0:
            effective_window -= 1
        min_window = order + 1
        if min_window % 2 == 0:
            min_window += 1
        if effective_window < min_window:
            return segment

        try:
            return savgol_filter(segment, window_length=effective_window, polyorder=order, mode="interp")
        except Exception:
            return segment

    @staticmethod
    def _compute_reengage_times(
        clean: np.ndarray,
        timestamps: np.ndarray,
        focus_threshold: float,
        off_threshold: float,
    ) -> list[float]:
        reengage_times: list[float] = []
        in_off_focus = False
        start_ts = 0.0

        for idx in range(len(clean)):
            value = clean[idx]
            if not np.isfinite(value):
                continue

            ts = float(timestamps[idx])
            if not in_off_focus and value > off_threshold:
                in_off_focus = True
                start_ts = ts
                continue

            if in_off_focus and value <= focus_threshold:
                reengage_times.append(max(0.0, ts - start_ts))
                in_off_focus = False

        return reengage_times

    def detect_micro_expression_events(self, df: pd.DataFrame, max_events: int = 200) -> list[dict]:
        events = []
        for au in self.AU_COLUMNS:
            if au not in df.columns:
                continue

            values = df[au].fillna(0.0).astype(float).to_numpy()
            if len(values) < 2:
                continue

            delta = np.diff(values, prepend=values[0])
            priority = self._get_au_priority(au)
            percentile = 95 if priority >= 2 else 97
            min_threshold = 0.4 if priority >= 3 else 0.5 if priority == 2 else 0.6
            threshold = max(float(np.percentile(np.abs(delta), percentile)), min_threshold)
            indices = np.where(delta > threshold)[0]

            for idx in indices:
                row = df.iloc[int(idx)]
                events.append(
                    {
                        "index": int(idx),
                        "au": au,
                        "au_meaning": self.AU_MEANINGS.get(au, "面部动作单元"),
                        "emotion_tags": self.AU_EMOTION_TAGS.get(au, ["其他"]),
                        "focus_level": self._get_focus_level(au),
                        "priority": int(priority),
                        "frame": int(row["frame"]),
                        "timestamp": float(row["timestamp"]),
                        "delta": float(delta[idx]),
                        "value": float(values[idx]),
                    }
                )

        events.sort(key=lambda x: (x.get("priority", 1), x["delta"]), reverse=True)
        return events[:max_events]

    def detect_abnormal_events(self, df: pd.DataFrame) -> list[dict]:
        if df is None or df.empty:
            return []

        timestamps = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0).astype(float).to_list()
        frames = pd.to_numeric(df["frame"], errors="coerce").fillna(0).astype(int).to_list()
        size = len(timestamps)

        au_keys = [
            "AU01_r",
            "AU02_r",
            "AU04_r",
            "AU05_r",
            "AU06_r",
            "AU07_r",
            "AU09_r",
            "AU10_r",
            "AU12_r",
            "AU14_r",
            "AU15_r",
            "AU17_r",
            "AU20_r",
            "AU23_r",
            "AU25_r",
            "AU26_r",
            "AU45_r",
        ]
        au_values = {key: self._extract_series(df, key, size) for key in au_keys}

        frame_step = self._estimate_frame_step(timestamps)
        au1_delta = self._compute_delta(au_values.get("AU01_r", [None] * size))
        au2_delta = self._compute_delta(au_values.get("AU02_r", [None] * size))
        au5_delta = self._compute_delta(au_values.get("AU05_r", [None] * size))

        events: list[dict] = []

        def au_at(key: str, idx: int) -> float | None:
            value = au_values.get(key, [None] * size)[idx]
            return value

        def has_value(value: float | None) -> bool:
            return value is not None and math.isfinite(value)

        def count_true(values: list[bool]) -> int:
            return sum(1 for value in values if value)

        def mouth_open_value(idx: int) -> float | None:
            au25 = au_at("AU25_r", idx)
            au26 = au_at("AU26_r", idx)
            values = [value for value in [au25, au26] if has_value(value)]
            if not values:
                return None
            return float(max(values))

        def is_mouth_open(idx: int, threshold: float) -> bool:
            value = mouth_open_value(idx)
            return value is not None and value > threshold

        def cond_long_eye(idx: int) -> bool:
            au45 = au_at("AU45_r", idx)
            return has_value(au45) and au45 > 0.5

        def cond_positive(idx: int) -> bool:
            au12 = au_at("AU12_r", idx)
            au6 = au_at("AU06_r", idx)
            if has_value(au12) and au12 >= self.ABNORMAL_POSITIVE_AU12_STRONG:
                return True
            return (
                has_value(au12)
                and au12 >= self.ABNORMAL_POSITIVE_AU12
                and has_value(au6)
                and au6 >= self.ABNORMAL_POSITIVE_AU6
            )

        def cond_sad(idx: int) -> bool:
            au15 = au_at("AU15_r", idx)
            au1 = au_at("AU01_r", idx)
            au4 = au_at("AU04_r", idx)
            au17 = au_at("AU17_r", idx)
            flags = [
                has_value(au15) and au15 > 0.5,
                has_value(au1) and au1 > 0.45,
                has_value(au4) and au4 > 0.45,
                has_value(au17) and au17 > 0.45,
            ]
            if is_mouth_open(idx, self.ABNORMAL_MOUTH_OPEN_STRONG_THRESHOLD):
                if not (has_value(au15) and au15 > 0.5):
                    return False
            return count_true(flags) >= 2

        def cond_aversion(idx: int) -> bool:
            au9 = au_at("AU09_r", idx)
            au10 = au_at("AU10_r", idx)
            au14 = au_at("AU14_r", idx)
            au15 = au_at("AU15_r", idx)
            core = (
                (has_value(au9) and au9 > self.ABNORMAL_AVERSION_AU9_AU10)
                or (has_value(au10) and au10 > self.ABNORMAL_AVERSION_AU9_AU10)
            )
            if not core:
                return False
            if is_mouth_open(idx, self.ABNORMAL_MOUTH_OPEN_THRESHOLD):
                strong_core = (
                    (has_value(au9) and au9 > self.ABNORMAL_AVERSION_STRONG)
                    or (has_value(au10) and au10 > self.ABNORMAL_AVERSION_STRONG)
                )
                if not strong_core:
                    return False
            flags = [
                has_value(au9) and au9 > self.ABNORMAL_AVERSION_AU9_AU10,
                has_value(au10) and au10 > self.ABNORMAL_AVERSION_AU9_AU10,
                has_value(au14) and au14 > 0.55,
                has_value(au15) and au15 > 0.55,
            ]
            return count_true(flags) >= 2

        def cond_stress(idx: int) -> bool:
            au4 = au_at("AU04_r", idx)
            au7 = au_at("AU07_r", idx)
            au23 = au_at("AU23_r", idx)
            au20 = au_at("AU20_r", idx)
            if not (
                has_value(au4)
                and au4 > self.ABNORMAL_STRESS_AU4
                and has_value(au7)
                and au7 > self.ABNORMAL_STRESS_AU7
            ):
                return False
            tension = (
                (has_value(au23) and au23 > self.ABNORMAL_LIP_TIGHT_THRESHOLD)
                or (has_value(au20) and au20 > 0.55)
            )
            if is_mouth_open(idx, self.ABNORMAL_MOUTH_OPEN_THRESHOLD) and not tension:
                return False
            return True

        def cond_surprise(idx: int) -> bool:
            au1 = au_at("AU01_r", idx)
            au2 = au_at("AU02_r", idx)
            au5 = au_at("AU05_r", idx)
            delta_au1 = au1_delta[idx]
            delta_au2 = au2_delta[idx]
            delta_au5 = au5_delta[idx]
            has_delta = (
                (has_value(delta_au1) and delta_au1 > 0.5)
                or (has_value(delta_au2) and delta_au2 > 0.5)
                or (has_value(delta_au5) and delta_au5 > 0.5)
            )
            return (
                has_delta
                and has_value(au1)
                and au1 > 0.5
                and has_value(au2)
                and au2 > 0.5
                and has_value(au5)
                and au5 > 0.5
            )

        def cond_confusion(idx: int) -> bool:
            au4 = au_at("AU04_r", idx)
            au1 = au_at("AU01_r", idx)
            return has_value(au4) and au4 > 0.5 and has_value(au1) and au1 > 0.5

        event_specs = [
            (
                "long_eye_closure_event",
                cond_long_eye,
                ["AU45_r"],
                {"AU45_r": 0.5},
                self.ABNORMAL_LONG_EYE_MIN_DURATION_SEC,
            ),
            (
                "positive_event",
                cond_positive,
                ["AU12_r", "AU06_r"],
                {"AU12_r": self.ABNORMAL_POSITIVE_AU12, "AU06_r": self.ABNORMAL_POSITIVE_AU6},
                self.ABNORMAL_POSITIVE_MIN_DURATION_SEC,
            ),
            (
                "sad_event",
                cond_sad,
                ["AU15_r", "AU01_r", "AU04_r", "AU17_r"],
                {
                    "AU15_r": 0.5,
                    "AU01_r": 0.45,
                    "AU04_r": 0.45,
                    "AU17_r": 0.45,
                },
                self.ABNORMAL_EVENT_MIN_DURATION_SEC,
            ),
            (
                "aversion_event",
                cond_aversion,
                ["AU09_r", "AU10_r", "AU14_r", "AU15_r"],
                {
                    "AU09_r": self.ABNORMAL_AVERSION_AU9_AU10,
                    "AU10_r": self.ABNORMAL_AVERSION_AU9_AU10,
                    "AU14_r": 0.55,
                    "AU15_r": 0.55,
                },
                self.ABNORMAL_AVERSION_MIN_DURATION_SEC,
            ),
            (
                "stress_event",
                cond_stress,
                ["AU04_r", "AU07_r"],
                {"AU04_r": 0.5, "AU07_r": 0.5},
                self.ABNORMAL_EVENT_MIN_DURATION_SEC,
            ),
            (
                "surprise_event",
                cond_surprise,
                ["AU01_r", "AU02_r", "AU05_r"],
                {
                    "AU01_r": 0.5,
                    "AU02_r": 0.5,
                    "AU05_r": 0.5,
                },
                self.ABNORMAL_EVENT_MIN_DURATION_SEC,
            ),
            (
                "confusion_event",
                cond_confusion,
                ["AU04_r", "AU01_r"],
                {"AU04_r": 0.5, "AU01_r": 0.5},
                self.ABNORMAL_EVENT_MIN_DURATION_SEC,
            ),
        ]

        for event_type, condition, related_aus, thresholds, min_duration in event_specs:
            flags = [bool(condition(idx)) for idx in range(size)]
            for start_idx, end_idx in self._build_segments(flags):
                duration = self._segment_duration(timestamps, start_idx, end_idx, frame_step)
                if duration < min_duration:
                    continue

                au_peak = {}
                for au_key in related_aus:
                    values = au_values.get(au_key, [])
                    peak = self._segment_peak(values, start_idx, end_idx)
                    if peak is None:
                        continue
                    au_peak[au_key] = peak

                au_list = []
                for au_key, peak in au_peak.items():
                    threshold = thresholds.get(au_key, 0.5)
                    if peak >= threshold:
                        au_list.append(au_key.replace("_r", ""))

                au_peak_value = max(au_peak.values()) if au_peak else 0.0
                intensity = float(au_peak_value)

                event = {
                    "type": event_type,
                    "index": int(start_idx),
                    "frame": int(frames[start_idx]) if start_idx < len(frames) else int(start_idx),
                    "timestamp": float(timestamps[start_idx]) if start_idx < len(timestamps) else 0.0,
                    "start": float(timestamps[start_idx]) if start_idx < len(timestamps) else 0.0,
                    "end": float(timestamps[end_idx]) if end_idx < len(timestamps) else 0.0,
                    "duration": float(duration),
                    "intensity": intensity,
                    "au_list": au_list,
                    "au_peak": {key.replace("_r", ""): float(value) for key, value in au_peak.items()},
                }
                events.append(event)

        events.sort(key=lambda item: (item.get("timestamp", 0.0), item.get("type", "")))
        return events

    @staticmethod
    def _extract_series(df: pd.DataFrame, key: str, size: int) -> list[float | None]:
        if key not in df.columns:
            return [None] * size
        series = pd.to_numeric(df[key], errors="coerce")
        output: list[float | None] = []
        for value in series.to_list():
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                output.append(None)
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                output.append(None)
                continue
            if not math.isfinite(number):
                output.append(None)
                continue
            output.append(number)
        if len(output) < size:
            output.extend([None] * (size - len(output)))
        if len(output) > size:
            output = output[:size]
        return output

    @staticmethod
    def _estimate_frame_step(timestamps: list[float]) -> float:
        if len(timestamps) < 2:
            return FrameService.ABNORMAL_FALLBACK_FRAME_SEC
        deltas = []
        for idx in range(1, len(timestamps)):
            delta = float(timestamps[idx]) - float(timestamps[idx - 1])
            if math.isfinite(delta) and delta > 1e-6:
                deltas.append(delta)
        if not deltas:
            return FrameService.ABNORMAL_FALLBACK_FRAME_SEC
        deltas.sort()
        mid = len(deltas) // 2
        if len(deltas) % 2 == 1:
            return float(deltas[mid])
        return float((deltas[mid - 1] + deltas[mid]) / 2.0)

    @staticmethod
    def _compute_delta(values: list[float | None]) -> list[float | None]:
        if not values:
            return []
        deltas: list[float | None] = [0.0]
        prev = values[0]
        for idx in range(1, len(values)):
            current = values[idx]
            if prev is None or current is None:
                deltas.append(None)
                prev = current
                continue
            deltas.append(abs(float(current) - float(prev)))
            prev = current
        return deltas

    @staticmethod
    def _build_segments(flags: list[bool]) -> list[tuple[int, int]]:
        segments: list[tuple[int, int]] = []
        start_idx = None
        for idx, flag in enumerate(flags):
            if flag and start_idx is None:
                start_idx = idx
                continue
            if not flag and start_idx is not None:
                segments.append((start_idx, idx - 1))
                start_idx = None
        if start_idx is not None:
            segments.append((start_idx, len(flags) - 1))
        return segments

    @staticmethod
    def _segment_duration(timestamps: list[float], start_idx: int, end_idx: int, frame_step: float) -> float:
        if not timestamps:
            return 0.0
        start_ts = float(timestamps[start_idx]) if start_idx < len(timestamps) else float(start_idx)
        end_ts = float(timestamps[end_idx]) if end_idx < len(timestamps) else float(end_idx)
        duration = max(0.0, end_ts - start_ts) + float(frame_step)
        return duration

    @staticmethod
    def _segment_peak(values: list[float | None], start_idx: int, end_idx: int) -> float | None:
        if not values:
            return None
        segment = [value for value in values[start_idx : end_idx + 1] if value is not None and math.isfinite(value)]
        if not segment:
            return None
        return float(max(segment))

    @staticmethod
    def _segment_mean(values: list[float | None], start_idx: int, end_idx: int) -> float | None:
        if not values:
            return None
        segment = [value for value in values[start_idx : end_idx + 1] if value is not None and math.isfinite(value)]
        if not segment:
            return None
        return float(sum(segment) / len(segment))

    @classmethod
    def _get_au_priority(cls, au: str) -> int:
        if au in cls.CORE_FOCUS_AUS:
            return 3
        if au in cls.RELATED_FOCUS_AUS:
            return 2
        return 1

    @classmethod
    def _get_focus_level(cls, au: str) -> str:
        if au in cls.CORE_FOCUS_AUS:
            return "core"
        if au in cls.RELATED_FOCUS_AUS:
            return "related"
        return "general"
