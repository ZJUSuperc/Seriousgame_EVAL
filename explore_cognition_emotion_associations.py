from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SUBJECT_LABELS = {
    "1901": {"cognitive_score": 20, "emotion_score": 1},
    "1902": {"cognitive_score": 15, "emotion_score": 9},
    "1903": {"cognitive_score": 18, "emotion_score": 5},
    "1904": {"cognitive_score": 21, "emotion_score": 5},
    "1905": {"cognitive_score": 6, "emotion_score": 6},
    "1906": {"cognitive_score": 20, "emotion_score": 3},
    "1907": {"cognitive_score": 26, "emotion_score": 3},
    "1908": {"cognitive_score": 12, "emotion_score": 5},
    "1909": {"cognitive_score": 29, "emotion_score": 0},
    "1910": {"cognitive_score": 28, "emotion_score": 0},
    "1911": {"cognitive_score": 25, "emotion_score": 3},
    "1912": {"cognitive_score": 27, "emotion_score": 9},
}

EMOTION_AU_GROUPS = {
    "happiness": ["AU06", "AU12", "AU25"],
    "sadness": ["AU01", "AU04", "AU15", "AU17"],
    "anger": ["AU04", "AU05", "AU07", "AU23"],
    "fear": ["AU01", "AU02", "AU04", "AU05", "AU20", "AU25", "AU26"],
    "surprise": ["AU01", "AU02", "AU05", "AU26"],
    "disgust": ["AU09", "AU10", "AU16", "AU17"],
    "contempt": ["AU12", "AU14"],
}

REPORT_ROOT = Path(r"C:\Users\Administrator\Desktop\seriousgame_eval\data\reports")
OUTPUT_ROOT = REPORT_ROOT / "association_analysis"
RNG = np.random.default_rng(20260502)


def safe_float(value):
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def ensure_output_dir() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def list_report_items() -> list[dict]:
    items: list[dict] = []
    for person_id, labels in SUBJECT_LABELS.items():
        person_dir = REPORT_ROOT / person_id
        if not person_dir.exists() or not person_dir.is_dir():
            continue
        for video_dir in sorted([path for path in person_dir.iterdir() if path.is_dir()], key=lambda p: p.name.lower()):
            stem = video_dir.name
            json_path = video_dir / f"{stem}.json"
            csv_path = video_dir / f"{stem}.csv"
            if not json_path.exists() or not csv_path.exists():
                continue
            items.append(
                {
                    "person_id": person_id,
                    "video_name": stem,
                    "video_dir": video_dir,
                    "json_path": json_path,
                    "csv_path": csv_path,
                    "cognitive_score": labels["cognitive_score"],
                    "emotion_score": labels["emotion_score"],
                }
            )
    return items


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_report_metrics(item: dict) -> dict:
    data = load_json(item["json_path"]) or {}
    overall = data.get("overall") if isinstance(data.get("overall"), dict) else {}
    affectnet_emotions = overall.get("affectnet_emotions") if isinstance(overall.get("affectnet_emotions"), dict) else {}
    affectnet_va = overall.get("affectnet_va") if isinstance(overall.get("affectnet_va"), dict) else {}
    attention = overall.get("attention") if isinstance(overall.get("attention"), dict) else {}

    emo_metrics = affectnet_emotions.get("metrics") if isinstance(affectnet_emotions.get("metrics"), dict) else {}
    emo_scores = affectnet_emotions.get("scores") if isinstance(affectnet_emotions.get("scores"), dict) else {}
    va_metrics = affectnet_va.get("metrics") if isinstance(affectnet_va.get("metrics"), dict) else {}
    va_scores = affectnet_va.get("scores") if isinstance(affectnet_va.get("scores"), dict) else {}
    att_metrics = attention.get("metrics") if isinstance(attention.get("metrics"), dict) else {}
    att_scores = attention.get("scores") if isinstance(attention.get("scores"), dict) else {}

    row = {
        "person_id": item["person_id"],
        "video_name": item["video_name"],
        "cognitive_score": item["cognitive_score"],
        "emotion_score": item["emotion_score"],
        "affect_score": safe_float(affectnet_emotions.get("score")),
        "va_score": safe_float(affectnet_va.get("score")),
        "attention_score": safe_float(attention.get("score")),
    }

    for key in [
        "tone",
        "pa",
        "na",
        "volatility",
        "switch_rate",
        "switch_count",
        "ambiguity_ratio",
        "dominance_ratio",
        "phase_change_rate",
        "phase_change_count",
        "phase_purity_ratio",
        "coverage",
        "margin_mean",
        "entropy_mean",
        "events_per_minute",
    ]:
        row[f"emo_metric__{key}"] = safe_float(emo_metrics.get(key))

    for key in [
        "tone_score",
        "confidence_score",
        "dominance_score",
        "phase_change_score",
        "phase_purity_score",
        "phase_change_raw_score",
        "deepface_process_score",
    ]:
        row[f"emo_score__{key}"] = safe_float(emo_scores.get(key))

    for key in [
        "valence_mean",
        "arousal_mean",
        "va_volatility",
        "activation_ratio",
        "coverage",
    ]:
        row[f"va_metric__{key}"] = safe_float(va_metrics.get(key))

    for key in [
        "valence_score",
        "arousal_score",
        "stability_score",
        "activation_score",
        "va_score",
        "emonet_score",
    ]:
        row[f"va_score__{key}"] = safe_float(va_scores.get(key))

    for key in [
        "total_duration_sec",
        "analysis_duration_sec",
        "valid_ratio",
        "focus_ratio",
        "screen_focus_ratio",
        "acceptable_focus_ratio",
        "off_focus_ratio",
        "stability_std",
        "reengage_time_median",
        "screen_focus_duration_median_sec",
        "screen_focus_duration_max_sec",
        "screen_focus_segment_count",
        "blink_ratio",
        "blink_count",
        "blink_count_per_min",
        "blink_excess_ratio",
    ]:
        row[f"attention_metric__{key}"] = safe_float(att_metrics.get(key))

    for key in [
        "focus_score",
        "fixation_score",
        "blink_score",
        "stability_score",
        "quality_score",
        "attention_score",
    ]:
        row[f"attention_score__{key}"] = safe_float(att_scores.get(key))

    return row


def pick_main_face(df: pd.DataFrame) -> pd.DataFrame:
    frame_df = df.copy()
    frame_df.columns = [str(col).strip() for col in frame_df.columns]
    if "success" in frame_df.columns:
        success = pd.to_numeric(frame_df["success"], errors="coerce").fillna(0)
        frame_df = frame_df[success >= 0.5].copy()
    if "confidence" in frame_df.columns:
        confidence = pd.to_numeric(frame_df["confidence"], errors="coerce").fillna(0)
        frame_df = frame_df[confidence >= 0.8].copy()
    if frame_df.empty:
        return frame_df
    if "face_id" in frame_df.columns:
        face_counts = frame_df["face_id"].value_counts()
        if not face_counts.empty:
            frame_df = frame_df[frame_df["face_id"] == face_counts.index[0]].copy()
    return frame_df


def segment_stats(active: np.ndarray, step_sec: float) -> tuple[float, float, float, float]:
    if active.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    starts = np.flatnonzero(active & np.concatenate(([True], ~active[:-1])))
    ends = np.flatnonzero(active & np.concatenate((~active[1:], [True])))
    if starts.size == 0 or ends.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    lengths = (ends - starts + 1).astype(float) * step_sec
    duration_total = float(lengths.sum())
    duration_mean = float(lengths.mean()) if lengths.size else 0.0
    duration_max = float(lengths.max()) if lengths.size else 0.0
    count = float(lengths.size)
    return count, duration_total, duration_mean, duration_max


def build_active_mask(df: pd.DataFrame, au_base: str) -> np.ndarray | None:
    c_col = f"{au_base}_c"
    r_col = f"{au_base}_r"
    if c_col in df.columns:
        values = pd.to_numeric(df[c_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        return values >= 0.5
    if r_col in df.columns:
        values = pd.to_numeric(df[r_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        return values >= 1.0
    return None


def build_strength_series(df: pd.DataFrame, au_base: str) -> np.ndarray | None:
    r_col = f"{au_base}_r"
    c_col = f"{au_base}_c"
    if r_col in df.columns:
        values = pd.to_numeric(df[r_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        return values
    if c_col in df.columns:
        values = pd.to_numeric(df[c_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        return values
    return None


def append_combo_segment_metrics(row: dict, key: str, active_mask: np.ndarray, step_sec: float, total_duration_sec: float) -> None:
    segment_count, segment_duration_total, segment_duration_mean, segment_duration_max = segment_stats(active_mask, step_sec)
    row[f"{key}__segment_count_per_min"] = float(segment_count / total_duration_sec * 60.0) if total_duration_sec > 1e-9 else 0.0
    row[f"{key}__segment_total_sec"] = float(segment_duration_total)
    row[f"{key}__segment_mean_sec"] = float(segment_duration_mean)
    row[f"{key}__segment_max_sec"] = float(segment_duration_max)


def extract_au_features(item: dict) -> dict:
    try:
        raw_df = pd.read_csv(item["csv_path"])
    except Exception:
        return {
            "person_id": item["person_id"],
            "video_name": item["video_name"],
            "cognitive_score": item["cognitive_score"],
            "emotion_score": item["emotion_score"],
        }

    df = pick_main_face(raw_df)
    row = {
        "person_id": item["person_id"],
        "video_name": item["video_name"],
        "cognitive_score": item["cognitive_score"],
        "emotion_score": item["emotion_score"],
    }
    if df.empty:
        return row

    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna().to_numpy(dtype=float)
        if ts.size >= 2:
            diffs = np.diff(ts)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            step_sec = float(np.median(diffs)) if diffs.size else 1.0 / 30.0
        else:
            step_sec = 1.0 / 30.0
    else:
        step_sec = 1.0 / 30.0
    total_duration_sec = max(float(len(df) * step_sec), 1e-9)

    au_r_cols = sorted([col for col in df.columns if str(col).startswith("AU") and str(col).endswith("_r")])
    au_c_cols = sorted([col for col in df.columns if str(col).startswith("AU") and str(col).endswith("_c")])
    au_bases = sorted(set([col[:-2] for col in au_r_cols + au_c_cols]))

    simultaneous_masks = []
    for au_base in au_bases:
        active = build_active_mask(df, au_base)
        if active is not None:
            simultaneous_masks.append(active.astype(float))
    if simultaneous_masks:
        active_matrix = np.vstack(simultaneous_masks)
        active_sum = active_matrix.sum(axis=0)
        row["au_global__simultaneous_active_mean"] = float(np.mean(active_sum))
        row["au_global__simultaneous_active_std"] = float(np.std(active_sum, ddof=0))
        row["au_global__simultaneous_active_p90"] = float(np.quantile(active_sum, 0.9))
    row["au_global__duration_sec"] = total_duration_sec

    for col in au_r_cols:
        au_base = col[:-2]
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        arr = values.to_numpy(dtype=float)
        diffs = np.diff(arr)
        active = arr >= 1.0
        segment_count, segment_duration_total, segment_duration_mean, segment_duration_max = segment_stats(active, step_sec)
        row[f"au__{au_base}__r_mean"] = float(np.mean(arr))
        row[f"au__{au_base}__r_std"] = float(np.std(arr, ddof=0))
        row[f"au__{au_base}__r_p90"] = float(np.quantile(arr, 0.9))
        row[f"au__{au_base}__r_active_ratio"] = float(np.mean(active))
        row[f"au__{au_base}__r_diff_std"] = float(np.std(diffs, ddof=0)) if diffs.size else 0.0
        row[f"au__{au_base}__r_transition_per_min"] = float(np.sum(active[1:] != active[:-1]) / total_duration_sec * 60.0) if active.size >= 2 else 0.0
        row[f"au__{au_base}__r_segment_count_per_min"] = float(segment_count / total_duration_sec * 60.0)
        row[f"au__{au_base}__r_segment_mean_sec"] = float(segment_duration_mean)
        row[f"au__{au_base}__r_segment_max_sec"] = float(segment_duration_max)
        row[f"au__{au_base}__r_energy"] = float(np.mean(arr ** 2))

    for col in au_c_cols:
        au_base = col[:-2]
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        arr = values.to_numpy(dtype=float)
        active = arr >= 0.5
        segment_count, segment_duration_total, segment_duration_mean, segment_duration_max = segment_stats(active, step_sec)
        row[f"au__{au_base}__c_mean"] = float(np.mean(arr))
        row[f"au__{au_base}__c_std"] = float(np.std(arr, ddof=0))
        row[f"au__{au_base}__c_p90"] = float(np.quantile(arr, 0.9))
        row[f"au__{au_base}__c_active_ratio"] = float(np.mean(active))
        row[f"au__{au_base}__c_transition_per_min"] = float(np.sum(active[1:] != active[:-1]) / total_duration_sec * 60.0) if active.size >= 2 else 0.0
        row[f"au__{au_base}__c_segment_count_per_min"] = float(segment_count / total_duration_sec * 60.0)
        row[f"au__{au_base}__c_segment_mean_sec"] = float(segment_duration_mean)
        row[f"au__{au_base}__c_segment_max_sec"] = float(segment_duration_max)

    pair_candidates = sorted([au for au in au_bases if build_active_mask(df, au) is not None])
    active_cache = {au: build_active_mask(df, au) for au in pair_candidates}
    strength_cache = {au: build_strength_series(df, au) for au in pair_candidates}
    for au_a, au_b in itertools.combinations(pair_candidates, 2):
        mask_a = active_cache.get(au_a)
        mask_b = active_cache.get(au_b)
        if mask_a is None or mask_b is None:
            continue
        union = mask_a | mask_b
        inter = mask_a & mask_b
        pair_key = f"au_pair__{au_a}__{au_b}"
        row[f"{pair_key}__coactive_ratio"] = float(np.mean(inter))
        row[f"{pair_key}__jaccard"] = float(np.mean(inter) / np.mean(union)) if np.mean(union) > 1e-9 else 0.0
        append_combo_segment_metrics(row, pair_key, inter, step_sec, total_duration_sec)
        strength_a = strength_cache.get(au_a)
        strength_b = strength_cache.get(au_b)
        if strength_a is not None and strength_b is not None and np.any(inter):
            pair_sum = strength_a + strength_b
            row[f"{pair_key}__strength_mean"] = float(np.mean(pair_sum[inter]))
            row[f"{pair_key}__strength_p90"] = float(np.quantile(pair_sum[inter], 0.9))
            row[f"{pair_key}__strength_std"] = float(np.std(pair_sum[inter], ddof=0))

    for au_a, au_b, au_c in itertools.combinations(pair_candidates, 3):
        mask_a = active_cache.get(au_a)
        mask_b = active_cache.get(au_b)
        mask_c = active_cache.get(au_c)
        if mask_a is None or mask_b is None or mask_c is None:
            continue
        inter3 = mask_a & mask_b & mask_c
        union3 = mask_a | mask_b | mask_c
        key = f"au_tri__{au_a}__{au_b}__{au_c}"
        row[f"{key}__coactive_ratio"] = float(np.mean(inter3))
        row[f"{key}__jaccard"] = float(np.mean(inter3) / np.mean(union3)) if np.mean(union3) > 1e-9 else 0.0
        append_combo_segment_metrics(row, key, inter3, step_sec, total_duration_sec)
        strength_a = strength_cache.get(au_a)
        strength_b = strength_cache.get(au_b)
        strength_c = strength_cache.get(au_c)
        if strength_a is not None and strength_b is not None and strength_c is not None and np.any(inter3):
            tri_sum = strength_a + strength_b + strength_c
            row[f"{key}__strength_mean"] = float(np.mean(tri_sum[inter3]))
            row[f"{key}__strength_p90"] = float(np.quantile(tri_sum[inter3], 0.9))
            row[f"{key}__strength_std"] = float(np.std(tri_sum[inter3], ddof=0))

    for emotion_name, group_aus in EMOTION_AU_GROUPS.items():
        available = [au for au in group_aus if au in active_cache and active_cache.get(au) is not None]
        if not available:
            continue
        active_stack = np.vstack([active_cache[au].astype(float) for au in available])
        active_bool = active_stack > 0.5
        row[f"emotion_proto__{emotion_name}__available_ratio"] = float(len(available) / len(group_aus))
        row[f"emotion_proto__{emotion_name}__any_active_ratio"] = float(np.mean(np.any(active_bool, axis=0)))
        row[f"emotion_proto__{emotion_name}__all_active_ratio"] = float(np.mean(np.all(active_bool, axis=0)))
        active_count = np.sum(active_bool, axis=0)
        row[f"emotion_proto__{emotion_name}__mean_active_count"] = float(np.mean(active_count))
        row[f"emotion_proto__{emotion_name}__active_count_p90"] = float(np.quantile(active_count, 0.9))
        full_mask = np.all(active_bool, axis=0)
        full_count, full_total_sec, full_mean_sec, full_max_sec = segment_stats(full_mask, step_sec)
        row[f"emotion_proto__{emotion_name}__full_pattern_count_per_min"] = float(full_count / total_duration_sec * 60.0)
        row[f"emotion_proto__{emotion_name}__full_pattern_total_sec"] = float(full_total_sec)
        row[f"emotion_proto__{emotion_name}__full_pattern_mean_sec"] = float(full_mean_sec)
        row[f"emotion_proto__{emotion_name}__full_pattern_max_sec"] = float(full_max_sec)

        strengths = [strength_cache[au] for au in available if strength_cache.get(au) is not None]
        if strengths:
            strength_stack = np.vstack(strengths)
            strength_frame_sum = np.sum(strength_stack, axis=0)
            row[f"emotion_proto__{emotion_name}__strength_mean"] = float(np.mean(strength_frame_sum[full_mask])) if np.any(full_mask) else 0.0
            row[f"emotion_proto__{emotion_name}__strength_p90"] = float(np.quantile(strength_frame_sum[full_mask], 0.9)) if np.any(full_mask) else 0.0
            row[f"emotion_proto__{emotion_name}__strength_std"] = float(np.std(strength_frame_sum[full_mask], ddof=0)) if np.any(full_mask) else 0.0
            row[f"emotion_proto__{emotion_name}__coactive_strength_mean"] = row[f"emotion_proto__{emotion_name}__strength_mean"]

        if len(available) >= 2:
            overlaps = []
            jaccards = []
            for au_a, au_b in itertools.combinations(available, 2):
                ma = active_cache[au_a]
                mb = active_cache[au_b]
                inter = np.mean(ma & mb)
                union = np.mean(ma | mb)
                overlaps.append(float(inter))
                jaccards.append(float(inter / union) if union > 1e-9 else 0.0)
            row[f"emotion_proto__{emotion_name}__pair_overlap_mean"] = float(np.mean(overlaps))
            row[f"emotion_proto__{emotion_name}__pair_jaccard_mean"] = float(np.mean(jaccards))

    if "gaze_angle_x" in df.columns and "gaze_angle_y" in df.columns:
        gx = pd.to_numeric(df["gaze_angle_x"], errors="coerce")
        gy = pd.to_numeric(df["gaze_angle_y"], errors="coerce")
        gaze = np.sqrt(gx.to_numpy(dtype=float) ** 2 + gy.to_numpy(dtype=float) ** 2)
        gaze = gaze[np.isfinite(gaze)]
        if gaze.size:
            row["gaze__mag_mean"] = float(np.mean(gaze))
            row["gaze__mag_std"] = float(np.std(gaze, ddof=0))
            row["gaze__mag_p90"] = float(np.quantile(gaze, 0.9))

    return row


def aggregate_by_person(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in df.columns if col not in {"person_id", "video_name"}]
    grouped = df.groupby("person_id", as_index=False)[numeric_cols].mean(numeric_only=True)
    grouped["video_count"] = df.groupby("person_id").size().values
    return grouped


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(arr: np.ndarray) -> np.ndarray:
    return pd.Series(arr).rank(method="average").to_numpy(dtype=float)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(x), rankdata(y))


def pearson_fast(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan"), float("nan")
    corr, pvalue = stats.pearsonr(x, y)
    return float(corr), float(pvalue)


def spearman_fast(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan"), float("nan")
    corr, pvalue = stats.spearmanr(x, y)
    return float(corr), float(pvalue)


def permutation_pvalue(x: np.ndarray, y: np.ndarray, method: str, iterations: int = 8000) -> float:
    if method == "pearson":
        observed = abs(pearson_corr(x, y))
    else:
        observed = abs(spearman_corr(x, y))
    if not math.isfinite(observed):
        return float("nan")
    hits = 0
    total = 0
    for _ in range(iterations):
        y_perm = RNG.permutation(y)
        value = abs(pearson_corr(x, y_perm) if method == "pearson" else spearman_corr(x, y_perm))
        if math.isfinite(value) and value >= observed - 1e-12:
            hits += 1
        total += 1
    return float((hits + 1) / (total + 1))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, method: str, iterations: int = 4000) -> tuple[float | None, float | None]:
    if x.size < 4:
        return None, None
    values = []
    idx = np.arange(x.size)
    for _ in range(iterations):
        sample_idx = RNG.choice(idx, size=idx.size, replace=True)
        xs = x[sample_idx]
        ys = y[sample_idx]
        value = pearson_corr(xs, ys) if method == "pearson" else spearman_corr(xs, ys)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None, None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def fdr_bh(pvalues: list[float | None]) -> list[float | None]:
    indexed = [(idx, p) for idx, p in enumerate(pvalues) if p is not None and math.isfinite(p)]
    result: list[float | None] = [None] * len(pvalues)
    if not indexed:
        return result
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    adjusted = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx, pvalue = indexed[rank]
        value = min(prev, pvalue * m / (rank + 1))
        adjusted[rank] = value
        prev = value
    for rank, (idx, _) in enumerate(indexed):
        result[idx] = float(min(max(adjusted[rank], 0.0), 1.0))
    return result


def run_correlation_table(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    domain: str,
    heavy_eval_top_n: int = 80,
) -> pd.DataFrame:
    rows: list[dict] = []
    for feature_col in feature_cols:
        series = pd.to_numeric(df[feature_col], errors="coerce")
        target = pd.to_numeric(df[target_col], errors="coerce")
        valid = series.notna() & target.notna()
        valid_count = int(valid.sum())
        if valid_count < 5:
            continue
        x = series[valid].to_numpy(dtype=float)
        y = target[valid].to_numpy(dtype=float)
        pearson, pearson_p_analytic = pearson_fast(x, y)
        spearman, spearman_p_analytic = spearman_fast(x, y)
        rows.append(
            {
                "domain": domain,
                "target": target_col,
                "feature": feature_col,
                "n": valid_count,
                "pearson_r": pearson,
                "pearson_p_analytic": pearson_p_analytic,
                "pearson_perm_p": None,
                "pearson_ci_low": None,
                "pearson_ci_high": None,
                "spearman_r": spearman,
                "spearman_p_analytic": spearman_p_analytic,
                "spearman_perm_p": None,
                "spearman_ci_low": None,
                "spearman_ci_high": None,
                "abs_max_r": max(abs(pearson) if math.isfinite(pearson) else 0.0, abs(spearman) if math.isfinite(spearman) else 0.0),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["pearson_fdr_analytic"] = fdr_bh(result["pearson_p_analytic"].tolist())
    result["spearman_fdr_analytic"] = fdr_bh(result["spearman_p_analytic"].tolist())
    result.sort_values(by=["abs_max_r", "spearman_p_analytic", "pearson_p_analytic"], ascending=[False, True, True], inplace=True)

    selected_index = result.head(min(heavy_eval_top_n, len(result))).index.tolist()
    for idx in selected_index:
        feature_col = str(result.at[idx, "feature"])
        series = pd.to_numeric(df[feature_col], errors="coerce")
        target = pd.to_numeric(df[target_col], errors="coerce")
        valid = series.notna() & target.notna()
        x = series[valid].to_numpy(dtype=float)
        y = target[valid].to_numpy(dtype=float)
        result.at[idx, "pearson_perm_p"] = permutation_pvalue(x, y, method="pearson", iterations=2500)
        result.at[idx, "spearman_perm_p"] = permutation_pvalue(x, y, method="spearman", iterations=2500)
        pearson_ci_low, pearson_ci_high = bootstrap_ci(x, y, method="pearson", iterations=1500)
        spearman_ci_low, spearman_ci_high = bootstrap_ci(x, y, method="spearman", iterations=1500)
        result.at[idx, "pearson_ci_low"] = pearson_ci_low
        result.at[idx, "pearson_ci_high"] = pearson_ci_high
        result.at[idx, "spearman_ci_low"] = spearman_ci_low
        result.at[idx, "spearman_ci_high"] = spearman_ci_high

    result["pearson_fdr"] = fdr_bh(result["pearson_perm_p"].tolist())
    result["spearman_fdr"] = fdr_bh(result["spearman_perm_p"].tolist())
    result.sort_values(by=["abs_max_r", "spearman_perm_p", "pearson_perm_p", "spearman_p_analytic"], ascending=[False, True, True, True], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def write_table(df: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_ROOT / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def summarize_top(corr_df: pd.DataFrame, top_n: int = 12) -> list[dict]:
    if corr_df.empty:
        return []
    columns = [
        "feature",
        "pearson_r",
        "pearson_perm_p",
        "pearson_fdr",
        "spearman_r",
        "spearman_perm_p",
        "spearman_fdr",
    ]
    rows = corr_df.loc[:, columns].head(top_n).to_dict(orient="records")
    return rows


def build_markdown_report(
    items: list[dict],
    participant_report_df: pd.DataFrame,
    participant_au_df: pd.DataFrame,
    expr_cog_df: pd.DataFrame,
    expr_emo_df: pd.DataFrame,
    proto_cog_df: pd.DataFrame,
    proto_emo_df: pd.DataFrame,
    au_cog_df: pd.DataFrame,
    au_emo_df: pd.DataFrame,
) -> str:
    lines = [
        "# 认知与情绪关联探索报告",
        "",
        "## 数据范围",
        f"- 受试者数量: {len(participant_report_df)}",
        f"- 视频数量: {len(items)}",
        f"- 报表根目录: `{REPORT_ROOT}`",
        "- 主要分析层级: 受试者级平均（先对同一受试者多个视频求平均，再与标签做相关性分析）",
        "- 认知分按用户说明视为数值越高越好；情绪分当前仅按原始数值做探索性关联，不预设高低好坏方向",
        "",
        "## 方法",
        "- 表情分析相关部分: 提取 `affectnet_emotions`、`affectnet_va`、`attention` 中的分数和关键指标",
        "- AU 分析部分: 从 OpenFace CSV 提取 AU 强度、出现比例、波动、转移频率、持续时长、双AU组合、三AU组合、共激活强度等特征",
        "- 统计检验: Pearson、Spearman 相关；置换检验 p 值；bootstrap 95% CI；FDR 做多重比较校正",
        "- 注意: 样本量仅 12 名受试者，因此结论应视为探索性线索，不应直接视为稳定因果结论",
        "",
        "## 受试者级概况",
    ]

    preview_cols = [
        "person_id",
        "cognitive_score",
        "emotion_score",
        "affect_score",
        "va_score",
        "attention_score",
        "attention_metric__focus_ratio",
        "attention_metric__screen_focus_ratio",
        "attention_metric__reengage_time_median",
        "attention_metric__screen_focus_duration_median_sec",
        "attention_metric__blink_ratio",
    ]
    preview_df = participant_report_df.loc[:, [col for col in preview_cols if col in participant_report_df.columns]].copy()
    lines.append("")
    lines.append("```text")
    lines.append(preview_df.round(4).to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## 表情分析指标与认知分")
    for row in summarize_top(expr_cog_df, top_n=10):
        lines.append(
            f"- `{row['feature']}`: Pearson={row['pearson_r']:.3f}, Spearman={row['spearman_r']:.3f}, "
            f"perm_p=({row['pearson_perm_p']:.4f}, {row['spearman_perm_p']:.4f})"
        )
    lines.append("")

    lines.append("## 表情分析指标与情绪分")
    for row in summarize_top(expr_emo_df, top_n=10):
        lines.append(
            f"- `{row['feature']}`: Pearson={row['pearson_r']:.3f}, Spearman={row['spearman_r']:.3f}, "
            f"perm_p=({row['pearson_perm_p']:.4f}, {row['spearman_perm_p']:.4f})"
        )
    lines.append("")

    lines.append("## 典型情绪 AU 组合与认知分")
    for row in summarize_top(proto_cog_df, top_n=10):
        lines.append(
            f"- `{row['feature']}`: Pearson={row['pearson_r']:.3f}, Spearman={row['spearman_r']:.3f}, "
            f"perm_p=({row['pearson_perm_p']:.4f}, {row['spearman_perm_p']:.4f})"
        )
    lines.append("")

    lines.append("## 典型情绪 AU 组合与情绪分")
    for row in summarize_top(proto_emo_df, top_n=10):
        lines.append(
            f"- `{row['feature']}`: Pearson={row['pearson_r']:.3f}, Spearman={row['spearman_r']:.3f}, "
            f"perm_p=({row['pearson_perm_p']:.4f}, {row['spearman_perm_p']:.4f})"
        )
    lines.append("")

    lines.append("## AU 特征与认知分")
    for row in summarize_top(au_cog_df, top_n=15):
        lines.append(
            f"- `{row['feature']}`: Pearson={row['pearson_r']:.3f}, Spearman={row['spearman_r']:.3f}, "
            f"perm_p=({row['pearson_perm_p']:.4f}, {row['spearman_perm_p']:.4f})"
        )
    lines.append("")

    lines.append("## AU 特征与情绪分")
    for row in summarize_top(au_emo_df, top_n=15):
        lines.append(
            f"- `{row['feature']}`: Pearson={row['pearson_r']:.3f}, Spearman={row['spearman_r']:.3f}, "
            f"perm_p=({row['pearson_perm_p']:.4f}, {row['spearman_perm_p']:.4f})"
        )
    lines.append("")

    lines.extend(
        [
            "## 初步判断",
            "- 认知分方面，若干子指标通常比总分更敏感，尤其是表情波动、恢复时间、主导性、以及某些嘴部/紧张相关 AU 特征。",
            "- 情绪分方面，整体分数相关性偏弱，但 AU 的切换频率、片段频率、嘴部开合、下颌活动以及高阶共激活特征值得继续关注。",
            "- 典型情绪 AU 原型组合的相关性可作为补充证据，用于判断快乐、悲伤、愤怒、恐惧、惊讶、厌恶、轻蔑等模式是否与标签方向一致。",
            "- 由于样本量小且同一受试者视频数量不均衡，建议后续继续增加样本，或对任务阶段做更细分分析。",
            "- 更稳妥的下一步通常是：筛选少量高相关特征，做受试者级可视化、留一法预测、以及按任务片段重新提特征。",
            "",
            "## 结果文件",
            f"- 表情指标逐视频: `{OUTPUT_ROOT / 'expression_metrics_video_level.csv'}`",
            f"- AU 特征逐视频: `{OUTPUT_ROOT / 'au_features_video_level.csv'}`",
            f"- 表情指标受试者级: `{OUTPUT_ROOT / 'expression_metrics_subject_level.csv'}`",
            f"- AU 特征受试者级: `{OUTPUT_ROOT / 'au_features_subject_level.csv'}`",
            f"- 表情-认知相关性: `{OUTPUT_ROOT / 'correlation_expression_vs_cognitive.csv'}`",
            f"- 表情-情绪相关性: `{OUTPUT_ROOT / 'correlation_expression_vs_emotion.csv'}`",
            f"- 情绪原型-认知相关性: `{OUTPUT_ROOT / 'correlation_emotion_prototypes_vs_cognitive.csv'}`",
            f"- 情绪原型-情绪相关性: `{OUTPUT_ROOT / 'correlation_emotion_prototypes_vs_emotion.csv'}`",
            f"- AU-认知相关性: `{OUTPUT_ROOT / 'correlation_au_vs_cognitive.csv'}`",
            f"- AU-情绪相关性: `{OUTPUT_ROOT / 'correlation_au_vs_emotion.csv'}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    ensure_output_dir()
    items = list_report_items()
    expression_rows = [extract_report_metrics(item) for item in items]
    au_rows = [extract_au_features(item) for item in items]

    expression_video_df = pd.DataFrame(expression_rows)
    au_video_df = pd.DataFrame(au_rows)
    expression_subject_df = aggregate_by_person(expression_video_df)
    au_subject_df = aggregate_by_person(au_video_df)

    expression_feature_cols = [
        col for col in expression_subject_df.columns
        if col not in {"person_id", "video_name", "cognitive_score", "emotion_score", "video_count"}
    ]
    au_feature_cols = [
        col for col in au_subject_df.columns
        if col not in {"person_id", "video_name", "cognitive_score", "emotion_score", "video_count"}
    ]
    prototype_feature_cols = [col for col in au_feature_cols if col.startswith("emotion_proto__")]
    generic_au_feature_cols = [col for col in au_feature_cols if not col.startswith("emotion_proto__")]

    expr_cog_df = run_correlation_table(expression_subject_df, "cognitive_score", expression_feature_cols, domain="expression")
    expr_emo_df = run_correlation_table(expression_subject_df, "emotion_score", expression_feature_cols, domain="expression")
    proto_cog_df = run_correlation_table(au_subject_df, "cognitive_score", prototype_feature_cols, domain="emotion_prototype")
    proto_emo_df = run_correlation_table(au_subject_df, "emotion_score", prototype_feature_cols, domain="emotion_prototype")
    au_cog_df = run_correlation_table(au_subject_df, "cognitive_score", generic_au_feature_cols, domain="au")
    au_emo_df = run_correlation_table(au_subject_df, "emotion_score", generic_au_feature_cols, domain="au")

    write_table(expression_video_df, "expression_metrics_video_level.csv")
    write_table(au_video_df, "au_features_video_level.csv")
    write_table(expression_subject_df, "expression_metrics_subject_level.csv")
    write_table(au_subject_df, "au_features_subject_level.csv")
    write_table(expr_cog_df, "correlation_expression_vs_cognitive.csv")
    write_table(expr_emo_df, "correlation_expression_vs_emotion.csv")
    write_table(proto_cog_df, "correlation_emotion_prototypes_vs_cognitive.csv")
    write_table(proto_emo_df, "correlation_emotion_prototypes_vs_emotion.csv")
    write_table(au_cog_df, "correlation_au_vs_cognitive.csv")
    write_table(au_emo_df, "correlation_au_vs_emotion.csv")

    report_text = build_markdown_report(
        items=items,
        participant_report_df=expression_subject_df,
        participant_au_df=au_subject_df,
        expr_cog_df=expr_cog_df,
        expr_emo_df=expr_emo_df,
        proto_cog_df=proto_cog_df,
        proto_emo_df=proto_emo_df,
        au_cog_df=au_cog_df,
        au_emo_df=au_emo_df,
    )
    report_path = OUTPUT_ROOT / "analysis_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"[INFO] videos={len(items)}")
    print(f"[INFO] subjects={len(expression_subject_df)}")
    print(f"[INFO] output={OUTPUT_ROOT}")
    print(f"[INFO] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
