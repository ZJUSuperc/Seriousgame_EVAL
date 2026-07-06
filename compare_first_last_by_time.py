from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\Administrator\Desktop\seriousgame_eval\data\reports")
OUT = ROOT / "_analysis_tmp"
SUBJECT_IDS = [str(x) for x in range(1901, 1913)]

DATE_RE = re.compile(r"(\d{8})")
COPY_RE = re.compile(r"\((\d+)\)")


def parse_video_order(name: str) -> tuple[str, int, str]:
    date_match = DATE_RE.search(name)
    date_text = date_match.group(1) if date_match else "00000000"
    copy_match = COPY_RE.search(name)
    copy_index = int(copy_match.group(1)) if copy_match else 0
    return date_text, copy_index, name


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_report(video_dir: Path, person_id: str) -> dict:
    stem = video_dir.name
    data = load_json(video_dir / f"{stem}.json")
    overall = data.get("overall") if isinstance(data.get("overall"), dict) else {}
    attention = overall.get("attention") if isinstance(overall.get("attention"), dict) else {}
    affect = overall.get("affectnet_emotions") if isinstance(overall.get("affectnet_emotions"), dict) else {}
    va = overall.get("affectnet_va") if isinstance(overall.get("affectnet_va"), dict) else {}
    attm = attention.get("metrics") if isinstance(attention.get("metrics"), dict) else {}
    atts = attention.get("scores") if isinstance(attention.get("scores"), dict) else {}
    afm = affect.get("metrics") if isinstance(affect.get("metrics"), dict) else {}
    vam = va.get("metrics") if isinstance(va.get("metrics"), dict) else {}
    row = {
        "person_id": person_id,
        "video_name": stem,
        "attention_score": attention.get("score"),
        "affect_score": affect.get("score"),
        "va_score": va.get("score"),
    }
    for key, value in attm.items():
        row[f"attention_metric__{key}"] = value
    for key, value in atts.items():
        row[f"attention_score__{key}"] = value
    for key, value in afm.items():
        row[f"emo_metric__{key}"] = value
    for key, value in vam.items():
        row[f"va_metric__{key}"] = value
    return row


def build_subject_video_rows() -> pd.DataFrame:
    rows = []
    for person_id in SUBJECT_IDS:
        person_dir = ROOT / person_id
        if not person_dir.exists():
            continue
        video_dirs = sorted([p for p in person_dir.iterdir() if p.is_dir()], key=lambda p: parse_video_order(p.name))
        for video_dir in video_dirs:
            rows.append(flatten_report(video_dir, person_id))
    return pd.DataFrame(rows)


def build_first_last_summary(video_df: pd.DataFrame, au_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    au_rows = []
    metric_cols = [c for c in video_df.columns if c not in {"person_id", "video_name"}]
    au_cols = [c for c in au_df.columns if c not in {"person_id", "video_name", "cognitive_score", "emotion_score"}]

    for person_id in SUBJECT_IDS:
        sub_v = video_df[video_df["person_id"] == person_id].copy()
        sub_a = au_df[au_df["person_id"] == person_id].copy()
        if sub_v.empty or sub_a.empty:
            continue
        sub_v.sort_values(by="video_name", key=lambda s: s.map(lambda x: parse_video_order(str(x))), inplace=True)
        sub_a.sort_values(by="video_name", key=lambda s: s.map(lambda x: parse_video_order(str(x))), inplace=True)

        first_v = sub_v.iloc[0]
        last_v = sub_v.iloc[-1]
        first_a = sub_a.iloc[0]
        last_a = sub_a.iloc[-1]

        metric_item = {
            "person_id": person_id,
            "first_video": first_v["video_name"],
            "last_video": last_v["video_name"],
        }
        for col in metric_cols:
            f = pd.to_numeric(pd.Series([first_v[col]]), errors="coerce").iloc[0]
            l = pd.to_numeric(pd.Series([last_v[col]]), errors="coerce").iloc[0]
            metric_item[f"{col}_first"] = f
            metric_item[f"{col}_last"] = l
            metric_item[f"{col}_delta"] = l - f if pd.notna(f) and pd.notna(l) else None
        metric_rows.append(metric_item)

        au_item = {
            "person_id": person_id,
            "first_video": first_a["video_name"],
            "last_video": last_a["video_name"],
        }
        for col in au_cols:
            f = pd.to_numeric(pd.Series([first_a[col]]), errors="coerce").iloc[0]
            l = pd.to_numeric(pd.Series([last_a[col]]), errors="coerce").iloc[0]
            au_item[f"{col}_first"] = f
            au_item[f"{col}_last"] = l
            au_item[f"{col}_delta"] = l - f if pd.notna(f) and pd.notna(l) else None
        au_rows.append(au_item)

    return pd.DataFrame(metric_rows), pd.DataFrame(au_rows)


def summarize_top_au_deltas(au_compare: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    rows = []
    delta_cols = [c for c in au_compare.columns if c.endswith("_delta")]
    for _, row in au_compare.iterrows():
        person_id = row["person_id"]
        values = []
        for col in delta_cols:
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            values.append((col[:-6], float(value), abs(float(value))))
        values.sort(key=lambda item: item[2], reverse=True)
        for feature, delta, abs_delta in values[:top_n]:
            rows.append({
                "person_id": person_id,
                "feature": feature,
                "delta": delta,
                "abs_delta": abs_delta,
            })
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    video_df = build_subject_video_rows()
    au_df = pd.read_csv(ROOT / "association_analysis" / "au_features_video_level.csv")

    metric_compare, au_compare = build_first_last_summary(video_df, au_df)
    top_au = summarize_top_au_deltas(au_compare, top_n=15)

    metric_path = OUT / "first_last_video_metrics_by_time.csv"
    au_path = OUT / "first_last_video_au_by_time.csv"
    top_au_path = OUT / "first_last_video_top_au_changes.csv"
    metric_compare.to_csv(metric_path, index=False, encoding="utf-8-sig")
    au_compare.to_csv(au_path, index=False, encoding="utf-8-sig")
    top_au.to_csv(top_au_path, index=False, encoding="utf-8-sig")

    preview_cols = [
        "person_id",
        "first_video",
        "last_video",
        "attention_score_delta",
        "affect_score_delta",
        "va_score_delta",
        "attention_metric__screen_focus_duration_max_sec_delta",
        "attention_metric__blink_ratio_delta",
        "emo_metric__tone_delta",
        "emo_metric__pa_delta",
        "emo_metric__na_delta",
        "va_metric__valence_mean_delta",
    ]
    print(metric_compare.loc[:, [c for c in preview_cols if c in metric_compare.columns]].round(4).to_string(index=False))
    print(f"\nmetrics={metric_path}")
    print(f"au={au_path}")
    print(f"top_au={top_au_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
