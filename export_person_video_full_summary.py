from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\Administrator\Desktop\seriousgame_eval\data\reports\association_analysis")
OUT_MD = ROOT / "每人每次视频_指标与关键AU汇总.md"
OUT_CSV = ROOT / "每人每次视频_指标与关键AU汇总.csv"


BASE_METRICS = [
    ("cognitive_score", "认知分"),
    ("emotion_score", "情绪分"),
    ("attention_score", "注意力总分"),
    ("attention_metric__focus_ratio", "高专注占比"),
    ("attention_metric__acceptable_focus_ratio", "注视屏幕占比"),
    ("attention_metric__screen_focus_duration_max_sec", "连续注视时长"),
    ("attention_metric__blink_ratio", "眨眼时长占比"),
    ("attention_metric__blink_count_per_min", "每分钟眨眼次数"),
    ("affect_score", "宏表情总分"),
    ("emo_metric__tone", "情绪基调"),
    ("emo_metric__pa", "正性情绪倾向"),
    ("emo_metric__na", "负性情绪倾向"),
    ("emo_metric__volatility", "情绪波动度"),
    ("emo_metric__switch_rate", "情绪切换频率"),
    ("emo_metric__dominance_ratio", "主导情绪占比"),
    ("va_score", "VA综合分"),
    ("va_metric__valence_mean", "平均愉悦度"),
    ("va_metric__arousal_mean", "平均唤醒度"),
    ("va_metric__va_volatility", "VA波动度"),
]

KEY_SINGLE_AU = [
    ("au__AU17__c_segment_max_sec", "AU17最长持续时间"),
    ("au__AU28__c_active_ratio", "AU28激活占比"),
    ("au__AU28__c_mean", "AU28平均激活"),
    ("au__AU01__c_transition_per_min", "AU01切换频率"),
    ("au__AU17__c_transition_per_min", "AU17切换频率"),
    ("au__AU23__r_mean", "AU23平均强度"),
]

KEY_MULTI_AU = [
    ("au_pair__AU15__AU28__segment_mean_sec", "AU15|AU28共激活平均持续时间"),
    ("au_pair__AU15__AU28__coactive_ratio", "AU15|AU28共激活比例"),
    ("au_tri__AU15__AU17__AU28__segment_mean_sec", "AU15|AU17|AU28共激活平均持续时间"),
    ("au_tri__AU15__AU23__AU28__segment_total_sec", "AU15|AU23|AU28共激活总时长"),
    ("au_tri__AU04__AU23__AU25__coactive_ratio", "AU04|AU23|AU25共激活比例"),
    ("au_tri__AU26__AU28__AU45__segment_mean_sec", "AU26|AU28|AU45共激活平均持续时间"),
    ("au_pair__AU01__AU45__strength_mean", "AU01|AU45共激活强度"),
]

KEY_PROTOTYPES = [
    ("emotion_proto__happiness__pair_jaccard_mean", "happiness重叠系数"),
    ("emotion_proto__happiness__full_pattern_total_sec", "happiness完整模式总时长"),
    ("emotion_proto__sadness__strength_mean", "sadness共激活强度"),
    ("emotion_proto__anger__strength_mean", "anger共激活强度"),
]


def format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def make_markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(format_value(v) for v in row.tolist()) + " |")
    return "\n".join([header, sep] + rows)


def main() -> int:
    expr = pd.read_csv(ROOT / "expression_metrics_video_level.csv")
    au = pd.read_csv(ROOT / "au_features_video_level.csv")
    merged = expr.merge(au, on=["person_id", "video_name", "cognitive_score", "emotion_score"], how="left")

    selected = [("person_id", "编号"), ("video_name", "视频名")] + BASE_METRICS + KEY_SINGLE_AU + KEY_MULTI_AU + KEY_PROTOTYPES
    cols = [src for src, _ in selected if src in merged.columns]
    table = merged.loc[:, cols].copy()
    rename_map = {src: dst for src, dst in selected if src in table.columns}
    table.rename(columns=rename_map, inplace=True)
    table.sort_values(by=["编号", "视频名"], inplace=True)
    table.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = [
        "# 每人每次视频：指标与关键AU汇总",
        "",
        "包含内容：",
        "- 认知分、情绪分",
        "- 注意力核心指标",
        "- 宏表情与VA核心指标",
        "- 关键单AU特征",
        "- 关键多AU组合特征",
        "- 关键表情原型组合特征",
        "",
    ]

    for pid, group in table.groupby("编号"):
        lines.append(f"## 编号 {pid}")
        lines.append("")
        lines.append(make_markdown_table(group.reset_index(drop=True)))
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] markdown={OUT_MD}")
    print(f"[INFO] csv={OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
