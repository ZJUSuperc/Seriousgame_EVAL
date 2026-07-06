from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\Administrator\Desktop\seriousgame_eval\data\reports\association_analysis")
OUT_MD = ROOT / "宏表情指标_按人按视频汇总.md"
OUT_CSV = ROOT / "宏表情指标_按人按视频汇总.csv"


METRICS = [
    ("affect_score", "宏表情总分", "整体宏表情分析综合分"),
    ("emo_metric__tone", "情绪基调", "整体情绪倾向，正负方向综合"),
    ("emo_metric__pa", "正性情绪倾向", "正向情绪活跃程度"),
    ("emo_metric__na", "负性情绪倾向", "负向情绪活跃程度"),
    ("emo_metric__volatility", "情绪波动度", "情绪状态变化幅度"),
    ("emo_metric__switch_rate", "情绪切换频率", "单位时间内情绪切换速度"),
    ("emo_metric__switch_count", "情绪切换次数", "整段视频内情绪切换总次数"),
    ("emo_metric__ambiguity_ratio", "情绪模糊比例", "情绪不明确的比例"),
    ("emo_metric__dominance_ratio", "主导情绪占比", "最主要情绪所占比例"),
    ("emo_metric__phase_change_rate", "阶段变化频率", "情绪阶段切换速度"),
    ("emo_metric__phase_change_count", "阶段变化次数", "情绪阶段切换总次数"),
    ("emo_metric__phase_purity_ratio", "阶段纯度", "单一情绪阶段的纯净程度"),
    ("emo_metric__margin_mean", "分类边际均值", "主情绪相对次情绪的区分度"),
    ("emo_metric__entropy_mean", "情绪熵均值", "情绪分布混乱程度"),
    ("emo_metric__events_per_minute", "情绪事件频率", "每分钟情绪事件数"),
    ("va_score", "VA综合分", "valence-arousal 综合评分"),
    ("va_metric__valence_mean", "平均愉悦度", "valence 平均值"),
    ("va_metric__arousal_mean", "平均唤醒度", "arousal 平均值"),
    ("va_metric__va_volatility", "VA波动度", "valence/arousal 变化幅度"),
    ("va_metric__activation_ratio", "高激活比例", "较高唤醒状态占比"),
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    video_df = pd.read_csv(ROOT / "expression_metrics_video_level.csv")
    cog_df = pd.read_csv(ROOT / "correlation_expression_vs_cognitive.csv")
    emo_df = pd.read_csv(ROOT / "correlation_expression_vs_emotion.csv")
    return video_df, cog_df, emo_df


def build_metric_summary(cog_df: pd.DataFrame, emo_df: pd.DataFrame) -> pd.DataFrame:
    cog_map = cog_df.set_index("feature")
    emo_map = emo_df.set_index("feature")
    rows = []
    for key, cn, desc in METRICS:
        row = {
            "字段名": key,
            "中文名": cn,
            "含义": desc,
            "认知_Pearson": None,
            "认知_Spearman": None,
            "认知_perm_p": None,
            "情绪_Pearson": None,
            "情绪_Spearman": None,
            "情绪_perm_p": None,
        }
        if key in cog_map.index:
            row["认知_Pearson"] = float(cog_map.at[key, "pearson_r"])
            row["认知_Spearman"] = float(cog_map.at[key, "spearman_r"])
            row["认知_perm_p"] = float(cog_map.at[key, "pearson_perm_p"]) if pd.notna(cog_map.at[key, "pearson_perm_p"]) else None
        if key in emo_map.index:
            row["情绪_Pearson"] = float(emo_map.at[key, "pearson_r"])
            row["情绪_Spearman"] = float(emo_map.at[key, "spearman_r"])
            row["情绪_perm_p"] = float(emo_map.at[key, "pearson_perm_p"]) if pd.notna(emo_map.at[key, "pearson_perm_p"]) else None
        rows.append(row)
    return pd.DataFrame(rows)


def build_video_table(video_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["person_id", "video_name", "cognitive_score", "emotion_score"] + [key for key, _, _ in METRICS]
    out = video_df.loc[:, keep_cols].copy()
    rename_map = {key: cn for key, cn, _ in METRICS}
    rename_map.update({
        "person_id": "编号",
        "video_name": "视频名",
        "cognitive_score": "认知分",
        "emotion_score": "情绪分",
    })
    out.rename(columns=rename_map, inplace=True)
    return out


def to_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    part = df if max_rows is None else df.head(max_rows)
    header = "| " + " | ".join(str(col) for col in part.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(part.columns)) + " |"
    rows = []
    for _, row in part.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep] + rows)


def main() -> int:
    video_df, cog_df, emo_df = load_inputs()
    metric_summary = build_metric_summary(cog_df, emo_df)
    video_table = build_video_table(video_df)
    video_table.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("# 宏表情指标中文说明、相关性与按人按视频结果")
    lines.append("")
    lines.append("## 1. 宏表情指标中文说明与相关性")
    lines.append("")
    lines.append(to_markdown_table(metric_summary))
    lines.append("")
    lines.append("说明：")
    lines.append("- 认知_Pearson / 认知_Spearman：该指标与认知分的相关系数")
    lines.append("- 情绪_Pearson / 情绪_Spearman：该指标与情绪分的相关系数")
    lines.append("- perm_p：对应相关性置换检验的经验 p 值，越小表示该相关越不容易由随机波动造成")
    lines.append("")
    lines.append("## 2. 各编号按视频结果表")
    lines.append("")

    for person_id, group in video_table.groupby("编号"):
        group = group.sort_values(by="视频名")
        lines.append(f"### 编号 {person_id}")
        lines.append("")
        lines.append(to_markdown_table(group))
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] markdown={OUT_MD}")
    print(f"[INFO] csv={OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
