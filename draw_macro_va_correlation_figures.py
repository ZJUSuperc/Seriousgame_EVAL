from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\Administrator\Desktop\seriousgame_eval\data\reports\association_analysis")
OUT = ROOT / "macro_va_figures"


METRIC_MAP = {
    "affect_score": "宏表情总分",
    "emo_metric__tone": "情绪基调",
    "emo_metric__pa": "正性情绪倾向",
    "emo_metric__na": "负性情绪倾向",
    "emo_metric__volatility": "情绪波动度",
    "emo_metric__switch_rate": "情绪切换频率",
    "emo_metric__switch_count": "情绪切换次数",
    "emo_metric__dominance_ratio": "主导情绪占比",
    "emo_metric__phase_purity_ratio": "阶段纯度",
    "emo_metric__margin_mean": "分类边际均值",
    "emo_metric__entropy_mean": "情绪熵均值",
    "emo_metric__events_per_minute": "情绪事件频率",
    "va_score": "VA综合分",
    "va_metric__valence_mean": "平均愉悦度",
    "va_metric__arousal_mean": "平均唤醒度",
    "va_metric__va_volatility": "VA波动度",
    "va_metric__activation_ratio": "高激活比例",
}


def ensure_dir() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def load_corr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["feature"].isin(METRIC_MAP)].copy()
    df["指标"] = df["feature"].map(METRIC_MAP)
    return df


def build_heatmap_matrix(cog_df: pd.DataFrame, emo_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    metric_order = [METRIC_MAP[k] for k in METRIC_MAP.keys()]
    cog_map = cog_df.set_index("指标")[value_col].to_dict()
    emo_map = emo_df.set_index("指标")[value_col].to_dict()
    matrix = pd.DataFrame(
        {
            "认知分": [cog_map.get(metric, np.nan) for metric in metric_order],
            "情绪分": [emo_map.get(metric, np.nan) for metric in metric_order],
        },
        index=metric_order,
    )
    return matrix


def draw_heatmap(matrix: pd.DataFrame, title: str, out_path: Path, is_p: bool = False) -> None:
    fig_w = 6.4
    fig_h = max(7.5, len(matrix.index) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    values = matrix.to_numpy(dtype=float)
    if is_p:
        im = ax.imshow(values, cmap="Reds_r", vmin=0.0, vmax=1.0, aspect="auto")
    else:
        im = ax.imshow(np.abs(values), cmap="Reds", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title(title, fontsize=13)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, fontsize=11)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=9)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                text = "-"
                color = "black"
            else:
                text = f"{value:.3f}"
                base = value if is_p else abs(value)
                color = "white" if base >= 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel("p值" if is_p else "|r|", rotation=270, labelpad=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def export_combined_table(cog_df: pd.DataFrame, emo_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, cn in METRIC_MAP.items():
        cog_row = cog_df[cog_df["feature"] == key]
        emo_row = emo_df[emo_df["feature"] == key]
        rows.append(
            {
                "字段名": key,
                "中文名": cn,
                "认知_Pearson_r": float(cog_row.iloc[0]["pearson_r"]) if not cog_row.empty else np.nan,
                "认知_Spearman_r": float(cog_row.iloc[0]["spearman_r"]) if not cog_row.empty else np.nan,
                "认知_Pearson_p": float(cog_row.iloc[0]["pearson_perm_p"]) if not cog_row.empty else np.nan,
                "认知_Spearman_p": float(cog_row.iloc[0]["spearman_perm_p"]) if not cog_row.empty else np.nan,
                "情绪_Pearson_r": float(emo_row.iloc[0]["pearson_r"]) if not emo_row.empty else np.nan,
                "情绪_Spearman_r": float(emo_row.iloc[0]["spearman_r"]) if not emo_row.empty else np.nan,
                "情绪_Pearson_p": float(emo_row.iloc[0]["pearson_perm_p"]) if not emo_row.empty else np.nan,
                "情绪_Spearman_p": float(emo_row.iloc[0]["spearman_perm_p"]) if not emo_row.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ensure_dir()
    setup_fonts()
    cog_df = load_corr(ROOT / "correlation_expression_vs_cognitive.csv")
    emo_df = load_corr(ROOT / "correlation_expression_vs_emotion.csv")

    pearson_r = build_heatmap_matrix(cog_df, emo_df, "pearson_r")
    spearman_r = build_heatmap_matrix(cog_df, emo_df, "spearman_r")
    pearson_p = build_heatmap_matrix(cog_df, emo_df, "pearson_perm_p")
    spearman_p = build_heatmap_matrix(cog_df, emo_df, "spearman_perm_p")

    draw_heatmap(pearson_r, "宏表情与VA指标 Pearson相关热图", OUT / "宏表情VA_Pearson_r热图.png", is_p=False)
    draw_heatmap(spearman_r, "宏表情与VA指标 Spearman相关热图", OUT / "宏表情VA_Spearman_r热图.png", is_p=False)
    draw_heatmap(pearson_p, "宏表情与VA指标 Pearson p值热图", OUT / "宏表情VA_Pearson_p热图.png", is_p=True)
    draw_heatmap(spearman_p, "宏表情与VA指标 Spearman p值热图", OUT / "宏表情VA_Spearman_p热图.png", is_p=True)

    combined = export_combined_table(cog_df, emo_df)
    combined.to_csv(OUT / "宏表情VA_相关性结果表.csv", index=False, encoding="utf-8-sig")
    (OUT / "宏表情VA_相关性结果表.md").write_text(combined.to_string(index=False), encoding="utf-8")

    print(f"[INFO] output={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
