from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\Administrator\Desktop\seriousgame_eval\data\reports\association_analysis")
OUT = ROOT / "attention_figures_selected"

SELECTED = {
    "attention_score": "注意力总分",
    "attention_metric__focus_ratio": "高专注占比",
    "attention_metric__screen_focus_ratio": "屏幕内注视占比",
    "attention_metric__acceptable_focus_ratio": "可接受注视占比",
    "attention_metric__off_focus_ratio": "偏离占比",
    "attention_metric__stability_std": "稳定性波动",
    "attention_metric__reengage_time_median": "回归专注中位时间",
    "attention_metric__screen_focus_duration_median_sec": "连续注视中位时长",
    "attention_metric__screen_focus_duration_max_sec": "最长连续注视时长",
    "attention_metric__screen_focus_segment_count": "连续注视片段数",
    "attention_metric__blink_ratio": "眨眼时长占比",
    "attention_metric__blink_count_per_min": "每分钟眨眼次数",
    "attention_metric__blink_excess_ratio": "额外眨眼占比",
    "attention_score__focus_score": "专注分",
    "attention_score__fixation_score": "持续注视分",
    "attention_score__blink_score": "眨眼分",
    "attention_score__stability_score": "稳定分",
    "attention_score__quality_score": "质量分",
}


def ensure_dir() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["feature"].isin(SELECTED)].copy()
    df["指标"] = df["feature"].map(SELECTED)
    order = [SELECTED[k] for k in SELECTED]
    df["指标"] = pd.Categorical(df["指标"], categories=order, ordered=True)
    df.sort_values("指标", inplace=True)
    return df


def build_matrix(cog_df: pd.DataFrame, emo_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    order = [SELECTED[k] for k in SELECTED]
    cog_map = cog_df.set_index("指标")[value_col].to_dict()
    emo_map = emo_df.set_index("指标")[value_col].to_dict()
    return pd.DataFrame(
        {
            "认知分": [cog_map.get(name, np.nan) for name in order],
            "情绪分": [emo_map.get(name, np.nan) for name in order],
        },
        index=order,
    )


def draw_combined_heatmap(pearson_matrix: pd.DataFrame, spearman_matrix: pd.DataFrame, out_path: Path, title: str, is_p: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 7.4), dpi=180, gridspec_kw={"width_ratios": [1, 1]})
    panels = [
        (axes[0], pearson_matrix, "Pearson"),
        (axes[1], spearman_matrix, "Spearman"),
    ]
    for ax, matrix, panel_title in panels:
        values = matrix.to_numpy(dtype=float)
        if is_p:
            im = ax.imshow(values, cmap="Reds_r", vmin=0.0, vmax=1.0, aspect="auto")
        else:
            im = ax.imshow(np.abs(values), cmap="Reds", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(panel_title, fontsize=12)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, fontsize=11)
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index, fontsize=8.5)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = values[i, j]
                text = f"{value:.3f}" if not np.isnan(value) else "-"
                base = value if is_p else abs(value)
                color = "white" if base >= 0.55 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.set_ylabel("p" if is_p else "|r|", rotation=270, labelpad=12)
        ax.set_xlabel("标签")
        ax.set_ylabel("指标")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ensure_dir()
    setup_fonts()
    cog = load_df(ROOT / "correlation_expression_vs_cognitive.csv")
    emo = load_df(ROOT / "correlation_expression_vs_emotion.csv")

    pearson_r = build_matrix(cog, emo, "pearson_r")
    spearman_r = build_matrix(cog, emo, "spearman_r")
    pearson_p = build_matrix(cog, emo, "pearson_perm_p")
    spearman_p = build_matrix(cog, emo, "spearman_perm_p")

    draw_combined_heatmap(pearson_r, spearman_r, OUT / "注意力_相关系数热图.png", "注意力指标相关系数热图", is_p=False)
    draw_combined_heatmap(pearson_p, spearman_p, OUT / "注意力_p值热图.png", "注意力指标显著性热图", is_p=True)

    table = pd.DataFrame(
        {
            "指标": list(pearson_r.index),
            "认知_Pearson_r": pearson_r["认知分"].to_list(),
            "情绪_Pearson_r": pearson_r["情绪分"].to_list(),
            "认知_Spearman_r": spearman_r["认知分"].to_list(),
            "情绪_Spearman_r": spearman_r["情绪分"].to_list(),
            "认知_Pearson_p": pearson_p["认知分"].to_list(),
            "情绪_Pearson_p": pearson_p["情绪分"].to_list(),
            "认知_Spearman_p": spearman_p["认知分"].to_list(),
            "情绪_Spearman_p": spearman_p["情绪分"].to_list(),
        }
    )
    table.to_csv(OUT / "注意力_相关结果表.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] output={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
