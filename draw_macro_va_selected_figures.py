from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\Administrator\Desktop\seriousgame_eval\data\reports\association_analysis")
OUT = ROOT / "macro_va_figures_selected"

SELECTED = {
    "affect_score": "宏表情总分",
    "emo_metric__tone": "情绪基调",
    "emo_metric__pa": "正性情绪倾向",
    "emo_metric__na": "负性情绪倾向",
    "emo_metric__volatility": "情绪波动度",
    "emo_metric__switch_rate": "情绪切换频率",
    "emo_metric__dominance_ratio": "主导情绪占比",
    "va_score": "VA综合得分",
    "va_metric__valence_mean": "平均愉悦度",
    "va_metric__arousal_mean": "平均唤醒度",
    "va_metric__va_volatility": "VA波动度",
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


def draw_r_heatmap(matrix: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.8), dpi=180)
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(np.abs(values), cmap="Reds", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("宏表情与VA指标相关系数热图", fontsize=13)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, fontsize=11)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = values[i, j]
            text = f"{value:.3f}" if not np.isnan(value) else "-"
            color = "white" if abs(value) >= 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel("|r|", rotation=270, labelpad=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_p_heatmap(matrix: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.8), dpi=180)
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="Reds_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("宏表情与VA指标显著性热图", fontsize=13)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, fontsize=11)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = values[i, j]
            text = f"{value:.3f}" if not np.isnan(value) else "-"
            color = "white" if value >= 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel("p", rotation=270, labelpad=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_combined_r_heatmap(pearson_matrix: pd.DataFrame, spearman_matrix: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.8), dpi=180, gridspec_kw={"width_ratios": [1, 1]})
    for ax, matrix, title in [
        (axes[0], pearson_matrix, "Pearson相关系数"),
        (axes[1], spearman_matrix, "Spearman相关系数"),
    ]:
        values = matrix.to_numpy(dtype=float)
        im = ax.imshow(np.abs(values), cmap="Reds", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(title, fontsize=12)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, fontsize=11)
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index, fontsize=9)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = values[i, j]
                text = f"{value:.3f}" if not np.isnan(value) else "-"
                color = "white" if abs(value) >= 0.55 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.set_ylabel("|r|", rotation=270, labelpad=12)
        ax.set_xlabel("标签")
        ax.set_ylabel("指标")
    fig.suptitle("宏表情与VA指标相关系数热图", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_combined_p_heatmap(pearson_matrix: pd.DataFrame, spearman_matrix: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.8), dpi=180, gridspec_kw={"width_ratios": [1, 1]})
    for ax, matrix, title in [
        (axes[0], pearson_matrix, "Pearson p值"),
        (axes[1], spearman_matrix, "Spearman p值"),
    ]:
        values = matrix.to_numpy(dtype=float)
        im = ax.imshow(values, cmap="Reds_r", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(title, fontsize=12)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, fontsize=11)
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index, fontsize=9)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = values[i, j]
                text = f"{value:.3f}" if not np.isnan(value) else "-"
                color = "white" if value >= 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.set_ylabel("p", rotation=270, labelpad=12)
        ax.set_xlabel("标签")
        ax.set_ylabel("指标")
    fig.suptitle("宏表情与VA指标显著性热图", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ensure_dir()
    setup_fonts()
    cog = load_df(ROOT / "correlation_expression_vs_cognitive.csv")
    emo = load_df(ROOT / "correlation_expression_vs_emotion.csv")

    pearson_r_matrix = build_matrix(cog, emo, "pearson_r")
    spearman_r_matrix = build_matrix(cog, emo, "spearman_r")
    pearson_p_matrix = build_matrix(cog, emo, "pearson_perm_p")
    spearman_p_matrix = build_matrix(cog, emo, "spearman_perm_p")

    draw_combined_r_heatmap(pearson_r_matrix, spearman_r_matrix, OUT / "宏表情VA_选定指标_相关系数热图.png")
    draw_combined_p_heatmap(pearson_p_matrix, spearman_p_matrix, OUT / "宏表情VA_选定指标_p值热图.png")

    table = pd.DataFrame(
        {
            "指标": list(pearson_r_matrix.index),
            "认知_Pearson_r": pearson_r_matrix["认知分"].to_list(),
            "情绪_Pearson_r": pearson_r_matrix["情绪分"].to_list(),
            "认知_Spearman_r": spearman_r_matrix["认知分"].to_list(),
            "情绪_Spearman_r": spearman_r_matrix["情绪分"].to_list(),
            "认知_Pearson_p": pearson_p_matrix["认知分"].to_list(),
            "情绪_Pearson_p": pearson_p_matrix["情绪分"].to_list(),
            "认知_Spearman_p": spearman_p_matrix["认知分"].to_list(),
            "情绪_Spearman_p": spearman_p_matrix["情绪分"].to_list(),
        }
    )
    table.to_csv(OUT / "宏表情VA_选定指标_相关结果表.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] output={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
