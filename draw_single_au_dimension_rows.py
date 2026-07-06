from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\Administrator\Desktop\seriousgame_eval\data\reports\association_analysis")
OUT = ROOT / "single_au_dimension_rows"


METRIC_GROUPS = {
    "强度均值": ["r_mean", "c_mean"],
    "波动程度": ["r_std", "c_std", "r_diff_std"],
    "激活占比": ["r_active_ratio", "c_active_ratio"],
    "平均持续时间": ["r_segment_mean_sec", "c_segment_mean_sec"],
}

TARGETS = {
    "cognitive_score": "认知分",
    "emotion_score": "情绪分",
}


def ensure_dir() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def summarize_group(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return np.nan
    return float(np.mean(finite))


def build_group_matrix(df: pd.DataFrame, target_col: str, method: str) -> pd.DataFrame:
    au_bases = sorted({col.split("__")[1] for col in df.columns if col.startswith("au__")})
    target = pd.to_numeric(df[target_col], errors="coerce")
    rows = []
    index_names = []

    for group_name, suffixes in METRIC_GROUPS.items():
        row = {}
        for au in au_bases:
            corr_values = []
            for suffix in suffixes:
                col = f"au__{au}__{suffix}"
                if col not in df.columns:
                    continue
                series = pd.to_numeric(df[col], errors="coerce")
                valid = series.notna() & target.notna()
                if valid.sum() < 5:
                    continue
                corr = series[valid].corr(target[valid], method=method)
                if pd.notna(corr):
                    corr_values.append(float(corr))
            row[au] = summarize_group(corr_values)
        rows.append(row)
        index_names.append(group_name)

    return pd.DataFrame(rows, index=index_names)


def draw_dual_heatmap(pearson_df: pd.DataFrame, spearman_df: pd.DataFrame, title: str, out_path: Path) -> None:
    cols = len(pearson_df.columns)
    fig_w = max(10.5, cols * 0.55)
    fig_h = max(5.8, len(pearson_df.index) * 1.05)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=180)

    for ax, matrix, panel_title in [
        (axes[0], pearson_df, "Pearson"),
        (axes[1], spearman_df, "Spearman"),
    ]:
        values = matrix.to_numpy(dtype=float)
        im = ax.imshow(np.abs(values), cmap="Reds", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(panel_title, fontsize=12)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, rotation=0, fontsize=8)
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index, fontsize=9)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = values[i, j]
                if np.isnan(value):
                    text = "-"
                    color = "black"
                else:
                    text = f"{value:.2f}"
                    color = "white" if abs(value) >= 0.55 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.set_ylabel("|r|", rotation=270, labelpad=12)
        ax.set_xlabel("AU")
        ax.set_ylabel("维度")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ensure_dir()
    setup_fonts()
    df = pd.read_csv(ROOT / "au_features_subject_level.csv")

    for target_col, target_cn in TARGETS.items():
        pearson_df = build_group_matrix(df, target_col, method="pearson")
        spearman_df = build_group_matrix(df, target_col, method="spearman")
        draw_dual_heatmap(
            pearson_df,
            spearman_df,
            f"单AU各维度与{target_cn}相关性热图",
            OUT / f"单AU各维度_{target_cn}_相关性热图.png",
        )
        out_table = pearson_df.copy()
        out_table.index.name = "维度"
        out_table.to_csv(OUT / f"单AU各维度_{target_cn}_Pearson结果表.csv", encoding="utf-8-sig")
        out_table2 = spearman_df.copy()
        out_table2.index.name = "维度"
        out_table2.to_csv(OUT / f"单AU各维度_{target_cn}_Spearman结果表.csv", encoding="utf-8-sig")

    print(f"[INFO] output={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
