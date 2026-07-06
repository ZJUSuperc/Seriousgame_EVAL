from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\Administrator\Desktop\seriousgame_eval\data\reports\association_analysis")
OUT = ROOT / "heatmaps_single_au"


CATEGORY_MAP = {
    "强度水平": ["r_mean", "r_p90", "c_mean", "c_p90", "r_energy"],
    "波动变化": ["r_std", "c_std", "r_diff_std"],
    "激活占比": ["r_active_ratio", "c_active_ratio"],
    "频率特征": ["r_transition_per_min", "c_transition_per_min", "r_segment_count_per_min", "c_segment_count_per_min"],
    "持续时间": ["r_segment_mean_sec", "c_segment_mean_sec", "r_segment_max_sec", "c_segment_max_sec"],
}

METRIC_CN = {
    "r_mean": "强度均值(r)",
    "r_p90": "强度P90(r)",
    "c_mean": "激活均值(c)",
    "c_p90": "激活P90(c)",
    "r_energy": "能量(r)",
    "r_std": "强度标准差(r)",
    "c_std": "激活标准差(c)",
    "r_diff_std": "差分波动(r)",
    "r_active_ratio": "激活占比(r)",
    "c_active_ratio": "激活占比(c)",
    "r_transition_per_min": "转移频率(r)",
    "c_transition_per_min": "转移频率(c)",
    "r_segment_count_per_min": "片段频率(r)",
    "c_segment_count_per_min": "片段频率(c)",
    "r_segment_mean_sec": "平均持续时间(r)",
    "c_segment_mean_sec": "平均持续时间(c)",
    "r_segment_max_sec": "最长持续时间(r)",
    "c_segment_max_sec": "最长持续时间(c)",
}

TARGET_CN = {
    "cognitive_score": "认知分",
    "emotion_score": "情绪分",
}


def ensure_dir() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def permutation_pvalue(x: np.ndarray, y: np.ndarray, corr_value: float, method: str, rng: np.random.Generator, iterations: int = 3000) -> float:
    if not np.isfinite(corr_value):
        return 1.0
    observed = abs(corr_value)
    count = 0
    for _ in range(iterations):
        yp = rng.permutation(y)
        if method == "pearson":
            value = np.corrcoef(x, yp)[0, 1]
        else:
            value = pd.Series(x).corr(pd.Series(yp), method="spearman")
        if np.isfinite(value) and abs(value) >= observed - 1e-12:
            count += 1
    return float((count + 1) / (iterations + 1))


def compute_single_au_tables(df: pd.DataFrame, target_col: str, suffixes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260506)
    au_bases = sorted({col.split("__")[1] for col in df.columns if col.startswith("au__")})
    r_rows = []
    p_rows = []
    target = pd.to_numeric(df[target_col], errors="coerce")

    for suffix in suffixes:
        pearson_row = {}
        spearman_row = {}
        pearson_p_row = {}
        spearman_p_row = {}
        for au in au_bases:
            col = f"au__{au}__{suffix}"
            if col not in df.columns:
                pearson_row[au] = np.nan
                spearman_row[au] = np.nan
                pearson_p_row[au] = np.nan
                spearman_p_row[au] = np.nan
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            valid = series.notna() & target.notna()
            if valid.sum() < 5:
                pearson_row[au] = np.nan
                spearman_row[au] = np.nan
                pearson_p_row[au] = np.nan
                spearman_p_row[au] = np.nan
                continue
            x = series[valid].to_numpy(dtype=float)
            y = target[valid].to_numpy(dtype=float)
            pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else np.nan
            spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
            pearson_p = permutation_pvalue(x, y, pearson, "pearson", rng)
            spearman_p = permutation_pvalue(x, y, spearman, "spearman", rng)
            pearson_row[au] = pearson
            spearman_row[au] = spearman
            pearson_p_row[au] = pearson_p
            spearman_p_row[au] = spearman_p
        base_name = METRIC_CN[suffix]
        r_rows.append(pd.Series(pearson_row, name=f"{base_name} | Pearson r"))
        r_rows.append(pd.Series(spearman_row, name=f"{base_name} | Spearman r"))
        p_rows.append(pd.Series(pearson_p_row, name=f"{base_name} | Pearson p"))
        p_rows.append(pd.Series(spearman_p_row, name=f"{base_name} | Spearman p"))

    return pd.DataFrame(r_rows), pd.DataFrame(p_rows)


def draw_heatmap(r_df: pd.DataFrame, p_df: pd.DataFrame, title: str, out_path: Path) -> None:
    cols = len(r_df.columns)
    fig_w = max(13, cols * 0.65)
    fig_h = max(5.5, (len(r_df.index) + len(p_df.index)) * 0.34)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=180, gridspec_kw={"height_ratios": [len(r_df.index), len(p_df.index)]})

    r_values = r_df.to_numpy(dtype=float)
    p_values = p_df.to_numpy(dtype=float)
    im_r = axes[0].imshow(r_values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    im_p = axes[1].imshow(p_values, cmap="YlOrRd_r", vmin=0.0, vmax=1.0, aspect="auto")

    for ax, df in [(axes[0], r_df), (axes[1], p_df)]:
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=0, fontsize=8)
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_yticklabels(df.index, fontsize=8)

    for i in range(r_values.shape[0]):
        for j in range(r_values.shape[1]):
            value = r_values[i, j]
            if np.isnan(value):
                continue
            color = "white" if abs(value) >= 0.55 else "black"
            axes[0].text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=color)

    for i in range(p_values.shape[0]):
        for j in range(p_values.shape[1]):
            value = p_values[i, j]
            if np.isnan(value):
                continue
            color = "white" if value >= 0.60 else "black"
            axes[1].text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6.5, color=color)

    axes[0].set_title(title, fontsize=14)
    axes[0].set_ylabel("相关系数")
    axes[1].set_ylabel("置换检验p值")
    axes[1].set_xlabel("AU")
    fig.colorbar(im_r, ax=axes[0], fraction=0.016, pad=0.01).ax.set_ylabel("r", rotation=270, labelpad=10)
    fig.colorbar(im_p, ax=axes[1], fraction=0.016, pad=0.01).ax.set_ylabel("p", rotation=270, labelpad=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_summary_file() -> None:
    lines = [
        "# 单AU特征分类说明",
        "",
        "当前单AU分析分为以下 5 类情况：",
        "",
        "1. 强度水平：看 AU 激活强不强，包括平均强度、高分位强度、能量等",
        "2. 波动变化：看 AU 变化是否剧烈，包括标准差、差分波动等",
        "3. 激活占比：看 AU 出现时间占比有多高",
        "4. 频率特征：看 AU 切换和出现得多不多，包括转移频率、片段频率",
        "5. 持续时间：看 AU 一次出现能持续多久，包括平均持续时间、最长持续时间",
        "",
        "具体字段：",
    ]
    for category, suffixes in CATEGORY_MAP.items():
        lines.append(f"- {category}：")
        for suffix in suffixes:
            lines.append(f"  - {suffix} -> {METRIC_CN[suffix]}")
    (OUT / "单AU特征分类说明.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dir()
    setup_fonts()
    write_summary_file()
    df = pd.read_csv(ROOT / "au_features_subject_level.csv")

    for target_col, target_cn in TARGET_CN.items():
        for category, suffixes in CATEGORY_MAP.items():
            r_df, p_df = compute_single_au_tables(df, target_col, suffixes)
            out_name = f"单AU_{target_cn}_{category}.png"
            draw_heatmap(r_df, p_df, f"单AU{category}与{target_cn}相关性热图", OUT / out_name)

    print(f"[INFO] single-AU heatmaps saved to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
