from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\Administrator\Desktop\seriousgame_eval\data\reports\association_analysis")
OUT = ROOT / "multi_au_figures_selected"

KEEP_SUFFIXES = (
    "coactive_ratio",
    "jaccard",
    "segment_total_sec",
    "segment_mean_sec",
    "strength_mean",
)


def ensure_dir() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def clean_combo_name(name: str) -> str:
    text = str(name)
    if text.startswith("au_pair__"):
        body = text[len("au_pair__"):].split("__")
        aus = "|".join(body[:-1])
        metric = body[-1]
        metric_cn = {
            "coactive_ratio": "共激活比例",
            "jaccard": "重叠系数",
            "segment_total_sec": "共激活总时长",
            "segment_mean_sec": "共激活平均持续时间",
            "strength_mean": "共激活强度",
        }.get(metric, metric)
        return f"{aus}|{metric_cn}"
    if text.startswith("au_tri__"):
        body = text[len("au_tri__"):].split("__")
        aus = "|".join(body[:-1])
        metric = body[-1]
        metric_cn = {
            "coactive_ratio": "共激活比例",
            "jaccard": "重叠系数",
            "segment_total_sec": "共激活总时长",
            "segment_mean_sec": "共激活平均持续时间",
            "strength_mean": "共激活强度",
        }.get(metric, metric)
        return f"{aus}|{metric_cn}"
    if text.startswith("emotion_proto__"):
        body = text[len("emotion_proto__"):].split("__")
        proto = body[0]
        metric = body[1]
        metric_cn = {
            "all_active_ratio": "全激活比例",
            "full_pattern_total_sec": "完整模式总时长",
            "full_pattern_mean_sec": "完整模式平均时长",
            "strength_mean": "共激活强度",
            "pair_jaccard_mean": "重叠系数",
            "coactive_strength_mean": "共激活强度",
        }.get(metric, metric)
        return f"{proto}|{metric_cn}"
    return text


def load_top_features(path: Path, top_n: int, prefix_filters: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(path)
    mask = df["feature"].astype(str).str.startswith(prefix_filters) & df["feature"].astype(str).str.endswith(KEEP_SUFFIXES)
    df = df[mask].copy()
    df["max_abs"] = df[["pearson_r", "spearman_r"]].abs().max(axis=1)
    df.sort_values(by="max_abs", ascending=False, inplace=True)
    df = df.head(top_n).copy()
    df["组合"] = df["feature"].map(clean_combo_name)
    return df


def load_top_prototypes(path: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    mask = df["feature"].astype(str).str.endswith((
        "all_active_ratio",
        "full_pattern_total_sec",
        "full_pattern_mean_sec",
        "strength_mean",
        "pair_jaccard_mean",
    ))
    df = df[mask].copy()
    df["max_abs"] = df[["pearson_r", "spearman_r"]].abs().max(axis=1)
    df.sort_values(by="max_abs", ascending=False, inplace=True)
    df = df.head(top_n).copy()
    df["组合"] = df["feature"].map(clean_combo_name)
    return df


def build_matrix(df: pd.DataFrame, value_cols: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "认知分" if "cognitive" in str(df.iloc[0]["target"]) else "情绪分": df[value_cols[0]].to_numpy(dtype=float),
            "认知分(S)" if "cognitive" in str(df.iloc[0]["target"]) else "情绪分(S)": df[value_cols[1]].to_numpy(dtype=float),
        },
        index=df["组合"],
    )


def draw_side_heatmap(pearson_vals: np.ndarray, spearman_vals: np.ndarray, labels: list[str], title: str, out_path: Path, is_p: bool = False) -> None:
    fig_h = max(5.5, len(labels) * 0.48)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, fig_h), dpi=180)
    for ax, values, panel_title in [
        (axes[0], pearson_vals.reshape(-1, 1), "Pearson"),
        (axes[1], spearman_vals.reshape(-1, 1), "Spearman"),
    ]:
        im = ax.imshow(values if is_p else np.abs(values), cmap="Reds_r" if is_p else "Reds", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(panel_title, fontsize=12)
        ax.set_xticks([0])
        ax.set_xticklabels(["值"], fontsize=10)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8.5)
        for i, val in enumerate(values[:, 0]):
            txt = f"{val:.3f}" if np.isfinite(val) else "-"
            base = val if is_p else abs(val)
            color = "white" if base >= 0.55 else "black"
            ax.text(0, i, txt, ha="center", va="center", fontsize=7.5, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cbar.ax.set_ylabel("p" if is_p else "|r|", rotation=270, labelpad=12)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def export_table(df: pd.DataFrame, out_name: str) -> None:
    keep = df.loc[:, ["组合", "pearson_r", "spearman_r", "pearson_perm_p", "spearman_perm_p"]].copy()
    keep.columns = ["组合", "Pearson_r", "Spearman_r", "Pearson_p", "Spearman_p"]
    keep.to_csv(OUT / out_name, index=False, encoding="utf-8-sig")


def main() -> int:
    ensure_dir()
    setup_fonts()

    multi_cog = load_top_features(ROOT / "correlation_au_vs_cognitive.csv", 12, ("au_pair__", "au_tri__"))
    multi_emo = load_top_features(ROOT / "correlation_au_vs_emotion.csv", 12, ("au_pair__", "au_tri__"))
    proto_cog = load_top_prototypes(ROOT / "correlation_emotion_prototypes_vs_cognitive.csv", 8)
    proto_emo = load_top_prototypes(ROOT / "correlation_emotion_prototypes_vs_emotion.csv", 8)

    draw_side_heatmap(multi_cog["pearson_r"].to_numpy(dtype=float), multi_cog["spearman_r"].to_numpy(dtype=float), multi_cog["组合"].tolist(), "高相关多AU组合与认知分相关系数图", OUT / "高相关多AU组合_认知_相关系数图.png", False)
    draw_side_heatmap(multi_cog["pearson_perm_p"].fillna(1.0).to_numpy(dtype=float), multi_cog["spearman_perm_p"].fillna(1.0).to_numpy(dtype=float), multi_cog["组合"].tolist(), "高相关多AU组合与认知分p值图", OUT / "高相关多AU组合_认知_p值图.png", True)
    draw_side_heatmap(multi_emo["pearson_r"].to_numpy(dtype=float), multi_emo["spearman_r"].to_numpy(dtype=float), multi_emo["组合"].tolist(), "高相关多AU组合与情绪分相关系数图", OUT / "高相关多AU组合_情绪_相关系数图.png", False)
    draw_side_heatmap(multi_emo["pearson_perm_p"].fillna(1.0).to_numpy(dtype=float), multi_emo["spearman_perm_p"].fillna(1.0).to_numpy(dtype=float), multi_emo["组合"].tolist(), "高相关多AU组合与情绪分p值图", OUT / "高相关多AU组合_情绪_p值图.png", True)

    draw_side_heatmap(proto_cog["pearson_r"].to_numpy(dtype=float), proto_cog["spearman_r"].to_numpy(dtype=float), proto_cog["组合"].tolist(), "表情相关原型组合与认知分相关系数图", OUT / "表情原型组合_认知_相关系数图.png", False)
    draw_side_heatmap(proto_cog["pearson_perm_p"].fillna(1.0).to_numpy(dtype=float), proto_cog["spearman_perm_p"].fillna(1.0).to_numpy(dtype=float), proto_cog["组合"].tolist(), "表情相关原型组合与认知分p值图", OUT / "表情原型组合_认知_p值图.png", True)
    draw_side_heatmap(proto_emo["pearson_r"].to_numpy(dtype=float), proto_emo["spearman_r"].to_numpy(dtype=float), proto_emo["组合"].tolist(), "表情相关原型组合与情绪分相关系数图", OUT / "表情原型组合_情绪_相关系数图.png", False)
    draw_side_heatmap(proto_emo["pearson_perm_p"].fillna(1.0).to_numpy(dtype=float), proto_emo["spearman_perm_p"].fillna(1.0).to_numpy(dtype=float), proto_emo["组合"].tolist(), "表情相关原型组合与情绪分p值图", OUT / "表情原型组合_情绪_p值图.png", True)

    export_table(multi_cog, "高相关多AU组合_认知_结果表.csv")
    export_table(multi_emo, "高相关多AU组合_情绪_结果表.csv")
    export_table(proto_cog, "表情原型组合_认知_结果表.csv")
    export_table(proto_emo, "表情原型组合_情绪_结果表.csv")

    print(f"[INFO] output={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
