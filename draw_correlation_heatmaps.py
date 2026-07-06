from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\Administrator\Desktop\seriousgame_eval\data\reports\association_analysis")
OUTPUT = ROOT / "heatmaps"


TITLE_MAP = {
    "correlation_expression_vs_cognitive.csv": "宏表情指标与认知分相关性热图",
    "correlation_expression_vs_emotion.csv": "宏表情指标与情绪分相关性热图",
    "correlation_au_vs_cognitive.csv": "AU特征与认知分相关性热图",
    "correlation_au_vs_emotion.csv": "AU特征与情绪分相关性热图",
    "correlation_emotion_prototypes_vs_cognitive.csv": "典型表情AU组合与认知分相关性热图",
    "correlation_emotion_prototypes_vs_emotion.csv": "典型表情AU组合与情绪分相关性热图",
}


TOKEN_MAP = {
    "affect_score": "宏表情总分",
    "va_score": "VA综合分",
    "attention_score": "注意力总分",
    "tone": "情绪基调",
    "pa": "正性情绪倾向",
    "na": "负性情绪倾向",
    "volatility": "波动度",
    "switch_rate": "切换频率",
    "switch_count": "切换次数",
    "ambiguity_ratio": "模糊比例",
    "dominance_ratio": "主导占比",
    "phase_change_rate": "阶段变化频率",
    "phase_change_count": "阶段变化次数",
    "phase_purity_ratio": "阶段纯度",
    "coverage": "覆盖率",
    "margin_mean": "分类边际均值",
    "entropy_mean": "情绪熵均值",
    "events_per_minute": "情绪事件频率",
    "valence_mean": "平均愉悦度",
    "arousal_mean": "平均唤醒度",
    "va_volatility": "VA波动度",
    "activation_ratio": "高激活比例",
    "valence_score": "愉悦度评分",
    "arousal_score": "唤醒度评分",
    "stability_score": "稳定性评分",
    "activation_score": "激活评分",
    "emonet_score": "VA模型总分",
    "total_duration_sec": "总时长(秒)",
    "analysis_duration_sec": "分析时长(秒)",
    "valid_ratio": "有效比例",
    "focus_ratio": "高专注占比",
    "screen_focus_ratio": "屏幕内注视占比",
    "acceptable_focus_ratio": "可接受注视占比",
    "off_focus_ratio": "偏离占比",
    "stability_std": "稳定性波动",
    "reengage_time_median": "回归专注中位时间",
    "screen_focus_duration_median_sec": "连续注视中位时长",
    "screen_focus_duration_max_sec": "最长连续注视时长",
    "screen_focus_segment_count": "连续注视片段数",
    "blink_ratio": "眨眼时长占比",
    "blink_count": "眨眼次数",
    "blink_count_per_min": "每分钟眨眼次数",
    "blink_excess_ratio": "额外眨眼占比",
    "focus_score": "专注分",
    "fixation_score": "持续注视分",
    "blink_score": "眨眼分",
    "quality_score": "质量分",
    "cognitive_score": "认知分",
    "emotion_score": "情绪分",
    "happiness": "快乐原型",
    "sadness": "悲伤原型",
    "anger": "愤怒原型",
    "fear": "恐惧原型",
    "surprise": "惊讶原型",
    "disgust": "厌恶原型",
    "contempt": "轻蔑原型",
    "available_ratio": "可用比例",
    "any_active_ratio": "任一激活比例",
    "all_active_ratio": "全组合激活比例",
    "mean_active_count": "平均激活个数",
    "active_count_p90": "激活个数P90",
    "full_pattern_count_per_min": "完整模式频率",
    "full_pattern_mean_sec": "完整模式平均时长",
    "full_pattern_max_sec": "完整模式最长时长",
    "strength_mean": "强度均值",
    "strength_p90": "强度P90",
    "coactive_strength_mean": "共激活强度均值",
    "pair_overlap_mean": "内部重叠均值",
    "pair_jaccard_mean": "内部Jaccard均值",
    "coactive_ratio": "共激活比例",
    "jaccard": "Jaccard",
    "product_mean": "强度乘积均值",
    "r_mean": "强度均值",
    "r_std": "强度标准差",
    "r_p90": "强度P90",
    "r_active_ratio": "激活比例",
    "r_diff_std": "差分波动",
    "r_transition_per_min": "转移频率",
    "r_segment_count_per_min": "片段频率",
    "r_segment_mean_sec": "片段平均时长",
    "r_segment_max_sec": "片段最长时长",
    "r_energy": "能量",
    "c_mean": "激活均值",
    "c_std": "激活标准差",
    "c_p90": "激活P90",
    "c_active_ratio": "激活比例",
    "c_transition_per_min": "转移频率",
    "c_segment_count_per_min": "片段频率",
    "c_segment_mean_sec": "片段平均时长",
    "c_segment_max_sec": "片段最长时长",
}


def ensure_output_dir() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def translate_non_au_feature(name: str) -> str:
    text = str(name)
    if text.startswith("au__") or text.startswith("au_pair__") or text.startswith("au_tri__"):
        return text.replace("__", " | ")
    prefixes = [
        "attention_metric__",
        "attention_score__",
        "emo_metric__",
        "emo_score__",
        "va_metric__",
        "va_score__",
        "emotion_proto__",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    parts = text.split("__")
    cn_parts = [TOKEN_MAP.get(part, part) for part in parts]
    return " | ".join(cn_parts)


def build_matrices(path: Path, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path).sort_values(by="abs_max_r", ascending=False).head(top_n).copy()
    df["feature_cn"] = df["feature"].map(translate_non_au_feature)
    r_matrix = df.loc[:, ["feature_cn", "pearson_r", "spearman_r"]].set_index("feature_cn").T
    r_matrix.index = ["Pearson r", "Spearman r"]
    p_matrix = df.loc[:, ["feature_cn", "pearson_perm_p", "spearman_perm_p"]].fillna(1.0).set_index("feature_cn").T
    p_matrix.index = ["Pearson p", "Spearman p"]
    return r_matrix, p_matrix


def draw_horizontal_heatmap(r_matrix: pd.DataFrame, p_matrix: pd.DataFrame, title: str, out_path: Path) -> None:
    cols = r_matrix.shape[1]
    fig_w = max(18, cols * 0.68)
    fig_h = 6.5
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=180, gridspec_kw={"height_ratios": [1, 1]})

    r_values = r_matrix.to_numpy(dtype=float)
    p_values = p_matrix.to_numpy(dtype=float)

    im_r = axes[0].imshow(r_values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    im_p = axes[1].imshow(p_values, cmap="YlOrRd_r", vmin=0.0, vmax=1.0, aspect="auto")

    for ax, matrix in [(axes[0], r_matrix), (axes[1], p_matrix)]:
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index, fontsize=10)

    for i in range(r_values.shape[0]):
        for j in range(r_values.shape[1]):
            color = "white" if abs(r_values[i, j]) >= 0.55 else "black"
            axes[0].text(j, i, f"{r_values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)

    for i in range(p_values.shape[0]):
        for j in range(p_values.shape[1]):
            color = "white" if p_values[i, j] >= 0.60 else "black"
            axes[1].text(j, i, f"{p_values[i, j]:.3f}", ha="center", va="center", fontsize=7, color=color)

    axes[0].set_title(title, fontsize=14)
    axes[0].set_ylabel("相关系数")
    axes[1].set_ylabel("置换检验p值")
    axes[1].set_xlabel("特征")

    cbar_r = fig.colorbar(im_r, ax=axes[0], fraction=0.018, pad=0.01)
    cbar_r.ax.set_ylabel("r", rotation=270, labelpad=10)
    cbar_p = fig.colorbar(im_p, ax=axes[1], fraction=0.018, pad=0.01)
    cbar_p.ax.set_ylabel("p", rotation=270, labelpad=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ensure_output_dir()
    setup_fonts()

    configs = [
        ("correlation_expression_vs_cognitive.csv", 18, "expression_vs_cognitive_top18.png"),
        ("correlation_expression_vs_emotion.csv", 18, "expression_vs_emotion_top18.png"),
        ("correlation_au_vs_cognitive.csv", 24, "au_vs_cognitive_top24.png"),
        ("correlation_au_vs_emotion.csv", 24, "au_vs_emotion_top24.png"),
        ("correlation_emotion_prototypes_vs_cognitive.csv", 20, "emotion_prototypes_vs_cognitive.png"),
        ("correlation_emotion_prototypes_vs_emotion.csv", 20, "emotion_prototypes_vs_emotion.png"),
    ]

    for file_name, top_n, out_name in configs:
        r_matrix, p_matrix = build_matrices(ROOT / file_name, top_n=top_n)
        draw_horizontal_heatmap(r_matrix, p_matrix, TITLE_MAP[file_name], OUTPUT / out_name)

    print(f"[INFO] heatmaps saved to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
