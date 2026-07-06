from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(r"C:\Users\Administrator\Desktop\seriousgame_eval\data\reports\association_analysis")
OUT = ROOT / "video_level_duplicate_label_analysis"


def ensure_dir() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def fdr_bh(pvalues: list[float | None]) -> list[float | None]:
    indexed = [(idx, p) for idx, p in enumerate(pvalues) if p is not None and math.isfinite(p)]
    result: list[float | None] = [None] * len(pvalues)
    if not indexed:
        return result
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    adjusted = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx, pvalue = indexed[rank]
        value = min(prev, pvalue * m / (rank + 1))
        adjusted[rank] = value
        prev = value
    for rank, (idx, _) in enumerate(indexed):
        result[idx] = float(min(max(adjusted[rank], 0.0), 1.0))
    return result


def run_corr(df: pd.DataFrame, target: str, domain: str) -> pd.DataFrame:
    rows = []
    feature_cols = [c for c in df.columns if c not in {"person_id", "video_name", "cognitive_score", "emotion_score"}]
    y = pd.to_numeric(df[target], errors="coerce")
    for feature in feature_cols:
        x = pd.to_numeric(df[feature], errors="coerce")
        valid = x.notna() & y.notna()
        n = int(valid.sum())
        if n < 5:
            continue
        xv = x[valid]
        yv = y[valid]
        if float(xv.std()) <= 1e-12 or float(yv.std()) <= 1e-12:
            continue
        pearson_r, pearson_p = stats.pearsonr(xv, yv)
        spearman_r, spearman_p = stats.spearmanr(xv, yv)
        rows.append(
            {
                "domain": domain,
                "target": target,
                "feature": feature,
                "n": n,
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "abs_max_r": max(abs(float(pearson_r)), abs(float(spearman_r))),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["pearson_fdr"] = fdr_bh(out["pearson_p"].tolist())
    out["spearman_fdr"] = fdr_bh(out["spearman_p"].tolist())
    out.sort_values(by=["abs_max_r", "spearman_p", "pearson_p"], ascending=[False, True, True], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def write_markdown_summary(outputs: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# 视频级重复标签探索性相关分析",
        "",
        "说明：",
        "- 该分析把每个视频都作为一条记录，并直接继承所属受试者的认知分与情绪分。",
        "- 因此同一受试者的多个视频共享同一标签，这会放大表面样本量，但不增加独立受试者数量。",
        "- 本结果仅用于探索性比较，不建议替代受试者级主分析。",
        "",
    ]
    for name, df in outputs.items():
        lines.append(f"## {name}")
        if df.empty:
            lines.append("无结果")
            lines.append("")
            continue
        preview = df.loc[:, ["feature", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "pearson_fdr", "spearman_fdr"]].head(20)
        lines.append(preview.to_string(index=False))
        lines.append("")
    (OUT / "video_level_duplicate_label_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dir()
    expr = pd.read_csv(ROOT / "expression_metrics_video_level.csv")
    au = pd.read_csv(ROOT / "au_features_video_level.csv")

    outputs = {
        "宏表情-认知": run_corr(expr, "cognitive_score", "expression"),
        "宏表情-情绪": run_corr(expr, "emotion_score", "expression"),
        "AU特征-认知": run_corr(au, "cognitive_score", "au"),
        "AU特征-情绪": run_corr(au, "emotion_score", "au"),
    }

    file_map = {
        "宏表情-认知": "video_level_expression_vs_cognitive.csv",
        "宏表情-情绪": "video_level_expression_vs_emotion.csv",
        "AU特征-认知": "video_level_au_vs_cognitive.csv",
        "AU特征-情绪": "video_level_au_vs_emotion.csv",
    }
    for key, df in outputs.items():
        df.to_csv(OUT / file_map[key], index=False, encoding="utf-8-sig")

    write_markdown_summary(outputs)
    print(f"[INFO] output={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
