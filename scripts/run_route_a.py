from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as `python scripts/run_route_a.py` from the repo root, where the
# script directory (`scripts/`) — not the repo root — is on sys.path. No-op when
# the repo root is already importable (e.g. under pytest).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from modeling.dataset import build_dataset
from modeling.features import extract_features, feature_names
from modeling.labels import get_labels, labeled_subject_ids
from modeling.loso import permutation_test, run_loso
from modeling.route_a import MODEL_FACTORIES, _design_matrix

ROOT = _ROOT


def _feature_importance(records, labels) -> dict[str, float]:
    X, y_cog, _ = _design_matrix(records, labels)
    if X.shape[0] < 2:
        return {n: 0.0 for n in feature_names()}
    rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y_cog)
    return {n: float(w) for n, w in zip(feature_names(), rf.feature_importances_)}


def main(reports_root: Path, out_path: Path, n_perm: int = 200) -> dict:
    subjects = labeled_subject_ids()
    records = build_dataset(reports_root, subjects, set(subjects))
    labels = get_labels()

    results: dict = {}
    for name, factory in MODEL_FACTORIES.items():
        fn = factory()
        metrics = run_loso(records, labels, fn)
        perm = permutation_test(records, labels, fn, n_permutations=n_perm, seed=0)
        results[name] = {
            "metrics": {t: metrics[t] for t in ("cognitive", "emotion")},
            "permutation": perm,
        }

    results["feature_importance"] = _feature_importance(records, labels)
    results["n_videos"] = len(records)
    results["n_subjects"] = len(subjects)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    # n_perm=100 (reduced from plan default 200) for tractability with RandomForest(300).
    # See plan2-report.md tractability note.
    res = main(
        REPORTS_ROOT := ROOT / "data" / "reports",
        ROOT / "data" / "modeling" / "route_a_results.json",
        n_perm=100,
    )
    for m in ("ridge", "svr", "rf"):
        c = res[m]["metrics"]["cognitive"]
        e = res[m]["metrics"]["emotion"]
        print(f"{m}: cog r={c['pearson_r']:.3f} R2={c['r2']:.3f} MAE={c['mae']:.2f} | "
              f"emo r={e['pearson_r']:.3f} R2={e['r2']:.3f} MAE={e['mae']:.2f}")
