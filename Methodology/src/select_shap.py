"""
Feature selection by SHAP importance + redundancy pruning + auto-stop.

Objectives: few features, high P/R/F1, no information loss, no redundant features.

Steps:
  1. Rank features by mean |SHAP value| for the critical class (model-based,
     interaction-aware relevance), computed on a Random Forest.
  2. Drop features highly correlated (|Spearman| > CORR_THR) with a higher-ranked
     one (removes redundancy).
  3. Add features in rank order and STOP once the next feature no longer improves
     CV F1 beyond EPS (no information gained); bounded by CAP.
The number of features is therefore data-driven.
"""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
import common


def shap_ranking(X, y, sample=8000, seed=0):
    Xs, ys = common.subsample(X, y, sample, seed)
    rf = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1)
    rf.fit(Xs.values, ys)
    sv = shap.TreeExplainer(rf).shap_values(Xs.values, check_additivity=False)
    arr = np.asarray(sv)
    if arr.ndim == 3:                       # (n, feat, class) or (class, n, feat)
        crit = arr[:, :, 1] if arr.shape[-1] == 2 else arr[1]
    elif isinstance(sv, list):
        crit = np.asarray(sv[1])
    else:
        crit = arr
    imp = np.abs(crit).mean(axis=0)
    return list(pd.Series(imp, index=Xs.columns).sort_values(ascending=False).index)


def main():
    X, y = common.load_xy()
    Xw = common.winsorize(X)
    print(f"Data: {Xw.shape[0]} rows, {Xw.shape[1]} candidate features")

    print("[1] SHAP importance (critical class)...")
    ranked = shap_ranking(Xw, y)

    print(f"[2] Redundancy pruning (|Spearman| > {common.CORR_THR})...")
    kept, dropped = common.redundancy_prune(ranked, Xw)
    print(f"    kept {len(kept)}, dropped {len(dropped)} redundant")

    print(f"[3] Auto-stop forward selection (cap {common.CAP}, eps {common.EPS})...")
    Xs, ys = common.subsample(Xw, y, 8000, seed=1)
    selected = common.forward_auto(kept, common.SurrogateScorer(Xs, ys))
    print(f"    selected {len(selected)} features: {selected}")
    common.save_features(selected, "shap_selected_features.csv")


if __name__ == "__main__":
    main()
