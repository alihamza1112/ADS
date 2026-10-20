"""
RQ1 - Fair evaluation harness ("can we beat ISA?").

Given any subset of feature names, train the SAME five classifiers the base
study used (RF, DT, KNN, MLP, NB) with an identical, leakage-free protocol and
report macro precision / recall / F1. This is the yard-stick every candidate
feature-selection method is measured against, so the comparison against ISA is
apples-to-apples on ONE dataset and ONE protocol.

Protocol (fixed for every subset so only the FEATURES differ):
  * Stratified k-fold cross-validation (default 5 folds).
  * MinMaxScaler fit on the TRAIN fold only, applied to the test fold
    (inside a Pipeline) -> no scaling leakage (the base study's script
    re-fit the scaler on the test set; we do not).
  * Macro-averaged precision / recall / F1, averaged across folds.

ISA baseline
  The base study's ISA selected 10 features for Dataset 1. Eight are present
  verbatim in the full pool; `isPedestrianScenario` is derived as
  (totalPedestrians > 0); `speedObstacleWithMinimumDistance` is not recoverable
  from the aggregated pool and is proxied by `feature_obstaclesMinimumSpeed`
  (documented substitution). See README.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Methodology" / "data" / "clean_features.csv"

# ISA's 10 features (base study), mapped onto the full-pool column names.
ISA_FEATURES = [
    "feature_ego_brake",
    "feature_ego_speed",
    "feature_scenarioTrafficLightDemand",
    "feature_totalNPCs",
    "feature_isPedestrianScenario",          # derived below if absent
    "feature_totalRoadUsers",
    "feature_obstaclesMinimumDistance",
    "feature_obstaclesMinimumSpeed",         # proxy for speedObstacleWithMinimumDistance
    "feature_volumeObstacleWithMinimumDistance",
    "feature_obstaclesMaximumSpeed",
]


def load_data():
    df = pd.read_csv(DATA)
    if "feature_isPedestrianScenario" not in df.columns \
            and "feature_totalPedestrians" in df.columns:
        df["feature_isPedestrianScenario"] = (df["feature_totalPedestrians"] > 0).astype(int)
    y = df["y"].values
    X = df.drop(columns=["y"])
    return X, y


def _models(random_state=1):
    return {
        "RF": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "DT": DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                             activation="relu", solver="adam", random_state=random_state),
        "NB": GaussianNB(),
    }


def evaluate_subset(X, y, features, folds=5, models=None, verbose=True):
    """Return {model: {'precision':.., 'recall':.., 'f1':..}} for a feature subset."""
    features = [f for f in features if f in X.columns]
    Xs = X[features].values
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scoring = {"precision": "precision_macro",
               "recall": "recall_macro",
               "f1": "f1_macro"}
    out = {}
    for name, clf in (models or _models()).items():
        pipe = Pipeline([("scale", MinMaxScaler((-1, 1))), ("clf", clf)])
        res = cross_validate(pipe, Xs, y, cv=cv, scoring=scoring, n_jobs=-1)
        out[name] = {k: float(np.mean(res[f"test_{k}"])) for k in scoring}
        if verbose:
            print(f"    {name:4s}  P={out[name]['precision']:.3f}  "
                  f"R={out[name]['recall']:.3f}  F1={out[name]['f1']:.3f}")
    return out


def summarize(result_dict):
    rows = [{"model": m, **v} for m, v in result_dict.items()]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="subsample N rows for a quick run (0 = full data)")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    X, y = load_data()
    if args.sample and args.sample < len(X):
        rng = np.random.RandomState(0)
        idx = rng.choice(len(X), args.sample, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
        print(f"[quick run on {args.sample} sampled rows]")

    print(f"\nISA baseline ({len([f for f in ISA_FEATURES if f in X.columns])} features):")
    isa = evaluate_subset(X, y, ISA_FEATURES, folds=args.folds)

    rng = np.random.RandomState(7)
    rand_feats = list(rng.choice([c for c in X.columns], size=10, replace=False))
    print(f"\nRandom baseline (10 features): {rand_feats}")
    rnd = evaluate_subset(X, y, rand_feats, folds=args.folds)

    isa_df = summarize(isa).assign(selection="ISA")
    rnd_df = summarize(rnd).assign(selection="Random")
    out = pd.concat([isa_df, rnd_df], ignore_index=True)
    out.to_csv(ROOT / "Methodology" / "results" / "baseline_isa_vs_random.csv", index=False)
    print("\nSaved results/baseline_isa_vs_random.csv")
    print(f"ISA mean F1={isa_df['f1'].mean():.3f}   Random mean F1={rnd_df['f1'].mean():.3f}")


if __name__ == "__main__":
    main()
