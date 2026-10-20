"""Shared helpers for the RQ1 feature-selection methods.

Selection objectives shared by every method:
  * few features,
  * high P / R / F1 (stop before information is lost),
  * no redundant (highly correlated) features.
The final feature count is chosen automatically by each method (data-driven),
bounded only by CAP.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Methodology" / "data" / "clean_features.csv"
RESULTS = ROOT / "Methodology" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CAP = 10          # hard upper bound on subset size
EPS = 0.002       # "no information loss" tolerance for the auto-stop rule
CORR_THR = 0.85   # redundancy threshold (|Spearman|)


def load_xy():
    df = pd.read_csv(DATA)
    if "feature_isPedestrianScenario" not in df.columns \
            and "feature_totalPedestrians" in df.columns:
        df["feature_isPedestrianScenario"] = (df["feature_totalPedestrians"] > 0).astype(int)
    y = df["y"].values
    X = df.drop(columns=["y"])
    return X, y


def subsample(X, y, n, seed=0):
    if not n or n >= len(X):
        return X.reset_index(drop=True), y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def winsorize(X, lo=0.01, hi=0.99):
    """Clip each feature to its 1st/99th percentile so outliers don't distort
    scaling or selection."""
    return X.clip(lower=X.quantile(lo), upper=X.quantile(hi), axis=1)


def redundancy_prune(ranked, X, thr=CORR_THR):
    """Walk a relevance-ranked list; drop a feature if |Spearman| with an
    already-kept (higher-ranked) feature exceeds `thr`. Returns (kept, dropped)."""
    corr = X[ranked].corr(method="spearman").abs()
    kept, dropped = [], []
    for f in ranked:
        if any(corr.loc[f, k] > thr for k in kept):
            dropped.append(f)
        else:
            kept.append(f)
    return kept, dropped


def redundancy(X, feats):
    feats = [f for f in feats if f in X.columns]
    if len(feats) < 2:
        return 0.0
    c = X[feats].corr(method="spearman").abs().values
    n = len(feats)
    return float((c.sum() - n) / (n * (n - 1)))


class SurrogateScorer:
    """Mean macro-F1 of RF+DT+KNN on a fixed holdout — a fast, deterministic
    surrogate for the full 5-classifier mean. Drives the selection search."""
    def __init__(self, X, y, seed=42):
        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y)
        self.clfs = [RandomForestClassifier(60, random_state=0, n_jobs=-1),
                     DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=0),
                     KNeighborsClassifier(5)]
        self._c = {}

    def score(self, feats):
        k = tuple(sorted(feats))
        if not k:
            return 0.0
        if k in self._c:
            return self._c[k]
        sc = MinMaxScaler((-1, 1)).fit(self.Xtr[list(k)])
        a = sc.transform(self.Xtr[list(k)]); b = sc.transform(self.Xte[list(k)])
        f1 = float(np.mean([f1_score(self.yte, c.fit(a, self.ytr).predict(b), average="macro")
                            for c in self.clfs]))
        self._c[k] = f1
        return f1


def forward_auto(ranked, scorer, cap=CAP, eps=EPS):
    """Add features in ranked order; keep the SMALLEST prefix whose score is
    within `eps` of the best score reached within `cap` (auto-stop: adding more
    features no longer adds information)."""
    scores, chosen = [], []
    for f in ranked[:cap]:
        chosen.append(f)
        scores.append(scorer.score(chosen))
    best = max(scores)
    for i, s in enumerate(scores):
        if s >= best - eps:
            return ranked[:i + 1]
    return ranked[:cap]


def knee_select(front, cap=CAP, eps=EPS):
    """From a list of {n_features, f1, features} subsets, pick the SMALLEST one
    (<= cap) whose F1 is within `eps` of the best (the knee of the trade-off)."""
    feas = [r for r in front if r["n_features"] <= cap]
    if not feas:
        feas = front
    best = max(r["f1"] for r in feas)
    keep = [r for r in feas if r["f1"] >= best - eps]
    return min(keep, key=lambda r: r["n_features"])


def save_features(feats, name):
    pd.Series(list(feats)).to_csv(RESULTS / name, index=False, header=["feature"])
    print(f"  saved {len(list(feats))} features -> results/{name}")
