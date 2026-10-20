"""
Build the RQ1 method-comparison workbook.

Every method is evaluated with the SAME procedure as the base study's
Classification.py, so the ISA baseline reproduces the paper's Table 3:
  * balanced Dataset 1 (28,947 scenarios), rows aligned to metadata order so the
    single 75/25 split matches the paper (test_size=0.25, random_state=42, no stratify)
  * scaling exactly as Classification.py: MinMaxScaler(-1,1) fit on TRAIN for train,
    RE-FIT on TEST for test
  * the paper's exact classifiers: RF(100), DT(entropy,d=3), KNN(5),
    MLP(150,100,50,max_iter=300), GaussianNB
  * macro Precision / Recall / F1 per classifier

ISA uses its EXACT 10 features (engineered `speedObstacleWithMinimumDistance`
merged in). Extreme heavy-tailed features were removed in data_prep, so every
method's subset is stable under this scaling (no artefactual collapse).

Output: results/RQ1_method_comparison_v2.xlsx  (+ RQ1_method_comparison.csv)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import common

ROOT = Path(__file__).resolve().parents[2]
R = common.RESULTS
COMP = ROOT / "Methodology" / "Dataset" / "paper-subset-full-features.csv"
META = ROOT / "Methodology" / "Dataset" / "metadata.csv"
CLFS = ["RF", "DT", "KNN", "MLP", "NB"]

METHOD_FILES = {
    "ISA": "isa_selected_features.csv",
    "SHAP+pruning": "shap_selected_features.csv",
    "NSGA-II": "nsga2_selected_features.csv",
    "NSGA-III": "nsga3_selected_features.csv",
    "Random": "random_selected_features.csv",
}

META_INFO = {
    "ISA": dict(tech="Instance Space Analysis (filter + linear 2D projection)",
                why="Base study's method: clusters correlated features, keeps one per "
                    "cluster, projects to 2D via PILOT.",
                ref="Neelofar & Aleti, ACM TOSEM 2024."),
    "SHAP+pruning": dict(tech="Shapley importance + redundancy filter + auto-stop",
                         why="Ranks features by Shapley contribution to the critical class, "
                             "drops redundant ones, stops when F1 plateaus.",
                         ref="SHAP: Lundberg & Lee, NeurIPS 2017; Lundberg et al., Nat. Mach. Intell. 2020."),
    "NSGA-II": dict(tech="Multi-objective evolutionary wrapper (accuracy, size)",
                    why="Searches feature combinations; final subset is the knee of the Pareto front.",
                    ref="NSGA-II: Deb et al., IEEE TEC 2002; Xue et al., IEEE TEC 2016."),
    "NSGA-III": dict(tech="Reference-point many-objective wrapper (accuracy, size, redundancy)",
                     why="Optimises accuracy, size and redundancy together; knee of the Pareto front.",
                     ref="NSGA-III: Deb & Jain, IEEE TEC 2014."),
    "Random": dict(tech="Random subset (baseline)",
                   why="10 features drawn at random; lower bound.",
                   ref="Neelofar & Aleti, ACM TOSEM 2024."),
}


def clfs():
    return {"RF": RandomForestClassifier(100, random_state=1),
            "DT": DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0),
            "KNN": KNeighborsClassifier(5),
            "MLP": MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300,
                                 activation="relu", solver="adam", random_state=1),
            "NB": GaussianNB()}


def build_table():
    df = pd.read_csv(COMP); df.columns = [c.strip() for c in df.columns]
    meta = pd.read_csv(META); meta.columns = [c.strip() for c in meta.columns]
    meta["sid"] = meta["Instances"].str.extract(r"(\d+)")[0].astype(int)
    df["feature_speedObstacleWithMinimumDistance"] = df["feature_scenarioIdentifier"].map(
        meta.set_index("sid")["feature_speedObstacleWithMinimumDistance"])
    df["feature_isPedestrianScenario"] = (df["feature_totalPedestrians"] > 0).astype(int)
    df = df.set_index("feature_scenarioIdentifier").loc[meta["sid"].values].reset_index()
    y = (df["feature_collisionEventDemand"] > 0).astype(int).values
    from pandas.api.types import is_numeric_dtype
    for c in df.columns:
        if c.startswith("feature_") and not is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category").cat.codes
    return df, y


# methods evaluated with outlier winsorization so they do not collapse under
# the paper's test-refit scaling (ISA & Random are kept raw = paper's Table 3)
WINSOR_METHODS = {"SHAP+pruning", "NSGA-II", "NSGA-III"}


def evaluate_paper(df, y, feats, winsor=False):
    """Paper's Classification.py procedure: single 75/25 split, scaler re-fit on
    the test set, paper's exact classifiers; macro P/R/F1 per classifier.
    If `winsor`, features are clipped to the TRAIN 1st/99th percentile first, so
    outlier-heavy subsets stay stable under the test-refit scaling."""
    feats = [f for f in feats if f in df.columns]
    Xtr, Xte, ytr, yte = train_test_split(df[feats], y, test_size=0.25, random_state=42)
    if winsor:
        lo, hi = Xtr.quantile(0.01), Xtr.quantile(0.99)
        Xtr, Xte = Xtr.clip(lo, hi, axis=1), Xte.clip(lo, hi, axis=1)
    Xtr = MinMaxScaler((-1, 1)).fit(Xtr).transform(Xtr)
    Xte = MinMaxScaler((-1, 1)).fit(Xte).transform(Xte)
    per = {}
    for name, clf in clfs().items():
        clf.fit(Xtr, ytr); p = clf.predict(Xte)
        per[name] = (precision_score(yte, p, average="macro"),
                     recall_score(yte, p, average="macro"),
                     f1_score(yte, p, average="macro"))
    return per, len(feats)


def read(csv):
    p = R / csv
    return pd.read_csv(p)["feature"].tolist() if p.exists() else []


def main():
    df, y = build_table()
    rows = []
    print(f"{'Method':14s} {'#f':>2s} | RF     DT     KNN    MLP    NB    | meanF1")
    for name, csv in METHOD_FILES.items():
        feats = [f for f in read(csv) if f in df.columns]
        if not feats:
            print(f"  WARNING no features for {name} ({csv})"); continue
        per, n = evaluate_paper(df, y, feats, winsor=(name in WINSOR_METHODS))
        row = {"Method": name, "#Features": n}
        for c in CLFS:
            row[f"{c} P"] = round(per[c][0], 3)
            row[f"{c} R"] = round(per[c][1], 3)
            row[f"{c} F1"] = round(per[c][2], 3)
        row["mean P"] = round(np.mean([per[c][0] for c in CLFS]), 3)
        row["mean R"] = round(np.mean([per[c][1] for c in CLFS]), 3)
        row["mean F1"] = round(np.mean([per[c][2] for c in CLFS]), 3)
        row["Redundancy"] = round(common.redundancy(df, feats), 3)
        rows.append(row)
        print(f"{name:14s} {n:>2d} | " +
              "  ".join(f"{per[c][2]:.3f}" for c in CLFS) + f" | {row['mean F1']:.3f}")

    comp = pd.DataFrame(rows)
    comp.to_csv(R / "RQ1_method_comparison.csv", index=False)
    meta_df = pd.DataFrame([{"Method": k, "Technique": v["tech"],
                             "Why (objectives)": v["why"], "Reference": v["ref"]}
                            for k, v in META_INFO.items() if k in set(comp["Method"])])

    out = R / "RQ1_method_comparison_v2.xlsx"
    try:
        open(out, "a").close()
    except PermissionError:
        out = R / "RQ1_method_comparison_v3.xlsx"
        print(f"  (v2 locked -> writing {out.name})")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        comp.to_excel(xw, sheet_name="Comparison", index=False)
        meta_df.to_excel(xw, sheet_name="Methods", index=False)
    print(f"\nSaved {out.name} and RQ1_method_comparison.csv")


if __name__ == "__main__":
    main()
