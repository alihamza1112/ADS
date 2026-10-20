"""
Feature selection by NSGA-III (reference-point many-objective wrapper).

Objectives: few features, high P/R/F1, no information loss, no redundant features.

NSGA-III (Deb & Jain, IEEE TEC 2014) optimises THREE objectives at once via
reference directions:
  1. (1 - surrogate F1)   -> high accuracy
  2. subset size          -> few features
  3. mean pairwise |Spearman| among selected features -> no redundancy
The final subset is the KNEE of the Pareto front: the smallest subset whose F1
is within EPS of the best (auto-stop before information loss), bounded by CAP.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
import common
from select_nsga2 import SparseBinary


class FS3(ElementwiseProblem):
    def __init__(self, feats, scorer, absc):
        self.feats, self.scorer, self.absc = feats, scorer, absc
        super().__init__(n_var=len(feats), n_obj=3, n_ieq_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *a, **k):
        m = np.asarray(x, bool)
        idx = np.where(m)[0]; n = len(idx)
        if n == 0:
            out["F"] = [1.0, 0.0, 1.0]; return
        f1 = self.scorer.score([self.feats[i] for i in idx])
        red = 0.0 if n == 1 else (self.absc[np.ix_(idx, idx)].sum() - n) / (n * (n - 1))
        out["F"] = [1.0 - f1, float(n), red]


def main():
    X, y = common.load_xy()
    Xw = common.winsorize(X)

    rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(Xw.values, y)
    ranked = list(pd.Series(rf.feature_importances_, index=Xw.columns)
                  .sort_values(ascending=False).index)[:25]
    pool, _ = common.redundancy_prune(ranked, Xw)
    absc = Xw[pool].corr(method="spearman").abs().values
    print(f"Candidate pool after redundancy pruning: {len(pool)} features")

    scorer = common.SurrogateScorer(Xw[pool], y)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=8)
    algo = NSGA3(ref_dirs=ref_dirs, sampling=SparseBinary(0.3),
                 crossover=TwoPointCrossover(), mutation=BitflipMutation(prob=0.1),
                 eliminate_duplicates=True)
    print(f"NSGA-III search (objectives: accuracy, size, redundancy; {len(ref_dirs)} ref dirs)...")
    res = minimize(FS3(pool, scorer, absc), algo, get_termination("n_gen", 25), seed=1, verbose=False)
    if res.X is None:
        raise SystemExit("NSGA-III found no solution.")

    front = {}
    for m in np.atleast_2d(res.X).astype(bool):
        sub = tuple(pool[i] for i in np.where(m)[0])
        if not sub:
            continue
        f1 = scorer.score(sub)
        if len(sub) not in front or f1 > front[len(sub)]["f1"]:
            front[len(sub)] = {"n_features": len(sub), "f1": f1, "features": list(sub)}
    rows = sorted(front.values(), key=lambda r: r["n_features"])
    pd.DataFrame(rows).to_csv(common.RESULTS / "nsga3_front.csv", index=False)

    best = common.knee_select(rows)
    print(f"Selected {best['n_features']} features (knee): {best['features']}")
    common.save_features(best["features"], "nsga3_selected_features.csv")


if __name__ == "__main__":
    main()
