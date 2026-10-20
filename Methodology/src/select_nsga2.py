"""
Feature selection by NSGA-II (multi-objective evolutionary wrapper).

Objectives: few features, high P/R/F1, no information loss, no redundant features.

  * Candidate pool = top features by RF importance, then redundancy-pruned
    (|Spearman| > CORR_THR) so no redundant features can be selected.
  * NSGA-II minimises two objectives simultaneously: (1 - surrogate F1) and
    subset size, returning a Pareto front.
  * The final subset is the KNEE of the front: the smallest subset whose F1 is
    within EPS of the best (auto-stop before information loss), bounded by CAP.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import common


class SparseBinary(Sampling):
    def __init__(self, p): super().__init__(); self.p = p
    def _do(self, prob, n, **kw):
        rng = np.random.RandomState(1)
        Xb = rng.random((n, prob.n_var)) < self.p
        for r in Xb:
            if not r.any():
                r[rng.randint(prob.n_var)] = True
        return Xb


class FS(ElementwiseProblem):
    """Same three objectives as NSGA-III (accuracy, size, redundancy) so the
    only difference from NSGA-III is the algorithm itself."""
    def __init__(self, feats, scorer, absc):
        self.feats, self.scorer, self.absc = feats, scorer, absc
        super().__init__(n_var=len(feats), n_obj=3, n_ieq_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *a, **k):
        m = np.asarray(x, bool); idx = np.where(m)[0]; n = len(idx)
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
    algo = NSGA2(pop_size=30, sampling=SparseBinary(0.3),
                 crossover=TwoPointCrossover(), mutation=BitflipMutation(prob=0.1),
                 eliminate_duplicates=True)
    print("NSGA-II search (objectives: accuracy, subset size, redundancy)...")
    res = minimize(FS(pool, scorer, absc), algo, get_termination("n_gen", 25), seed=1, verbose=False)
    if res.X is None:
        raise SystemExit("NSGA-II found no solution.")

    front = {}
    for m in np.atleast_2d(res.X).astype(bool):
        sub = tuple(pool[i] for i in np.where(m)[0])
        if not sub:
            continue
        f1 = scorer.score(sub)
        if len(sub) not in front or f1 > front[len(sub)]["f1"]:
            front[len(sub)] = {"n_features": len(sub), "f1": f1, "features": list(sub)}
    rows = sorted(front.values(), key=lambda r: r["n_features"])
    pd.DataFrame(rows).to_csv(common.RESULTS / "nsga2_front.csv", index=False)

    best = common.knee_select(rows)
    print(f"Selected {best['n_features']} features (knee): {best['features']}")
    common.save_features(best["features"], "nsga2_selected_features.csv")


if __name__ == "__main__":
    main()
