"""
Random feature subset (baseline).

Draws 10 features uniformly at random. Any real selection method should do
clearly better than this.
"""
import numpy as np
import common


def main():
    X, _ = common.load_xy()
    rng = np.random.RandomState(7)
    feats = list(rng.choice(list(X.columns), common.CAP, replace=False))
    print(f"Random feature set ({len(feats)} features)")
    common.save_features(feats, "random_selected_features.csv")


if __name__ == "__main__":
    main()
