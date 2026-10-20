"""
RQ1 - Data preparation for feature selection.

Builds ONE clean, labelled table from the base study's full feature pool
(complete-feature-set.csv, 60,000 unique scenarios, all 61+ input features)
so that ANY feature-selection method can be run and compared fairly on the
same data and the same protocol.

Design decisions (kept explicit for the thesis write-up):
  * Base table  : complete-feature-set.csv  -- the only file that carries
                  ALL candidate input features together with an outcome.
  * Label (y)   : derived from feature_collisionEvent / feature_collisionEventDemand
                    0 = NotOccurred            -> safe
                    1 = BreakSafetyDistance    -> unsafe (critical)
                    2 = Occurred (collision)   -> unsafe (critical)
                  Binary target: y = 1 (critical) if event in {1,2}, else 0.
                  This matches the base paper's definition of "unsafe": a
                  collision OR a safety-distance violation.
  * Excluded    : the scenario id, and every OUTCOME / post-hoc column, so no
                  label leakage can occur during selection.
  * Candidate features: every remaining feature_* column that is not constant.

Outputs (written to Methodology/data/):
  clean_features.csv   -- candidate input features + binary label `y`
  feature_manifest.csv -- one row per candidate feature: group, dtype,
                          #unique values, |Spearman correlation| with y
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]          # ...\Thesis
DATASET = ROOT / "Methodology" / "Dataset" / "complete-feature-set.csv"
METADATA = ROOT / "Methodology" / "Dataset" / "metadata.csv"   #  paper's Dataset 1 (ISA subset)
SUBSET_CSV = ROOT / "Methodology" / "Dataset" / "paper-subset-full-features.csv"
OUT = ROOT / "Methodology" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# Columns that must never enter feature selection ---------------------------
ID_COLS = ["feature_scenarioIdentifier"]
LEAKY_COLS = [                       # outcome / post-hoc / diversity metrics
    "feature_collisionEvent",
    "feature_collisionEventDemand",
    "feature_collisionProbabilityTimeStamp",
    "feature_collisionProbability",
    "feature_diversity",
    "feature_diversity_",
    "feature_diversityAVG",
    "feature_demand",
]
# Absolute ego position is map/route-specific: a model can memorise WHERE on a
# particular map crashes happen rather than WHAT makes a scenario critical, so
# it does not generalise to new maps. Excluded from the candidate pool for all
# methods (documented for the thesis).
MAP_SPECIFIC_COLS = [
    "feature_ego_positionx",
    "feature_ego_positiony",
    "feature_ego_positionz",
    "feature_ego_comPosition",
]
# Drop quasi-constant columns: a single value covering >= this share of rows
# carries almost no information and lets a search "game" the fast proxy.
QUASI_CONSTANT_FRAC = 0.98


def feature_group(name: str) -> str:
    """Coarse semantic group, used later for redundancy / reporting."""
    n = name.lower()
    if "ego" in n:
        return "ego"
    if any(k in n for k in ("rain", "fog", "wetness", "time")):
        return "weather_light"
    if any(k in n for k in ("trafficlight", "sidewalk", "npc", "pedestrian",
                            "roadusers", "staticobstacle", "totalobstacles")):
        return "road_traffic"
    if "distance" in n:
        return "obstacle_distance"
    if "speed" in n:
        return "obstacle_speed"
    if "velocity" in n:
        return "obstacle_velocity"
    if "acceleration" in n:
        return "obstacle_acceleration"
    if "volume" in n:
        return "obstacle_volume"
    return "other"


def main() -> None:
    print(f"Reading {DATASET} ...")
    df = pd.read_csv(DATASET)
    df.columns = [c.strip() for c in df.columns]
    print(f"  full pool shape: {df.shape}")

    # --- restrict to the base paper's balanced Dataset 1 -----------------
    # The paper used 28,946 balanced scenarios (~45% critical). They are a
    # subset of the 60k pool; metadata.csv identifies them by scenario id.
    # We keep the FULL feature set for exactly those scenarios so results are
    # directly comparable to the paper AND all features remain selectable.
    meta = pd.read_csv(METADATA)
    meta.columns = [c.strip() for c in meta.columns]
    sids = set(meta["Instances"].str.extract(r"(\d+)")[0].astype(int))
    df = df[df["feature_scenarioIdentifier"].isin(sids)].reset_index(drop=True)
    print(f"  restricted to paper's Dataset 1: {df.shape} "
          f"({100*(df['feature_collisionEventDemand']>0).mean():.1f}% critical)")
    df.to_csv(SUBSET_CSV, index=False)
    print(f"  wrote full-feature subset -> {SUBSET_CSV}")

    # --- label ------------------------------------------------------------
    event = df["feature_collisionEventDemand"].astype(int)   # 0/1/2
    y = (event > 0).astype(int)                              # 1 = critical
    print(f"  label balance: safe={int((y == 0).sum())}, "
          f"critical={int((y == 1).sum())} "
          f"({100 * y.mean():.1f}% critical)")

    # --- candidate features ----------------------------------------------
    drop = set(ID_COLS + LEAKY_COLS + MAP_SPECIFIC_COLS)
    candidates = [c for c in df.columns if c not in drop]
    print(f"  excluding {len(MAP_SPECIFIC_COLS)} map-specific position columns")

    X = df[candidates].copy()
    # drop constant columns (zero information)
    const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const:
        print(f"  dropping {len(const)} constant columns: {const}")
        X = X.drop(columns=const)

    # drop quasi-constant columns (dominant value covers >= QUASI_CONSTANT_FRAC)
    quasi = []
    for c in X.columns:
        top = X[c].value_counts(dropna=False, normalize=True).iloc[0]
        if top >= QUASI_CONSTANT_FRAC:
            quasi.append((c, round(float(top), 4)))
    if quasi:
        print(f"  dropping {len(quasi)} quasi-constant columns: {quasi}")
        X = X.drop(columns=[c for c, _ in quasi])

    # every candidate is numeric in this dataset; assert and coerce safely
    from pandas.api.types import is_numeric_dtype
    non_numeric = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if non_numeric:
        print(f"  WARNING non-numeric candidates (label-encoding): {non_numeric}")
        for c in non_numeric:
            X[c] = X[c].astype("category").cat.codes

    # --- manifest ---------------------------------------------------------
    rows = []
    for c in X.columns:
        rho, _ = spearmanr(X[c], y)
        rows.append({
            "feature": c,
            "group": feature_group(c),
            "dtype": str(X[c].dtype),
            "n_unique": int(X[c].nunique()),
            "abs_spearman_with_y": round(abs(rho) if rho == rho else 0.0, 4),
        })
    manifest = (pd.DataFrame(rows)
                .sort_values("abs_spearman_with_y", ascending=False)
                .reset_index(drop=True))

    clean = X.copy()
    clean["y"] = y.values

    clean.to_csv(OUT / "clean_features.csv", index=False)
    manifest.to_csv(OUT / "feature_manifest.csv", index=False)

    print(f"\n  candidate features kept: {X.shape[1]}")
    print(f"  wrote {OUT / 'clean_features.csv'}  ({clean.shape})")
    print(f"  wrote {OUT / 'feature_manifest.csv'}")
    print("\n  Top 15 features by |Spearman| with critical label:")
    print(manifest.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
