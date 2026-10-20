"""
ISA feature set (baseline).

The 10 features identified by Instance Space Analysis in the base study
(Neelofar & Aleti 2024, Dataset 1). These are given (already known), not
re-selected here; this script just writes them out so the comparison evaluates
ISA like any other method. Uses the EXACT feature names (incl. the engineered
`speedObstacleWithMinimumDistance`) so ISA reproduces the paper's Table 3.
"""
import common

ISA_EXACT = [
    "feature_ego_brake",
    "feature_ego_speed",
    "feature_scenarioTrafficLightDemand",
    "feature_totalNPCs",
    "feature_isPedestrianScenario",
    "feature_totalRoadUsers",
    "feature_obstaclesMinimumDistance",
    "feature_speedObstacleWithMinimumDistance",
    "feature_volumeObstacleWithMinimumDistance",
    "feature_obstaclesMaximumSpeed",
]


def main():
    print(f"ISA feature set ({len(ISA_EXACT)} features)")
    common.save_features(ISA_EXACT, "isa_selected_features.csv")


if __name__ == "__main__":
    main()
