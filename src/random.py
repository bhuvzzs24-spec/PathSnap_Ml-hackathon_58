from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __package__:
    from .logistic import (
        CLEANED_TRAIN_DIR,
        CANDIDATE_POOL_SIZE,
        CANDIDATES_PER_OBSERVATION,
        FEATURE_NAMES,
        STUDENT_TEST_DIR,
        TRAIN_DIR,
        build_trip_candidates,
    )
else:
    from logistic import (
        CLEANED_TRAIN_DIR,
        CANDIDATE_POOL_SIZE,
        CANDIDATES_PER_OBSERVATION,
        FEATURE_NAMES,
        STUDENT_TEST_DIR,
        TRAIN_DIR,
        build_trip_candidates,
    )


MODEL_PATH = Path(__file__).resolve().parent / "random_forest_map_matcher.joblib"
RESULTS_PATH = Path(__file__).resolve().parent / "random_forest_student_test_results.csv"

SEED = 42
N_ESTIMATORS = 200
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 1


def main():
    if not CLEANED_TRAIN_DIR.is_dir():
        raise FileNotFoundError(
            f"Cleaned training data not found: {CLEANED_TRAIN_DIR}. "
            "Run Student folder\\cleaning.py first."
        )
    if not TRAIN_DIR.is_dir() or not STUDENT_TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected both training and student-test folders under "
            f"{TRAIN_DIR.parent}"
        )

    train_trip_dirs = sorted(path for path in TRAIN_DIR.iterdir() if path.is_dir())
    test_trip_dirs = sorted(path for path in STUDENT_TEST_DIR.iterdir() if path.is_dir())
    if not train_trip_dirs or not test_trip_dirs:
        raise FileNotFoundError("Training or student-test trip folders were not found.")

    started = time.perf_counter()
    training_features = []
    training_labels = []
    training_trip_ids = []
    train_coverage = []

    for trip_dir in train_trip_dirs:
        features, _, _, labels, covered = build_trip_candidates(
            trip_dir,
            use_cleaned_track=True,
        )
        train_coverage.append(float(covered.mean()))
        if covered.any():
            training_features.append(
                features[covered].reshape(-1, len(FEATURE_NAMES))
            )
            training_labels.append(labels[covered].reshape(-1).astype(np.int8))
            training_trip_ids.append(trip_dir.name)

    if not training_features:
        raise ValueError("No training GPS observations had a reference-route candidate.")

    x_train = np.concatenate(training_features)
    y_train = np.concatenate(training_labels)
    if np.unique(y_train).size != 2:
        raise ValueError(
            "Candidate training data must contain both positive and negative labels."
        )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    positive_probability_index = int(np.flatnonzero(model.classes_ == 1)[0])
    trip_results = []
    for trip_dir in test_trip_dirs:
        features, candidate_ids, target_ids, labels, covered = build_trip_candidates(
            trip_dir,
            use_cleaned_track=False,
        )
        all_features = features.reshape(-1, len(FEATURE_NAMES))
        positive_probabilities = model.predict_proba(all_features)[
            :, positive_probability_index
        ].reshape(features.shape[:2])
        selected = np.argmax(positive_probabilities, axis=1)
        predicted_ids = candidate_ids[np.arange(len(candidate_ids)), selected]
        correct = predicted_ids == target_ids
        covered_count = int(covered.sum())
        correct_covered = int((correct & covered).sum())
        trip_results.append(
            {
                "trip_id": trip_dir.name,
                "observations": len(target_ids),
                "candidate_covered_observations": covered_count,
                "correct_on_covered_observations": correct_covered,
                "candidate_coverage": float(covered.mean()),
                "top1_accuracy_on_covered": (
                    correct_covered / covered_count if covered_count else float("nan")
                ),
                "top1_accuracy_all_observations": float(correct.mean()),
            }
        )

    result_frame = pd.DataFrame(trip_results)
    total_observations = int(result_frame["observations"].sum())
    total_covered = int(result_frame["candidate_covered_observations"].sum())
    total_correct_covered = int(
        result_frame["correct_on_covered_observations"].sum()
    )
    result_frame.to_csv(RESULTS_PATH, index=False)
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "candidates_per_observation": CANDIDATES_PER_OBSERVATION,
            "label_method": "nearest reference-route arc to each GPS observation",
            "training_trip_ids": training_trip_ids,
            "seed": SEED,
            "n_estimators": N_ESTIMATORS,
        },
        MODEL_PATH,
    )

    print(f"Training trips: {len(train_trip_dirs)}")
    print(f"Training candidate rows: {len(y_train):,}")
    print(f"Training candidate positive rate: {y_train.mean():.4f}")
    print(f"Mean training candidate coverage: {np.mean(train_coverage):.2%}")
    print(f"Student-test trips: {len(test_trip_dirs)}")
    print(result_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "Overall student-test candidate coverage: "
        f"{total_covered / total_observations:.2%}"
    )
    print(
        "Overall top-1 accuracy on covered observations: "
        f"{total_correct_covered / total_covered:.4f}"
        if total_covered
        else "Overall top-1 accuracy on covered observations: N/A"
    )
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Student-test results saved to: {RESULTS_PATH}")
    print(f"Elapsed time: {time.perf_counter() - started:.1f} seconds")


if __name__ == "__main__":
    main()
