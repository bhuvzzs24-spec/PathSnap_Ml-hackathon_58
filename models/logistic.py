from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Student folder"
CLEANED_TRAIN_DIR = DATA_ROOT / "cleaned_train"
TRAIN_DIR = DATA_ROOT / "train"
STUDENT_TEST_DIR = DATA_ROOT / "student_test"
MODEL_PATH = Path(__file__).resolve().parent / "logistic_map_matcher.joblib"

CANDIDATE_POOL_SIZE = 32
CANDIDATES_PER_OBSERVATION = 8
FEATURE_NAMES = [
    "distance_to_road_m",
    "distance_rank",
    "distance_gap_m",
    "segment_length_m",
    "projection_fraction",
    "heading_cosine",
    "gps_speed_mps",
    "time_gap_s",
    "previous_distance_m",
    "next_distance_m",
    "previous_heading_cosine",
    "next_heading_cosine",
    "endpoint_distance_m",
    "step_distance_m",
    "local_turn_cosine",
]


def read_numeric_file(path, columns):
    data = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        header=None,
        names=columns,
    )
    return data.apply(pd.to_numeric, errors="coerce")


def read_valid_track(path):
    track = read_numeric_file(path, ["lon", "lat", "time"])
    valid = (
        track.notna().all(axis=1)
        & track["lon"].between(-180, 180)
        & track["lat"].between(-90, 90)
        & (track["time"] >= 0)
    )
    track = track.loc[valid]
    if track.empty:
        raise ValueError(f"No valid GPS observations in {path}")

    return (
        track["lon"].to_numpy(dtype=np.float64),
        track["lat"].to_numpy(dtype=np.float64),
        track["time"].to_numpy(dtype=np.float64),
    )


def project_coordinates(lon, lat, reference_latitude):
    x = lon * (111_320.0 * np.cos(np.deg2rad(reference_latitude)))
    y = lat * 110_540.0
    return np.column_stack((x, y))


def point_to_segment_features(points, candidate_ids, segment_starts, segment_vectors):
    vectors = segment_vectors[candidate_ids]
    starts = segment_starts[candidate_ids]
    lengths_squared = np.einsum("nki,nki->nk", vectors, vectors)
    lengths_squared = np.maximum(lengths_squared, 1e-12)

    offsets = points[:, None, :] - starts
    fractions = np.einsum("nki,nki->nk", offsets, vectors) / lengths_squared
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = starts + fractions[:, :, None] * vectors
    distances = np.linalg.norm(points[:, None, :] - projections, axis=2)

    return distances, fractions


def nearest_route_arc_labels(points, route_arc_ids, segment_starts, segment_vectors):
    route_ids = np.unique(route_arc_ids)
    starts = segment_starts[route_ids]
    vectors = segment_vectors[route_ids]
    lengths_squared = np.maximum(np.einsum("ij,ij->i", vectors, vectors), 1e-12)
    labels = np.empty(len(points), dtype=np.int64)

    # Chunk the calculation to bound temporary memory on long trips.
    for start in range(0, len(points), 512):
        end = min(start + 512, len(points))
        offsets = points[start:end, None, :] - starts[None, :, :]
        fractions = np.einsum("nri,ri->nr", offsets, vectors) / lengths_squared
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = starts[None, :, :] + fractions[:, :, None] * vectors[None, :, :]
        distances_squared = np.einsum(
            "nri,nri->nr", points[start:end, None, :] - projections,
            points[start:end, None, :] - projections,
        )
        labels[start:end] = route_ids[np.argmin(distances_squared, axis=1)]

    return labels


def build_trip_candidates(trip_dir, use_cleaned_track):
    trip_id = trip_dir.name
    track_path = (
        CLEANED_TRAIN_DIR / trip_id / f"{trip_id}.track"
        if use_cleaned_track
        else trip_dir / f"{trip_id}.track"
    )
    if not track_path.is_file():
        raise FileNotFoundError(
            f"Missing track file {track_path}. Run Student folder\\cleaning.py first "
            "to create the cleaned training tracks."
        )

    node_path = trip_dir / f"{trip_id}.nodes"
    arc_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"
    for path in (node_path, arc_path, route_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required trip file: {path}")

    lon, lat, timestamps = read_valid_track(track_path)
    nodes = read_numeric_file(node_path, ["lon", "lat"])
    arcs = read_numeric_file(arc_path, ["from_node", "to_node"])
    route = read_numeric_file(route_path, ["arc_id"])["arc_id"]

    if nodes.empty or nodes.isna().any().any():
        raise ValueError(f"Invalid road-network nodes in trip {trip_id}")
    if arcs.empty or arcs.isna().any().any():
        raise ValueError(f"Invalid road-network arcs in trip {trip_id}")
    if (
        (arcs[["from_node", "to_node"]] % 1 != 0).any().any()
        or not arcs["from_node"].between(0, len(nodes) - 1).all()
        or not arcs["to_node"].between(0, len(nodes) - 1).all()
    ):
        raise ValueError(f"Arc endpoint outside the node table in trip {trip_id}")
    if (
        route.empty
        or route.isna().any()
        or (route % 1 != 0).any()
        or not route.between(0, len(arcs) - 1).all()
    ):
        raise ValueError(f"Invalid reference-route arc ID in trip {trip_id}")

    node_coordinates = project_coordinates(
        nodes["lon"].to_numpy(dtype=np.float64),
        nodes["lat"].to_numpy(dtype=np.float64),
        float(np.median(lat)),
    )
    points = project_coordinates(lon, lat, float(np.median(lat)))

    arc_from = arcs["from_node"].to_numpy(dtype=np.int64)
    arc_to = arcs["to_node"].to_numpy(dtype=np.int64)
    segment_starts = node_coordinates[arc_from]
    segment_vectors = node_coordinates[arc_to] - segment_starts
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)

    route_arc_ids = route.to_numpy(dtype=np.int64)
    target_arc_ids = nearest_route_arc_labels(
        points, route_arc_ids, segment_starts, segment_vectors
    )

    midpoints = segment_starts + segment_vectors / 2.0
    tree = cKDTree(midpoints)
    pool_size = min(CANDIDATE_POOL_SIZE, len(midpoints))
    nearest_midpoints = tree.query(points, k=pool_size, workers=-1)[1]
    if pool_size == 1:
        nearest_midpoints = nearest_midpoints[:, None]

    distances, fractions = point_to_segment_features(
        points, nearest_midpoints, segment_starts, segment_vectors
    )
    candidate_count = min(CANDIDATES_PER_OBSERVATION, pool_size)
    closest_positions = np.argsort(distances, axis=1)[:, :candidate_count]
    candidate_ids = np.take_along_axis(nearest_midpoints, closest_positions, axis=1)
    distances = np.take_along_axis(distances, closest_positions, axis=1)
    fractions = np.take_along_axis(fractions, closest_positions, axis=1)

    previous_points = points.copy()
    next_points = points.copy()
    previous_movement = np.zeros_like(points)
    next_movement = np.zeros_like(points)
    time_gaps = np.zeros(len(points), dtype=np.float64)
    if len(points) > 1:
        previous_points[1:] = points[:-1]
        next_points[:-1] = points[1:]
        previous_movement[1:] = points[1:] - points[:-1]
        next_movement[:-1] = points[1:] - points[:-1]
        time_gaps[1:] = timestamps[1:] - timestamps[:-1]
        time_gaps[0] = time_gaps[1]

    previous_distances, _ = point_to_segment_features(
        previous_points, candidate_ids, segment_starts, segment_vectors
    )
    next_distances, _ = point_to_segment_features(
        next_points, candidate_ids, segment_starts, segment_vectors
    )
    step_distances = np.linalg.norm(previous_movement, axis=1)
    if len(points) > 1:
        step_distances[0] = np.linalg.norm(next_movement[0])

    center_movement = previous_movement + next_movement
    center_magnitude = np.linalg.norm(center_movement, axis=1)
    cancelled_heading = center_magnitude <= 1e-6
    center_movement[cancelled_heading] = previous_movement[cancelled_heading]
    missing_previous = np.linalg.norm(previous_movement, axis=1) <= 1e-6
    center_movement[missing_previous] = next_movement[missing_previous]
    missing_next = np.linalg.norm(next_movement, axis=1) <= 1e-6
    center_movement[missing_next] = previous_movement[missing_next]

    road_headings = np.arctan2(
        segment_vectors[candidate_ids, 1],
        segment_vectors[candidate_ids, 0],
    )
    road_headings = np.where(
        segment_lengths[candidate_ids] > 1e-6,
        road_headings,
        0.0,
    )

    def heading_cosine(movement_vectors):
        headings = np.arctan2(movement_vectors[:, 1], movement_vectors[:, 0])
        alignment = np.cos(headings[:, None] - road_headings)
        available = np.linalg.norm(movement_vectors, axis=1) > 1e-6
        return np.where(available[:, None], alignment, 0.0)

    current_heading_cosine = heading_cosine(center_movement)
    previous_heading_cosine = heading_cosine(previous_movement)
    next_heading_cosine = heading_cosine(next_movement)
    previous_available = np.linalg.norm(previous_movement, axis=1) > 1e-6
    next_available = np.linalg.norm(next_movement, axis=1) > 1e-6
    previous_heading_cosine = np.where(
        previous_available[:, None],
        previous_heading_cosine,
        current_heading_cosine,
    )
    next_heading_cosine = np.where(
        next_available[:, None],
        next_heading_cosine,
        current_heading_cosine,
    )

    movement_distance = np.linalg.norm(previous_movement, axis=1)
    if len(points) > 1:
        movement_distance[0] = step_distances[0]
    speeds = np.divide(
        movement_distance,
        time_gaps,
        out=np.zeros_like(movement_distance),
        where=time_gaps > 0,
    )
    rank_scale = max(distances.shape[1] - 1, 1)
    distance_ranks = (
        np.argsort(np.argsort(distances, axis=1), axis=1) / rank_scale
    )
    distance_gaps = distances - distances.min(axis=1, keepdims=True)
    segment_lengths_for_candidates = segment_lengths[candidate_ids]
    endpoint_distances = (
        np.minimum(fractions, 1.0 - fractions) * segment_lengths_for_candidates
    )

    previous_magnitudes = np.linalg.norm(previous_movement, axis=1)
    next_magnitudes = np.linalg.norm(next_movement, axis=1)
    turn_denominator = previous_magnitudes * next_magnitudes
    local_turn_cosines = np.divide(
        np.einsum("ni,ni->n", previous_movement, next_movement),
        turn_denominator,
        out=np.ones_like(turn_denominator),
        where=turn_denominator > 1e-6,
    )
    local_turn_cosines = np.clip(local_turn_cosines, -1.0, 1.0)

    features = np.stack(
        [
            distances,
            np.broadcast_to(distance_ranks, distances.shape),
            distance_gaps,
            segment_lengths_for_candidates,
            fractions,
            current_heading_cosine,
            np.broadcast_to(speeds[:, None], distances.shape),
            np.broadcast_to(time_gaps[:, None], distances.shape),
            previous_distances,
            next_distances,
            previous_heading_cosine,
            next_heading_cosine,
            endpoint_distances,
            np.broadcast_to(step_distances[:, None], distances.shape),
            np.broadcast_to(local_turn_cosines[:, None], distances.shape),
        ],
        axis=2,
    ).astype(np.float32)

    candidate_arc_ids = candidate_ids.astype(np.int64)
    labels = candidate_arc_ids == target_arc_ids[:, None]
    covered = labels.any(axis=1)
    return features, candidate_arc_ids, target_arc_ids, labels, covered


def main():
    if not CLEANED_TRAIN_DIR.is_dir():
        raise FileNotFoundError(
            f"Cleaned training data not found: {CLEANED_TRAIN_DIR}. "
            "Run Student folder\\cleaning.py first."
        )
    if not TRAIN_DIR.is_dir() or not STUDENT_TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected both training and student-test folders under {DATA_ROOT}"
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
            trip_dir, use_cleaned_track=True
        )
        train_coverage.append(float(covered.mean()))
        if covered.any():
            training_features.append(features[covered].reshape(-1, len(FEATURE_NAMES)))
            training_labels.append(labels[covered].reshape(-1).astype(np.int8))
            training_trip_ids.append(trip_dir.name)

    if not training_features:
        raise ValueError("No training GPS observations had a reference-route candidate.")

    x_train = np.concatenate(training_features)
    y_train = np.concatenate(training_labels)
    if np.unique(y_train).size != 2:
        raise ValueError("Candidate training data must contain both positive and negative labels.")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=500,
            random_state=42,
        ),
    )
    model.fit(x_train, y_train)

    positive_probability_index = int(
        np.flatnonzero(model[-1].classes_ == 1)[0]
    )
    trip_results = []
    for trip_dir in test_trip_dirs:
        features, candidate_ids, target_ids, labels, covered = build_trip_candidates(
            trip_dir, use_cleaned_track=False
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
    result_frame.to_csv(Path(__file__).resolve().parent / "logistic_student_test_results.csv", index=False)
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "candidates_per_observation": CANDIDATES_PER_OBSERVATION,
            "label_method": "nearest reference-route arc to each GPS observation",
            "training_trip_ids": training_trip_ids,
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
    print(
        "Student-test metrics saved to: "
        f"{Path(__file__).resolve().parent / 'logistic_student_test_results.csv'}"
    )
    print(f"Elapsed time: {time.perf_counter() - started:.1f} seconds")


if __name__ == "__main__":
    main()
