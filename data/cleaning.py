from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
CLEANED_DIR = ROOT / "cleaned_train"
SUMMARY_PATH = ROOT / "cleaning_summary.csv"


def read_file(path, columns):
    """Read one whitespace-separated dataset file without a header."""
    data = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=columns,
    )
    return data.apply(pd.to_numeric, errors="coerce")


def clean_trip(trip_dir):
    trip_id = trip_dir.name
    files = {
        "track": trip_dir / f"{trip_id}.track",
        "nodes": trip_dir / f"{trip_id}.nodes",
        "arcs": trip_dir / f"{trip_id}.arcs",
        "route": trip_dir / f"{trip_id}.route",
    }

    missing_files = [str(path) for path in files.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Trip {trip_id} is missing required file(s): {', '.join(missing_files)}"
        )

    track = read_file(files["track"], ["lon", "lat", "time"])
    nodes = read_file(files["nodes"], ["lon", "lat"])
    arcs = read_file(files["arcs"], ["from_node", "to_node"])
    route = read_file(files["route"], ["arc_id"])

    valid_gps = (
        track.notna().all(axis=1)
        & track["lon"].between(-180, 180)
        & track["lat"].between(-90, 90)
        & (track["time"] >= 0)
    )

    time_gaps = track["time"].diff()
    invalid_arc = (
        arcs.isna().any(axis=1)
        | (arcs["from_node"] % 1 != 0)
        | (arcs["to_node"] % 1 != 0)
        | ~arcs["from_node"].between(0, len(nodes) - 1)
        | ~arcs["to_node"].between(0, len(nodes) - 1)
    )

    route_ids = route["arc_id"]
    invalid_route = (
        route_ids.isna()
        | (route_ids % 1 != 0)
        | ~route_ids.between(0, len(arcs) - 1)
    )

    cleaned_trip_dir = CLEANED_DIR / trip_id
    cleaned_trip_dir.mkdir(parents=True, exist_ok=True)
    cleaned_track_path = cleaned_trip_dir / f"{trip_id}.track"
    track.loc[valid_gps, ["lon", "lat", "time"]].to_csv(
        cleaned_track_path,
        sep="\t",
        header=False,
        index=False,
    )

    return {
        "trip_id": trip_id,
        "gps_rows": len(track),
        "invalid_gps_rows": int((~valid_gps).sum()),
        "cleaned_gps_rows": int(valid_gps.sum()),
        "non_increasing_timestamps": int((time_gaps <= 0).sum()),
        "gaps_over_2_seconds": int((time_gaps > 2).sum()),
        "invalid_node_coordinate_rows": int(
            (
                nodes.isna().any(axis=1)
                | ~nodes["lon"].between(-180, 180)
                | ~nodes["lat"].between(-90, 90)
            ).sum()
        ),
        "invalid_arc_rows": int(invalid_arc.sum()),
        "route_rows": len(route),
        "invalid_route_arc_ids": int(invalid_route.sum()),
        "cleaned_track_file": str(cleaned_track_path.relative_to(ROOT)),
    }


def main():
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

    trip_dirs = sorted(path for path in TRAIN_DIR.iterdir() if path.is_dir())
    if not trip_dirs:
        raise FileNotFoundError(f"No training trip folders found in {TRAIN_DIR}")

    summary = pd.DataFrame(clean_trip(trip_dir) for trip_dir in trip_dirs)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"Processed {len(summary)} training trips.")
    print(f"Cleaned GPS copies: {CLEANED_DIR}")
    print(f"Audit summary: {SUMMARY_PATH}")
    print(
        summary[
            [
                "trip_id",
                "gps_rows",
                "invalid_gps_rows",
                "cleaned_gps_rows",
                "invalid_arc_rows",
                "invalid_route_arc_ids",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
