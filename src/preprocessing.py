"""Reusable GPS-track validation and ordering helpers.

The full trip/candidate preprocessing used for model training remains in
logistic.py (build_trip_candidates); these helpers are kept separate for
clarity and do not replace that tested pipeline.
"""
import numpy as np

def clean_track_array(track):
    """Validate [longitude, latitude, timestamp], sort by time, drop duplicate times."""
    arr = np.asarray(track, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Expected an N x 3 [lon, lat, timestamp] array")
    valid = (
        np.isfinite(arr[:, :3]).all(axis=1)
        & (arr[:,  0] >= -180) & (arr[:, 0] <= 180)
        & (arr[:, 1] >= -90) & (arr[:, 1] <= 90)
        & (arr[:, 2] >= 0)
    )
    arr = arr[valid]
    arr = arr[np.argsort(arr[:, 2], kind="stable")]
    if len(arr):
        arr = arr[np.r_[True, np.diff(arr[:, 2]) > 0]]
    if not len(arr):
        raise ValueError("No valid GPS observations remain after cleaning")
    return arr
