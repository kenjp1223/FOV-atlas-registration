"""
BigWarp landmark I/O.

BigWarp exports landmarks as a headerless CSV with 6 columns:
    Name, Active, Moving-X, Moving-Y, Fixed-X, Fixed-Y

Where:
    Moving = atlas slice (source that was warped in BigWarp)
    Fixed  = target histology image (reference)

All coordinates are in pixel space.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_landmarks(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a BigWarp landmark CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the headerless BigWarp CSV.

    Returns
    -------
    moving_pts : np.ndarray, shape (N, 2)
        Pixel coordinates in atlas slice space (X, Y).
    fixed_pts : np.ndarray, shape (N, 2)
        Pixel coordinates in target image space (X, Y).
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["name", "active", "mov_x", "mov_y", "fix_x", "fix_y"],
        dtype={"name": str, "active": str},
    )
    # Filter to active landmarks only
    active = df[df["active"].str.strip().str.lower() == "true"]

    moving_pts = active[["mov_x", "mov_y"]].to_numpy(dtype=float)
    fixed_pts = active[["fix_x", "fix_y"]].to_numpy(dtype=float)

    return moving_pts, fixed_pts


def save_landmarks(
    path: str | Path,
    moving_pts: np.ndarray,
    fixed_pts: np.ndarray,
    names: list[str] | None = None,
) -> None:
    """Save landmarks in BigWarp's headerless CSV format.

    Parameters
    ----------
    moving_pts : np.ndarray, shape (N, 2)
    fixed_pts : np.ndarray, shape (N, 2)
    names : optional list of N strings; defaults to Pt-0, Pt-1, ...
    """
    n = len(moving_pts)
    if names is None:
        names = [f"Pt-{i}" for i in range(n)]

    rows = []
    for i in range(n):
        rows.append([
            f'"{names[i]}"',
            '"true"',
            f'"{moving_pts[i, 0]}"',
            f'"{moving_pts[i, 1]}"',
            f'"{fixed_pts[i, 0]}"',
            f'"{fixed_pts[i, 1]}"',
        ])

    with open(path, "w") as f:
        for row in rows:
            f.write(",".join(row) + "\n")
