"""
Export utilities — results table (CSV), overlay images (TIFF).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def export_points_table(
    results: dict,
    output_path: str | Path,
    point_names: list[str] | None = None,
    target_resolution: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Export the full point mapping results to a CSV file.

    Parameters
    ----------
    results : dict from point_mapper.map_points_full_pipeline
    output_path : path to output .csv
    point_names : optional list of names for each point
    target_resolution : (x_um_per_pixel, y_um_per_pixel) for the target image

    Returns
    -------
    pd.DataFrame — also written to output_path
    """
    n = len(results["target_px"])
    if point_names is None:
        point_names = [f"pt_{i}" for i in range(n)]

    target_px = results["target_px"]
    atlas_slice_px = results["atlas_slice_px"]
    atlas_voxel = results["atlas_voxel"]
    apdv_mm = results["apdv_mm"]

    rows = []
    for i in range(n):
        row = {
            "point_name": point_names[i],
            "target_px_x": target_px[i, 0],
            "target_px_y": target_px[i, 1],
        }
        if target_resolution is not None:
            row["target_um_x"] = target_px[i, 0] * target_resolution[0]
            row["target_um_y"] = target_px[i, 1] * target_resolution[1]

        row.update({
            "atlas_slice_px_x": atlas_slice_px[i, 0],
            "atlas_slice_px_y": atlas_slice_px[i, 1],
            "atlas_voxel_z": atlas_voxel[i, 0],
            "atlas_voxel_y": atlas_voxel[i, 1],
            "atlas_voxel_x": atlas_voxel[i, 2],
            "AP_mm": round(apdv_mm[i, 0], 4),
            "ML_mm": round(apdv_mm[i, 1], 4),
            "DV_mm": round(apdv_mm[i, 2], 4),
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def export_overlay_image(
    target_image: np.ndarray,
    atlas_slice: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.4,
) -> None:
    """Save a simple alpha-blend overlay of target and atlas slice as TIFF.

    Both images are normalised to [0, 255] uint8 before blending.

    Parameters
    ----------
    target_image : np.ndarray, (H, W) or (H, W, 3)
    atlas_slice  : np.ndarray, (H, W) — must match target spatial dims
    output_path  : output .tif path
    alpha        : weight of atlas_slice in the blend (0=target only, 1=atlas only)
    """
    import tifffile

    def to_uint8(arr):
        arr = arr.astype(float)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255
        return arr.astype(np.uint8)

    t = to_uint8(target_image if target_image.ndim == 2 else target_image[..., 0])
    a = to_uint8(atlas_slice)

    # Resize atlas slice to target size if needed
    if t.shape != a.shape:
        from skimage.transform import resize
        a = (resize(a.astype(float), t.shape, anti_aliasing=True) * 255).astype(np.uint8)

    blended = ((1 - alpha) * t + alpha * a).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output_path), blended)


def export_warped_image(
    warped: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save the inverse-warped target image as TIFF."""
    import tifffile
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output_path), warped)
