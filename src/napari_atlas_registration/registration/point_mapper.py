"""
Point mapper — transforms points through the full pipeline:

    target pixel (X, Y)
        -> [inverse TPS]
    atlas slice pixel (X, Y)
        -> [unproject through rotation + slice index]
    3D atlas voxel (Z, Y, X)
        -> [bregma offset]
    AP / ML / DV in mm
"""

from __future__ import annotations

import numpy as np

from .tps import TPSTransform


def target_to_atlas_slice(
    points_xy: np.ndarray,
    inverse_tps: TPSTransform,
) -> np.ndarray:
    """Map points from target image space to atlas slice pixel space.

    Parameters
    ----------
    points_xy : np.ndarray, shape (N, 2)  — (X, Y) pixel coords in target image
    inverse_tps : fitted TPSTransform from fixed->moving

    Returns
    -------
    np.ndarray, shape (N, 2) — (X, Y) pixel coords in atlas slice space
    """
    return inverse_tps(points_xy)


def atlas_slice_to_voxel(
    slice_pts_xy: np.ndarray,
    slice_index: int,
) -> np.ndarray:
    """Convert 2D atlas slice pixel coords to 3D atlas voxel coords.

    The slice was extracted along the Z axis, so Z = slice_index.
    The (X, Y) of the slice map directly to the (X, Y) of the volume
    at that Z level.

    Parameters
    ----------
    slice_pts_xy : np.ndarray, shape (N, 2)  — (X, Y) in atlas slice pixels
    slice_index : int

    Returns
    -------
    np.ndarray, shape (N, 3)  — (Z, Y, X) in atlas voxel coords
    """
    n = len(slice_pts_xy)
    z_col = np.full((n, 1), slice_index, dtype=float)
    # slice_pts_xy is (X, Y) -> store as (Z, Y, X)
    voxels = np.hstack([z_col, slice_pts_xy[:, 1:2], slice_pts_xy[:, 0:1]])
    return voxels


def map_points_full_pipeline(
    target_points_xy: np.ndarray,
    inverse_tps: TPSTransform,
    slice_index: int,
    bregma_calculator,  # coordinates.bregma.BregmaCalculator
) -> dict:
    """Run the full point mapping pipeline.

    Parameters
    ----------
    target_points_xy : np.ndarray, shape (N, 2)
    inverse_tps : TPSTransform (fixed->moving)
    slice_index : int
    bregma_calculator : BregmaCalculator

    Returns
    -------
    dict with keys:
        target_px      : (N, 2)  original target coords
        atlas_slice_px : (N, 2)  coords in atlas slice
        atlas_voxel    : (N, 3)  (Z, Y, X) voxel coords
        apdv_mm        : (N, 3)  (AP, ML, DV) mm relative to bregma
    """
    atlas_slice_px = target_to_atlas_slice(target_points_xy, inverse_tps)
    atlas_voxel = atlas_slice_to_voxel(atlas_slice_px, slice_index)
    apdv_mm = bregma_calculator.voxel_to_apdv(atlas_voxel)

    return {
        "target_px": target_points_xy,
        "atlas_slice_px": atlas_slice_px,
        "atlas_voxel": atlas_voxel,
        "apdv_mm": apdv_mm,
    }
