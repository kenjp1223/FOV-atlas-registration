"""
Thin-plate spline (TPS) transform — forward and inverse.

Strategy
--------
Forward transform:  moving (atlas slice) -> fixed (target)
Inverse transform:  fixed (target) -> moving (atlas slice)

BigWarp's forward warp is moving->fixed. To invert it for:
  - Point mapping: fit a NEW TPS with fixed_pts as source and moving_pts as
    target (swap the roles). This is the "swapped TPS" approximation.
    Works well when landmarks are distributed across the image.
  - Image warping: use the pull/inverse approach — for each output pixel
    in moving space, ask "where does it come from in fixed space?" using
    the forward TPS. scikit-image.transform.warp() handles this natively.

TPS implementation uses scipy's RBF interpolation with the thin-plate kernel
(r^2 * log(r)), which is equivalent to a standard 2D TPS.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator


class TPSTransform:
    """2D thin-plate spline transform fitted from control point pairs.

    Parameters
    ----------
    src_pts : np.ndarray, shape (N, 2)  — source control points (X, Y)
    dst_pts : np.ndarray, shape (N, 2)  — destination control points (X, Y)
    smoothing : float
        RBF smoothing factor. 0 = interpolating (passes through all points).
        Small positive value adds regularisation; useful when landmarks are
        noisy or very close together.
    """

    def __init__(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        smoothing: float = 0.0,
    ) -> None:
        self._rbf_x = RBFInterpolator(
            src_pts, dst_pts[:, 0], kernel="thin_plate_spline", smoothing=smoothing
        )
        self._rbf_y = RBFInterpolator(
            src_pts, dst_pts[:, 1], kernel="thin_plate_spline", smoothing=smoothing
        )
        self.src_pts = src_pts
        self.dst_pts = dst_pts

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        """Map points from source space to destination space.

        Parameters
        ----------
        pts : np.ndarray, shape (N, 2) — (X, Y) in source space

        Returns
        -------
        np.ndarray, shape (N, 2) — (X, Y) in destination space
        """
        pts = np.atleast_2d(pts)
        x_out = self._rbf_x(pts)
        y_out = self._rbf_y(pts)
        return np.column_stack([x_out, y_out])


def build_forward_transform(
    moving_pts: np.ndarray,
    fixed_pts: np.ndarray,
    smoothing: float = 0.0,
) -> TPSTransform:
    """Build the forward TPS: atlas slice -> target image.

    moving_pts are the source (atlas), fixed_pts are the destination (target).
    """
    return TPSTransform(moving_pts, fixed_pts, smoothing=smoothing)


def build_inverse_transform(
    moving_pts: np.ndarray,
    fixed_pts: np.ndarray,
    smoothing: float = 0.0,
) -> TPSTransform:
    """Build the inverse TPS: target image -> atlas slice.

    Achieved by swapping src/dst: fit TPS from fixed_pts -> moving_pts.
    """
    return TPSTransform(fixed_pts, moving_pts, smoothing=smoothing)


def warp_image_to_moving(
    target_image: np.ndarray,
    moving_pts: np.ndarray,
    fixed_pts: np.ndarray,
    output_shape: tuple[int, int] | None = None,
    smoothing: float = 0.0,
) -> np.ndarray:
    """Warp the target image into atlas slice (moving) space.

    Uses the pull/inverse approach: for each pixel in output (moving) space,
    the forward TPS tells us where to sample from in the target (fixed) image.

    Parameters
    ----------
    target_image : np.ndarray, shape (H, W) or (H, W, C)
    moving_pts : np.ndarray, shape (N, 2)
    fixed_pts  : np.ndarray, shape (N, 2)
    output_shape : (H, W) of output; defaults to target_image.shape[:2]
    smoothing : float

    Returns
    -------
    np.ndarray, same dtype as target_image, shape output_shape (+ C if multichannel)
    """
    from skimage.transform import warp

    if output_shape is None:
        output_shape = target_image.shape[:2]

    forward_tps = build_forward_transform(moving_pts, fixed_pts, smoothing=smoothing)

    def _inverse_map(coords):
        # coords shape: (N, 2) in (row, col) = (Y, X) order
        # swap to (X, Y), transform, swap back to (row, col)
        xy = coords[:, ::-1]          # (N, 2) X, Y
        xy_src = forward_tps(xy)       # map moving -> fixed (forward)
        return xy_src[:, ::-1]         # back to (row, col)

    warped = warp(
        target_image,
        _inverse_map,
        output_shape=output_shape,
        order=1,
        preserve_range=True,
        cval=0,
    )
    return warped.astype(target_image.dtype)
