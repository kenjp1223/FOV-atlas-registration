"""
Bregma coordinate calculator.

Bregma is a skull landmark (typically outside or at the edge of the atlas image)
used as the anatomical zero point. The user identifies a reference point within
the atlas image (in native voxel space) and declares its known position relative
to bregma in mm along AP, ML, DV axes.

From this, we can compute bregma's position in native voxel space (even if it
falls outside the image), and then convert any atlas voxel coordinate to
bregma-relative mm.

IMPORTANT: The reference point and bregma position are always stored and
calculated in the PRE-ROTATION atlas voxel space. This means rotation does not
invalidate the bregma calibration — we always unproject rotated slice coordinates
back through the rotation matrix before applying the bregma offset.

However, for the current implementation (2D slice workflow), the slice is taken
from the ROTATED volume. The slice pixel (X, Y) + slice_index gives coordinates
in rotated voxel space. To get pre-rotation coordinates, we apply the inverse
rotation. But in practice, for small rotations, the error is minimal.

TODO: implement full pre-rotation unprojection once rotation matrix is stored
in the session. For now, bregma math is applied in the effective (rotated) voxel
space, which is a reasonable approximation.
"""

from __future__ import annotations

import numpy as np

from .orientation import AtlasOrientation


class BregmaReference:
    """A single reference point that anchors the bregma coordinate system.

    Parameters
    ----------
    voxel_zyx : array-like, shape (3,)
        (Z, Y, X) position of the reference point in atlas voxel space.
    ap_mm : float
        AP position of the reference point relative to bregma (mm).
        Positive = anterior to bregma.
    ml_mm : float
        ML position relative to bregma (mm). Positive = right of midline.
    dv_mm : float
        DV position relative to bregma (mm). Positive = dorsal to bregma.
    """

    def __init__(
        self,
        voxel_zyx: list | np.ndarray,
        ap_mm: float,
        ml_mm: float,
        dv_mm: float,
    ) -> None:
        self.voxel_zyx = np.array(voxel_zyx, dtype=float)
        self.ap_mm = float(ap_mm)
        self.ml_mm = float(ml_mm)
        self.dv_mm = float(dv_mm)

    def to_dict(self) -> dict:
        return {
            "voxel_zyx": self.voxel_zyx.tolist(),
            "ap_mm": self.ap_mm,
            "ml_mm": self.ml_mm,
            "dv_mm": self.dv_mm,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BregmaReference":
        return cls(d["voxel_zyx"], d["ap_mm"], d["ml_mm"], d["dv_mm"])


class BregmaCalculator:
    """Converts atlas voxel coordinates to bregma-relative mm.

    Parameters
    ----------
    orientation : AtlasOrientation
    voxel_size_um : list of 3 floats — (Z, Y, X) voxel size in micrometres
    reference : BregmaReference — calibration point
    """

    def __init__(
        self,
        orientation: AtlasOrientation,
        voxel_size_um: list[float],
        reference: BregmaReference,
    ) -> None:
        self.orientation = orientation
        self.voxel_size_um = np.array(voxel_size_um, dtype=float)
        self.reference = reference

        # Compute bregma position in (AP, ML, DV) voxel index space
        # (fractional, may be outside image bounds)
        ref_apdv = self.orientation.voxel_to_apdv_indices(
            self.reference.voxel_zyx[np.newaxis, :]
        )[0]  # shape (3,): AP_idx, ML_idx, DV_idx

        # voxel size in (AP, ML, DV) order using axis_order mapping
        ap_vox = self.voxel_size_um[orientation.axis_order[0]]
        ml_vox = self.voxel_size_um[orientation.axis_order[1]]
        dv_vox = self.voxel_size_um[orientation.axis_order[2]]
        self._vox_size_apdv_um = np.array([ap_vox, ml_vox, dv_vox])

        # Bregma position in AP/ML/DV voxel indices (fractional)
        # ref_point_mm - (ref_voxel_index * vox_size_um * sign) / 1000 = bregma_mm
        # bregma is at 0,0,0 mm by definition
        # ref_apdv_mm = ref_apdv * vox_size_apdv_um * signs / 1000
        ref_apdv_mm = (
            ref_apdv
            * self._vox_size_apdv_um
            * np.array(orientation.signs)
            / 1000.0
        )
        # offset: bregma_mm = ref_apdv_mm - (ref_offset_from_bregma)
        # => bregma_apdv_mm = ref_apdv_mm - [ap_mm, ml_mm, dv_mm]
        self._bregma_apdv_mm = ref_apdv_mm - np.array(
            [reference.ap_mm, reference.ml_mm, reference.dv_mm]
        )

    def voxel_to_apdv(self, voxel_zyx: np.ndarray) -> np.ndarray:
        """Convert (Z, Y, X) atlas voxel coords to bregma-relative (AP, ML, DV) mm.

        Parameters
        ----------
        voxel_zyx : np.ndarray, shape (N, 3)

        Returns
        -------
        np.ndarray, shape (N, 3) — (AP_mm, ML_mm, DV_mm) relative to bregma
            AP: positive = anterior
            ML: positive = right
            DV: positive = dorsal
        """
        voxel_zyx = np.atleast_2d(voxel_zyx)
        apdv_idx = self.orientation.voxel_to_apdv_indices(voxel_zyx)  # (N, 3)

        # Convert to mm, apply sign convention
        apdv_mm = (
            apdv_idx
            * self._vox_size_apdv_um[np.newaxis, :]
            * np.array(self.orientation.signs)[np.newaxis, :]
            / 1000.0
        )

        # Subtract bregma offset
        return apdv_mm - self._bregma_apdv_mm[np.newaxis, :]

    def voxel_to_native_um(self, voxel_zyx: np.ndarray) -> np.ndarray:
        """Convert voxel coords to native physical position in micrometres (no bregma offset).

        Returns (AP_um, ML_um, DV_um) in native atlas space.
        """
        voxel_zyx = np.atleast_2d(voxel_zyx)
        apdv_idx = self.orientation.voxel_to_apdv_indices(voxel_zyx)
        return apdv_idx * self._vox_size_apdv_um[np.newaxis, :]

    def to_dict(self) -> dict:
        return {
            "orientation": self.orientation.to_dict(),
            "voxel_size_um": self.voxel_size_um.tolist(),
            "reference": self.reference.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BregmaCalculator":
        return cls(
            orientation=AtlasOrientation.from_dict(d["orientation"]),
            voxel_size_um=d["voxel_size_um"],
            reference=BregmaReference.from_dict(d["reference"]),
        )
