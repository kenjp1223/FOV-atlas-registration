"""
Atlas orientation — defines how the native (Z, Y, X) voxel axes map to
anatomical (AP, ML, DV) axes.

The user declares the viewing orientation of the atlas stack:
    "coronal"   — slices are coronal sections (front-to-back)
    "sagittal"  — slices are sagittal sections (left-to-right)
    "axial"     — slices are axial/horizontal sections (top-to-bottom)

From this we derive:
    which voxel axis corresponds to AP, ML, DV
    whether each axis is positive-anterior or positive-posterior, etc.

Convention used here (matches Allen Mouse Brain Atlas):
    AP  — anterior is positive (or negative, depending on atlas)
    ML  — right is positive (lateral from midline)
    DV  — dorsal is positive (or ventral, depending on atlas)

The orientation definition also stores sign conventions so that the
bregma calculator can produce correctly signed mm values.

Each entry in ORIENTATIONS is:
    "axis_order": which of (Z, Y, X) maps to (AP, ML, DV)
    "signs":      multiply voxel coordinate by this to get the positive direction
                  e.g. -1 means increasing voxel index = more negative in that direction
"""

from __future__ import annotations

import numpy as np

# Orientation definitions: maps voxel axis indices (0=Z, 1=Y, 2=X) to AP/ML/DV
# and stores sign conventions.
#
# axis_order: [ap_axis, ml_axis, dv_axis] — indices into (Z, Y, X)
# signs:      [ap_sign, ml_sign, dv_sign] — +1 or -1
ORIENTATIONS: dict[str, dict] = {
    "coronal": {
        # Coronal stack: Z axis = AP (anterior/posterior slice index)
        #                Y axis = DV (dorsal-ventral within slice)
        #                X axis = ML (medial-lateral within slice)
        "axis_order": [0, 2, 1],   # AP=Z, ML=X, DV=Y
        "signs": [1, 1, -1],        # AP+ = anterior; ML+ = right; DV+ = dorsal (Y increases down)
        "description": "Coronal sections (Z = anterior-posterior)",
    },
    "sagittal": {
        # Sagittal stack: Z axis = ML
        #                 Y axis = DV
        #                 X axis = AP
        "axis_order": [2, 0, 1],   # AP=X, ML=Z, DV=Y
        "signs": [1, 1, -1],
        "description": "Sagittal sections (Z = medial-lateral)",
    },
    "axial": {
        # Axial/horizontal stack: Z axis = DV
        #                          Y axis = AP
        #                          X axis = ML
        "axis_order": [1, 2, 0],   # AP=Y, ML=X, DV=Z
        "signs": [1, 1, -1],
        "description": "Axial/horizontal sections (Z = dorsal-ventral)",
    },
}

ORIENTATION_NAMES = list(ORIENTATIONS.keys())


class AtlasOrientation:
    """Encapsulates axis mapping for a given atlas orientation.

    Parameters
    ----------
    orientation : str
        One of "coronal", "sagittal", "axial".
    """

    def __init__(self, orientation: str = "coronal") -> None:
        orientation = orientation.lower()
        if orientation not in ORIENTATIONS:
            raise ValueError(
                f"Unknown orientation '{orientation}'. "
                f"Choose from: {ORIENTATION_NAMES}"
            )
        self.name = orientation
        cfg = ORIENTATIONS[orientation]
        self.axis_order: list[int] = cfg["axis_order"]   # [ap_axis, ml_axis, dv_axis]
        self.signs: list[int] = cfg["signs"]              # [ap_sign, ml_sign, dv_sign]
        self.description: str = cfg["description"]

    def voxel_to_apdv_indices(self, voxel_zyx: np.ndarray) -> np.ndarray:
        """Extract (AP, ML, DV) voxel indices from (Z, Y, X) voxel coords.

        Parameters
        ----------
        voxel_zyx : np.ndarray, shape (N, 3) — columns are Z, Y, X

        Returns
        -------
        np.ndarray, shape (N, 3) — columns are AP, ML, DV (unsigned voxel indices)
        """
        voxel_zyx = np.atleast_2d(voxel_zyx)
        ap = voxel_zyx[:, self.axis_order[0]]
        ml = voxel_zyx[:, self.axis_order[1]]
        dv = voxel_zyx[:, self.axis_order[2]]
        return np.column_stack([ap, ml, dv])

    def to_dict(self) -> dict:
        return {"orientation": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "AtlasOrientation":
        return cls(d["orientation"])
