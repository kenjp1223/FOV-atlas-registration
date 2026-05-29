"""
Allen CCFv3 → Paxinos-Franklin bregma-relative stereotaxic coordinates.

References
----------
- Bohan Zhao, "Aligning Allen CCF to Paxinos-Franklin atlas"
  https://bohanzhao.com/atlas/
- Cortex Lab, "AllenCCF"
  https://github.com/cortex-lab/allenCCF

Input voxel order:
    ap_idx, dv_idx, ml_idx  =  Allen volume index order [AP, DV, ML]

Output convention:
    AP+ = anterior to bregma
    ML+ = right of midline
    DV+ = ventral from bregma
"""

import numpy as np


def ccf25_to_bregma(
    ap_idx,
    dv_idx,
    ml_idx,
    resolution_um: float = 25,
    bregma_ap_um:  float = 5400,
    bregma_dv_um:  float = 332,
    bregma_ml_um:  float = 5739,
    dv_scale:      float = 0.9434,
    tilt_deg:      float = 5.0,
):
    """
    Convert Allen CCFv3 voxel indices to bregma-relative mm.

    Parameters
    ----------
    ap_idx, dv_idx, ml_idx : array-like
        Allen CCFv3 voxel indices in [AP, DV, ML] order.
    resolution_um : float
        Voxel size in µm (default 25).
    bregma_ap_um, bregma_dv_um, bregma_ml_um : float
        Bregma position in Allen CCF physical space (µm).
    dv_scale : float
        DV scale factor calibrated from atlas landmarks (default 0.9434).
    tilt_deg : float
        AP-DV tilt correction in degrees (default 5.0).

    Returns
    -------
    ap_mm, ml_mm, dv_mm : np.ndarray
    """
    ap_um = np.asarray(ap_idx) * resolution_um
    dv_um = np.asarray(dv_idx) * resolution_um
    ml_um = np.asarray(ml_idx) * resolution_um

    ap = bregma_ap_um - ap_um          # Allen AP increases posteriorly → flip
    dv = (dv_um - bregma_dv_um) * dv_scale
    ml = ml_um - bregma_ml_um

    theta  = np.deg2rad(tilt_deg)
    ap_rot = ap * np.cos(theta) - dv * np.sin(theta)
    dv_rot = ap * np.sin(theta) + dv * np.cos(theta)

    return ap_rot / 1000, ml / 1000, dv_rot / 1000


def ccf25_array_to_bregma(coords: np.ndarray) -> np.ndarray:
    """
    Vectorised wrapper.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Columns must be [AP_idx, DV_idx, ML_idx].

    Returns
    -------
    np.ndarray, shape (N, 3)  — columns [AP_mm, ML_mm, DV_mm]
    """
    coords = np.asarray(coords)
    ap_mm, ml_mm, dv_mm = ccf25_to_bregma(
        coords[:, 0], coords[:, 1], coords[:, 2]
    )
    return np.column_stack([ap_mm, ml_mm, dv_mm])
