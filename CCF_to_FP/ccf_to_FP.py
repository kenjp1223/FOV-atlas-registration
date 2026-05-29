import numpy as np

def ccf25_to_paxinos_bregma(
    ap_idx,
    dv_idx,
    ml_idx,
    resolution_um=25,
    bregma_ap_um=5400,
    bregma_dv_um=332,
    bregma_ml_um=5739,
    dv_scale=0.9434,
    tilt_deg=5.0,
):
    """
    Convert Allen CCFv3 25 um voxel coordinates to approximate
    Paxinos/Franklin-style bregma-relative stereotaxic coordinates.

    Input:
        ap_idx, dv_idx, ml_idx : voxel indices in 25 um Allen CCF space.
                                 Expected order: AP, DV, ML.

    Output:
        ap_mm, ml_mm, dv_mm : bregma-relative coordinates in mm.
                              AP+ = anterior
                              ML+ = right
                              DV+ = ventral
    """

    # Convert voxel index to Allen CCF microns
    ap_um = np.asarray(ap_idx) * resolution_um
    dv_um = np.asarray(dv_idx) * resolution_um
    ml_um = np.asarray(ml_idx) * resolution_um

    # Center on estimated bregma in CCF axes
    # In Allen CCF AP coordinate, larger AP is more posterior.
    ap_post_um = ap_um - bregma_ap_um      # + posterior
    dv_vent_um = dv_um - bregma_dv_um      # + ventral
    ml_right_um = ml_um - bregma_ml_um     # + right, assuming Allen ML increases left-to-right

    # Rotate AP-DV plane by 5 degrees.
    # This follows the common Neuropixels/IBL-style correction:
    # anterior CCF is tilted ventrally relative to stereotaxic space.
    theta = np.deg2rad(tilt_deg)

    ap_post_rot_um = ap_post_um * np.cos(theta) - dv_vent_um * np.sin(theta)
    dv_vent_rot_um = ap_post_um * np.sin(theta) + dv_vent_um * np.cos(theta)

    # Apply DV scaling
    dv_vent_rot_um = dv_vent_rot_um * dv_scale

    # Convert to Paxinos/Franklin-style bregma-relative mm
    # AP+ = anterior, so flip posterior axis.
    ap_mm = -ap_post_rot_um / 1000
    ml_mm = ml_right_um / 1000
    dv_mm = dv_vent_rot_um / 1000

    return ap_mm, ml_mm, dv_mm