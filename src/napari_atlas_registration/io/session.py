"""
Session I/O — reads prism_alignment _settings.json and saves plugin state.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_prism_settings(path) -> dict:
    """Load a _settings.json exported by prism_alignment/gui.py."""
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    rot = raw.get("rotation_degrees", {})
    sp  = raw.get("atlas_spacing_um", {})

    ref_point = None
    rp = raw.get("reference_point")
    if rp and rp.get("voxel"):
        v = rp["voxel"]
        ref_point = [v.get("x"), v.get("y"), v.get("z")]

    return {
        "rotation": {
            "rx": rot.get("rx_pitch", 0.0),
            "ry": rot.get("ry_yaw",   0.0),
            "rz": rot.get("rz_roll",  0.0),
        },
        "z_index":         raw.get("z_index", 0),
        "voxel_size_um":   [sp.get("x", 25.0), sp.get("y", 25.0), sp.get("z", 25.0)],
        "atlas_shape_zyx": raw.get("atlas_shape_zyx", []),
        "atlas_path":      raw.get("reference_atlas"),
        "target_path":     raw.get("target_image"),
        "ref_point_voxel": ref_point,
        "flip_h":          raw.get("flip_horizontal", False),
        "flip_v":          raw.get("flip_vertical",   False),
        "orientation":     raw.get("orientation", "coronal"),
        "target_resolution": raw.get("target_resolution", {}),
        "bregma_references": raw.get("bregma_references", []),
        "_raw":            raw,
    }


def save_plugin_settings(path, state: dict) -> None:
    """Save supplementary plugin settings alongside the prism settings JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_plugin_settings(path) -> dict:
    """Load a *_plugin.json saved by save_plugin_settings."""
    with open(path) as f:
        return json.load(f)


def build_plugin_state(
    bregma_references=None,
    target_x_um_per_pixel: float = 1.0,
    target_y_um_per_pixel: float = 1.0,
    bigwarp_landmarks_path=None,
    orientation: str = "coronal",
) -> dict:
    return {
        "orientation": orientation,
        "target_resolution": {
            "x_um_per_pixel": target_x_um_per_pixel,
            "y_um_per_pixel": target_y_um_per_pixel,
        },
        "bregma_references": bregma_references or [],
        "bigwarp_landmarks_path": bigwarp_landmarks_path,
    }
