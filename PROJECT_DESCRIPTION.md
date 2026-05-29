# prism_alignment — Project Description

## Overview

A two-part brain atlas registration pipeline. The original tool (`prism_alignment`) is a standalone Tkinter GUI. A napari plugin (`napari-atlas-registration`) extends it with inverse warping, coordinate readout, and a unified napari-based interface.

**DO NOT modify these original files:**
- `gui.py`, `reslice.py`, `imagej/`, `launch_gui.bat`, root `pyproject.toml`

---

## Repository Structure

```
prism_alignment/                        ← root (original package, do not modify)
├── gui.py                              ← original Tkinter GUI (rotation + slice)
├── reslice.py                          ← SimpleITK rotation engine (used as library)
├── imagej/                             ← ImageJ/BigWarp launcher scripts
├── launch_gui.bat                      ← launches original Tkinter GUI
├── launch_napari.bat                   ← launches napari plugin (NEW)
├── pyproject.toml                      ← brain-reslice package definition
└── napari-atlas-registration/          ← napari plugin (NEW, all new work goes here)
    ├── pyproject.toml                  ← plugin package, requires-python = ">=3.10"
    └── src/napari_atlas_registration/
        ├── napari.yaml                 ← registers AtlasSetupWidget + InverseWarpWidget
        ├── __init__.py                 ← lazy Qt imports
        ├── _widget_rotation.py         ← Widget 1: Atlas Setup (743 lines)
        ├── _widget_inverse_warp.py     ← Widget 2: Inverse Warp (408 lines)
        ├── atlas/                      ← deprecated stubs (use reslice.py instead)
        ├── coordinates/
        │   ├── orientation.py          ← AtlasOrientation: coronal/sagittal/axial axis mapping
        │   └── bregma.py               ← BregmaReference + BregmaCalculator
        ├── io/
        │   ├── session.py              ← load_prism_settings(), save/load plugin state
        │   └── export.py               ← export helpers
        └── registration/
            ├── bigwarp_io.py           ← parse BigWarp landmark CSV (headerless 6-col)
            ├── tps.py                  ← TPSTransform (scipy RBF), forward + inverse
            └── point_mapper.py         ← map points through TPS
```

---

## Pipeline (3 Steps)

### Step 1 — Atlas Setup (`_widget_rotation.py` → `AtlasSetupWidget`)
- Load 3D atlas TIFF (single or multi-channel) and a 2D target histology image
- Interactive rotation sliders (rx/ry/rz pitch/yaw/roll) with live oblique-slice preview
  - Preview uses fast oblique sampling via `scipy.ndimage.map_coordinates` (~50× faster than full volume rotation)
  - Export uses SimpleITK BSpline via `reslice.py` (high quality, physically correct for anisotropic voxels)
- Set voxel spacing (presets: 25×25×25, 10×10×10, 20×20×50 µm, or manual)
- Set atlas orientation (coronal / sagittal / axial) — defines which axis = AP/ML/DV
- Set target image resolution (x/y µm per pixel)
- **Bregma reference mode**: click a point on the atlas slice, declare its known AP/ML/DV offset from bregma in mm; stored in pre-rotation voxel space
- Flip horizontal/vertical
- Export: saves `_settings.json` (compatible with original `gui.py` format) + per-channel atlas slice TIFFs + rotation affine `.npy`
- Launch ImageJ/BigWarp from within the widget

### Step 2 — BigWarp (external, ImageJ)
- User places landmarks manually between the exported atlas slice and the target histology
- Saves a BigWarp landmark CSV (headerless, 6 columns: `Name, Active, Moving-X, Moving-Y, Fixed-X, Fixed-Y`)
  - Moving = atlas slice space, Fixed = target image space

### Step 3 — Inverse Warp (`_widget_inverse_warp.py` → `InverseWarpWidget`)
- Load `_settings.json` (recovers rotation, z_index, bregma references, resolution)
- Load BigWarp landmark CSV
- Load target image (optional, for overlay)
- Build inverse TPS: `TPSTransform(fixed_pts → moving_pts)` — swapped roles = target→atlas
- Warp target image into atlas slice space (pull warping via `skimage.transform.warp`)
- Load query points from CSV (x,y columns) or click on target image layer
- Map points: target pixel → atlas slice pixel → 3D atlas voxel → AP/ML/DV mm (bregma-relative)
- Display results table with AP/ML/DV columns
- Export results CSV and overlay TIFF

---

## Key Technical Details

### Oblique slice algorithm
Same as `gui.py`'s `_compute_slice()`. Rotates a sampling grid instead of the volume:
```python
def _oblique_slice(atlas_arr, spacing, z_idx, rx, ry, rz, order=1):
    R = Rotation.from_euler("XYZ", [rx, ry, rz], degrees=True)
    # build grid of physical coords at slice z_idx in rotated space
    # apply R.inv() to get back to original voxel space
    # sample with map_coordinates
```

### TPS inverse transform
```python
# Forward: moving (atlas) -> fixed (target)
TPSTransform(src=moving_pts, dst=fixed_pts)
# Inverse: fixed (target) -> moving (atlas) — swap src/dst
TPSTransform(src=fixed_pts, dst=moving_pts)
```

### Bregma coordinate system
- User clicks a reference point on the atlas and enters its known AP/ML/DV mm offset from bregma
- `BregmaCalculator.voxel_to_apdv(voxel_zyx)` converts any atlas voxel to bregma-relative mm
- Orientation-aware: `AtlasOrientation` maps (Z,Y,X) voxel axes to (AP, ML, DV)
  - Coronal: AP=Z, ML=X, DV=Y (signs: +1, +1, -1)
  - Sagittal: AP=X, ML=Z, DV=Y
  - Axial: AP=Y, ML=X, DV=Z

### Session handoff (`_settings.json`)
`load_prism_settings(path)` reads the JSON exported by `gui.py` and normalises it:
```python
{
  "rotation": {"rx": ..., "ry": ..., "rz": ...},
  "z_index": int,
  "voxel_size_um": [x, y, z],
  "atlas_path": str,
  "target_path": str,
  "ref_point_voxel": [x, y, z] | None,
  "flip_h": bool, "flip_v": bool,
  "orientation": "coronal" | "sagittal" | "axial",
  "target_resolution": {"x_um_per_pixel": float, "y_um_per_pixel": float},
  "bregma_references": [...],
}
```

---

## Environment

- **Python**: ≥ 3.10
- **Package manager**: `uv` (not pip)
- **Install**: `cd napari-atlas-registration && uv sync --dev && uv pip install -e .`
- **Run**: `launch_napari.bat` (double-click) or `uv run napari` from `napari-atlas-registration/`
- **Qt backend**: PyQt6
- **Key dependencies**: napari, SimpleITK, scipy, scikit-image, tifffile, pandas, magicgui, brain-reslice (path dep on `../reslice.py`)

---

## Known Issues / Next Steps

- Null bytes can appear in widget files if the Edit tool writes large files — strip with `data.rstrip(b'\x00')`
- `from __future__ import annotations` must NOT be used in widget files — breaks napari's viewer injection (needs real `napari.Viewer` type at runtime)
- Widget constructor must be `def __init__(self, napari_viewer: napari.Viewer)` for napari to inject the viewer
- Bregma unprojection is currently approximate (done in rotated voxel space); full pre-rotation unprojection via stored affine is a TODO
- Atlas region lookup (identify brain region from atlas label volume at a coordinate) is not yet implemented
