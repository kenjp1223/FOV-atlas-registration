# napari-atlas-registration

A napari plugin for atlas-to-histology registration pipelines.

## Pipeline overview

```
1. prism_alignment/gui.py  (existing Tkinter app — run separately)
   ├── Load atlas (3D TIFF) + target (2D TIFF)
   ├── Set voxel spacing
   ├── Rotate atlas (pitch / roll / yaw) with live oblique-slice preview
   ├── Select Z slice index
   ├── Set reference point (click on atlas)
   ├── Export: aligned_slice.tif + affine.npy + _settings.json
   └── One-click open in ImageJ for BigWarp

2. BigWarp (ImageJ)
   ├── Open: atlas slice (moving) + target (fixed)
   ├── Place landmarks
   └── Export landmarks CSV

3. napari-atlas-registration — Inverse Warp widget  (this plugin)
   ├── Load prism_alignment _settings.json (rotation, z_index, voxel size)
   ├── Load BigWarp landmarks CSV
   ├── Load target image
   ├── Build inverse TPS (target → atlas slice)
   ├── Warp target image into atlas space (optional)
   ├── Load or click points on target image
   ├── Set bregma reference + AP/ML/DV offset
   ├── Map points → atlas slice → 3D voxel → AP/ML/DV mm
   ├── View results table
   └── Export CSV + overlay TIFF
```

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
# Create venv and install with dev dependencies
uv sync --dev

# Launch napari inside the managed environment
uv run napari
```

To add an optional format (e.g. CZI or ND2 support):

```bash
uv sync --extra czi
uv sync --extra nd2
```

Then find the widgets under **Plugins → Atlas Registration**.

## Atlas config JSON

```json
{
    "orientation": "coronal",
    "voxel_size_um": [25.0, 25.0, 25.0],
    "channels": {
        "mri":        "mri.tif",
        "annotation": "annotation.tif",
        "boundary":   "boundary.tif"
    }
}
```

All channel paths are relative to the config file location.

## Session JSON

The session file records rotation angles, slice index, bregma reference(s),
target image path and resolution. It is the handoff between the Atlas Setup
widget and the Inverse Warp widget.

## Coordinate conventions

- **AP**: positive = anterior to bregma
- **ML**: positive = right of midline
- **DV**: positive = dorsal to bregma

Bregma references are stored in pre-rotation atlas voxel space.

## Notes on the inverse TPS

BigWarp does not natively support inverse image warps. This plugin fits a
swapped thin-plate spline (TPS) using `fixed_pts → moving_pts` from the
BigWarp landmark file. This approximation is accurate when landmarks are
well-distributed across the section.
