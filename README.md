# FOV-atlas-registration

A napari plugin that registers cells imaged in a small field-of-view (FOV) all
the way through to stereotaxic brain atlas coordinates (Allen CCFv3 / Paxinos-Franklin).

Everything runs inside napari — no BigWarp, no Fiji, no external tools required.

---

## Pipeline

For **N cells** with pixel coordinates in FOV space, the plugin produces
per-cell brain region labels and stereotaxic coordinates (AP / ML / DV mm
relative to bregma).

```
Inputs
  FOV image          — small widefield / confocal acquisition
  Cell CSV           — (x, y) pixel coords in FOV space
  Section image      — whole brain section the FOV was taken from
  3D Atlas (CCF)     — Allen CCFv3 TIFF volume(s), 10 or 25 µm
  Annotation volume  — Allen CCFv3 annotation_25.nrrd / annotation.tif
```

### Step 1 — FOV → Section  `[ TODO ]`
> Widget: **FOV Alignment**

Rigid alignment (translate / rotate / scale) of the FOV image onto the full
brain section. Both images open in independent napari windows for side-by-side
comparison. Transform is applied to the cell CSV coordinates.

```
T_rigid[FOV → Section]
(x_fov, y_fov)  →  (x_section, y_section)
```

### Step 2 — Section → Rotated Atlas slice  `[ TODO ]`
> Widget: **Section–Atlas Alignment**

Affine / landmark-based alignment of the section image to the rotated atlas
slice. Landmarks are placed interactively inside napari (replaces BigWarp).
A thin-plate spline (TPS) is fitted to the landmarks and applied to the
section cell coordinates.

```
T_affine[Section → Atlas slice]
(x_section, y_section)  →  (x_atlas_slice, y_atlas_slice)
```

### Step 3 — Atlas rotation  `[ DONE ]`
> Widget: **Atlas Setup**

Interactive Rx / Ry / Rz sliders + Z-slice selector produce a live oblique
preview of the 3-D atlas. The rotation that best matches the cutting angle of
the section is saved together with the slice index and voxel spacing.

```
Saved: rx, ry, rz (degrees)  |  z_index  |  rotation_matrix_3x3  |  voxel_spacing_um
(x_atlas_slice, y_atlas_slice)  →  (x_rot, y_rot, z_rot)  [rotated 3D voxel]
```

### Step 4 — Rotated atlas coords → Original CCF coords  `[ IN PROGRESS ]`

The saved rotation matrix is inverted and applied to each rotated-atlas voxel
to recover the original CCFv3 voxel index.

```
R_inv  ×  (x_rot, y_rot, z_rot)  →  (ap_idx, dv_idx, ml_idx)  [CCF voxel]
```

### Step 5 — Region lookup  `[ TODO ]`

The CCF annotation volume is queried at each voxel to retrieve the Allen
structure ID, acronym, and full region name.

```
annotation[ap_idx, dv_idx, ml_idx]  →  structure_id, acronym, full_name
```

### Step 6 — CCF → Bregma-relative stereotaxic coords  `[ DONE ]`
> `coordinates/ccf_to_FP.py`

Converts CCF voxel indices to Paxinos-Franklin / stereotaxic mm with a 5°
AP–DV tilt correction and DV scale factor calibrated to the Allen CCFv3 25 µm
atlas.

```
(ap_idx, dv_idx, ml_idx)  →  (AP_mm, ML_mm, DV_mm)  relative to bregma
  AP+  = anterior   |   ML+  = right   |   DV+  = ventral
```

```
Final output CSV (one row per cell)
  x_fov · y_fov · x_section · y_section · ap_idx · dv_idx · ml_idx
  · structure_id · acronym · AP_mm · ML_mm · DV_mm
```

---

## Current widget status

| Widget | Step | Status |
|--------|------|--------|
| Atlas Setup | Step 3 — atlas rotation + oblique slice | **Done** |
| Target Image | loads section image into second window | **Done** |
| FOV Alignment | Step 1 — FOV→Section rigid transform | **TODO** |
| Section–Atlas Alignment | Step 2 — Section→Atlas affine/TPS (replaces BigWarp) | **TODO** |
| Inverse Warp | Steps 4–6 — CCF lookup + bregma coords | **In progress** |

---

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd napari-atlas-registration
uv sync --dev
uv run napari
```

Optional image format support:

```bash
uv sync --extra czi   # Zeiss CZI
uv sync --extra nd2   # Nikon ND2
```

Open widgets from **Plugins → Atlas Registration**.

---

## Coordinate conventions

| Axis | Positive direction |
|------|--------------------|
| AP   | anterior to bregma |
| ML   | right of midline   |
| DV   | ventral to bregma  |

Atlas orientation default: **coronal** (AP = Z axis of CCF volume).

---

## Session file (`_settings.json`)

Saved by the Atlas Setup widget. Records everything needed to resume or hand
off between steps:

```json
{
  "rotation_degrees":    { "rx_pitch": 0.0, "ry_yaw": 0.0, "rz_roll": 0.0 },
  "rotation_matrix_3x3": [[...], [...], [...]],
  "z_index":             240,
  "atlas_spacing_um":    { "x": 25.0, "y": 25.0, "z": 25.0 },
  "atlas_shape_zyx":     [528, 320, 456],
  "orientation":         "coronal",
  "target_resolution":   { "x_um_per_pixel": 0.65, "y_um_per_pixel": 0.65 },
  "flip_horizontal":     false,
  "flip_vertical":       false
}
```
