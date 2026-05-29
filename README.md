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

### Step 1 — Atlas rotation  `[ DONE ]`
> Widget: **1 — Atlas Setup**

Rotate the 3-D CCF atlas to match the cutting angle of the section.
Interactive Rx / Ry / Rz sliders + Z-slice selector with live oblique preview.
Exports the rotated atlas slice as a TIFF and a settings JSON carrying the
rotation matrix, z-index, and voxel spacing — used by Steps 3 and 4.

```
Saved: rx, ry, rz (°) | z_index | rotation_matrix_3x3 | voxel_spacing_um
→ atlas_slice.tif  +  _settings.json
```

### Step 2 — FOV → Section  `[ DONE ]`
> Widget: **2 — FOV Alignment**

Rigid alignment (translate / rotate / scale) of the small FOV image onto the
full brain section. Both images open in independent napari windows.
Landmark pairs are placed sequentially (click FOV → click section → row in table).
Transform is applied to the cell CSV coordinates.

```
T_rigid[FOV → Section]
(x_fov, y_fov)  →  (x_section, y_section)
Saves: landmarks.csv | transform.json | cells_section.csv
```

### Step 3 — Section → Rotated Atlas slice  `[ DONE ]`
> Widget: **3 — Section-Atlas Alignment**

TPS (thin-plate spline) alignment of the section image to the exported atlas
slice. Replaces BigWarp — everything runs inside napari.
Section in main viewer, atlas slice in second window.
Landmark pairs placed sequentially; TPS fitted and applied to cell coordinates.

```
T_TPS[Section → Atlas slice]
(x_section, y_section)  →  (x_atlas_slice, y_atlas_slice)
Saves: landmarks.csv | session.json | cells_atlas_slice.csv
```

### Step 4 — Rotated atlas coords → Original CCF coords  `[ TODO ]`

Apply the inverse rotation matrix (stored in `_settings.json`) to unproject
each atlas-slice pixel to the original CCFv3 voxel index.

```
(x_atlas_slice, y_atlas_slice, z_index)  →  [R_inv]  →  (ap_idx, dv_idx, ml_idx)
```

### Step 5 — Region lookup  `[ TODO ]`

Query the CCF annotation volume at each voxel to get Allen structure ID,
acronym, and full region name.

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
  x_fov · y_fov · x_section · y_section · x_atlas_slice · y_atlas_slice
  · ap_idx · dv_idx · ml_idx · structure_id · acronym · AP_mm · ML_mm · DV_mm
```

---

## Current widget status

| Widget | Step | Status |
|--------|------|--------|
| 1 — Atlas Setup | Atlas rotation + oblique slice export | **Done** |
| 2 — FOV Alignment | FOV → Section rigid transform + landmark pairing | **Done** |
| 3 — Section-Atlas Alignment | Section → Atlas TPS (replaces BigWarp) | **Done** |
| Target Image | Helper — loads section into second window | **Done** |
| Steps 4–6 | CCF unprojection + region lookup + bregma coords | **TODO** |

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
