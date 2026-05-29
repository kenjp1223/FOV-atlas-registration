# FOV-atlas-registration

A napari plugin that registers cells imaged in a small field-of-view (FOV) all
the way through to stereotaxic brain atlas coordinates (Allen CCFv3 / Paxinos-Franklin).

Everything runs inside napari — no BigWarp, no Fiji, no external tools required.

---

## Pipeline overview

For **N cells** with pixel coordinates in FOV space, the plugin produces
per-cell brain region labels and stereotaxic coordinates (AP / ML / DV mm
relative to bregma).

```
Inputs
  FOV image             — small widefield / confocal acquisition
  Cell CSV              — (x, y) pixel coords in FOV space
  Section image         — whole brain section the FOV was taken from
  3D Atlas TIFF         — Allen CCFv3 volume, 10 or 25 µm
  annotation.tif        — Allen CCFv3 annotation volume (25 µm)
  Allen_annotation_labels.csv — region id → name, acronym

Output (cells_final.csv, one row per cell)
  x_fov · y_fov
  x_section · y_section
  x_rot · y_rot · z_rot   (rotated atlas slice pixel coords)
  x_ccf · y_ccf · z_ccf   (original CCF voxel indices)
  region_id · region_acronym · region_name · parent_acronym
  AP_mm · ML_mm · DV_mm   (bregma-relative, Paxinos-Franklin)
```

---

## Step-by-step

### Step 1 — FOV → Section alignment
> Widget: **1 — FOV Alignment**

Rigid alignment (translate / rotate / uniform scale) of the FOV image onto the
full brain section. Both images open in independent napari windows for
side-by-side comparison. Landmark pairs are placed sequentially (click FOV →
click section → row added to table). The rigid transform is fitted by
least-squares and applied to all cell coordinates.

```
T_rigid[FOV → Section]
(x_fov, y_fov)  →  (x_section, y_section)

Saves: landmarks.csv | transform.json | cells_section.csv
```

### Step 2 — Atlas rotation + Section alignment
> Widget: **2 — Atlas Registration**

Two-phase widget. Both phases use the same window pair throughout:
- **Main viewer** — rotating atlas slice (live oblique preview)
- **Second window** — section image (reference)

**Phase 1 — Rotation:** Adjust Rx / Ry / Rz sliders and Z-slice until the
atlas slice visually matches the section. Click *Save & lock rotation* to
export the atlas slice TIFF and settings JSON, and unlock Phase 2.

**Phase 2 — Landmark alignment:** Click *Add pair* → click a landmark on the
atlas slice (main) → click the matching point on the section (second) → pair
saved to table. Repeat for ≥ 4 pairs (6–10 recommended). Click *Compute TPS*
to fit a thin-plate spline transform in both directions. Apply to cells.

The save also computes the unrotation: `(x_rot, y_rot, z_rot)` →
`R_inv` → `(x_ccf, y_ccf, z_ccf)` original CCF voxel indices.

```
Phase 1: rx, ry, rz (°) | z_index | rotation_matrix_3x3 | voxel_spacing_um
Phase 2: T_TPS[Section → Atlas slice]
(x_section, y_section)  →  (x_rot, y_rot)  +  z_rot = z_index
R_inv → (x_ccf, y_ccf, z_ccf)

Saves: _settings.json | _landmarks.csv | _cells_atlas_slice.csv
       _atlas_warped_to_section.tif (*) | _section_warped_to_atlas.tif (*)
       (*) image warp visualisation — under review
```

### Step 3 — Region lookup + Bregma coordinates
> Widget: **3 — Coordinate Readout**

Loads the cells CSV (with `x_ccf, y_ccf, z_ccf`) and:

1. **Region lookup** — queries `annotation.tif` at `annotation[ap, dv, ml]`
   → structure ID → joins Allen labels CSV → region name, acronym,
   parent acronym.

2. **AP / ML / DV** — calls `ccf25_to_bregma()` with the user-specified axis
   mapping (configurable dropdowns; preset buttons for Coronal / Sagittal /
   Axial).

```
annotation[ap_idx, dv_idx, ml_idx]  →  region_id, acronym, name
(ap_idx, dv_idx, ml_idx)  →  AP_mm, ML_mm, DV_mm  (bregma-relative)

Saves: cells_final.csv
```

---

## Widget status

| Widget | Description | Status |
|--------|-------------|--------|
| 1 — FOV Alignment | Rigid FOV→Section, sequential landmark pairs | ✅ Done |
| 2 — Atlas Registration | Atlas rotation (Phase 1) + TPS alignment (Phase 2) | ✅ Done |
| 3 — Coordinate Readout | CCF region lookup + AP/ML/DV mm | ✅ Done |
| Image warp overlay | Atlas↔section image warp visualisation | ⚠ Under review |

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

## CCF resources

Place the following files in a `CCF3/` folder at the project root
(already in `.gitignore` — files are too large to commit):

```
CCF3/
  annotation_25_coronal.tif   — Allen CCFv3 annotation volume (25 µm)
  Allen_annotation_labels.csv — region id → name, acronym, parent
  template_25_coronal.tif     — (optional) anatomy reference
```

Paths are configurable in the widget; the plugin defaults to `CCF3/`.

---

## Coordinate conventions

| Axis | Positive direction |
|------|--------------------|
| AP   | anterior to bregma |
| ML   | right of midline   |
| DV   | ventral to bregma  |

Default axis mapping for `annotation_25_coronal.tif` (Allen CCFv3 coronal):

| CCF array axis | Anatomical axis |
|---------------|-----------------|
| axis 0 (Z)    | AP              |
| axis 1 (Y)    | DV              |
| axis 2 (X)    | ML              |

---

## Bregma coordinate conversion

The conversion from Allen CCFv3 voxel indices to Paxinos-Franklin stereotaxic
coordinates applies a **5° AP–DV tilt correction** and a **DV scale factor**
(0.9434) calibrated against atlas landmarks, following:

- **Bohan Zhao** — *Aligning Allen CCF to Paxinos-Franklin atlas*  
  <https://bohanzhao.com/atlas/>

- **Cortex Lab** — *AllenCCF: alignment tools for the Allen CCF*  
  <https://github.com/cortex-lab/allenCCF>

Bregma position in Allen CCFv3 25 µm space: AP = 5400 µm, DV = 332 µm, ML = 5739 µm.

Implementation: `src/napari_atlas_registration/coordinates/ccf_to_FP.py`

---

## Session files

**`_settings.json`** — saved by Phase 1 of the Atlas Registration widget:

```json
{
  "rotation_degrees":    { "rx_pitch": 0.0, "ry_yaw": 0.0, "rz_roll": 0.0 },
  "rotation_matrix_3x3": [[...], [...], [...]],
  "z_index":             240,
  "atlas_spacing_um":    { "x": 25.0, "y": 25.0, "z": 25.0 },
  "atlas_shape_zyx":     [528, 320, 456],
  "orientation":         "coronal",
  "flip_horizontal":     false,
  "flip_vertical":       false
}
```

**`cells_atlas_slice.csv`** — saved by Phase 2, passed to Step 3:

```
x_fov, y_fov, x_section, y_section,
x_rot, y_rot, z_rot,        ← rotated atlas slice pixel coords + z_index
x_ccf, y_ccf, z_ccf         ← original CCF voxel indices (after R_inv)
```
