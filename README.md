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
  cell_id                  (e.g. my_fov_cell_index_42 — stat.npy row preserved)
  stat_idx                 (original stat.npy row index; omitted for CSV input)
  x_fov · y_fov
  x_section · y_section
  x_rot · y_rot · z_rot   (rotated atlas slice pixel coords)
  x_ccf · y_ccf · z_ccf   (original CCF voxel indices)
  region_id · region_acronym · region_name · parent_acronym
  AP_mm · ML_mm · DV_mm   (bregma-relative, Paxinos-Franklin)
```

---

## suite2p integration

The plugin has first-class support for [suite2p](https://github.com/MouseLand/suite2p)
outputs. In Step 1 the cell-loading toggle is **ON** by default:

- **stat.npy** — `med` (median pixel [y, x]) is used as the per-cell centroid
  for all downstream warping. Original row order in `stat.npy` is preserved
  throughout; cells are never reordered.
- **iscell.npy** — if present in the same folder as `stat.npy`, it is loaded
  automatically and only ROIs where `iscell[:,0] == 1` are kept.
- **Cell masks** — `ypix` / `xpix` from each ROI are painted into a napari
  Labels layer (`cell_masks`) so you can visually inspect cell footprints
  overlaid on the FOV image. The label value equals the cell's 1-based position
  in the filtered list.
- **Cell IDs** — the original `stat.npy` row index (`stat_idx`) is carried
  through all intermediate CSVs. In the final export (Step 3) each row gets a
  human-readable `cell_id` of the form `{prefix}_cell_index_{stat_idx}`, where
  the prefix defaults to the FOV image filename stem.

To use a plain CSV instead, uncheck **"Load cells from suite2p stat.npy"** in
Step 1.

---

## Step-by-step

### Step 1 — FOV → Section alignment
> **Plugins → Atlas Registration → 1 — FOV Alignment**

Rigid alignment (translate / rotate / uniform scale) of the FOV image onto the
full brain section. Both images open in independent napari windows for
side-by-side comparison. Landmark pairs are placed sequentially (click FOV →
click section → row added to table). The rigid transform is fitted by
least-squares and applied to all cell coordinates.

**Cell input (choose one):**
- **suite2p** (default): load `stat.npy`; `iscell.npy` auto-detected in same
  folder. Cell ROI masks shown as a Labels layer on the FOV.
- **CSV**: uncheck the toggle and load a CSV with `x` and `y` columns.

```
T_rigid[FOV → Section]
(x_fov, y_fov)  →  (x_section, y_section)

Saves: landmarks.csv | transform.json | cells_section.csv
```

### Step 2 — Atlas rotation + Section alignment
> **Plugins → Atlas Registration → 2 — Atlas Registration**

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
> **Plugins → Atlas Registration → 3 — Coordinate Readout**

Loads the cells CSV (with `x_ccf, y_ccf, z_ccf`) and:

1. **Region lookup** — queries `annotation.tif` at `annotation[ap, dv, ml]`
   → structure ID → joins Allen labels CSV → region name, acronym,
   parent acronym.

2. **AP / ML / DV** — calls `ccf25_to_bregma()` with the user-specified axis
   mapping (configurable dropdowns; preset buttons for Coronal / Sagittal /
   Axial).

3. **Cell ID export** — set a prefix string (auto-populated from the FOV
   filename). The saved CSV gains a `cell_id` column:
   `{prefix}_cell_index_{stat_idx}` where `stat_idx` is the original row in
   `stat.npy`. Row order matches `stat.npy` exactly.

```
annotation[ap_idx, dv_idx, ml_idx]  →  region_id, acronym, name
(ap_idx, dv_idx, ml_idx)  →  AP_mm, ML_mm, DV_mm  (bregma-relative)

Saves: cells_final.csv  (first column: cell_id)
```

---

## Widget status

| Widget | Description | Status |
|--------|-------------|--------|
| 1 — FOV Alignment | Rigid FOV→Section, sequential landmark pairs | ✅ Done |
| 2 — Atlas Registration | Atlas rotation (Phase 1) + TPS alignment (Phase 2) | ✅ Done |
| 3 — Coordinate Readout | CCF region lookup + AP/ML/DV mm | ✅ Done |

---

## Installation

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager

Install `uv` if you don't have it:

```bash
pip install uv
```

### Clone and install

```bash
git clone https://github.com/kenjp1223/FOV-atlas-registration.git
cd FOV-atlas-registration
uv sync --dev
```

Optional image format support (install as needed):

```bash
uv sync --extra czi   # Zeiss CZI files
uv sync --extra nd2   # Nikon ND2 files
```

### Launch napari

```bash
uv run napari
```

### Open the widgets

In napari, go to **Plugins → Atlas Registration** and select the step you want:

- **1 — FOV Alignment**
- **2 — Atlas Registration**
- **3 — Coordinate Readout**

Each widget can be docked in the napari window. Run them in order (Step 1 → 2 → 3).

### CCF resource files

Use the following link to download the Allen CCFv3 files and place them in a `CCF3/` folder at the project root
https://drive.google.com/drive/folders/1AHzQ8sRubvUSm6bT7_oip5ZmQcuitV5g?usp=drive_link


```
CCF3/
  annotation_25_coronal.tif   — Allen CCFv3 annotation volume (25 µm)
  Allen_annotation_labels.csv — region id → name, acronym, parent
  template_25_coronal.tif     — (optional) anatomy reference
```

Paths are configurable inside the widget; the plugin defaults to `CCF3/`.

### Example files

Use the following link to download example image files and cell coordinate files
https://drive.google.com/drive/folders/1AHzQ8sRubvUSm6bT7_oip5ZmQcuitV5g?usp=drive_link


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
