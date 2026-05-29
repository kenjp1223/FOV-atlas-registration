"""
Step 3 — Coordinate Readout
============================
Takes the cells CSV from Step 2 (Atlas Registration) which contains CCF voxel
indices (x_ccf, y_ccf, z_ccf) and produces:

  • Brain region lookup  — queries annotation.tif at each voxel
  • AP / ML / DV mm     — via ccf_to_FP (Paxinos-Franklin bregma-relative coords)

Inputs
------
  cells_atlas_slice.csv  — output of Step 2, must have x_ccf, y_ccf, z_ccf columns
  annotation.tif         — Allen CCFv3 annotation volume (25 µm coronal)
  Allen_annotation_labels.csv — id → name, acronym, parent_acronym columns

The annotation volume and labels CSV paths default to CCF3/ inside the project
folder but can be overridden.

Output
------
  cells_final.csv — all original columns plus:
    region_id, region_name, region_acronym, parent_acronym,
    AP_mm, ML_mm, DV_mm
"""

import json
import numpy as np
import napari
from pathlib import Path
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Default paths (relative to this file → project root / CCF3)
# ---------------------------------------------------------------------------
_DEFAULT_CCF3 = Path(__file__).parents[3] / "CCF3"
_DEFAULT_ANNOTATION = _DEFAULT_CCF3 / "annotation_25_coronal.tif"
_DEFAULT_LABELS     = _DEFAULT_CCF3 / "Allen_annotation_labels.csv"


from .coordinates.ccf_to_FP import ccf25_to_bregma as _ccf25_to_bregma


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class CoordinateReadoutWidget(QWidget):
    """Step 3 — region lookup + AP/ML/DV stereotaxic coordinates."""

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer       = napari_viewer
        self._cells_df     = None
        self._annotation   = None   # (Z, Y, X) uint32 array
        self._labels_df    = None   # id → name, acronym, …
        self._results_df   = None
        self._build_ui()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)
        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_ccf_group())
        layout.addWidget(self._build_axis_group())
        layout.addWidget(self._build_run_group())
        layout.addWidget(self._build_table_group())
        layout.addWidget(self._build_export_group())
        self._status = QLabel("Load inputs to begin.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    def _build_input_group(self) -> QGroupBox:
        box = QGroupBox("Inputs")
        layout = QVBoxLayout(box)

        btn_cells = QPushButton("Load cells CSV (x_ccf, y_ccf, z_ccf)…")
        btn_cells.clicked.connect(self._on_load_cells)
        layout.addWidget(btn_cells)

        self._cells_info = QLabel("No cells loaded")
        self._cells_info.setWordWrap(True)
        layout.addWidget(self._cells_info)
        return box

    def _build_ccf_group(self) -> QGroupBox:
        box = QGroupBox("CCF resources")
        layout = QVBoxLayout(box)

        # Annotation volume
        ann_row = QHBoxLayout()
        self._ann_edit = QLineEdit(str(_DEFAULT_ANNOTATION))
        btn_ann = QPushButton("Browse…")
        btn_ann.setMaximumWidth(70)
        btn_ann.clicked.connect(self._on_browse_annotation)
        ann_row.addWidget(QLabel("Annotation:"))
        ann_row.addWidget(self._ann_edit)
        ann_row.addWidget(btn_ann)
        layout.addLayout(ann_row)

        # Labels CSV
        lbl_row = QHBoxLayout()
        self._lbl_edit = QLineEdit(str(_DEFAULT_LABELS))
        btn_lbl = QPushButton("Browse…")
        btn_lbl.setMaximumWidth(70)
        btn_lbl.clicked.connect(self._on_browse_labels)
        lbl_row.addWidget(QLabel("Labels CSV:"))
        lbl_row.addWidget(self._lbl_edit)
        lbl_row.addWidget(btn_lbl)
        layout.addLayout(lbl_row)

        # Resolution
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Annotation resolution (µm):"))
        self._res_edit = QLineEdit("25")
        self._res_edit.setMaximumWidth(60)
        res_row.addWidget(self._res_edit)
        res_row.addStretch()
        layout.addLayout(res_row)

        btn_load = QPushButton("Load annotation + labels")
        btn_load.clicked.connect(self._on_load_ccf)
        layout.addWidget(btn_load)

        self._ccf_info = QLabel("Not loaded")
        self._ccf_info.setWordWrap(True)
        layout.addWidget(self._ccf_info)
        return box

    def _build_axis_group(self) -> QGroupBox:
        """
        Map x_ccf / y_ccf / z_ccf  →  AP / DV / ML indices
        required by ccf_to_FP(ap_idx, dv_idx, ml_idx).

        Default (coronal Allen CCF, annotation_25_coronal.tif):
          AP ← z_ccf   (axis 0 of annotation, anterior-posterior slice)
          DV ← y_ccf   (axis 1, dorsal-ventral rows)
          ML ← x_ccf   (axis 2, medial-lateral columns)

        The annotation lookup uses the same mapping:
          annotation[AP_axis, DV_axis, ML_axis]
        """
        box = QGroupBox("Axis mapping  (x/y/z_ccf → AP/DV/ML)")
        layout = QVBoxLayout(box)

        note = QLabel(
            "ccf_to_FP expects:  coords[:,0]=AP  coords[:,1]=DV  coords[:,2]=ML\n"
            "Choose which of your x_ccf/y_ccf/z_ccf columns maps to each anatomical axis.\n"
            "Default is for annotation_25_coronal.tif (Allen CCFv3 coronal)."
        )
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 10px; color: #555;")
        layout.addWidget(note)

        COLS = ["x_ccf", "y_ccf", "z_ccf"]
        form = QFormLayout()

        self._ap_combo = QComboBox(); self._ap_combo.addItems(COLS)
        self._dv_combo = QComboBox(); self._dv_combo.addItems(COLS)
        self._ml_combo = QComboBox(); self._ml_combo.addItems(COLS)

        form.addRow("AP index  (ant–post)  ←", self._ap_combo)
        form.addRow("DV index  (dors–vent) ←", self._dv_combo)
        form.addRow("ML index  (med–lat)   ←", self._ml_combo)
        layout.addLayout(form)

        # Preset buttons
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets:"))
        btn_cor = QPushButton("Coronal")
        btn_sag = QPushButton("Sagittal")
        btn_ax  = QPushButton("Axial")
        btn_cor.setMaximumWidth(70); btn_sag.setMaximumWidth(70); btn_ax.setMaximumWidth(70)
        btn_cor.clicked.connect(lambda: self._apply_axis_preset("coronal"))
        btn_sag.clicked.connect(lambda: self._apply_axis_preset("sagittal"))
        btn_ax.clicked.connect( lambda: self._apply_axis_preset("axial"))
        preset_row.addWidget(btn_cor); preset_row.addWidget(btn_sag); preset_row.addWidget(btn_ax)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        # Set default: coronal
        self._apply_axis_preset("coronal")
        return box

    def _apply_axis_preset(self, orientation: str) -> None:
        """Set axis combos to the standard mapping for each orientation."""
        # Allen CCFv3 conventions:
        #   Coronal   annotation(AP, DV, ML): AP=z_ccf, DV=y_ccf, ML=x_ccf
        #   Sagittal  annotation(ML, DV, AP): AP=z_ccf, DV=y_ccf, ML=x_ccf  ← same array order usually
        #   Axial     annotation(DV, AP, ML): AP=y_ccf, DV=z_ccf, ML=x_ccf
        mapping = {
            "coronal":  ("z_ccf", "y_ccf", "x_ccf"),
            "sagittal": ("x_ccf", "y_ccf", "z_ccf"),
            "axial":    ("y_ccf", "z_ccf", "x_ccf"),
        }
        ap, dv, ml = mapping.get(orientation, ("z_ccf", "y_ccf", "x_ccf"))
        self._ap_combo.setCurrentText(ap)
        self._dv_combo.setCurrentText(dv)
        self._ml_combo.setCurrentText(ml)

    def _build_run_group(self) -> QGroupBox:
        box = QGroupBox("Run")
        layout = QVBoxLayout(box)
        btn = QPushButton("Run: region lookup + AP/ML/DV")
        btn.clicked.connect(self._on_run)
        layout.addWidget(btn)
        return box

    def _build_table_group(self) -> QGroupBox:
        box = QGroupBox("Results")
        layout = QVBoxLayout(box)
        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels([
            "Cell", "x_ccf", "y_ccf", "z_ccf",
            "Acronym", "Region name",
            "AP (mm)", "ML (mm)",  # DV added as col 8
        ])
        self._table.setColumnCount(9)
        self._table.setHorizontalHeaderLabels([
            "Cell", "x_ccf", "y_ccf", "z_ccf",
            "Acronym", "Region name",
            "AP (mm)", "ML (mm)", "DV (mm)",
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self._table.setMinimumHeight(200)
        layout.addWidget(self._table)
        return box

    def _build_export_group(self) -> QGroupBox:
        box = QGroupBox("Export")
        layout = QVBoxLayout(box)
        btn = QPushButton("Save results CSV…")
        btn.clicked.connect(self._on_export)
        layout.addWidget(btn)
        return box

    # ---------------------------------------------------------------- Load --

    def _on_load_cells(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load cells CSV", "", "CSV (*.csv);;All files (*)")
        if not path: return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            cols = {c.lower() for c in df.columns}
            if not {"x_ccf", "y_ccf", "z_ccf"}.issubset(cols):
                QMessageBox.critical(self, "Missing columns",
                    "CSV must have x_ccf, y_ccf, z_ccf columns.\n"
                    "Run Step 2 (Atlas Registration) → Save session first.")
                return
            self._cells_df = df
            self._cells_info.setText(
                f"{Path(path).name}  —  {len(df)} cells")
            self._set_status(f"Loaded {len(df)} cells.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_browse_annotation(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Annotation volume", "", "TIFF (*.tif *.tiff);;All files (*)")
        if path: self._ann_edit.setText(path)

    def _on_browse_labels(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Annotation labels CSV", "", "CSV (*.csv);;All files (*)")
        if path: self._lbl_edit.setText(path)

    def _on_load_ccf(self) -> None:
        try:
            import tifffile, pandas as pd
            ann_path = Path(self._ann_edit.text())
            lbl_path = Path(self._lbl_edit.text())
            if not ann_path.exists():
                QMessageBox.critical(self, "Not found", str(ann_path)); return
            if not lbl_path.exists():
                QMessageBox.critical(self, "Not found", str(lbl_path)); return

            self._set_status("Loading annotation volume…")
            self._annotation = tifffile.imread(str(ann_path))
            # Ensure uint32
            self._annotation = self._annotation.astype(np.int64)

            self._labels_df = pd.read_csv(str(lbl_path))
            # Normalise column names to lowercase
            self._labels_df.columns = [c.lower() for c in self._labels_df.columns]
            # Build fast lookup dict: id → row
            self._id_to_row = self._labels_df.set_index("id").to_dict("index")

            nz, ny, nx = self._annotation.shape
            self._ccf_info.setText(
                f"Annotation: {nx}×{ny}×{nz} vox  "
                f"| Labels: {len(self._labels_df)} regions  "
                f"| Res: {self._res_edit.text()} µm")
            self._set_status("CCF resources loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error loading CCF", str(e))

    # ----------------------------------------------------------------- Run --

    def _on_run(self) -> None:
        if self._cells_df is None:
            QMessageBox.warning(self, "No cells", "Load cells CSV first."); return
        if self._annotation is None or self._labels_df is None:
            QMessageBox.warning(self, "No CCF", "Load annotation + labels first."); return

        try:
            import pandas as pd
            self._set_status("Running…")
            df   = self._cells_df.copy()
            ann  = self._annotation
            nz, ny, nx = ann.shape

            x_ccf = df["x_ccf"].to_numpy(dtype=float)
            y_ccf = df["y_ccf"].to_numpy(dtype=float)
            z_ccf = df["z_ccf"].to_numpy(dtype=float)
            col_map = {"x_ccf": x_ccf, "y_ccf": y_ccf, "z_ccf": z_ccf}

            # Apply user-defined axis mapping
            ap_idx = col_map[self._ap_combo.currentText()]
            dv_idx = col_map[self._dv_combo.currentText()]
            ml_idx = col_map[self._ml_combo.currentText()]

            # Clamp to annotation bounds
            # annotation axes: (axis0, axis1, axis2) = (AP, DV, ML) for coronal
            ap_c = np.clip(ap_idx.astype(int), 0, nz - 1)
            dv_c = np.clip(dv_idx.astype(int), 0, ny - 1)
            ml_c = np.clip(ml_idx.astype(int), 0, nx - 1)

            # Region lookup: annotation[ap, dv, ml]
            region_ids = ann[ap_c, dv_c, ml_c]

            region_names    = []
            region_acronyms = []
            parent_acronyms = []

            for rid in region_ids:
                row = self._id_to_row.get(int(rid))
                if row:
                    region_names.append(row.get("name", "unknown"))
                    region_acronyms.append(row.get("acronym", "?"))
                    parent_acronyms.append(row.get("parent_acronym", ""))
                else:
                    region_names.append("unknown")
                    region_acronyms.append("?")
                    parent_acronyms.append("")

            df["region_id"]       = region_ids
            df["region_acronym"]  = region_acronyms
            df["region_name"]     = region_names
            df["parent_acronym"]  = parent_acronyms

            # AP/ML/DV via ccf_to_FP — pass in the remapped indices
            res_um = float(self._res_edit.text())
            ap_mm, ml_mm, dv_mm = _ccf25_to_bregma(
                ap_c, dv_c, ml_c, resolution_um=res_um)

            df["AP_mm"] = np.round(ap_mm, 4)
            df["ML_mm"] = np.round(ml_mm, 4)
            df["DV_mm"] = np.round(dv_mm, 4)

            self._results_df = df
            self._populate_table(df)
            self._set_status(
                f"Done — {len(df)} cells  |  "
                f"{df['region_acronym'].nunique()} unique regions")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            raise

    def _populate_table(self, df) -> None:
        import pandas as pd
        self._table.setRowCount(len(df))
        for i, row in df.iterrows():
            vals = [
                str(i),
                str(row.get("x_ccf", "")),
                str(row.get("y_ccf", "")),
                str(row.get("z_ccf", "")),
                str(row.get("region_acronym", "")),
                str(row.get("region_name", "")),
                f"{row.get('AP_mm', 0):.3f}",
                f"{row.get('ML_mm', 0):.3f}",
                f"{row.get('DV_mm', 0):.3f}",
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(i, j, item)

    # --------------------------------------------------------------- Export --

    def _on_export(self) -> None:
        if self._results_df is None:
            QMessageBox.warning(self, "No results", "Run first."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save results CSV", "", "CSV (*.csv)")
        if not path: return
        self._results_df.to_csv(path, index=False)
        self._set_status(f"Saved: {Path(path).name}")
        QMessageBox.information(self, "Saved", path)

    def _set_status(self, msg): self._status.setText(msg)
