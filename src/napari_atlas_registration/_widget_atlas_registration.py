"""
Step 2 — Atlas Registration  (Rotation + Section Alignment)
=============================================================
Two-phase widget. Both phases use the same window layout throughout:
  Main viewer   → rotating atlas slice (live preview)
  Second window → section image (reference) + cells from Step 1

Phase 1 — Atlas Rotation
  Adjust Rx/Ry/Rz and Z-slice until the atlas slice matches the section.
  "Save Phase 1" locks the rotation and exports the atlas slice + settings JSON.
  Phase 2 becomes visible only after Phase 1 is saved.
  An existing settings JSON can be loaded to restore a previous rotation.

Phase 2 — Landmark Alignment + TPS
  Click "Add pair" → click atlas (main) → click section (second) → pair saved.
  TPS is fitted in both directions:
    _tps_fwd  : atlas  → section  (for warping atlas image into section space)
    _tps_inv  : section→ atlas    (for warping section image into atlas space,
                                   and for mapping cell coordinates)
  Visualisation buttons:
    "Warp atlas → section space" + "Show overlay" checkbox
    "Warp section → atlas space" + "Show overlay" checkbox
  Save session writes:
    _settings.json, _landmarks.csv, _cells_atlas_slice.csv
    _atlas_warped_to_section.tif, _section_warped_to_atlas.tif

Coordinate note (Q6)
--------------------
x_atlas_slice, y_atlas_slice  are pixel coords in the ROTATED atlas slice (2D).
z_rot_atlas = z_index is added as a constant column so downstream steps can
form the 3-D rotated-atlas voxel (x_rot, y_rot, z_rot) and then apply R_inv
to recover the original CCF voxel indices.
"""

import json
import numpy as np
import napari
from datetime import datetime
from pathlib import Path
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSizePolicy, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QVBoxLayout, QWidget,
)
from scipy.spatial.transform import Rotation
from skimage.transform import warp as sk_warp

import tifffile

from ._widget_rotation import _load_image_any, _collapse_to_2d, _collapse_to_3d, _oblique_slice
from .registration.tps import TPSTransform

_IDLE     = "idle"
_WAIT_ATL = "waiting_atlas"
_WAIT_SEC = "waiting_section"


class AtlasRegistrationWidget(QWidget):
    """Step 2 — Atlas rotation (Phase 1) + Section alignment (Phase 2)."""

    SPACING_PRESETS = {
        "10×10×10": [10.0, 10.0, 10.0],
        "25×25×25": [25.0, 25.0, 25.0],
        "20×20×50": [20.0, 20.0, 50.0],
        "5×5×10":   [5.0,  5.0,  10.0],
    }

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = napari_viewer

        # Atlas
        self._atlas_channels = {}
        self._atlas_path     = None
        self._spacing        = [25.0, 25.0, 25.0]
        self._phase1_locked  = False
        self._atlas_slice_arr = None   # current high-quality slice (saved in Phase 1)

        # Section / cells
        self._section_path  = None
        self._section_arr   = None
        self._second_viewer = None
        self._cell_df       = None
        self._cell_pts_sec  = None   # (N,2) x,y section pixels

        # Landmark pairs
        self._pairs          = []
        self._pair_state     = _IDLE
        self._pending_atl_xy = None
        self._atl_n_before   = 0
        self._sec_n_before   = 0
        self._atl_lm_layer   = None
        self._sec_lm_layer   = None

        # TPS (fitted in Phase 2)
        self._tps_fwd  = None   # atlas  → section
        self._tps_inv  = None   # section→ atlas

        # Results
        self._cell_pts_atlas      = None
        self._atlas_warped_arr    = None   # atlas slice warped to section space
        self._section_warped_arr  = None   # section warped to atlas slice space

        # Debounce
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview)

        self._build_ui()

    # ================================================================ UI build

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._tabs = QTabWidget()

        # ── Tab 1: Phase 1 ──────────────────────────────────────────────────
        tab1 = QWidget()
        t1 = QVBoxLayout(tab1)
        t1.setAlignment(Qt.AlignTop)
        t1.addWidget(self._build_atlas_group())
        t1.addWidget(self._build_section_group())
        t1.addWidget(self._build_rotation_group())
        t1.addWidget(self._build_phase1_save_group())
        self._tabs.addTab(tab1, "Phase 1 — Rotation")

        # ── Tab 2: Phase 2 ──────────────────────────────────────────────────
        tab2 = QWidget()
        t2 = QVBoxLayout(tab2)
        t2.setAlignment(Qt.AlignTop)
        t2.addWidget(self._build_landmark_group())
        t2.addWidget(self._build_tps_group())
        t2.addWidget(self._build_warp_group())
        t2.addWidget(self._build_save_group())
        self._tabs.addTab(tab2, "Phase 2 — Alignment")
        self._tabs.setTabEnabled(1, False)   # locked until Phase 1 saved

        layout.addWidget(self._tabs)

        self._status = QLabel("Load atlas and section to begin.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    # ── Atlas ────────────────────────────────────────────────────────────────

    def _build_atlas_group(self) -> QGroupBox:
        box = QGroupBox("Atlas volume")
        layout = QVBoxLayout(box)

        row = QHBoxLayout()
        self._btn_load_atlas = QPushButton("Load atlas TIFF…")
        self._btn_load_atlas.clicked.connect(self._on_load_atlas)
        self._btn_add_ch = QPushButton("Add channel…")
        self._btn_add_ch.clicked.connect(self._on_add_channel)
        row.addWidget(self._btn_load_atlas); row.addWidget(self._btn_add_ch)
        layout.addLayout(row)

        sp_row = QHBoxLayout()
        sp_row.addWidget(QLabel("Spacing X/Y/Z (µm):"))
        self._sp_x = self._dspin(0.001, 1e5, 25.0)
        self._sp_y = self._dspin(0.001, 1e5, 25.0)
        self._sp_z = self._dspin(0.001, 1e5, 25.0)
        for w in (self._sp_x, self._sp_y, self._sp_z):
            sp_row.addWidget(w)
        layout.addLayout(sp_row)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets:"))
        for label in self.SPACING_PRESETS:
            b = QPushButton(label); b.setMaximumWidth(90)
            b.clicked.connect(lambda _, l=label: self._apply_preset(l))
            preset_row.addWidget(b)
        layout.addLayout(preset_row)

        self._atlas_info = QLabel("No atlas loaded")
        self._atlas_info.setWordWrap(True)
        layout.addWidget(self._atlas_info)
        return box

    # ── Section ──────────────────────────────────────────────────────────────

    def _build_section_group(self) -> QGroupBox:
        box = QGroupBox("Section image  (second window)")
        layout = QVBoxLayout(box)

        btn_sec = QPushButton("Load section image…")
        btn_sec.clicked.connect(self._on_load_section)
        layout.addWidget(btn_sec)

        csv_row = QHBoxLayout()
        btn_csv = QPushButton("Load cell CSV (x_section, y_section)…")
        btn_csv.clicked.connect(self._on_load_csv)
        self._btn_clear_cells = QPushButton("Clear cells")
        self._btn_clear_cells.clicked.connect(self._on_clear_cells)
        self._btn_clear_cells.setEnabled(False)
        csv_row.addWidget(btn_csv); csv_row.addWidget(self._btn_clear_cells)
        layout.addLayout(csv_row)

        self._section_info = QLabel("No section loaded")
        self._section_info.setWordWrap(True)
        layout.addWidget(self._section_info)
        return box

    # ── Rotation ─────────────────────────────────────────────────────────────

    def _build_rotation_group(self) -> QGroupBox:
        box = QGroupBox("Phase 1 — Atlas rotation")
        form = QFormLayout(box)

        self._rx_spin, rx_w = self._angle_row()
        self._ry_spin, ry_w = self._angle_row()
        self._rz_spin, rz_w = self._angle_row()
        form.addRow("Rx (pitch):", rx_w)
        form.addRow("Ry (yaw):",   ry_w)
        form.addRow("Rz (roll):",  rz_w)

        self._z_spin = QSpinBox(); self._z_spin.setRange(0, 9999)
        self._z_spin.valueChanged.connect(self._schedule_preview)
        self._z_phys = QLabel("0 µm")
        z_w = QWidget(); z_r = QHBoxLayout(z_w); z_r.setContentsMargins(0,0,0,0)
        z_r.addWidget(self._z_spin); z_r.addWidget(self._z_phys)
        form.addRow("Z slice:", z_w)

        self._flip_h = QCheckBox("Flip H"); self._flip_v = QCheckBox("Flip V")
        self._flip_h.toggled.connect(self._schedule_preview)
        self._flip_v.toggled.connect(self._schedule_preview)
        flip_w = QWidget(); flip_r = QHBoxLayout(flip_w); flip_r.setContentsMargins(0,0,0,0)
        flip_r.addWidget(self._flip_h); flip_r.addWidget(self._flip_v)
        form.addRow("", flip_w)

        self._orientation = QComboBox()
        self._orientation.addItems(["coronal", "sagittal", "axial"])
        form.addRow("Orientation:", self._orientation)

        btn_reset = QPushButton("Reset rotation")
        btn_reset.clicked.connect(self._reset_rotation)
        form.addRow(btn_reset)
        return box

    # ── Phase 1 save / load ──────────────────────────────────────────────────

    def _build_phase1_save_group(self) -> QGroupBox:
        box = QGroupBox("Phase 1 — Save / Load rotation")
        layout = QVBoxLayout(box)

        col_row = QHBoxLayout()

        self._btn_save_p1 = QPushButton("Save & lock rotation…")
        self._btn_save_p1.clicked.connect(self._on_save_phase1)
        self._btn_load_p1 = QPushButton("Load rotation JSON…")
        self._btn_load_p1.clicked.connect(self._on_load_phase1)
        col_row.addWidget(self._btn_save_p1)
        col_row.addWidget(self._btn_load_p1)
        layout.addLayout(col_row)

        self._phase1_status = QLabel("Phase 1 not saved")
        self._phase1_status.setStyleSheet("font-weight: bold; color: #888;")
        layout.addWidget(self._phase1_status)
        return box

    # ── Landmark pairs ───────────────────────────────────────────────────────

    def _build_landmark_group(self) -> QGroupBox:
        box = QGroupBox("Phase 2 — Landmark pairs  (atlas slice → section)")
        layout = QVBoxLayout(box)

        btn_load_lm = QPushButton("Load landmarks from CSV…")
        btn_load_lm.clicked.connect(self._on_load_landmarks_csv)
        layout.addWidget(btn_load_lm)

        self._pair_status = QLabel("Add landmark pairs to fit TPS.")
        self._pair_status.setWordWrap(True)
        self._pair_status.setStyleSheet("font-weight: bold; color: #555;")
        layout.addWidget(self._pair_status)

        btn_row = QHBoxLayout()
        self._btn_add_pair = QPushButton("Add pair")
        self._btn_add_pair.setEnabled(False)
        self._btn_add_pair.clicked.connect(self._on_start_pair)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel_pair)
        btn_row.addWidget(self._btn_add_pair); btn_row.addWidget(self._btn_cancel)
        layout.addLayout(btn_row)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["#", "RotAtl x", "RotAtl y", "Sec x", "Sec y", ""])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Fixed)
        self._table.setColumnWidth(5, 54)
        self._table.setMinimumHeight(130)
        self._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._table.itemSelectionChanged.connect(self._on_table_selection_changed)
        layout.addWidget(self._table)
        return box

    # ── TPS ──────────────────────────────────────────────────────────────────

    def _build_tps_group(self) -> QGroupBox:
        box = QGroupBox("TPS transform")
        layout = QVBoxLayout(box)
        btn = QPushButton("Compute TPS  (needs ≥ 4 pairs)")
        btn.clicked.connect(self._on_compute_tps)
        layout.addWidget(btn)
        self._tps_info = QLabel("No transform computed")
        self._tps_info.setWordWrap(True)
        layout.addWidget(self._tps_info)

        btn_apply = QPushButton("Apply TPS → map cells to atlas slice space")
        btn_apply.clicked.connect(self._on_apply_tps)
        layout.addWidget(btn_apply)
        return box

    # ── Warp visualisation ───────────────────────────────────────────────────

    def _build_warp_group(self) -> QGroupBox:
        box = QGroupBox("Image warping  ⚠ experimental")
        layout = QVBoxLayout(box)

        warn = QLabel(
            "⚠  Warp visualisation is under review — output may be incorrect.\n"
            "Cell coordinate mapping (above) is unaffected."
        )
        warn.setWordWrap(True)
        warn.setStyleSheet("color: #b05000; font-style: italic; font-size: 10px;")
        layout.addWidget(warn)

        # Atlas → section
        atl_row = QHBoxLayout()
        btn_atl2sec = QPushButton("Warp atlas → section space")
        btn_atl2sec.clicked.connect(self._on_warp_atlas_to_section)
        self._chk_atl_overlay = QCheckBox("Show section overlay")
        self._chk_atl_overlay.toggled.connect(self._on_atl_overlay_toggled)
        atl_row.addWidget(btn_atl2sec); atl_row.addWidget(self._chk_atl_overlay)
        layout.addLayout(atl_row)

        # Section → atlas
        sec_row = QHBoxLayout()
        btn_sec2atl = QPushButton("Warp section → atlas space")
        btn_sec2atl.clicked.connect(self._on_warp_section_to_atlas)
        self._chk_sec_overlay = QCheckBox("Show atlas overlay")
        self._chk_sec_overlay.toggled.connect(self._on_sec_overlay_toggled)
        sec_row.addWidget(btn_sec2atl); sec_row.addWidget(self._chk_sec_overlay)
        layout.addLayout(sec_row)

        return box

    # ── Save session ─────────────────────────────────────────────────────────

    def _build_save_group(self) -> QGroupBox:
        box = QGroupBox("Save Phase 2 results")
        layout = QVBoxLayout(box)
        btn = QPushButton("Save session…")
        btn.clicked.connect(self._on_save_session)
        layout.addWidget(btn)
        info = QLabel(
            "Saves: _settings.json · _landmarks.csv\n"
            "       _cells_atlas_slice.csv  (x_rot, y_rot, z_rot columns)\n"
            "       _atlas_warped_to_section.tif\n"
            "       _section_warped_to_atlas.tif"
        )
        info.setWordWrap(True); info.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(info)
        return box

    # ============================================================ Helpers

    def _dspin(self, lo, hi, val, dec=2):
        w = QDoubleSpinBox(); w.setRange(lo, hi); w.setValue(val); w.setDecimals(dec)
        return w

    def _angle_row(self):
        spin = QDoubleSpinBox(); spin.setRange(-180, 180); spin.setValue(0)
        spin.setSingleStep(0.5); spin.setDecimals(1); spin.setSuffix(" °")
        spin.valueChanged.connect(self._schedule_preview)
        dec = QPushButton("<"); inc = QPushButton(">")
        dec.setMaximumWidth(26); inc.setMaximumWidth(26)
        dec.clicked.connect(lambda: spin.setValue(round(max(-180, spin.value()-1), 1)))
        inc.clicked.connect(lambda: spin.setValue(round(min( 180, spin.value()+1), 1)))
        w = QWidget(); r = QHBoxLayout(w); r.setContentsMargins(0,0,0,0)
        r.addWidget(dec); r.addWidget(spin); r.addWidget(inc)
        return spin, w

    def _set_rotation_locked(self, locked: bool) -> None:
        for w in (self._rx_spin, self._ry_spin, self._rz_spin,
                  self._z_spin, self._flip_h, self._flip_v,
                  self._orientation, self._btn_load_atlas, self._btn_add_ch,
                  self._sp_x, self._sp_y, self._sp_z):
            w.setEnabled(not locked)

    def _set_status(self, msg): self._status.setText(msg)

    def _ro_item(self, text):
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    # ============================================================ Atlas load

    def _on_load_atlas(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load atlas TIFF", "", "TIFF (*.tif *.tiff);;All files (*)")
        if not path: return
        try:
            arr = tifffile.imread(path).astype(np.float32)
            arr = _collapse_to_3d(arr)
            name = Path(path).stem
            self._atlas_channels = {name: arr}
            self._atlas_path = Path(path)
            nz, ny, nx = arr.shape
            self._z_spin.setMaximum(nz - 1)
            self._z_spin.setValue(nz // 2)
            sx, sy, sz = self._spacing
            self._atlas_info.setText(
                f"{self._atlas_path.name}  {nx}×{ny}×{nz} vox  "
                f"{nx*sx/1000:.1f}×{ny*sy/1000:.1f}×{nz*sz/1000:.1f} mm")
            self._update_preview()
            self._update_pair_btn_state()
            self._set_status(f"Atlas loaded: {self._atlas_path.name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_add_channel(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Add atlas channel", "", "TIFF (*.tif *.tiff);;All files (*)")
        if not path: return
        try:
            arr = _collapse_to_3d(_load_image_any(path))
            self._atlas_channels[Path(path).stem] = arr
            self._update_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _apply_preset(self, label) -> None:
        sx, sy, sz = self.SPACING_PRESETS[label]
        self._sp_x.setValue(sx); self._sp_y.setValue(sy); self._sp_z.setValue(sz)
        self._spacing = [sx, sy, sz]; self._schedule_preview()

    # ============================================================ Section load

    def _on_load_section(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load section image", "",
            "Images (*.tif *.tiff *.png *.jpg *.czi *.nd2);;All files (*)")
        if not path: return
        try:
            arr = _collapse_to_2d(_load_image_any(path))
            if arr.nbytes > 400 * 1024 * 1024:
                f = int(np.ceil((arr.nbytes / (400*1024*1024)) ** 0.5))
                arr = arr[::f, ::f]
            self._section_path = Path(path)
            self._section_arr  = arr
            if self._second_viewer is not None:
                try: self._second_viewer.close()
                except Exception: pass
            self._second_viewer = napari.Viewer(
                title=f"Section — {self._section_path.name}")
            self._second_viewer.add_image(arr, name="section", colormap="gray")
            self._second_viewer.reset_view()
            # Re-display cells if already loaded
            if self._cell_pts_sec is not None:
                self._show_cells_on_section()
            self._section_info.setText(
                f"{self._section_path.name}  {arr.shape[1]}×{arr.shape[0]} px")
            self._update_pair_btn_state()
            self._set_status(f"Section loaded: {self._section_path.name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load cell CSV", "", "CSV (*.csv);;All files (*)")
        if not path: return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            xcol = cols.get("x_section") or cols.get("x")
            ycol = cols.get("y_section") or cols.get("y")
            if xcol is None or ycol is None:
                QMessageBox.critical(self, "Error",
                    "CSV must have 'x_section'/'y_section' or 'x'/'y' columns.")
                return
            self._cell_pts_sec = df[[xcol, ycol]].to_numpy(dtype=float)
            self._cell_df = df
            self._show_cells_on_section()
            self._btn_clear_cells.setEnabled(True)
            self._section_info.setText(
                self._section_info.text().split("\n")[0] +
                f"\nCells: {len(self._cell_pts_sec)}")
            self._set_status(f"Loaded {len(self._cell_pts_sec)} cells.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _show_cells_on_section(self) -> None:
        """Display cell points in the second viewer (section space)."""
        if self._second_viewer is None or self._cell_pts_sec is None: return
        pts_yx = self._cell_pts_sec[:, ::-1]
        if "cells_section" in self._second_viewer.layers:
            self._second_viewer.layers["cells_section"].data = pts_yx
        else:
            self._second_viewer.add_points(
                pts_yx, name="cells_section", size=8,
                face_color="red", border_color="white", opacity=0.8)

    def _on_clear_cells(self) -> None:
        self._cell_pts_sec = None; self._cell_df = None
        if self._second_viewer and "cells_section" in self._second_viewer.layers:
            self._second_viewer.layers.remove("cells_section")
        self._btn_clear_cells.setEnabled(False)
        self._section_info.setText(self._section_info.text().split("\n")[0])
        self._set_status("Cells cleared.")

    # ============================================================ Preview

    def _schedule_preview(self, *_):
        if not self._phase1_locked:
            self._preview_timer.start(60)

    def _update_preview(self) -> None:
        if not self._atlas_channels: return
        self._spacing = [self._sp_x.value(), self._sp_y.value(), self._sp_z.value()]
        z  = self._z_spin.value()
        rx = self._rx_spin.value(); ry = self._ry_spin.value(); rz = self._rz_spin.value()
        self._z_phys.setText(f"{z * self._spacing[2]:.0f} µm")
        for name, arr in self._atlas_channels.items():
            sl = _oblique_slice(arr, self._spacing, z, rx, ry, rz, order=1)
            if self._flip_h.isChecked(): sl = np.fliplr(sl)
            if self._flip_v.isChecked(): sl = np.flipud(sl)
            lname = f"atlas_{name}"
            if lname in self._viewer.layers:
                self._viewer.layers[lname].data = sl
            else:
                self._viewer.add_image(sl, name=lname, colormap="gray")

    def _reset_rotation(self) -> None:
        for s in (self._rx_spin, self._ry_spin, self._rz_spin): s.setValue(0.0)
        if self._atlas_channels:
            self._z_spin.setValue(list(self._atlas_channels.values())[0].shape[0] // 2)

    # ============================================================ Phase 1 save/load

    def _on_save_phase1(self) -> None:
        if not self._atlas_channels:
            QMessageBox.warning(self, "No atlas", "Load atlas first."); return
        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not out_dir: return
        try:
            out  = Path(out_dir)
            stem = self._atlas_path.stem if self._atlas_path else "atlas"
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            pfx  = f"{stem}_{ts}"

            rx, ry, rz = self._rx_spin.value(), self._ry_spin.value(), self._rz_spin.value()
            z_idx = self._z_spin.value()
            sx, sy, sz = self._sp_x.value(), self._sp_y.value(), self._sp_z.value()
            self._spacing = [sx, sy, sz]
            R_mat = Rotation.from_euler("XYZ", [rx, ry, rz], degrees=True).as_matrix()
            primary = list(self._atlas_channels.values())[0]

            # High-quality atlas slice
            sl = _oblique_slice(primary, self._spacing, z_idx, rx, ry, rz, order=3)
            if self._flip_h.isChecked(): sl = np.fliplr(sl)
            if self._flip_v.isChecked(): sl = np.flipud(sl)
            self._atlas_slice_arr = sl.copy()
            slice_path = out / f"{pfx}_atlas_slice.tif"
            tifffile.imwrite(str(slice_path), sl.astype(np.float32))

            # Update main viewer with high-quality slice
            for name in self._atlas_channels:
                lname = f"atlas_{name}"
                if lname in self._viewer.layers:
                    self._viewer.layers[lname].data = sl

            # Settings JSON
            nz, ny, nx = primary.shape
            settings = {
                "timestamp":           datetime.now().isoformat(),
                "atlas_path":          str(self._atlas_path) if self._atlas_path else None,
                "section_image":       str(self._section_path) if self._section_path else None,
                "atlas_shape_zyx":     list(primary.shape),
                "atlas_spacing_um":    {"x": sx, "y": sy, "z": sz},
                "rotation_degrees":    {"rx_pitch": rx, "ry_yaw": ry, "rz_roll": rz},
                "rotation_matrix_3x3": R_mat.tolist(),
                "z_index":             z_idx,
                "z_physical_um":       z_idx * sz,
                "flip_horizontal":     self._flip_h.isChecked(),
                "flip_vertical":       self._flip_v.isChecked(),
                "orientation":         self._orientation.currentText(),
                "atlas_slice_tif":     str(slice_path),
            }
            self._current_settings = settings
            settings_path = out / f"{pfx}_settings.json"
            with open(settings_path, "w") as f:
                json.dump(settings, f, indent=2)
            self._settings_path = settings_path

            # Lock and reveal Phase 2
            self._phase1_locked = True
            self._set_rotation_locked(True)
            self._tabs.setTabEnabled(1, True)
            self._tabs.setCurrentIndex(1)
            self._update_pair_btn_state()
            self._phase1_status.setText(f"✓ Saved: {settings_path.name}")
            self._phase1_status.setStyleSheet("font-weight: bold; color: #2a8a2a;")
            self._set_status(f"Phase 1 saved.  Proceed to Phase 2.")
        except Exception as e:
            QMessageBox.critical(self, "Error saving Phase 1", str(e))

    def _on_load_phase1(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load rotation settings JSON", "", "JSON (*.json);;All files (*)")
        if not path: return
        try:
            with open(path) as f:
                s = json.load(f)
            # Restore only the four values needed to reproduce the rotation
            rot = s.get("rotation_degrees", {})
            self._rx_spin.setValue(rot.get("rx_pitch", 0))
            self._ry_spin.setValue(rot.get("ry_yaw",   0))
            self._rz_spin.setValue(rot.get("rz_roll",  0))
            self._z_spin.setValue(s.get("z_index", 0))
            self._current_settings = s
            self._settings_path = Path(path)
            self._phase1_status.setText(f"Loaded: {Path(path).name}  — adjust & save to lock")
            self._phase1_status.setStyleSheet("font-weight: bold; color: #a07000;")
            self._set_status(f"Rotation restored from {Path(path).name}. Adjust if needed, then Save & lock.")
        except Exception as e:
            QMessageBox.critical(self, "Error loading Phase 1", str(e))

    # ============================================================ Landmark pairing

    def _update_pair_btn_state(self) -> None:
        ready = (self._phase1_locked and
                 self._second_viewer is not None and
                 self._pair_state == _IDLE)
        self._btn_add_pair.setEnabled(ready)

    def _on_start_pair(self) -> None:
        self._pair_state = _WAIT_ATL; self._pending_atl_xy = None
        if self._atl_lm_layer is None or "atl_landmarks" not in self._viewer.layers:
            self._atl_lm_layer = self._viewer.add_points(
                name="atl_landmarks", size=14, face_color="cyan", border_color="white")
        self._atl_n_before = len(self._atl_lm_layer.data)
        self._atl_lm_layer.mode = "add"
        try: self._atl_lm_layer.events.data.disconnect(self._on_atl_clicked)
        except Exception: pass
        self._atl_lm_layer.events.data.connect(self._on_atl_clicked)
        self._btn_add_pair.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._pair_status.setText("Step 1/2 — Click landmark on Atlas slice (main window)")
        self._pair_status.setStyleSheet("font-weight: bold; color: #1a7abf;")

    def _on_atl_clicked(self, event=None) -> None:
        if self._pair_state != _WAIT_ATL: return
        if self._atl_lm_layer is None: return
        if len(self._atl_lm_layer.data) <= self._atl_n_before: return
        yx = self._atl_lm_layer.data[self._atl_n_before]
        self._pending_atl_xy = [float(yx[1]), float(yx[0])]
        self._pair_state = _WAIT_SEC
        self._atl_lm_layer.mode = "pan_zoom"
        try: self._atl_lm_layer.events.data.disconnect(self._on_atl_clicked)
        except Exception: pass
        if self._sec_lm_layer is None or "sec_landmarks" not in self._second_viewer.layers:
            self._sec_lm_layer = self._second_viewer.add_points(
                name="sec_landmarks", size=14, face_color="yellow", border_color="white")
        self._sec_n_before = len(self._sec_lm_layer.data)
        self._sec_lm_layer.mode = "add"
        try: self._sec_lm_layer.events.data.disconnect(self._on_sec_clicked)
        except Exception: pass
        self._sec_lm_layer.events.data.connect(self._on_sec_clicked)
        self._pair_status.setText("Step 2/2 — Click matching landmark on Section (second window)")
        self._pair_status.setStyleSheet("font-weight: bold; color: #c07000;")

    def _on_sec_clicked(self, event=None) -> None:
        if self._pair_state != _WAIT_SEC: return
        if self._sec_lm_layer is None: return
        if len(self._sec_lm_layer.data) <= self._sec_n_before: return
        yx = self._sec_lm_layer.data[self._sec_n_before]
        sec_xy = [float(yx[1]), float(yx[0])]
        self._pairs.append({"atl": self._pending_atl_xy, "sec": sec_xy})
        self._sec_lm_layer.mode = "pan_zoom"
        try: self._sec_lm_layer.events.data.disconnect(self._on_sec_clicked)
        except Exception: pass
        self._pair_state = _IDLE; self._pending_atl_xy = None
        self._btn_cancel.setEnabled(False); self._update_pair_btn_state()
        self._update_table(); self._refresh_lm_layers()
        self._pair_status.setText(f"{len(self._pairs)} pair(s) — Add more or compute TPS.")
        self._pair_status.setStyleSheet("font-weight: bold; color: #2a8a2a;")

    def _on_cancel_pair(self) -> None:
        if (self._pair_state == _WAIT_SEC and self._atl_lm_layer is not None
                and len(self._atl_lm_layer.data) > self._atl_n_before):
            self._atl_lm_layer.data = self._atl_lm_layer.data[:self._atl_n_before]
        try: self._atl_lm_layer.events.data.disconnect(self._on_atl_clicked)
        except Exception: pass
        try: self._sec_lm_layer.events.data.disconnect(self._on_sec_clicked)
        except Exception: pass
        if self._atl_lm_layer: self._atl_lm_layer.mode = "pan_zoom"
        if self._sec_lm_layer: self._sec_lm_layer.mode = "pan_zoom"
        self._pair_state = _IDLE; self._pending_atl_xy = None
        self._btn_cancel.setEnabled(False); self._update_pair_btn_state()
        self._pair_status.setText(f"{len(self._pairs)} pair(s).")
        self._pair_status.setStyleSheet("font-weight: bold; color: #555;")

    def _update_table(self) -> None:
        self._table.setRowCount(len(self._pairs))
        for i, p in enumerate(self._pairs):
            self._table.setItem(i, 0, self._ro_item(str(i+1)))
            self._table.setItem(i, 1, self._ro_item(f"{p['atl'][0]:.1f}"))
            self._table.setItem(i, 2, self._ro_item(f"{p['atl'][1]:.1f}"))
            self._table.setItem(i, 3, self._ro_item(f"{p['sec'][0]:.1f}"))
            self._table.setItem(i, 4, self._ro_item(f"{p['sec'][1]:.1f}"))
            btn = QPushButton("✕"); btn.setFixedWidth(42)
            btn.clicked.connect(lambda _, idx=i: self._delete_pair(idx))
            self._table.setCellWidget(i, 5, btn)

    def _delete_pair(self, idx: int) -> None:
        if 0 <= idx < len(self._pairs):
            self._pairs.pop(idx); self._update_table(); self._refresh_lm_layers()

    def _on_table_selection_changed(self) -> None:
        """Highlight the selected landmark pair in both napari viewers."""
        row = self._table.currentRow()
        if row < 0 or row >= len(self._pairs):
            return
        try:
            if self._atl_lm_layer is not None and "atl_landmarks" in self._viewer.layers:
                self._atl_lm_layer.selected_data = {row}
                self._viewer.layers.selection.active = self._atl_lm_layer
        except Exception:
            pass
        try:
            if (self._sec_lm_layer is not None and self._second_viewer is not None
                    and "sec_landmarks" in self._second_viewer.layers):
                self._sec_lm_layer.selected_data = {row}
                self._second_viewer.layers.selection.active = self._sec_lm_layer
        except Exception:
            pass

    def _refresh_lm_layers(self) -> None:
        atl = np.array([[p["atl"][1], p["atl"][0]] for p in self._pairs]) if self._pairs else np.empty((0,2))
        sec = np.array([[p["sec"][1], p["sec"][0]] for p in self._pairs]) if self._pairs else np.empty((0,2))
        if self._atl_lm_layer is not None and "atl_landmarks" in self._viewer.layers:
            self._atl_lm_layer.data = atl
        if self._sec_lm_layer is not None and self._second_viewer is not None:
            if "sec_landmarks" in self._second_viewer.layers:
                self._sec_lm_layer.data = sec

    def _on_load_landmarks_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load landmarks CSV", "", "CSV (*.csv);;All files (*)")
        if not path: return
        try:
            import pandas as pd
            df = pd.read_csv(path); df.columns = [c.lower() for c in df.columns]
            required = {"atlas_x", "atlas_y", "sec_x", "sec_y"}
            if not required.issubset(set(df.columns)):
                QMessageBox.critical(self, "Bad CSV",
                    f"Expected: {required}\nFound: {list(df.columns)}"); return
            self._pairs = [
                {"atl": [float(r["atlas_x"]), float(r["atlas_y"])],
                 "sec": [float(r["sec_x"]),   float(r["sec_y"])]}
                for _, r in df.iterrows()]
            self._update_table(); self._refresh_lm_layers()
            self._pair_status.setText(f"{len(self._pairs)} pair(s) loaded.")
            self._pair_status.setStyleSheet("font-weight: bold; color: #2a8a2a;")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ============================================================ TPS

    def _on_compute_tps(self) -> None:
        if len(self._pairs) < 4:
            QMessageBox.warning(self, "Too few pairs",
                "Need ≥ 4 pairs. 6–10+ recommended."); return
        atl_pts = np.array([p["atl"] for p in self._pairs])
        sec_pts = np.array([p["sec"] for p in self._pairs])
        try:
            # Forward: atlas → section  (for warping atlas image into section space)
            self._tps_fwd = TPSTransform(src_pts=atl_pts, dst_pts=sec_pts)
            # Inverse: section → atlas  (for mapping cells and warping section image)
            self._tps_inv = TPSTransform(src_pts=sec_pts, dst_pts=atl_pts)
            # Residual check
            pred_fwd = self._tps_fwd(atl_pts)
            res = float(np.sqrt(((pred_fwd - sec_pts)**2).sum(axis=1)).mean())
            self._tps_info.setText(
                f"TPS fitted on {len(self._pairs)} pairs  |  residual: {res:.2f} px")
            self._set_status("TPS computed.")
        except Exception as e:
            QMessageBox.critical(self, "TPS error", str(e))

    def _on_apply_tps(self) -> None:
        if self._tps_inv is None:
            QMessageBox.warning(self, "No TPS", "Compute TPS first."); return
        if self._cell_pts_sec is None:
            QMessageBox.warning(self, "No cells", "Load cell CSV first."); return
        z_idx = self._z_spin.value()
        self._cell_pts_atlas = self._tps_inv(self._cell_pts_sec)
        pts_yx = self._cell_pts_atlas[:, ::-1]
        if "cells_atlas" in self._viewer.layers:
            self._viewer.layers["cells_atlas"].data = pts_yx
        else:
            self._viewer.add_points(pts_yx, name="cells_atlas", size=8,
                                    face_color="red", border_color="white", opacity=0.8)
        self._set_status(
            f"{len(self._cell_pts_atlas)} cells mapped → rotated atlas slice space "
            f"(z_rot = {z_idx}).")

    # ============================================================ Warp visualisation

    # =========================================================================
    # TODO: Image warp visualisation is currently under review.
    #
    # Known issue: warp output appears incorrect (partially/fully black or
    # misaligned) for some landmark configurations and image sizes.
    #
    # Suspected causes to investigate:
    #   1. Coordinate axis convention mismatch between skimage.warp (row=y, col=x)
    #      and the TPS which was fitted in (x, y) order.
    #   2. Downsampling of the output grid — coordinates are scaled to full section
    #      space but there may be an off-by-one or aspect-ratio error.
    #   3. TPS extrapolation outside the convex hull of landmarks producing garbage
    #      values that skimage clips to 0.
    #
    # The cell coordinate mapping (_on_apply_tps) is NOT affected — it calls the
    # TPS directly on the cell points without any image warping.
    # =========================================================================

    def _on_warp_atlas_to_section(self) -> None:
        """Warp atlas slice into section space and display in second window."""
        if self._tps_fwd is None:
            QMessageBox.warning(self, "No TPS", "Compute TPS first."); return
        if self._atlas_slice_arr is None:
            QMessageBox.warning(self, "No atlas slice",
                "Save Phase 1 first (exports the atlas slice)."); return
        if self._section_arr is None:
            QMessageBox.warning(self, "No section", "Load section image first."); return
        try:
            atlas_sl = self._atlas_slice_arr
            out_shape = self._section_arr.shape[:2]

            # Reduce output to max 2048px for speed — TPS at full section size is very slow
            MAX_PX = 2048
            scale       = min(1.0, MAX_PX / max(out_shape))
            small_shape = (max(1, int(out_shape[0]*scale)),
                           max(1, int(out_shape[1]*scale)))
            scale_h = out_shape[0] / small_shape[0]
            scale_w = out_shape[1] / small_shape[1]

            def _inv_map_atl2sec(coords):
                # coords: (2, small_rows, small_cols) — output grid in downsampled space
                # Scale up to full section pixel space before applying TPS
                spatial = coords.shape[1:]
                y_full  = coords[0].ravel() * scale_h   # → full section row coords
                x_full  = coords[1].ravel() * scale_w   # → full section col coords
                pts_xy  = np.column_stack([x_full, y_full])  # (N,2) x,y in section space
                src_xy  = self._tps_inv(pts_xy)              # → rotated atlas pixel coords
                result  = np.empty_like(coords)
                result[0] = src_xy[:, 1].reshape(spatial)    # y in atlas slice
                result[1] = src_xy[:, 0].reshape(spatial)    # x in atlas slice
                return result

            # Sample from full-resolution atlas slice, output at small_shape
            warped_small = sk_warp(atlas_sl.astype(np.float32),
                                   _inv_map_atl2sec, output_shape=small_shape,
                                   order=1, preserve_range=True, cval=0)
            # Upsample back to section size for display
            if scale < 1.0:
                from skimage.transform import resize as sk_resize
                warped = sk_resize(warped_small, out_shape,
                                   preserve_range=True, anti_aliasing=False)
            else:
                warped = warped_small
            self._atlas_warped_arr = warped.astype(np.float32)

            if self._second_viewer is None:
                QMessageBox.warning(self, "No second viewer",
                    "Load section image first."); return
            if "atlas_warped" in self._second_viewer.layers:
                self._second_viewer.layers["atlas_warped"].data = self._atlas_warped_arr
            else:
                self._second_viewer.add_image(
                    self._atlas_warped_arr, name="atlas_warped",
                    colormap="green", blending="additive", opacity=0.5)
            self._set_status("Atlas warped → section space (green layer in second window).")
        except Exception as e:
            QMessageBox.critical(self, "Warp error", str(e))

    def _on_atl_overlay_toggled(self, checked: bool) -> None:
        if self._second_viewer and "atlas_warped" in self._second_viewer.layers:
            self._second_viewer.layers["atlas_warped"].visible = checked

    def _on_warp_section_to_atlas(self) -> None:
        """Warp section image into atlas slice space and display in main window."""
        if self._tps_inv is None:
            QMessageBox.warning(self, "No TPS", "Compute TPS first."); return
        if self._section_arr is None:
            QMessageBox.warning(self, "No section", "Load section image first."); return
        if self._atlas_slice_arr is None:
            QMessageBox.warning(self, "No atlas slice",
                "Save Phase 1 first."); return
        try:
            out_shape = self._atlas_slice_arr.shape[:2]

            # Atlas output is typically small (e.g. 528×456) — no downsampling needed.
            # For each atlas pixel, _tps_fwd maps → section coords; sample from section_arr.
            def _inv_map_sec2atl(coords):
                # coords: (2, atlas_rows, atlas_cols) — output grid in atlas space
                spatial = coords.shape[1:]
                y_flat  = coords[0].ravel()                   # atlas row coords (y)
                x_flat  = coords[1].ravel()                   # atlas col coords (x)
                pts_xy  = np.column_stack([x_flat, y_flat])   # (N,2) in atlas pixel space
                src_xy  = self._tps_fwd(pts_xy)               # → section pixel coords
                result  = np.empty_like(coords)
                result[0] = src_xy[:, 1].reshape(spatial)     # y in section
                result[1] = src_xy[:, 0].reshape(spatial)     # x in section
                return result

            warped = sk_warp(self._section_arr.astype(np.float32),
                             _inv_map_sec2atl, output_shape=out_shape,
                             order=1, preserve_range=True, cval=0)
            self._section_warped_arr = warped.astype(np.float32)

            if "section_warped" in self._viewer.layers:
                self._viewer.layers["section_warped"].data = self._section_warped_arr
            else:
                self._viewer.add_image(
                    self._section_warped_arr, name="section_warped",
                    colormap="magenta", blending="additive", opacity=0.5)
            self._set_status("Section warped → atlas space (magenta layer in main window).")
        except Exception as e:
            QMessageBox.critical(self, "Warp error", str(e))

    def _on_sec_overlay_toggled(self, checked: bool) -> None:
        if "section_warped" in self._viewer.layers:
            self._viewer.layers["section_warped"].visible = checked

    # ============================================================ Save session

    def _on_save_session(self) -> None:
        if not self._pairs:
            QMessageBox.warning(self, "No landmarks", "Add landmark pairs first."); return
        if self._tps_inv is None:
            QMessageBox.warning(self, "No TPS", "Compute TPS first."); return
        if self._cell_pts_atlas is None:
            QMessageBox.warning(self, "Not applied", "Apply TPS to cells first."); return

        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not out_dir: return

        import pandas as pd
        out  = Path(out_dir)
        stem = self._atlas_path.stem if self._atlas_path else "atlas"
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        pfx  = f"{stem}_{ts}"
        z_idx = self._z_spin.value()

        # 1 ── Landmarks CSV  (rename to rot_atlas for clarity)
        lm_df = pd.DataFrame([
            {"pair_id": i+1,
             "rot_atlas_x": p["atl"][0], "rot_atlas_y": p["atl"][1],
             "sec_x":       p["sec"][0], "sec_y":       p["sec"][1]}
            for i, p in enumerate(self._pairs)])
        lm_path = out / f"{pfx}_landmarks.csv"
        lm_df.to_csv(lm_path, index=False)

        # 2 ── Cell coordinates CSV: rotated atlas coords + unrotated CCF voxels
        df_out = self._cell_df.copy() if self._cell_df is not None else pd.DataFrame()
        if self._cell_pts_sec is not None:
            df_out["x_section"] = self._cell_pts_sec[:, 0]
            df_out["y_section"] = self._cell_pts_sec[:, 1]
        df_out["x_rot"] = self._cell_pts_atlas[:, 0]
        df_out["y_rot"] = self._cell_pts_atlas[:, 1]
        df_out["z_rot"] = float(z_idx)

        # Unrotate: (x_rot, y_rot, z_rot) → original CCF voxel indices
        # Reproduces the oblique-slice centering used in _oblique_slice()
        settings   = getattr(self, "_current_settings", {})
        R_mat      = settings.get("rotation_matrix_3x3")
        sp         = settings.get("atlas_spacing_um", {})
        atlas_shape = settings.get("atlas_shape_zyx")

        if R_mat is not None and atlas_shape is not None:
            sx = sp.get("x", self._sp_x.value())
            sy = sp.get("y", self._sp_y.value())
            sz = sp.get("z", self._sp_z.value())
            nz, ny, nx = atlas_shape
            cx = nx * sx / 2.0
            cy = ny * sy / 2.0
            cz = nz * sz / 2.0

            R     = Rotation.from_matrix(np.array(R_mat))
            R_inv = R.inv()

            # Physical coords in rotated frame (centred)
            u = self._cell_pts_atlas[:, 0] * sx - cx
            v = self._cell_pts_atlas[:, 1] * sy - cy
            w = np.full(len(u), z_idx * sz - cz)
            pts_rot = np.column_stack([u, v, w])

            # Apply inverse rotation → original physical coords (centred)
            pts_orig = R_inv.apply(pts_rot)

            # Convert back to voxel indices
            df_out["x_ccf"] = np.round((pts_orig[:, 0] + cx) / sx).astype(int)
            df_out["y_ccf"] = np.round((pts_orig[:, 1] + cy) / sy).astype(int)
            df_out["z_ccf"] = np.round((pts_orig[:, 2] + cz) / sz).astype(int)
        else:
            self._set_status("Warning: rotation matrix not found — CCF voxels not computed.")

        cells_path = out / f"{pfx}_cells_atlas_slice.csv"
        df_out.to_csv(cells_path, index=False)

        # 3 ── Session JSON (references all output files + rotation settings)
        session = {
            "step": "atlas_registration",
            "atlas_path":          str(self._atlas_path) if self._atlas_path else None,
            "section_image":       str(self._section_path) if self._section_path else None,
            "settings_json":       str(self._settings_path) if hasattr(self, "_settings_path") else None,
            "landmarks_csv":       str(lm_path),
            "cells_atlas_csv":     str(cells_path),
            "z_index":             z_idx,
            "rotation_matrix_3x3": getattr(self, "_current_settings", {}).get("rotation_matrix_3x3"),
            "note_coordinates":    (
                "x_rot/y_rot/z_rot = pixel coords in the ROTATED atlas slice (z_rot = z_index). "
                "x_ccf/y_ccf/z_ccf = original CCF voxel indices after applying R_inv. "
                "CCF convention: X=medial-lateral, Y=dorsal-ventral, Z=anterior-posterior."
            ),
        }
        session_path = out / f"{pfx}_session.json"
        with open(session_path, "w") as f:
            json.dump(session, f, indent=2)

        saved_files = [lm_path.name, session_path.name, cells_path.name]

        # 4 ── Warped images (if available)
        if self._atlas_warped_arr is not None:
            p = out / f"{pfx}_atlas_warped_to_section.tif"
            tifffile.imwrite(str(p), self._atlas_warped_arr)
            saved_files.append(p.name)
        if self._section_warped_arr is not None:
            p = out / f"{pfx}_section_warped_to_atlas.tif"
            tifffile.imwrite(str(p), self._section_warped_arr)
            saved_files.append(p.name)

        self._set_status(f"Session saved to {out}")
        QMessageBox.information(self, "Session saved",
            "\n".join(f"  {f}" for f in saved_files) + "\n\n"
            "x_rot, y_rot, z_rot in cells CSV = rotated atlas voxel coords.\n"
            "Pass cells CSV to the coordinate conversion step.")
