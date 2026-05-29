"""
Step 2 — Section → Atlas Slice Alignment Widget
================================================
Aligns the brain section image to the rotated atlas slice using a thin-plate
spline (TPS) transform fitted from manually placed landmark pairs.

This replaces BigWarp.  Everything runs inside napari.

Workflow
--------
1. Load section image      → main napari viewer
2. Load cell CSV           → Points layer on the section (x_section, y_section)
3. Load atlas slice        → independent second napari window
4. Load atlas settings JSON (exported by Atlas Setup widget) — carries rotation
   matrix, z_index, voxel spacing needed for Steps 3-4
5. Add landmark pairs (sequential guided, same as Step 1):
     Click section (main) → click atlas slice (second) → pair saved to table
6. Compute TPS transform   → section space → atlas slice pixel space
7. Apply → cell dots appear on atlas slice in second window
8. Save session:
     landmarks.csv, session.json, cells_atlas_slice.csv  (→ input for Step 3)

Coordinate note
---------------
All coordinates are in pixel space.  The atlas slice pixel (x, y) combined
with z_index and the rotation matrix (stored in the atlas settings JSON) gives
the 3-D CCF voxel in Step 4.
"""

import json
import numpy as np
import napari
from pathlib import Path
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QVBoxLayout, QWidget,
)

from ._widget_rotation import _load_image_any, _collapse_to_2d
from .registration.tps import TPSTransform

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
_IDLE     = "idle"
_WAIT_SEC = "waiting_section"
_WAIT_ATL = "waiting_atlas"


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class SectionAtlasWidget(QWidget):
    """Step 2 — Section → Atlas slice TPS alignment."""

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = napari_viewer

        self._section_path  = None
        self._atlas_path    = None
        self._settings_path = None
        self._atlas_settings = {}
        self._cell_df       = None
        self._cell_pts_sec  = None   # (N,2) x,y in section pixels
        self._tps           = None   # fitted TPSTransform
        self._second_viewer = None
        self._sec_lm_layer  = None
        self._atl_lm_layer  = None

        self._pairs = []             # [{"sec":[x,y], "atl":[x,y]}, ...]
        self._pair_state    = _IDLE
        self._pending_sec_xy = None
        self._sec_n_before  = 0
        self._atl_n_before  = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)
        layout.addWidget(self._build_load_group())
        layout.addWidget(self._build_landmark_group())
        layout.addWidget(self._build_transform_group())
        layout.addWidget(self._build_save_group())
        self._status = QLabel("Ready")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    def _build_load_group(self) -> QGroupBox:
        box = QGroupBox("Load")
        layout = QVBoxLayout(box)

        btn_sec = QPushButton("Load section image…")
        btn_sec.clicked.connect(self._on_load_section)
        layout.addWidget(btn_sec)

        btn_csv = QPushButton("Load cell CSV (x_section, y_section)…")
        btn_csv.clicked.connect(self._on_load_csv)
        layout.addWidget(btn_csv)

        btn_atl = QPushButton("Load atlas slice image…")
        btn_atl.clicked.connect(self._on_load_atlas_slice)
        layout.addWidget(btn_atl)

        btn_json = QPushButton("Load atlas settings JSON…")
        btn_json.clicked.connect(self._on_load_settings)
        layout.addWidget(btn_json)

        self._load_info = QLabel("Nothing loaded")
        self._load_info.setWordWrap(True)
        layout.addWidget(self._load_info)
        return box

    def _build_landmark_group(self) -> QGroupBox:
        box = QGroupBox("Landmark pairs")
        layout = QVBoxLayout(box)

        btn_load_lm = QPushButton("Load landmarks from CSV…")
        btn_load_lm.clicked.connect(self._on_load_landmarks_csv)
        layout.addWidget(btn_load_lm)

        self._pair_status = QLabel("Load section and atlas slice to begin.")
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

        btn_row.addWidget(self._btn_add_pair)
        btn_row.addWidget(self._btn_cancel)
        layout.addLayout(btn_row)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["#", "Sec x", "Sec y", "Atlas x", "Atlas y", ""]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Fixed)
        self._table.setColumnWidth(5, 54)
        self._table.setMinimumHeight(140)
        self._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(self._table)
        return box

    def _build_transform_group(self) -> QGroupBox:
        box = QGroupBox("TPS Transform")
        layout = QVBoxLayout(box)

        btn_compute = QPushButton("Compute TPS transform")
        btn_compute.clicked.connect(self._on_compute_tps)
        layout.addWidget(btn_compute)

        self._tps_info = QLabel("No transform computed")
        self._tps_info.setWordWrap(True)
        layout.addWidget(self._tps_info)

        btn_apply = QPushButton("Apply → show cells on atlas slice")
        btn_apply.clicked.connect(self._on_apply_transform)
        layout.addWidget(btn_apply)
        return box

    def _build_save_group(self) -> QGroupBox:
        box = QGroupBox("Save results")
        layout = QVBoxLayout(box)

        btn = QPushButton("Save session (landmarks + cells)…")
        btn.clicked.connect(self._on_save_session)
        layout.addWidget(btn)

        info = QLabel(
            "Saves:\n"
            "  • landmarks.csv         — reload pairs next time\n"
            "  • session.json          — paths + atlas settings\n"
            "  • cells_atlas_slice.csv — input for Step 3"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(info)
        return box

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _on_load_section(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load section image", "",
            "Images (*.tif *.tiff *.png *.jpg *.czi *.nd2);;All files (*)"
        )
        if not path:
            return
        try:
            arr = _collapse_to_2d(_load_image_any(path))
            self._section_path = Path(path)
            if "section" in self._viewer.layers:
                self._viewer.layers["section"].data = arr
            else:
                self._viewer.add_image(arr, name="section", colormap="gray")
            self._viewer.reset_view()
            self._update_load_info()
            self._update_pair_btn_state()
            self._set_status(f"Section loaded: {self._section_path.name}  {arr.shape[1]}×{arr.shape[0]} px")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load cell CSV", "", "CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            # Accept x_section/y_section or plain x/y
            xcol = cols.get("x_section") or cols.get("x")
            ycol = cols.get("y_section") or cols.get("y")
            if xcol is None or ycol is None:
                QMessageBox.critical(
                    self, "Error",
                    "CSV must have 'x_section'/'y_section' or 'x'/'y' columns."
                )
                return
            self._cell_pts_sec = df[[xcol, ycol]].to_numpy(dtype=float)
            self._cell_df = df
            pts_yx = self._cell_pts_sec[:, ::-1]
            if "cells_section" in self._viewer.layers:
                self._viewer.layers["cells_section"].data = pts_yx
            else:
                self._viewer.add_points(
                    pts_yx, name="cells_section", size=8,
                    face_color="red", border_color="white", opacity=0.8
                )
            self._update_load_info()
            self._set_status(f"Loaded {len(self._cell_pts_sec)} cells.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_load_atlas_slice(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load atlas slice", "",
            "Images (*.tif *.tiff *.png);;All files (*)"
        )
        if not path:
            return
        try:
            arr = _collapse_to_2d(_load_image_any(path))
            self._atlas_path = Path(path)
            if self._second_viewer is not None:
                try:
                    self._second_viewer.close()
                except Exception:
                    pass
            self._second_viewer = napari.Viewer(
                title=f"Atlas slice — {self._atlas_path.name}"
            )
            self._second_viewer.add_image(arr, name="atlas_slice", colormap="gray")
            self._second_viewer.reset_view()
            self._update_load_info()
            self._update_pair_btn_state()
            self._set_status(f"Atlas slice loaded: {self._atlas_path.name}  {arr.shape[1]}×{arr.shape[0]} px")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_load_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load atlas settings JSON", "", "JSON (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                self._atlas_settings = json.load(f)
            self._settings_path = Path(path)
            rot = self._atlas_settings.get("rotation_degrees", {})
            self._update_load_info()
            self._set_status(
                f"Settings: {self._settings_path.name}  "
                f"z={self._atlas_settings.get('z_index','?')}  "
                f"Rx={rot.get('rx_pitch',0):.1f}° "
                f"Ry={rot.get('ry_yaw',0):.1f}° "
                f"Rz={rot.get('rz_roll',0):.1f}°"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _update_load_info(self) -> None:
        lines = []
        if self._section_path:
            lines.append(f"Section: {self._section_path.name}")
        if self._cell_pts_sec is not None:
            lines.append(f"Cells: {len(self._cell_pts_sec)}")
        if self._atlas_path:
            lines.append(f"Atlas slice: {self._atlas_path.name}")
        if self._settings_path:
            lines.append(f"Settings: {self._settings_path.name}")
        self._load_info.setText("\n".join(lines) if lines else "Nothing loaded")

    def _update_pair_btn_state(self) -> None:
        ready = self._section_path is not None and self._second_viewer is not None
        self._btn_add_pair.setEnabled(ready and self._pair_state == _IDLE)

    # ------------------------------------------------------------------
    # Sequential landmark pairing
    # ------------------------------------------------------------------

    def _on_start_pair(self) -> None:
        self._pair_state = _WAIT_SEC
        self._pending_sec_xy = None

        if self._sec_lm_layer is None or "sec_landmarks" not in self._viewer.layers:
            self._sec_lm_layer = self._viewer.add_points(
                name="sec_landmarks", size=14,
                face_color="cyan", border_color="white",
            )
        self._sec_n_before = len(self._sec_lm_layer.data)
        self._sec_lm_layer.mode = "add"
        try:
            self._sec_lm_layer.events.data.disconnect(self._on_sec_clicked)
        except Exception:
            pass
        self._sec_lm_layer.events.data.connect(self._on_sec_clicked)

        self._btn_add_pair.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._pair_status.setText(
            "Step 1/2 — Click a landmark on the Section image (main window)"
        )
        self._pair_status.setStyleSheet("font-weight: bold; color: #1a7abf;")

    def _on_sec_clicked(self, event=None) -> None:
        if self._pair_state != _WAIT_SEC:
            return
        if self._sec_lm_layer is None:
            return
        if len(self._sec_lm_layer.data) <= self._sec_n_before:
            return

        yx = self._sec_lm_layer.data[self._sec_n_before]
        self._pending_sec_xy = [float(yx[1]), float(yx[0])]

        self._pair_state = _WAIT_ATL
        self._sec_lm_layer.mode = "pan_zoom"
        try:
            self._sec_lm_layer.events.data.disconnect(self._on_sec_clicked)
        except Exception:
            pass

        if self._atl_lm_layer is None or "atl_landmarks" not in self._second_viewer.layers:
            self._atl_lm_layer = self._second_viewer.add_points(
                name="atl_landmarks", size=14,
                face_color="yellow", border_color="white",
            )
        self._atl_n_before = len(self._atl_lm_layer.data)
        self._atl_lm_layer.mode = "add"
        try:
            self._atl_lm_layer.events.data.disconnect(self._on_atl_clicked)
        except Exception:
            pass
        self._atl_lm_layer.events.data.connect(self._on_atl_clicked)

        self._pair_status.setText(
            "Step 2/2 — Click the matching landmark on the Atlas slice (second window)"
        )
        self._pair_status.setStyleSheet("font-weight: bold; color: #c07000;")

    def _on_atl_clicked(self, event=None) -> None:
        if self._pair_state != _WAIT_ATL:
            return
        if self._atl_lm_layer is None:
            return
        if len(self._atl_lm_layer.data) <= self._atl_n_before:
            return

        yx = self._atl_lm_layer.data[self._atl_n_before]
        atl_xy = [float(yx[1]), float(yx[0])]

        self._pairs.append({"sec": self._pending_sec_xy, "atl": atl_xy})

        self._atl_lm_layer.mode = "pan_zoom"
        try:
            self._atl_lm_layer.events.data.disconnect(self._on_atl_clicked)
        except Exception:
            pass

        self._pair_state = _IDLE
        self._pending_sec_xy = None
        self._btn_cancel.setEnabled(False)
        self._update_pair_btn_state()
        self._update_table()
        self._refresh_lm_layers()
        self._pair_status.setText(
            f"{len(self._pairs)} pair(s) — ready.  Add more or compute TPS."
        )
        self._pair_status.setStyleSheet("font-weight: bold; color: #2a8a2a;")

    def _on_cancel_pair(self) -> None:
        if (self._pair_state == _WAIT_ATL
                and self._sec_lm_layer is not None
                and len(self._sec_lm_layer.data) > self._sec_n_before):
            self._sec_lm_layer.data = self._sec_lm_layer.data[:self._sec_n_before]
        try:
            self._sec_lm_layer.events.data.disconnect(self._on_sec_clicked)
        except Exception:
            pass
        try:
            self._atl_lm_layer.events.data.disconnect(self._on_atl_clicked)
        except Exception:
            pass
        if self._sec_lm_layer is not None:
            self._sec_lm_layer.mode = "pan_zoom"
        if self._atl_lm_layer is not None:
            self._atl_lm_layer.mode = "pan_zoom"

        self._pair_state = _IDLE
        self._pending_sec_xy = None
        self._btn_cancel.setEnabled(False)
        self._update_pair_btn_state()
        self._pair_status.setText(f"{len(self._pairs)} pair(s).")
        self._pair_status.setStyleSheet("font-weight: bold; color: #555;")

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _update_table(self) -> None:
        self._table.setRowCount(len(self._pairs))
        for i, p in enumerate(self._pairs):
            self._table.setItem(i, 0, self._ro_item(str(i + 1)))
            self._table.setItem(i, 1, self._ro_item(f"{p['sec'][0]:.1f}"))
            self._table.setItem(i, 2, self._ro_item(f"{p['sec'][1]:.1f}"))
            self._table.setItem(i, 3, self._ro_item(f"{p['atl'][0]:.1f}"))
            self._table.setItem(i, 4, self._ro_item(f"{p['atl'][1]:.1f}"))
            btn = QPushButton("✕")
            btn.setFixedWidth(42)
            btn.clicked.connect(lambda _, idx=i: self._delete_pair(idx))
            self._table.setCellWidget(i, 5, btn)

    def _ro_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _delete_pair(self, idx: int) -> None:
        if 0 <= idx < len(self._pairs):
            self._pairs.pop(idx)
            self._update_table()
            self._refresh_lm_layers()
            self._pair_status.setText(f"{len(self._pairs)} pair(s).")

    def _refresh_lm_layers(self) -> None:
        sec_pts = np.array([[p["sec"][1], p["sec"][0]] for p in self._pairs]) if self._pairs else np.empty((0, 2))
        atl_pts = np.array([[p["atl"][1], p["atl"][0]] for p in self._pairs]) if self._pairs else np.empty((0, 2))
        if self._sec_lm_layer is not None and "sec_landmarks" in self._viewer.layers:
            self._sec_lm_layer.data = sec_pts
        if self._atl_lm_layer is not None and self._second_viewer is not None:
            if "atl_landmarks" in self._second_viewer.layers:
                self._atl_lm_layer.data = atl_pts

    # ------------------------------------------------------------------
    # TPS transform
    # ------------------------------------------------------------------

    def _on_compute_tps(self) -> None:
        if len(self._pairs) < 4:
            QMessageBox.warning(
                self, "Too few pairs",
                "TPS needs at least 4 landmark pairs for a stable fit.\n"
                "More landmarks (6-10+) give better accuracy."
            )
            return
        src = np.array([p["sec"] for p in self._pairs])  # section x,y
        dst = np.array([p["atl"] for p in self._pairs])  # atlas x,y
        try:
            self._tps = TPSTransform(src_pts=src, dst_pts=dst)
            self._tps_info.setText(
                f"TPS fitted on {len(self._pairs)} landmark pairs.\n"
                f"Residual (mean landmark error): "
                f"{self._tps_residual(src, dst):.2f} px"
            )
            self._set_status("TPS transform computed.")
        except Exception as e:
            QMessageBox.critical(self, "TPS error", str(e))

    def _tps_residual(self, src: np.ndarray, dst: np.ndarray) -> float:
        pred = self._tps(src)
        return float(np.sqrt(((pred - dst) ** 2).sum(axis=1)).mean())

    def _on_apply_transform(self) -> None:
        if self._tps is None:
            QMessageBox.warning(self, "No transform", "Compute TPS first.")
            return
        if self._cell_pts_sec is None:
            QMessageBox.warning(self, "No cells", "Load cell CSV first.")
            return
        if self._second_viewer is None:
            QMessageBox.warning(self, "No atlas", "Load atlas slice first.")
            return

        self._cell_pts_atlas = self._tps(self._cell_pts_sec)  # (N,2) x,y atlas pixels
        pts_yx = self._cell_pts_atlas[:, ::-1]

        if "cells_atlas" in self._second_viewer.layers:
            self._second_viewer.layers["cells_atlas"].data = pts_yx
        else:
            self._second_viewer.add_points(
                pts_yx, name="cells_atlas", size=8,
                face_color="red", border_color="white", opacity=0.8
            )
        self._set_status(f"{len(self._cell_pts_atlas)} cells mapped → atlas slice space.")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _on_load_landmarks_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load landmarks CSV", "", "CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            required = {"sec_x", "sec_y", "atlas_x", "atlas_y"}
            if not required.issubset(set(df.columns)):
                QMessageBox.critical(
                    self, "Bad CSV",
                    f"Expected columns: {required}\nFound: {list(df.columns)}"
                )
                return
            self._pairs = [
                {"sec": [float(r["sec_x"]), float(r["sec_y"])],
                 "atl": [float(r["atlas_x"]), float(r["atlas_y"])]}
                for _, r in df.iterrows()
            ]
            self._update_table()
            self._refresh_lm_layers()
            self._pair_status.setText(
                f"{len(self._pairs)} pair(s) loaded from {Path(path).name}"
            )
            self._pair_status.setStyleSheet("font-weight: bold; color: #2a8a2a;")
            self._set_status(f"Loaded {len(self._pairs)} landmark pairs.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_save_session(self) -> None:
        if not self._pairs:
            QMessageBox.warning(self, "No landmarks", "Add landmark pairs first.")
            return
        if self._tps is None:
            QMessageBox.warning(self, "No transform", "Compute TPS first.")
            return
        if not hasattr(self, "_cell_pts_atlas"):
            QMessageBox.warning(self, "Not applied", "Apply transform to cells first.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not out_dir:
            return

        import pandas as pd
        out  = Path(out_dir)
        stem = self._section_path.stem if self._section_path else "section"

        # 1 ── Landmark pairs CSV
        lm_df = pd.DataFrame([
            {"pair_id": i + 1,
             "sec_x": p["sec"][0], "sec_y": p["sec"][1],
             "atlas_x": p["atl"][0], "atlas_y": p["atl"][1]}
            for i, p in enumerate(self._pairs)
        ])
        lm_path = out / f"{stem}_landmarks.csv"
        lm_df.to_csv(lm_path, index=False)

        # 2 ── Session JSON
        session = {
            "step": "section_to_atlas",
            "section_image":      str(self._section_path)  if self._section_path  else None,
            "atlas_slice_image":  str(self._atlas_path)    if self._atlas_path    else None,
            "atlas_settings_json": str(self._settings_path) if self._settings_path else None,
            "n_landmark_pairs":   len(self._pairs),
            "landmarks_csv":      str(lm_path),
            "cells_atlas_csv":    str(out / f"{stem}_cells_atlas_slice.csv"),
            "atlas_settings":     self._atlas_settings,
        }
        session_path = out / f"{stem}_section_atlas_session.json"
        with open(session_path, "w") as f:
            json.dump(session, f, indent=2)

        # 3 ── Cell coordinates CSV (input for Step 3)
        df_out = self._cell_df.copy() if self._cell_df is not None else pd.DataFrame()
        df_out["x_section"]    = self._cell_pts_sec[:, 0]
        df_out["y_section"]    = self._cell_pts_sec[:, 1]
        df_out["x_atlas_slice"] = self._cell_pts_atlas[:, 0]
        df_out["y_atlas_slice"] = self._cell_pts_atlas[:, 1]
        cells_path = out / f"{stem}_cells_atlas_slice.csv"
        df_out.to_csv(cells_path, index=False)

        self._set_status(f"Session saved to {out}")
        QMessageBox.information(
            self, "Session saved",
            f"  {lm_path.name}\n"
            f"  {session_path.name}\n"
            f"  {cells_path.name}\n\n"
            f"Pass {cells_path.name} to Step 3 (Atlas coordinate conversion)."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self._status.setText(msg)
