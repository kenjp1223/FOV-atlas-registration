"""
Widget 2 — Inverse Warp & Coordinate Readout

Post-BigWarp analysis panel:
  - Load BigWarp landmark CSV
  - Load session (to recover rotation, slice index, bregma, resolution)
  - Compute inverse TPS (target -> atlas slice)
  - Warp target image into moving (atlas slice) space
  - Load points from CSV or click on the target image
  - Transform points to atlas slice space -> 3D voxel -> AP/ML/DV mm
  - Show results table in the panel
"""

import napari
import numpy as np
from pathlib import Path
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class InverseWarpWidget(QWidget):
    """Inverse warp and coordinate readout widget."""

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = napari_viewer
        self._moving_pts: np.ndarray | None = None
        self._fixed_pts:  np.ndarray | None = None
        self._inverse_tps = None
        self._session: dict = {}
        self._bregma_calc = None
        self._mapped_results: dict | None = None
        self._points_layer = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        layout.addWidget(self._build_load_group())
        layout.addWidget(self._build_warp_group())
        layout.addWidget(self._build_points_group())
        layout.addWidget(self._build_results_group())
        layout.addWidget(self._build_export_group())

    def _build_load_group(self) -> QGroupBox:
        box = QGroupBox("Load")
        form = QFormLayout()
        box.setLayout(form)

        self._btn_load_session = QPushButton("Load session JSON…")
        self._btn_load_session.clicked.connect(self._on_load_session)
        form.addRow(self._btn_load_session)

        self._btn_load_landmarks = QPushButton("Load BigWarp landmarks CSV…")
        self._btn_load_landmarks.clicked.connect(self._on_load_landmarks)
        form.addRow(self._btn_load_landmarks)

        self._btn_load_target = QPushButton("Load target image…")
        self._btn_load_target.clicked.connect(self._on_load_target)
        form.addRow(self._btn_load_target)

        self._load_status = QLabel("Nothing loaded")
        form.addRow(self._load_status)

        return box

    def _build_warp_group(self) -> QGroupBox:
        box = QGroupBox("Inverse Warp")
        layout = QVBoxLayout()
        box.setLayout(layout)

        self._btn_build_tps = QPushButton("Build inverse TPS transform")
        self._btn_build_tps.clicked.connect(self._on_build_tps)
        layout.addWidget(self._btn_build_tps)

        self._btn_warp_image = QPushButton("Warp target image → atlas space")
        self._btn_warp_image.clicked.connect(self._on_warp_image)
        layout.addWidget(self._btn_warp_image)

        self._warp_status = QLabel("TPS not built yet")
        layout.addWidget(self._warp_status)

        return box

    def _build_points_group(self) -> QGroupBox:
        box = QGroupBox("Points in Target Space")
        layout = QVBoxLayout()
        box.setLayout(layout)

        self._btn_load_pts_csv = QPushButton("Load points CSV (x, y columns)…")
        self._btn_load_pts_csv.clicked.connect(self._on_load_points_csv)
        layout.addWidget(self._btn_load_pts_csv)

        self._btn_click_pts = QPushButton("Click points on target image")
        self._btn_click_pts.setCheckable(True)
        self._btn_click_pts.toggled.connect(self._on_click_mode_toggled)
        layout.addWidget(self._btn_click_pts)

        self._btn_map_points = QPushButton("Map points → atlas coords")
        self._btn_map_points.clicked.connect(self._on_map_points)
        layout.addWidget(self._btn_map_points)

        self._pts_status = QLabel("No points loaded")
        layout.addWidget(self._pts_status)

        return box

    def _build_results_group(self) -> QGroupBox:
        box = QGroupBox("Results")
        layout = QVBoxLayout()
        box.setLayout(layout)

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels([
            "Point", "Target X", "Target Y",
            "Slice X", "Slice Y",
            "AP (mm)", "ML (mm)", # "DV (mm)" added as 8th
        ])
        # add DV column
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels([
            "Point", "Target X", "Target Y",
            "Slice X", "Slice Y",
            "AP (mm)", "ML (mm)", "DV (mm)",
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)

        return box

    def _build_export_group(self) -> QGroupBox:
        box = QGroupBox("Export")
        layout = QHBoxLayout()
        box.setLayout(layout)

        btn_csv = QPushButton("Export results CSV…")
        btn_csv.clicked.connect(self._on_export_csv)
        btn_overlay = QPushButton("Export overlay TIFF…")
        btn_overlay.clicked.connect(self._on_export_overlay)
        layout.addWidget(btn_csv)
        layout.addWidget(btn_overlay)

        return box

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load prism_alignment settings JSON", "",
            "JSON (*.json);;All files (*)"
        )
        if not path:
            return
        from .io.session import load_prism_settings
        try:
            self._prism = load_prism_settings(path)
            # Also try loading a sibling *_plugin.json if it exists
            plugin_path = Path(path).with_name(
                Path(path).stem.replace("_settings", "") + "_plugin.json"
            )
            if plugin_path.exists():
                from .io.session import load_plugin_settings
                self._plugin_settings = load_plugin_settings(plugin_path)
            else:
                self._plugin_settings = {}
            self._session = {**self._prism, **self._plugin_settings}
            self._build_bregma_calc()
            n_ref = len(self._session.get("bregma_references", []))
            self._load_status.setText(
                f"Loaded: {Path(path).name}  |  "
                f"z={self._prism['z_index']}  "
                f"Rx={self._prism['rotation']['rx']:.1f}°  "
                f"{'bregma set' if n_ref else 'no bregma yet'}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error loading settings", str(e))

    def _on_load_landmarks(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open landmarks CSV", "", "CSV (*.csv);;All (*)")
        if not path:
            return
        from .registration.bigwarp_io import load_landmarks
        try:
            self._moving_pts, self._fixed_pts = load_landmarks(path)
            n = len(self._moving_pts)
            self._load_status.setText(f"Landmarks: {n} active points")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_load_target(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open target image", "",
            "Images (*.tif *.tiff *.png *.jpg *.czi *.nd2);;All files (*)"
        )
        if not path:
            return
        from ._widget_rotation import _load_image_any, _collapse_to_2d
        try:
            self._target_image = _collapse_to_2d(_load_image_any(path))
            self._target_image_path = path
            if "target" not in self._viewer.layers:
                self._viewer.add_image(self._target_image, name="target")
            else:
                self._viewer.layers["target"].data = self._target_image
            self._load_status.setText(f"Target: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_build_tps(self) -> None:
        if self._moving_pts is None or self._fixed_pts is None:
            QMessageBox.warning(self, "No landmarks", "Load landmark CSV first.")
            return
        from .registration.tps import build_inverse_transform
        try:
            self._inverse_tps = build_inverse_transform(self._moving_pts, self._fixed_pts)
            self._warp_status.setText(f"Inverse TPS built ({len(self._moving_pts)} landmarks)")
        except Exception as e:
            QMessageBox.critical(self, "TPS error", str(e))

    def _on_warp_image(self) -> None:
        if self._inverse_tps is None:
            QMessageBox.warning(self, "No TPS", "Build inverse TPS first.")
            return
        if not hasattr(self, "_target_image"):
            QMessageBox.warning(self, "No target", "Load target image first.")
            return
        from .registration.tps import warp_image_to_moving
        try:
            warped = warp_image_to_moving(
                self._target_image,
                self._moving_pts,
                self._fixed_pts,
            )
            if "warped_target" in self._viewer.layers:
                self._viewer.layers["warped_target"].data = warped
            else:
                self._viewer.add_image(warped, name="warped_target", colormap="gray", opacity=0.6)
            self._warp_status.setText("Warped image added to viewer")
        except Exception as e:
            QMessageBox.critical(self, "Warp error", str(e))

    def _on_load_points_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open points CSV", "", "CSV (*.csv)")
        if not path:
            return
        import pandas as pd
        try:
            df = pd.read_csv(path)
            # Accept 'x','y' or 'X','Y' column names
            xcol = next((c for c in df.columns if c.lower() == "x"), None)
            ycol = next((c for c in df.columns if c.lower() == "y"), None)
            if xcol is None or ycol is None:
                QMessageBox.critical(self, "Error", "CSV must have 'x' and 'y' columns.")
                return
            pts = df[[xcol, ycol]].to_numpy(dtype=float)
            self._set_points(pts)
            self._pts_status.setText(f"{len(pts)} points loaded from CSV")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_click_mode_toggled(self, checked: bool) -> None:
        if checked:
            self._btn_click_pts.setText("Done adding points")
            if self._points_layer is None or self._points_layer not in self._viewer.layers:
                self._points_layer = self._viewer.add_points(
                    name="target_points", size=12, face_color="red"
                )
            self._points_layer.mode = "add"
        else:
            self._btn_click_pts.setText("Click points on target image")
            if self._points_layer is not None:
                pts_yx = self._points_layer.data
                if len(pts_yx) > 0:
                    # napari Points: (row, col) = (Y, X) -> convert to (X, Y)
                    pts_xy = pts_yx[:, ::-1]
                    self._target_points = pts_xy
                    self._pts_status.setText(f"{len(pts_xy)} points from clicks")

    def _on_map_points(self) -> None:
        pts = getattr(self, "_target_points", None)
        if pts is None or len(pts) == 0:
            QMessageBox.warning(self, "No points", "Load or click points first.")
            return
        if self._inverse_tps is None:
            QMessageBox.warning(self, "No TPS", "Build inverse TPS first.")
            return
        if self._bregma_calc is None:
            QMessageBox.warning(self, "No session", "Load session with bregma reference first.")
            return

        slice_index = self._session.get("z_index", 0)
        from .registration.point_mapper import map_points_full_pipeline
        try:
            self._mapped_results = map_points_full_pipeline(
                pts, self._inverse_tps, slice_index, self._bregma_calc
            )
            self._populate_table(self._mapped_results)
        except Exception as e:
            QMessageBox.critical(self, "Mapping error", str(e))

    def _on_export_csv(self) -> None:
        if self._mapped_results is None:
            QMessageBox.warning(self, "No results", "Map points first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save results CSV", "", "CSV (*.csv)")
        if not path:
            return
        tres = self._session.get("target_resolution", {})
        rx = tres.get("x_um_per_pixel")
        ry = tres.get("y_um_per_pixel")
        res = (rx, ry) if rx is not None and ry is not None else None
        from .io.export import export_points_table
        export_points_table(self._mapped_results, path, target_resolution=res)
        QMessageBox.information(self, "Exported", f"Saved to {path}")

    def _on_export_overlay(self) -> None:
        if not hasattr(self, "_target_image"):
            QMessageBox.warning(self, "No image", "Load target image first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save overlay TIFF", "", "TIFF (*.tif)")
        if not path:
            return
        # Find first atlas slice layer in viewer
        atlas_slice = None
        for layer in self._viewer.layers:
            if "slice" in layer.name.lower() and hasattr(layer, "data"):
                atlas_slice = layer.data
                break
        if atlas_slice is None:
            QMessageBox.warning(self, "No atlas slice", "No atlas slice layer found in viewer.")
            return
        from .io.export import export_overlay_image
        export_overlay_image(self._target_image, atlas_slice, path)
        QMessageBox.information(self, "Exported", f"Overlay saved to {path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_points(self, pts_xy: np.ndarray) -> None:
        """Set target points (X, Y) and display in viewer."""
        self._target_points = pts_xy
        pts_yx = pts_xy[:, ::-1]  # napari needs (row, col) = (Y, X)
        if self._points_layer is None or self._points_layer not in self._viewer.layers:
            self._points_layer = self._viewer.add_points(
                pts_yx, name="target_points", size=12, face_color="red"
            )
        else:
            self._points_layer.data = pts_yx

    def _build_bregma_calc(self) -> None:
        refs = self._session.get("bregma_references", [])
        if not refs:
            self._bregma_calc = None
            return
        from .coordinates.orientation import AtlasOrientation
        from .coordinates.bregma import BregmaCalculator, BregmaReference
        orientation = AtlasOrientation(
            self._session.get("orientation", "coronal")
        )
        voxel_size = self._session.get("voxel_size_um", [25, 25, 25])
        ref_data = refs[0]  # use first reference point
        ref = BregmaReference(
            ref_data["voxel_zyx"],
            ref_data["ap_mm"],
            ref_data["ml_mm"],
            ref_data["dv_mm"],
        )
        self._bregma_calc = BregmaCalculator(orientation, voxel_size, ref)

    def _populate_table(self, results: dict) -> None:
        n = len(results["target_px"])
        self._table.setRowCount(n)
        for i in range(n):
            vals = [
                f"pt_{i}",
                f"{results['target_px'][i,0]:.1f}",
                f"{results['target_px'][i,1]:.1f}",
                f"{results['atlas_slice_px'][i,0]:.1f}",
                f"{results['atlas_slice_px'][i,1]:.1f}",
                f"{results['apdv_mm'][i,0]:.3f}",
                f"{results['apdv_mm'][i,1]:.3f}",
                f"{results['apdv_mm'][i,2]:.3f}",
            ]
            for j, v in enumerate(vals):
                self._table.setItem(i, j, QTableWidgetItem(v))
