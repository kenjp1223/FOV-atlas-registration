"""
Target Image Widget — separate dock panel for loading the histology target image.

Loads the target image into the napari viewer as a layer named "target".
Resolution (um/pixel) is stored on the layer's metadata so the Atlas Setup
and Inverse Warp widgets can read it without needing a direct reference.
"""

import napari
import numpy as np
from pathlib import Path
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox,
)

from ._widget_rotation import _load_image_any, _collapse_to_2d


class TargetImageWidget(QWidget):
    """Dock widget for loading the target histology image and setting resolution."""

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = napari_viewer
        self._target_path = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        # Load button
        btn = QPushButton("Load target image…")
        btn.clicked.connect(self._on_load)
        layout.addWidget(btn)

        # Resolution
        res_group = QGroupBox("Resolution (µm / pixel)")
        form = QFormLayout(res_group)
        self._res_x = self._dspin(0.001, 1e5, 1.0)
        self._res_y = self._dspin(0.001, 1e5, 1.0)
        self._res_x.valueChanged.connect(self._on_res_changed)
        self._res_y.valueChanged.connect(self._on_res_changed)
        form.addRow("X (um/px):", self._res_x)
        form.addRow("Y (um/px):", self._res_y)
        layout.addWidget(res_group)

        # Second window button
        self._btn_second = QPushButton("Open in second window")
        self._btn_second.setEnabled(False)
        self._btn_second.clicked.connect(self._on_open_second_window)
        layout.addWidget(self._btn_second)

        # Status
        self._info = QLabel("No target loaded")
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

    def _dspin(self, lo, hi, val, dec=4):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        w.setDecimals(dec)
        return w

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load target image", "",
            "Images (*.tif *.tiff *.png *.jpg *.czi *.nd2);;All files (*)"
        )
        if not path:
            return
        try:
            arr = _collapse_to_2d(_load_image_any(path))
            # Downsample if very large (>200 MB)
            if arr.nbytes > 200 * 1024 * 1024:
                f = int(np.ceil((arr.nbytes / (200 * 1024 * 1024)) ** 0.5))
                arr = arr[::f, ::f]
            self._target_path = Path(path)
            self._target_arr = arr
            h, w = arr.shape
            self._btn_second.setEnabled(True)
            self._on_res_changed()
            self._open_second_window(arr, title=self._target_path.name)
            self._info.setText(
                f"{self._target_path.name}\n"
                f"{w} × {h} px  |  "
                f"X={self._res_x.value():.4f} µm/px  "
                f"Y={self._res_y.value():.4f} µm/px"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error loading target", str(e))

    def _open_second_window(self, arr, title: str) -> None:
        """Open image in an independent second napari viewer window."""
        # Close any existing second viewer first
        if hasattr(self, "_second_viewer"):
            try:
                self._second_viewer.close()
            except Exception:
                pass
        v2 = napari.Viewer(title=f"Reference — {title}")
        v2.add_image(arr, name=title, colormap="gray")
        v2.reset_view()
        self._second_viewer = v2  # keep reference to prevent GC

    def _on_open_second_window(self) -> None:
        """Manual re-open button — re-opens the second window if closed."""
        if not hasattr(self, "_target_arr"):
            return
        title = self._target_path.name if self._target_path else "Target"
        self._open_second_window(self._target_arr, title)

    def _on_res_changed(self) -> None:
        # Store resolution in viewer shared metadata so other widgets can read it
        if not hasattr(self._viewer, "_atlas_reg_meta"):
            self._viewer._atlas_reg_meta = {}
        self._viewer._atlas_reg_meta.update({
            "x_um_per_pixel": self._res_x.value(),
            "y_um_per_pixel": self._res_y.value(),
            "target_path": str(self._target_path) if self._target_path else None,
        })
        if self._target_path and hasattr(self, "_target_arr"):
            h, w = self._target_arr.shape[:2]
            self._info.setText(
                f"{self._target_path.name}\n"
                f"{w} × {h} px  |  "
                f"X={self._res_x.value():.4f} µm/px  "
                f"Y={self._res_y.value():.4f} µm/px"
            )
