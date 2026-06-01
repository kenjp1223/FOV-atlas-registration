"""
Atlas Setup Widget — napari port of prism_alignment/gui.py
==========================================================
Replicates all functionality of gui.py inside a napari dock widget.
Original files are NOT modified. Heavy rotation (SimpleITK BSpline)
is called from reslice.py on export only; interactive preview uses
the fast oblique-sampling trick from gui.py.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import napari
import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image_any(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(str(path)).astype(np.float32)
    elif suffix == ".czi":
        import czifile
        return czifile.imread(str(path)).squeeze().astype(np.float32)
    elif suffix == ".nd2":
        import nd2
        return nd2.imread(str(path)).astype(np.float32)
    else:
        from PIL import Image
        return np.array(Image.open(str(path))).astype(np.float32)


def _collapse_to_2d(arr):
    """Return a 2-D float32 array from any image shape.

    Handles:
      (H, W)          — already 2D, pass through
      (H, W, C)       — channel-last (RGB/RGBA/multi-channel): luminance-weight to grayscale
      (T, H, W)       — time/z stack: take first frame
      (T, H, W, C)    — time + channel: first frame then grayscale
      (1, H, W, 1)    — squeeze unit axes first
    """
    arr = np.squeeze(arr)          # remove any size-1 axes
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        # Distinguish (H, W, C) from (T, H, W) by checking if last dim is small
        if arr.shape[-1] <= 4:    # channel-last: C = 1, 2, 3, or 4
            if arr.shape[-1] == 1:
                return arr[..., 0].astype(np.float32)
            # RGB/RGBA → luminance (ITU-R BT.601 for first 3 channels)
            weights = np.array([0.299, 0.587, 0.114, 0.0], dtype=np.float32)
            w = weights[:arr.shape[-1]].copy()
            w /= w.sum() if w.sum() > 0 else 1.0
            return (arr.astype(np.float32) * w).sum(axis=-1)
        else:                     # (T, H, W): take first frame
            return arr[0].astype(np.float32)
    # 4-D or higher after squeeze: drop leading axes until 2D
    while arr.ndim > 2:
        arr = arr[0]
    return arr.astype(np.float32)


def _collapse_to_3d(arr):
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    return arr


def _oblique_slice(atlas_arr, spacing, z_idx, rx, ry, rz, order=1):
    """
    Fast oblique slice — same algorithm as gui.py _compute_slice().
    Samples the original atlas volume along a rotated plane.
    order=1 for preview, order=3 for export.
    """
    nz, ny, nx = atlas_arr.shape
    sx, sy, sz = spacing

    if rx == 0.0 and ry == 0.0 and rz == 0.0:
        idx = int(np.clip(z_idx, 0, nz - 1))
        return atlas_arr[idx].copy()

    cx = nx * sx / 2.0
    cy = ny * sy / 2.0
    cz = nz * sz / 2.0

    R = Rotation.from_euler("XYZ", [rx, ry, rz], degrees=True)

    u = np.linspace(0.0, nx * sx, nx, endpoint=False) - cx
    v = np.linspace(0.0, ny * sy, ny, endpoint=False) - cy
    w = float(z_idx) * sz - cz

    uu, vv = np.meshgrid(u, v)
    pts_rot = np.stack([uu.ravel(), vv.ravel(), np.full(uu.size, w)], axis=1)
    pts_orig = R.inv().apply(pts_rot)

    ix = (pts_orig[:, 0] + cx) / sx
    iy = (pts_orig[:, 1] + cy) / sy
    iz = (pts_orig[:, 2] + cz) / sz

    out = map_coordinates(
        atlas_arr.astype(np.float64), [iz, iy, ix],
        order=order, mode="constant", cval=0.0,
    )
    return out.reshape(ny, nx).astype(np.float32)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class AtlasSetupWidget(QWidget):
    """Atlas rotation + slice setup widget for napari. Replaces gui.py."""

    SPACING_PRESETS = {
        "10x10x10": [10.0, 10.0, 10.0],
        "20x20x50": [20.0, 20.0, 50.0],
        "25x25x25": [25.0, 25.0, 25.0],
        "5x5x10":   [5.0,  5.0,  10.0],
    }

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self._viewer = napari_viewer

        self._atlas_channels = {}   # name -> (Z,Y,X) float32
        self._atlas_path = None
        self._spacing = [20.0, 20.0, 50.0]   # X, Y, Z µm
        self._last_json_path = None

        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout()
        root.setAlignment(Qt.AlignTop)
        self.setLayout(root)
        root.addWidget(self._build_atlas_group())
        root.addWidget(self._build_rotation_group())
        root.addWidget(self._build_display_group())
        root.addWidget(self._build_export_group())
        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

    def _build_atlas_group(self):
        box = QGroupBox("Atlas")
        layout = QVBoxLayout(box)

        row1 = QHBoxLayout()
        btn_load = QPushButton("Load atlas TIFF...")
        btn_load.clicked.connect(self._on_load_atlas)
        btn_add = QPushButton("Add channel...")
        btn_add.clicked.connect(self._on_add_channel)
        row1.addWidget(btn_load)
        row1.addWidget(btn_add)
        layout.addLayout(row1)

        sp_row = QHBoxLayout()
        sp_row.addWidget(QLabel("Spacing X/Y/Z (um):"))
        self._sp_x = self._dspin(0.001, 1e5, 20.0)
        self._sp_y = self._dspin(0.001, 1e5, 20.0)
        self._sp_z = self._dspin(0.001, 1e5, 50.0)
        for w in (self._sp_x, self._sp_y, self._sp_z):
            sp_row.addWidget(w)
        layout.addLayout(sp_row)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets:"))
        for label in self.SPACING_PRESETS:
            b = QPushButton(label)
            b.setMaximumWidth(85)
            b.clicked.connect(lambda _, l=label: self._apply_preset(l))
            preset_row.addWidget(b)
        layout.addLayout(preset_row)

        self._atlas_info = QLabel("No atlas loaded")
        self._atlas_info.setWordWrap(True)
        layout.addWidget(self._atlas_info)
        return box

    def _build_rotation_group(self):
        box = QGroupBox("Rotation & Slice")
        form = QFormLayout(box)
        self._rx_spin, rx_w = self._angle_row()
        self._ry_spin, ry_w = self._angle_row()
        self._rz_spin, rz_w = self._angle_row()
        form.addRow("Rx deg (pitch):", rx_w)
        form.addRow("Ry deg (yaw):",   ry_w)
        form.addRow("Rz deg (roll):",  rz_w)

        self._z_spin = QSpinBox()
        self._z_spin.setRange(0, 9999)
        self._z_spin.valueChanged.connect(self._schedule_preview)
        self._z_phys_label = QLabel("0.0 um")
        z_w = QWidget()
        z_row = QHBoxLayout(z_w)
        z_row.setContentsMargins(0, 0, 0, 0)
        z_row.addWidget(self._z_spin)
        z_row.addWidget(self._z_phys_label)
        form.addRow("Z slice:", z_w)

        btn_reset = QPushButton("Reset all")
        btn_reset.clicked.connect(self._reset_controls)
        form.addRow(btn_reset)
        return box

    def _build_display_group(self):
        box = QGroupBox("Display")
        layout = QHBoxLayout(box)
        self._flip_h = QCheckBox("Flip H")
        self._flip_v = QCheckBox("Flip V")
        self._flip_h.toggled.connect(self._schedule_preview)
        self._flip_v.toggled.connect(self._schedule_preview)
        layout.addWidget(self._flip_h)
        layout.addWidget(self._flip_v)
        return box

    def _build_bregma_group(self):
        box = QGroupBox("Bregma Reference")
        layout = QVBoxLayout(box)

        self._btn_refpoint = QPushButton("Enter reference point mode")
        self._btn_refpoint.setCheckable(True)
        self._btn_refpoint.toggled.connect(self._on_refpoint_mode_toggled)
        layout.addWidget(self._btn_refpoint)

        hint = QLabel(
            "Click on the atlas slice layer to place a reference point,\n"
            "then enter its known position relative to bregma and confirm."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        self._bregma_ap = self._dspin(-200, 200, 0.0, dec=3, suffix=" mm")
        self._bregma_ml = self._dspin(-200, 200, 0.0, dec=3, suffix=" mm")
        self._bregma_dv = self._dspin(-200, 200, 0.0, dec=3, suffix=" mm")
        form.addRow("AP from bregma (+ant):", self._bregma_ap)
        form.addRow("ML from bregma (+right):", self._bregma_ml)
        form.addRow("DV from bregma (+dorsal):", self._bregma_dv)
        layout.addLayout(form)

        self._btn_confirm_ref = QPushButton("Confirm reference point")
        self._btn_confirm_ref.setEnabled(False)
        self._btn_confirm_ref.clicked.connect(self._on_confirm_refpoint)
        layout.addWidget(self._btn_confirm_ref)

        self._bregma_status = QLabel("No reference points set")
        layout.addWidget(self._bregma_status)
        return box

    def _build_export_group(self):
        box = QGroupBox("Export")
        layout = QVBoxLayout(box)

        ori_row = QHBoxLayout()
        ori_row.addWidget(QLabel("Atlas orientation:"))
        self._orientation_combo = QComboBox()
        self._orientation_combo.addItems(["coronal", "sagittal", "axial"])
        ori_row.addWidget(self._orientation_combo)
        layout.addLayout(ori_row)

        btn_export = QPushButton("Export aligned slice + settings...")
        btn_export.clicked.connect(self._on_export)
        layout.addWidget(btn_export)

        btn_imagej = QPushButton("Open in ImageJ (BigWarp)...")
        btn_imagej.clicked.connect(self._on_open_imagej)
        layout.addWidget(btn_imagej)

        self._fiji_path = QLineEdit()
        self._fiji_path.setPlaceholderText("Fiji path (blank = auto-detect)")
        layout.addWidget(self._fiji_path)
        return box

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dspin(self, lo, hi, val, dec=2, suffix=""):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        w.setDecimals(dec)
        if suffix:
            w.setSuffix(suffix)
        return w

    def _angle_row(self):
        spin = QDoubleSpinBox()
        spin.setRange(-180.0, 180.0)
        spin.setValue(0.0)
        spin.setSingleStep(0.5)
        spin.setDecimals(1)
        spin.setSuffix(" deg")
        spin.valueChanged.connect(self._schedule_preview)
        dec_btn = QPushButton("<")
        inc_btn = QPushButton(">")
        dec_btn.setMaximumWidth(26)
        inc_btn.setMaximumWidth(26)
        dec_btn.clicked.connect(lambda: spin.setValue(round(max(-180, spin.value()-1.0), 1)))
        inc_btn.clicked.connect(lambda: spin.setValue(round(min(180,  spin.value()+1.0), 1)))
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(dec_btn)
        row.addWidget(spin)
        row.addWidget(inc_btn)
        return spin, w

    def _status(self, msg):
        self._status_label.setText(msg)

    # ------------------------------------------------------------------
    # Viewer events
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _on_load_atlas(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load atlas TIFF", "", "TIFF (*.tif *.tiff);;All files (*)"
        )
        if not path:
            return
        try:
            self._status("Loading atlas...")
            import tifffile
            arr = tifffile.imread(path).astype(np.float32)
            arr = _collapse_to_3d(arr)
            name = Path(path).stem
            self._atlas_channels = {name: arr}
            self._atlas_path = Path(path)
            self._on_atlas_loaded(arr)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_add_channel(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Add atlas channel", "",
            "TIFF (*.tif *.tiff);;NIfTI (*.nii *.nii.gz);;All files (*)"
        )
        if not path:
            return
        try:
            arr = _load_image_any(path)
            arr = _collapse_to_3d(arr)
            name = Path(path).stem
            self._atlas_channels[name] = arr
            self._status(f"Added channel: {name}")
            self._update_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_atlas_loaded(self, arr):
        nz, ny, nx = arr.shape
        sx, sy, sz = self._spacing
        self._z_spin.setMaximum(nz - 1)
        self._z_spin.setValue(nz // 2)
        self._atlas_info.setText(
            f"{self._atlas_path.name}  {nx}x{ny}x{nz} vox  "
            f"{sx:.0f}x{sy:.0f}x{sz:.0f} um  "
            f"FOV {nx*sx/1000:.2f}x{ny*sy/1000:.2f}x{nz*sz/1000:.2f} mm"
        )
        self._update_preview()
        self._status("Atlas loaded.")

    def _get_target_resolution(self) -> dict:
        """Resolution is stored in the viewer's shared metadata dict under 'target_resolution'."""
        meta = getattr(self._viewer, "_atlas_reg_meta", {})
        return {
            "x_um_per_pixel": meta.get("x_um_per_pixel", 1.0),
            "y_um_per_pixel": meta.get("y_um_per_pixel", 1.0),
        }

    # ------------------------------------------------------------------
    # Spacing
    # ------------------------------------------------------------------

    def _apply_preset(self, label):
        sx, sy, sz = self.SPACING_PRESETS[label]
        self._sp_x.setValue(sx)
        self._sp_y.setValue(sy)
        self._sp_z.setValue(sz)
        self._spacing = [sx, sy, sz]
        self._schedule_preview()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _schedule_preview(self, *_):
        self._preview_timer.start(60)

    def _update_preview(self):
        if not self._atlas_channels:
            return
        self._spacing = [self._sp_x.value(), self._sp_y.value(), self._sp_z.value()]
        z   = self._z_spin.value()
        rx  = self._rx_spin.value()
        ry  = self._ry_spin.value()
        rz  = self._rz_spin.value()
        sz  = self._spacing[2]
        self._z_phys_label.setText(f"{z * sz:.1f} um")

        for name, arr in self._atlas_channels.items():
            sl = _oblique_slice(arr, self._spacing, z, rx, ry, rz, order=1)
            if self._flip_h.isChecked():
                sl = np.fliplr(sl)
            if self._flip_v.isChecked():
                sl = np.flipud(sl)
            layer_name = f"atlas_{name}"
            if layer_name in self._viewer.layers:
                self._viewer.layers[layer_name].data = sl
            else:
                self._viewer.add_image(sl, name=layer_name, colormap="gray",
                                       blending="additive")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export(self):
        if not self._atlas_channels:
            QMessageBox.warning(self, "No atlas", "Load an atlas first.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select export folder")
        if not out_dir:
            return

        out = Path(out_dir)
        stem = self._atlas_path.stem if self._atlas_path else "atlas"
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        pfx = f"{stem}_{ts}"

        try:
            self._status("Exporting (high-quality BSpline slice)...")
            rx = self._rx_spin.value()
            ry = self._ry_spin.value()
            rz = self._rz_spin.value()
            z_idx = self._z_spin.value()
            sx, sy, sz = self._spacing
            flip_h = self._flip_h.isChecked()
            flip_v = self._flip_v.isChecked()

            import tifffile
            channel_paths = {}
            primary_arr = list(self._atlas_channels.values())[0]
            for ch_name, ch_arr in self._atlas_channels.items():
                sl = _oblique_slice(ch_arr, self._spacing, z_idx, rx, ry, rz, order=3)
                if flip_h: sl = np.fliplr(sl)
                if flip_v: sl = np.flipud(sl)
                ch_path = out / f"{pfx}_slice_{ch_name}.tif"
                tifffile.imwrite(str(ch_path), sl.astype(np.float32))
                channel_paths[ch_name] = str(ch_path)

            from scipy.spatial.transform import Rotation as _R
            R_mat = _R.from_euler("XYZ", [rx, ry, rz], degrees=True).as_matrix()
            nz, ny, nx = primary_arr.shape
            cx = nx*sx/2.0; cy = ny*sy/2.0; cz = nz*sz/2.0
            centre = np.array([cx, cy, cz])
            affine = np.eye(4)
            affine[:3, :3] = R_mat
            affine[:3, 3]  = centre - R_mat @ centre
            np.save(str(out / f"{pfx}_affine.npy"), affine)

            target_meta = self._get_target_resolution()
            target_path = (
                self._viewer.layers["target"].metadata.get("path")
                if "target" in self._viewer.layers else None
            )

            settings = {
                "timestamp":          datetime.now().isoformat(),
                "target_image":       target_path,
                "reference_atlas":    str(self._atlas_path)  if self._atlas_path  else None,
                "atlas_channels":     channel_paths,
                "atlas_shape_zyx":    list(primary_arr.shape),
                "atlas_spacing_um":   {"x": sx, "y": sy, "z": sz},
                "atlas_fov_mm":       {"x": nx*sx/1000, "y": ny*sy/1000, "z": nz*sz/1000},
                "rotation_degrees":   {"rx_pitch": rx, "ry_yaw": ry, "rz_roll": rz},
                "rotation_euler_convention": "intrinsic XYZ (pitch->yaw->roll)",
                "rotation_matrix_3x3": R_mat.tolist(),
                "affine_matrix_4x4_um": affine.tolist(),
                "z_index":            z_idx,
                "z_physical_um":      z_idx * sz,
                "flip_horizontal":    flip_h,
                "flip_vertical":      flip_v,
                "orientation":        self._orientation_combo.currentText(),
                "target_resolution":  target_meta,
                "output_files": {
                    "settings_json":     str(out / f"{pfx}_settings.json"),
                    "affine_matrix_npy": str(out / f"{pfx}_affine.npy"),
                },
            }

            json_path = out / f"{pfx}_settings.json"
            with open(json_path, "w") as f:
                json.dump(settings, f, indent=2)

            self._last_json_path = json_path
            self._status(f"Exported to {out}")
            QMessageBox.information(
                self, "Export complete",
                f"Saved to: {out}\n\n"
                + "\n".join(f"  {k}: {Path(v).name}"
                            for k, v in channel_paths.items())
                + f"\n  settings: {json_path.name}"
                + "\n\nOpen in ImageJ and run BigWarp."
            )
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))
            raise

    def _on_open_imagej(self):
        if self._last_json_path is None:
            QMessageBox.warning(self, "No export", "Export first.")
            return
        # Launcher lives at prism_alignment/imagej/open_in_imagej.py
        # Plugin is at prism_alignment/napari-atlas-registration/src/...
        launcher = Path(__file__).parents[4] / "imagej" / "open_in_imagej.py"
        if not launcher.exists():
            QMessageBox.critical(
                self, "Launcher not found",
                f"Expected:\n{launcher}"
            )
            return
        cmd = [sys.executable, str(launcher), "--json", str(self._last_json_path)]
        fiji = self._fiji_path.text().strip()
        if fiji:
            cmd += ["--fiji", fiji]
        try:
            import platform as _pl
            if _pl.system() == "Windows":
                subprocess.Popen(cmd, creationflags=subprocess.DETACHED_PROCESS)
            else:
                subprocess.Popen(cmd, start_new_session=True)
            self._status("ImageJ opened -- run Plugins > BigWarp > Big Warp")
        except Exception as e:
            QMessageBox.critical(self, "Launch error", str(e))

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _reset_controls(self):
        for s in (self._rx_spin, self._ry_spin, self._rz_spin):
            s.setValue(0.0)
        self._flip_h.setChecked(False)
        self._flip_v.setChecked(False)
        if self._atlas_channels:
            nz = list(self._atlas_channels.values())[0].shape[0]
            self._z_spin.setValue(nz // 2)
        self._status("Controls reset.")
