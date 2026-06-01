"""
Step 1 — FOV → Section Alignment Widget
========================================
Rigid registration (translate + rotate + scale) of a small FOV image onto the
full brain section image.

Workflow
--------
1. Load FOV image  → displayed in main napari viewer
2. Load cell CSV   → displayed as Points on the FOV
3. Load section    → opens in an independent second napari window
4. Add landmark pairs (sequential, guided):
     Click "Add pair" → click FOV window → click Section window → pair saved
5. Compute rigid transform (least-squares similarity: translate + rotate + scale)
6. Apply transform → cell dots appear on section in second window
7. Export: transformed cell CSV + transform JSON

NOTE on napari view operations:
  Do NOT use napari's built-in transpose/flip toolbar buttons while placing
  landmarks — those are camera-only operations and will misalign coordinates.
  Use the Flip H / Flip V controls in the Atlas Setup widget instead (those
  ARE recorded in the session JSON).
"""

import json
import numpy as np
import napari
from pathlib import Path
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QVBoxLayout, QWidget,
)

from ._widget_rotation import _load_image_any, _collapse_to_2d


# ---------------------------------------------------------------------------
# Rigid transform math
# ---------------------------------------------------------------------------

def _compute_rigid_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (translate + rotate + uniform scale).

    src : (N, 2)  FOV pixel coords (x, y)
    dst : (N, 2)  Section pixel coords (x, y)
    Returns M : (2, 3) affine  [R*s | t]
    """
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    src_n = src - src_c
    dst_n = dst - dst_c

    src_scale = np.sqrt((src_n ** 2).sum(axis=1).mean())
    dst_scale = np.sqrt((dst_n ** 2).sum(axis=1).mean())
    scale = dst_scale / (src_scale + 1e-12)

    src_n /= (src_scale + 1e-12)
    dst_n /= (dst_scale + 1e-12)

    H = src_n.T @ dst_n
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_c - scale * (R @ src_c)
    M = np.zeros((2, 3))
    M[:2, :2] = scale * R
    M[:2, 2] = t
    return M


def _apply_transform(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(pts)
    return (M[:, :2] @ pts.T + M[:, 2:3]).T


def _transform_params(M: np.ndarray) -> dict:
    scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    angle_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return {
        "scale": scale,
        "rotation_deg": angle_deg,
        "translation_x": float(M[0, 2]),
        "translation_y": float(M[1, 2]),
        "matrix_2x3": M.tolist(),
    }


# ---------------------------------------------------------------------------
# Landmark pair state
# ---------------------------------------------------------------------------

# Pairing state machine
_IDLE       = "idle"
_WAIT_FOV   = "waiting_fov"
_WAIT_SEC   = "waiting_section"


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class FOVAlignmentWidget(QWidget):
    """Step 1 — FOV → Section rigid alignment."""

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = napari_viewer

        self._fov_path     = None
        self._section_path = None
        self._section_arr  = None
        self._cell_pts_fov = None
        self._cell_df      = None
        self._stat_raw     = None   # full stat array (all ROIs, unfiltered)
        self._cell_stat    = None   # suite2p stat array (filtered cells)
        self._iscell       = None   # raw iscell array (n_rois, 2)

        # FOV display orientation (never touches stored coordinates)
        self._fov_arr_orig     = None   # raw loaded FOV array, never modified
        self._fov_display_k    = 0      # np.rot90 k: 0=0°, 1=90°CCW, 2=180°, 3=270°CCW
        self._fov_display_flip_h = False
        self._fov_display_flip_v = False
        self._transform_M  = None
        self._second_viewer = None

        # Landmark pair storage: list of {"fov": [x,y], "sec": [x,y]}
        self._pairs = []

        # Points layers (display only — rebuilt from _pairs)
        self._fov_lm_layer = None
        self._sec_lm_layer = None

        # State machine
        self._pair_state = _IDLE
        self._pending_fov_xy = None  # x,y of the clicked FOV point

        # Debounce timer to avoid double-firing on data events
        self._click_timer = QTimer()
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(50)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)
        layout.addWidget(self._build_load_group())
        layout.addWidget(self._build_display_group())
        layout.addWidget(self._build_landmark_group())
        layout.addWidget(self._build_transform_group())
        layout.addWidget(self._build_export_group())
        self._status = QLabel("Ready")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    def _build_load_group(self) -> QGroupBox:
        box = QGroupBox("Load")
        layout = QVBoxLayout(box)

        # ── FOV source ────────────────────────────────────────────────────
        from qtpy.QtWidgets import QButtonGroup, QRadioButton
        fov_src_label = QLabel("FOV image source:")
        fov_src_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(fov_src_label)

        self._radio_fov_image  = QRadioButton("Image file  (tif / png / jpg / czi / nd2)")
        self._radio_fov_suite2p = QRadioButton("suite2p .npy  (reg_outputs.npy  or  ops.npy  → meanImg)")
        self._radio_fov_image.setChecked(True)

        self._fov_radio_group = QButtonGroup(self)
        self._fov_radio_group.addButton(self._radio_fov_image,   0)
        self._fov_radio_group.addButton(self._radio_fov_suite2p, 1)
        self._fov_radio_group.buttonClicked.connect(self._on_fov_source_toggled)

        layout.addWidget(self._radio_fov_image)
        layout.addWidget(self._radio_fov_suite2p)

        self._btn_fov_image = QPushButton("Load FOV image…")
        self._btn_fov_image.clicked.connect(self._on_load_fov)
        layout.addWidget(self._btn_fov_image)

        self._btn_fov_suite2p = QPushButton("Load reg_outputs.npy / ops.npy…")
        self._btn_fov_suite2p.clicked.connect(self._on_load_suite2p_npy)
        self._btn_fov_suite2p.setVisible(False)
        layout.addWidget(self._btn_fov_suite2p)
        # ─────────────────────────────────────────────────────────────────

        # ── suite2p / CSV toggle ──────────────────────────────────────────
        self._chk_suite2p = QCheckBox("Load cells from suite2p stat.npy")
        self._chk_suite2p.setChecked(True)
        self._chk_suite2p.toggled.connect(self._on_suite2p_toggled)
        layout.addWidget(self._chk_suite2p)

        # suite2p buttons (shown when checkbox is ON)
        self._btn_statnpy = QPushButton("Load stat.npy…")
        self._btn_statnpy.clicked.connect(self._on_load_statnpy)
        layout.addWidget(self._btn_statnpy)

        self._btn_iscell = QPushButton("Load iscell.npy…")
        self._btn_iscell.clicked.connect(self._on_load_iscell)
        layout.addWidget(self._btn_iscell)

        self._iscell_info = QLabel("iscell.npy: not loaded")
        self._iscell_info.setWordWrap(True)
        self._iscell_info.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self._iscell_info)

        # CSV button (shown when checkbox is OFF)
        self._btn_csv = QPushButton("Load cell CSV (x, y)…")
        self._btn_csv.clicked.connect(self._on_load_csv)
        self._btn_csv.setVisible(False)
        layout.addWidget(self._btn_csv)
        # ─────────────────────────────────────────────────────────────────

        btn_sec = QPushButton("Load section image…")
        btn_sec.clicked.connect(self._on_load_section)
        layout.addWidget(btn_sec)

        self._load_info = QLabel("Nothing loaded")
        self._load_info.setWordWrap(True)
        layout.addWidget(self._load_info)
        return box

    def _build_display_group(self) -> QGroupBox:
        box = QGroupBox("FOV Display Orientation")
        layout = QVBoxLayout(box)

        note = QLabel(
            "Rotate/flip the FOV display to match the section for easier landmarking.\n"
            "All coordinates are always stored in the original image space."
        )
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(note)

        row1 = QHBoxLayout()
        btn_ccw = QPushButton("↺ Rotate CCW")
        btn_cw  = QPushButton("↻ Rotate CW")
        btn_ccw.clicked.connect(self._on_rotate_ccw)
        btn_cw.clicked.connect(self._on_rotate_cw)
        row1.addWidget(btn_ccw)
        row1.addWidget(btn_cw)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        btn_fh = QPushButton("⇔ Flip H")
        btn_fv = QPushButton("⇕ Flip V")
        btn_fh.clicked.connect(self._on_flip_h)
        btn_fv.clicked.connect(self._on_flip_v)
        row2.addWidget(btn_fh)
        row2.addWidget(btn_fv)
        layout.addLayout(row2)

        btn_reset = QPushButton("Reset orientation")
        btn_reset.clicked.connect(self._on_reset_display)
        layout.addWidget(btn_reset)

        self._orient_info = QLabel("Orientation: 0° | No flip")
        self._orient_info.setStyleSheet("font-size: 10px; color: #555;")
        layout.addWidget(self._orient_info)
        return box

    # ------------------------------------------------------------------
    # Display orientation controls
    # ------------------------------------------------------------------

    def _on_rotate_cw(self)  -> None:
        self._fov_display_k = (self._fov_display_k - 1) % 4
        self._refresh_fov_display()

    def _on_rotate_ccw(self) -> None:
        self._fov_display_k = (self._fov_display_k + 1) % 4
        self._refresh_fov_display()

    def _on_flip_h(self) -> None:
        self._fov_display_flip_h = not self._fov_display_flip_h
        self._refresh_fov_display()

    def _on_flip_v(self) -> None:
        self._fov_display_flip_v = not self._fov_display_flip_v
        self._refresh_fov_display()

    def _on_reset_display(self) -> None:
        self._fov_display_k     = 0
        self._fov_display_flip_h = False
        self._fov_display_flip_v = False
        self._refresh_fov_display()

    def _refresh_fov_display(self) -> None:
        """Re-render FOV layer, cell points, and masks with current display transform."""
        if self._fov_arr_orig is None:
            return

        # Update orientation label
        rot_label = {0: "0°", 1: "90° CCW", 2: "180°", 3: "270° CCW"}
        flips = []
        if self._fov_display_flip_h: flips.append("Flip H")
        if self._fov_display_flip_v: flips.append("Flip V")
        self._orient_info.setText(
            f"Orientation: {rot_label[self._fov_display_k]} | "
            f"{' | '.join(flips) if flips else 'No flip'}"
        )

        arr = self._apply_display_transform(self._fov_arr_orig)
        if "fov" in self._viewer.layers:
            self._viewer.layers["fov"].data = arr
        else:
            self._viewer.add_image(arr, name="fov", colormap="gray")

        # Cell centroids — kept in original space; transform for display
        if self._cell_pts_fov is not None:
            yx_orig = self._cell_pts_fov[:, ::-1]          # xy→yx original
            yx_disp = self._transform_yx_for_display(yx_orig)
            if "cells_fov" in self._viewer.layers:
                self._viewer.layers["cells_fov"].data = yx_disp

        # Cell masks
        if self._cell_stat is not None:
            self._show_stat_masks(self._cell_stat)

        # Landmarks (stored in original space; transform for display)
        self._refresh_lm_layers()

    # ------------------------------------------------------------------
    # Display transform math
    # ------------------------------------------------------------------

    def _apply_display_transform(self, arr: np.ndarray) -> np.ndarray:
        """Apply current rotation + flips to an image array."""
        arr = np.rot90(arr, self._fov_display_k)
        if self._fov_display_flip_h:
            arr = np.fliplr(arr)
        if self._fov_display_flip_v:
            arr = np.flipud(arr)
        return arr

    def _transform_yx_for_display(self, yx: np.ndarray) -> np.ndarray:
        """Map (N,2) yx coordinates from original FOV space → display space."""
        if self._fov_arr_orig is None or len(yx) == 0:
            return yx
        H, W = self._fov_arr_orig.shape[:2]
        y = yx[:, 0].astype(float).copy()
        x = yx[:, 1].astype(float).copy()

        k = self._fov_display_k
        if k == 1:          # 90° CCW: y'=W-1-x, x'=y  → shape (W,H)
            y, x = W - 1 - x, y.copy()
            H, W = W, H
        elif k == 2:        # 180°:   y'=H-1-y, x'=W-1-x
            y, x = H - 1 - y, W - 1 - x
        elif k == 3:        # 90° CW: y'=x, x'=H-1-y   → shape (W,H)
            y, x = x.copy(), H - 1 - y
            H, W = W, H

        if self._fov_display_flip_h:
            x = W - 1 - x
        if self._fov_display_flip_v:
            y = H - 1 - y

        return np.stack([y, x], axis=1)

    def _inverse_transform_yx(self, y: float, x: float) -> tuple[float, float]:
        """Map a single (y, x) click from display space → original FOV space.

        Uses numpy's own inverse operations on an indicator pixel — correct
        by construction regardless of rotation/flip combination.
        """
        if self._fov_arr_orig is None:
            return y, x

        disp = self._apply_display_transform(
            np.zeros(self._fov_arr_orig.shape[:2], dtype=np.uint8))
        disp_H, disp_W = disp.shape

        yi = int(np.clip(round(y), 0, disp_H - 1))
        xi = int(np.clip(round(x), 0, disp_W - 1))

        # Plant a marker and apply the exact inverse of the display transform
        tmp = np.zeros((disp_H, disp_W), dtype=np.uint8)
        tmp[yi, xi] = 1

        # Inverse order: undo flips first, then undo rotation
        if self._fov_display_flip_v:
            tmp = np.flipud(tmp)
        if self._fov_display_flip_h:
            tmp = np.fliplr(tmp)
        if self._fov_display_k != 0:
            tmp = np.rot90(tmp, 4 - self._fov_display_k)

        pos = np.argwhere(tmp == 1)
        if len(pos) == 0:
            return y, x
        return float(pos[0][0]), float(pos[0][1])

    def _build_landmark_group(self) -> QGroupBox:
        box = QGroupBox("Landmark pairs")
        layout = QVBoxLayout(box)

        btn_load_lm = QPushButton("Load landmarks from CSV…")
        btn_load_lm.clicked.connect(self._on_load_landmarks_csv)
        layout.addWidget(btn_load_lm)

        # Guided status label
        self._pair_status = QLabel("Load FOV and Section images to begin.")
        self._pair_status.setWordWrap(True)
        self._pair_status.setStyleSheet("font-weight: bold; color: #555;")
        layout.addWidget(self._pair_status)

        # Add / Cancel buttons
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

        # Pair table: # | FOV x | FOV y | Section x | Section y | Delete
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(["#", "FOV x", "FOV y", "Sec x", "Sec y", ""])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Fixed)
        self._table.setColumnWidth(5, 54)
        self._table.setMinimumHeight(140)
        self._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(self._table)

        return box

    def _build_transform_group(self) -> QGroupBox:
        box = QGroupBox("Transform")
        layout = QVBoxLayout(box)

        btn_compute = QPushButton("Compute rigid transform")
        btn_compute.clicked.connect(self._on_compute_transform)
        layout.addWidget(btn_compute)

        self._transform_info = QLabel("No transform computed")
        self._transform_info.setWordWrap(True)
        layout.addWidget(self._transform_info)

        btn_apply = QPushButton("Apply → show cells on section")
        btn_apply.clicked.connect(self._on_apply_transform)
        layout.addWidget(btn_apply)
        return box

    def _build_export_group(self) -> QGroupBox:
        box = QGroupBox("Save results")
        layout = QVBoxLayout(box)

        # Primary save — writes all three files at once
        btn_save = QPushButton("Save session (landmarks + transform + cells)…")
        btn_save.clicked.connect(self._on_save_session)
        layout.addWidget(btn_save)

        info = QLabel(
            "Saves:\n"
            "  • landmarks.csv      — reload pairs next time\n"
            "  • transform.json     — matrix, scale, rotation\n"
            "  • cells_section.csv  — input for Step 2"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(info)

        return box

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _on_fov_source_toggled(self, btn=None) -> None:
        suite2p_mode = self._radio_fov_suite2p.isChecked()
        self._btn_fov_image.setVisible(not suite2p_mode)
        self._btn_fov_suite2p.setVisible(suite2p_mode)

    def _on_load_suite2p_npy(self) -> None:
        """Load FOV image from a suite2p .npy dict (reg_outputs.npy or ops.npy).

        Scans for image-like keys and lets the user pick which one to display.
        Candidate keys in priority order:
          meanImg_chan2  — anatomical/structural channel (best for atlas alignment)
          meanImg        — functional channel mean
          meanImgE       — contrast-enhanced mean
          refImg         — registration reference frame
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Load suite2p .npy (reg_outputs or ops)", "",
            "NumPy (*.npy);;All files (*)"
        )
        if not path:
            return
        try:
            data = np.load(path, allow_pickle=True).item()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load {Path(path).name}:\n{e}")
            return

        # Always use refImg; fall back to meanImg if absent
        FALLBACK_KEYS = ["refImg", "meanImg"]
        key_used = next((k for k in FALLBACK_KEYS if k in data and
                         hasattr(data[k], "shape") and data[k].ndim == 2), None)
        if key_used is None:
            QMessageBox.critical(
                self, "No image found",
                f"{Path(path).name} contains neither 'meanImg' nor 'refImg'.\n"
                f"Found keys: {list(data.keys())}"
            )
            return

        arr = data[key_used].astype(np.float32)
        try:
            self._fov_path = Path(path)
            self._fov_arr_orig = arr
            self._fov_display_k = 0
            self._fov_display_flip_h = False
            self._fov_display_flip_v = False
            if "fov" in self._viewer.layers:
                self._viewer.layers["fov"].data = arr
            else:
                self._viewer.add_image(arr, name="fov", colormap="gray")
            self._viewer.reset_view()
            self._update_load_info()
            self._update_pair_btn_state()
            if self._cell_stat is not None:
                self._show_stat_masks(self._cell_stat)
            self._set_status(
                f"FOV loaded from {Path(path).name} [{key_used}]  "
                f"{arr.shape[1]}×{arr.shape[0]} px"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_load_fov(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load FOV image", "",
            "Images (*.tif *.tiff *.png *.jpg *.czi *.nd2);;All files (*)"
        )
        if not path:
            return
        try:
            arr = _collapse_to_2d(_load_image_any(path))
            self._fov_path = Path(path)
            self._fov_arr_orig = arr
            self._fov_display_k = 0
            self._fov_display_flip_h = False
            self._fov_display_flip_v = False
            if "fov" in self._viewer.layers:
                self._viewer.layers["fov"].data = arr
            else:
                self._viewer.add_image(arr, name="fov", colormap="gray")
            self._viewer.reset_view()
            self._update_load_info()
            self._update_pair_btn_state()
            if self._cell_stat is not None:
                self._show_stat_masks(self._cell_stat)
            self._set_status(f"FOV loaded: {self._fov_path.name}  {arr.shape[1]}×{arr.shape[0]} px")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_suite2p_toggled(self, checked: bool) -> None:
        self._btn_statnpy.setVisible(checked)
        self._btn_iscell.setVisible(checked)
        self._iscell_info.setVisible(checked)
        self._btn_csv.setVisible(not checked)

    def _on_load_iscell(self) -> None:
        """Explicitly load an iscell.npy file.

        If stat.npy is already loaded, immediately re-applies the filter
        and refreshes the display.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Load iscell.npy", "",
            "NumPy (*.npy);;All files (*)"
        )
        if not path:
            return
        try:
            iscell = np.load(path, allow_pickle=True)
            if iscell.ndim != 2 or iscell.shape[1] < 1:
                QMessageBox.critical(self, "Bad file",
                    "iscell.npy must be shape (n_rois, 2).")
                return
            self._iscell = iscell
            n_cells = int(iscell[:, 0].sum())
            n_total = len(iscell)
            self._iscell_info.setText(
                f"iscell.npy: {n_cells} / {n_total} ROIs are cells"
            )
            self._iscell_info.setStyleSheet("font-size: 10px; color: #2a8a2a;")
            self._set_status(f"iscell.npy loaded: {n_cells}/{n_total} cells")

            # Re-apply filter if stat is already loaded
            if self._stat_raw is not None:
                if len(iscell) != len(self._stat_raw):
                    QMessageBox.critical(
                        self, "Size mismatch",
                        f"iscell.npy has {len(iscell)} ROIs but stat.npy has "
                        f"{len(self._stat_raw)} — they must be from the same suite2p run."
                    )
                    self._iscell = None
                    self._iscell_info.setText("iscell.npy: size mismatch — not applied")
                    self._iscell_info.setStyleSheet("font-size: 10px; color: #cc0000;")
                    return
                self._apply_stat_filter()
        except Exception as e:
            QMessageBox.critical(self, "Error loading iscell.npy", str(e))

    def _on_load_statnpy(self) -> None:
        """Load cell coordinates from a suite2p stat.npy file.

        Uses iscell.npy if already loaded, otherwise auto-detects it in
        the same folder. Preserves original row order (no resorting).
        Extracts med=[y,x] as the per-cell centroid coordinate.
        Displays cell ROI masks as a Labels layer if FOV image is loaded.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Load suite2p stat.npy", "",
            "NumPy (*.npy);;All files (*)"
        )
        if not path:
            return
        try:
            self._stat_raw = np.load(path, allow_pickle=True)

            # Use explicitly loaded iscell, else auto-detect in same folder
            if self._iscell is not None:
                iscell_note = " (filtered by iscell.npy)"
            else:
                iscell_path = Path(path).parent / "iscell.npy"
                if iscell_path.exists():
                    self._iscell = np.load(str(iscell_path), allow_pickle=True)
                    n_cells = int(self._iscell[:, 0].sum())
                    n_total = len(self._iscell)
                    self._iscell_info.setText(
                        f"iscell.npy: {n_cells} / {n_total} ROIs are cells  (auto-detected)"
                    )
                    self._iscell_info.setStyleSheet("font-size: 10px; color: #2a8a2a;")
                    iscell_note = " (filtered by iscell.npy, auto-detected)"
                else:
                    iscell_note = " (no iscell.npy — all ROIs kept)"

            self._apply_stat_filter()
            self._update_load_info()
            self._set_status(
                f"Loaded {len(self._cell_stat)} cells from {Path(path).name}{iscell_note}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error loading stat.npy", str(e))

    def _apply_stat_filter(self) -> None:
        """Filter stat_raw by iscell and refresh Points + Labels layers."""
        import pandas as pd

        stat = self._stat_raw
        if self._iscell is not None:
            if len(self._iscell) != len(stat):
                QMessageBox.critical(
                    self, "Size mismatch",
                    f"stat.npy has {len(stat)} ROIs but iscell.npy has "
                    f"{len(self._iscell)} — they must be from the same "
                    f"suite2p run.\n\nProceeding without iscell filter."
                )
                self._iscell = None
                self._iscell_info.setText(
                    f"iscell.npy: size mismatch ({len(self._iscell) if self._iscell is not None else '?'} "
                    f"vs {len(stat)}) — ignored"
                )
                self._iscell_info.setStyleSheet("font-size: 10px; color: #cc0000;")
            cell_mask = self._iscell[:, 0].astype(bool) if self._iscell is not None else np.ones(len(stat), dtype=bool)
        else:
            cell_mask = np.ones(len(stat), dtype=bool)

        # Keep original indices — do NOT sort or reorder
        original_indices = np.where(cell_mask)[0]
        cells_stat = stat[cell_mask]

        # med is [y, x]; store in original space, display in transformed space
        meds = np.array([s["med"] for s in cells_stat], dtype=float)  # (N,2) yx original
        self._cell_pts_fov = meds[:, ::-1]  # (x, y) original — used for transform computation
        self._cell_df = pd.DataFrame({"stat_idx": original_indices})
        self._cell_stat = cells_stat

        # Display in current display space
        meds_disp = self._transform_yx_for_display(meds)
        if "cells_fov" in self._viewer.layers:
            self._viewer.layers["cells_fov"].data = meds_disp
        else:
            self._viewer.add_points(
                meds_disp, name="cells_fov", size=8,
                face_color="red", border_color="white", opacity=0.8
            )

        # Refresh Labels layer
        self._show_stat_masks(cells_stat)

    def _show_stat_masks(self, cells_stat: np.ndarray) -> None:
        """Create/update a Labels layer with per-cell ROI masks.

        Each cell's pixels (from ypix / xpix) are painted with its 1-based index.
        Requires the 'fov' image layer to be present for shape information.
        """
        try:
            fov_layer = self._viewer.layers["fov"]
        except KeyError:
            return  # FOV not loaded yet; masks will be added after FOV loads

        # Build labels in original image space using original dimensions
        if self._fov_arr_orig is None:
            return
        H, W = self._fov_arr_orig.shape[:2]
        labels = np.zeros((H, W), dtype=np.int32)

        for i, s in enumerate(cells_stat):
            ypix = s["ypix"]
            xpix = s["xpix"]
            valid = (ypix >= 0) & (ypix < H) & (xpix >= 0) & (xpix < W)
            labels[ypix[valid], xpix[valid]] = i + 1  # 1-based; 0 = background

        # Apply display transform so masks align with the (possibly rotated) FOV
        labels = self._apply_display_transform(labels)

        if "cell_masks" in self._viewer.layers:
            self._viewer.layers["cell_masks"].data = labels
        else:
            self._viewer.add_labels(labels, name="cell_masks", opacity=0.4)

    def _on_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load cell CSV", "", "CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            xcol = next((c for c in df.columns if c.lower() == "x"), None)
            ycol = next((c for c in df.columns if c.lower() == "y"), None)
            if xcol is None or ycol is None:
                QMessageBox.critical(self, "Error", "CSV must have 'x' and 'y' columns.")
                return
            self._cell_pts_fov = df[[xcol, ycol]].to_numpy(dtype=float)
            self._cell_df = df
            pts_yx = self._cell_pts_fov[:, ::-1]
            if "cells_fov" in self._viewer.layers:
                self._viewer.layers["cells_fov"].data = pts_yx
            else:
                self._viewer.add_points(
                    pts_yx, name="cells_fov", size=8,
                    face_color="red", border_color="white", opacity=0.8
                )
            self._update_load_info()
            self._set_status(f"Loaded {len(self._cell_pts_fov)} cells from {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_load_section(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load section image", "",
            "Images (*.tif *.tiff *.png *.jpg *.czi *.nd2);;All files (*)"
        )
        if not path:
            return
        try:
            arr = _collapse_to_2d(_load_image_any(path))
            if arr.nbytes > 400 * 1024 * 1024:
                f = int(np.ceil((arr.nbytes / (400 * 1024 * 1024)) ** 0.5))
                arr = arr[::f, ::f]
            self._section_path = Path(path)
            self._section_arr  = arr

            if self._second_viewer is not None:
                try:
                    self._second_viewer.close()
                except Exception:
                    pass
            self._second_viewer = napari.Viewer(title=f"Section — {self._section_path.name}")
            self._second_viewer.add_image(arr, name="section", colormap="gray")
            self._second_viewer.reset_view()

            self._update_load_info()
            self._update_pair_btn_state()
            self._set_status(f"Section loaded: {self._section_path.name}  {arr.shape[1]}×{arr.shape[0]} px")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _update_load_info(self) -> None:
        lines = []
        if self._fov_path:
            lines.append(f"FOV: {self._fov_path.name}")
        if self._cell_pts_fov is not None:
            lines.append(f"Cells: {len(self._cell_pts_fov)}")
        if self._section_path:
            lines.append(f"Section: {self._section_path.name}")
        self._load_info.setText("\n".join(lines) if lines else "Nothing loaded")

    def _update_pair_btn_state(self) -> None:
        ready = self._fov_path is not None and self._second_viewer is not None
        self._btn_add_pair.setEnabled(ready and self._pair_state == _IDLE)

    # ------------------------------------------------------------------
    # Sequential landmark pairing state machine
    # ------------------------------------------------------------------

    def _on_start_pair(self) -> None:
        """Begin adding a new landmark pair — wait for FOV click first."""
        self._pair_state = _WAIT_FOV
        self._pending_fov_xy = None

        # Ensure FOV landmark layer exists and is in add mode
        if self._fov_lm_layer is None or "fov_landmarks" not in self._viewer.layers:
            self._fov_lm_layer = self._viewer.add_points(
                name="fov_landmarks", size=14,
                face_color="cyan", border_color="white",
            )
        # Record count BEFORE the click so we can index the new point reliably
        self._fov_n_before = len(self._fov_lm_layer.data)
        self._fov_lm_layer.mode = "add"
        # Disconnect old, connect fresh
        try:
            self._fov_lm_layer.events.data.disconnect(self._on_fov_clicked)
        except Exception:
            pass
        self._fov_lm_layer.events.data.connect(self._on_fov_clicked)

        self._btn_add_pair.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._pair_status.setText("Step 1/2 — Click a landmark on the FOV image (main window)")
        self._pair_status.setStyleSheet("font-weight: bold; color: #1a7abf;")

    def _on_fov_clicked(self, event=None) -> None:
        if self._pair_state != _WAIT_FOV:
            return
        if self._fov_lm_layer is None:
            return
        # Wait until a genuinely NEW point appears
        if len(self._fov_lm_layer.data) <= self._fov_n_before:
            return

        # Index the new point directly — don't use [-1] which can be stale
        yx = self._fov_lm_layer.data[self._fov_n_before]
        # Inverse-transform from display space → original FOV space
        y_orig, x_orig = self._inverse_transform_yx(float(yx[0]), float(yx[1]))
        self._pending_fov_xy = [x_orig, y_orig]  # store as xy in original space

        # Switch state → wait for section click
        self._pair_state = _WAIT_SEC
        self._fov_lm_layer.mode = "pan_zoom"
        try:
            self._fov_lm_layer.events.data.disconnect(self._on_fov_clicked)
        except Exception:
            pass

        # Prepare section landmark layer
        if self._sec_lm_layer is None or "sec_landmarks" not in self._second_viewer.layers:
            self._sec_lm_layer = self._second_viewer.add_points(
                name="sec_landmarks", size=14,
                face_color="yellow", border_color="white",
            )
        # Record section count BEFORE click
        self._sec_n_before = len(self._sec_lm_layer.data)
        self._sec_lm_layer.mode = "add"
        try:
            self._sec_lm_layer.events.data.disconnect(self._on_sec_clicked)
        except Exception:
            pass
        self._sec_lm_layer.events.data.connect(self._on_sec_clicked)

        self._pair_status.setText("Step 2/2 — Click the matching landmark on the Section image (second window)")
        self._pair_status.setStyleSheet("font-weight: bold; color: #c07000;")

    def _on_sec_clicked(self, event=None) -> None:
        if self._pair_state != _WAIT_SEC:
            return
        if self._sec_lm_layer is None:
            return
        # Wait until a genuinely NEW point appears
        if len(self._sec_lm_layer.data) <= self._sec_n_before:
            return

        yx = self._sec_lm_layer.data[self._sec_n_before]
        sec_xy = [float(yx[1]), float(yx[0])]

        # Commit the pair
        pair = {"fov": self._pending_fov_xy, "sec": sec_xy}
        self._pairs.append(pair)

        # Clean up
        self._sec_lm_layer.mode = "pan_zoom"
        try:
            self._sec_lm_layer.events.data.disconnect(self._on_sec_clicked)
        except Exception:
            pass

        self._pair_state = _IDLE
        self._pending_fov_xy = None
        self._btn_cancel.setEnabled(False)
        self._update_pair_btn_state()
        self._update_table()
        self._refresh_lm_layers()
        self._pair_status.setText(f"{len(self._pairs)} pair(s) — ready.  Add more or compute transform.")
        self._pair_status.setStyleSheet("font-weight: bold; color: #2a8a2a;")

    def _on_cancel_pair(self) -> None:
        """Cancel a pair currently in progress."""
        # Remove any pending point from the FOV layer
        if (self._pair_state == _WAIT_SEC
                and self._fov_lm_layer is not None
                and len(self._fov_lm_layer.data) > 0):
            self._fov_lm_layer.data = self._fov_lm_layer.data[:-1]

        try:
            self._fov_lm_layer.events.data.disconnect(self._on_fov_clicked)
        except Exception:
            pass
        try:
            self._sec_lm_layer.events.data.disconnect(self._on_sec_clicked)
        except Exception:
            pass

        if self._fov_lm_layer is not None:
            self._fov_lm_layer.mode = "pan_zoom"
        if self._sec_lm_layer is not None:
            self._sec_lm_layer.mode = "pan_zoom"

        self._pair_state = _IDLE
        self._pending_fov_xy = None
        self._btn_cancel.setEnabled(False)
        self._update_pair_btn_state()
        self._pair_status.setText(f"{len(self._pairs)} pair(s).  Add more or compute transform.")
        self._pair_status.setStyleSheet("font-weight: bold; color: #555;")

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def _update_table(self) -> None:
        self._table.setRowCount(len(self._pairs))
        for i, p in enumerate(self._pairs):
            self._table.setItem(i, 0, self._ro_item(str(i + 1)))
            self._table.setItem(i, 1, self._ro_item(f"{p['fov'][0]:.1f}"))
            self._table.setItem(i, 2, self._ro_item(f"{p['fov'][1]:.1f}"))
            self._table.setItem(i, 3, self._ro_item(f"{p['sec'][0]:.1f}"))
            self._table.setItem(i, 4, self._ro_item(f"{p['sec'][1]:.1f}"))
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
        """Rebuild the Points layers from self._pairs.

        Pairs store FOV coordinates in ORIGINAL image space.
        Transform to display space before painting on the (possibly rotated/flipped) layer.
        """
        if self._pairs:
            yx_orig = np.array([[p["fov"][1], p["fov"][0]] for p in self._pairs])  # yx original
            fov_pts = self._transform_yx_for_display(yx_orig)
            sec_pts = np.array([[p["sec"][1], p["sec"][0]] for p in self._pairs])
        else:
            fov_pts = np.empty((0, 2))
            sec_pts = np.empty((0, 2))

        if self._fov_lm_layer is not None and "fov_landmarks" in self._viewer.layers:
            self._fov_lm_layer.data = fov_pts
        if self._sec_lm_layer is not None and self._second_viewer is not None:
            if "sec_landmarks" in self._second_viewer.layers:
                self._sec_lm_layer.data = sec_pts

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _on_compute_transform(self) -> None:
        if len(self._pairs) < 2:
            QMessageBox.warning(self, "Too few pairs", "Need at least 2 landmark pairs.")
            return
        src = np.array([p["fov"] for p in self._pairs])
        dst = np.array([p["sec"] for p in self._pairs])
        try:
            self._transform_M = _compute_rigid_transform(src, dst)
            p = _transform_params(self._transform_M)
            self._transform_info.setText(
                f"Scale: {p['scale']:.4f}  |  Rotation: {p['rotation_deg']:.2f}°\n"
                f"Tx: {p['translation_x']:.1f}  Ty: {p['translation_y']:.1f}"
            )
            self._set_status("Rigid transform computed.")
        except Exception as e:
            QMessageBox.critical(self, "Transform error", str(e))

    def _on_apply_transform(self) -> None:
        if self._transform_M is None:
            QMessageBox.warning(self, "No transform", "Compute transform first.")
            return
        if self._cell_pts_fov is None:
            QMessageBox.warning(self, "No cells", "Load cell CSV first.")
            return
        if self._second_viewer is None:
            QMessageBox.warning(self, "No section", "Load section first.")
            return

        self._cell_pts_section = _apply_transform(self._transform_M, self._cell_pts_fov)
        pts_yx = self._cell_pts_section[:, ::-1]

        if "cells_section" in self._second_viewer.layers:
            self._second_viewer.layers["cells_section"].data = pts_yx
        else:
            self._second_viewer.add_points(
                pts_yx, name="cells_section", size=8,
                face_color="red", border_color="white", opacity=0.8
            )
        self._set_status(f"{len(self._cell_pts_section)} cells mapped → section space.")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _on_load_landmarks_csv(self) -> None:
        """Load a previously saved landmarks CSV and restore pairs."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load landmarks CSV", "", "CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            required = {"fov_x", "fov_y", "sec_x", "sec_y"}
            if not required.issubset({c.lower() for c in df.columns}):
                QMessageBox.critical(
                    self, "Bad CSV",
                    f"Expected columns: {required}\nFound: {list(df.columns)}"
                )
                return
            # normalise column names to lowercase
            df.columns = [c.lower() for c in df.columns]
            self._pairs = [
                {"fov": [float(r["fov_x"]), float(r["fov_y"])],
                 "sec": [float(r["sec_x"]), float(r["sec_y"])]}
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
            QMessageBox.critical(self, "Error loading landmarks", str(e))

    def _on_save_session(self) -> None:
        """Save all results needed for reproduction and for Step 2."""
        if not self._pairs:
            QMessageBox.warning(self, "No landmarks", "Add landmark pairs first.")
            return
        if self._transform_M is None:
            QMessageBox.warning(self, "No transform", "Compute transform first.")
            return
        if not hasattr(self, "_cell_pts_section"):
            QMessageBox.warning(self, "Not applied", "Apply transform to cells first.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not out_dir:
            return

        import pandas as pd
        out  = Path(out_dir)
        stem = self._fov_path.stem if self._fov_path else "fov"

        # 1 ── Landmark pairs CSV (for reloading)
        lm_df = pd.DataFrame([
            {"pair_id": i + 1,
             "fov_x": p["fov"][0], "fov_y": p["fov"][1],
             "sec_x": p["sec"][0], "sec_y": p["sec"][1]}
            for i, p in enumerate(self._pairs)
        ])
        lm_path = out / f"{stem}_landmarks.csv"
        lm_df.to_csv(lm_path, index=False)

        # 2 ── Transform JSON (for reproducibility + Step 2 header info)
        params = _transform_params(self._transform_M)
        params["fov_image"]        = str(self._fov_path)     if self._fov_path     else None
        params["section_image"]    = str(self._section_path) if self._section_path else None
        params["n_landmark_pairs"] = len(self._pairs)
        params["landmarks_csv"]    = str(lm_path)
        params["cells_section_csv"] = str(out / f"{stem}_cells_section.csv")
        tf_path = out / f"{stem}_fov_to_section_transform.json"
        with open(tf_path, "w") as f:
            json.dump(params, f, indent=2)

        # 3 ── Cell coordinates CSV (input for Step 2)
        df_out = self._cell_df.copy() if self._cell_df is not None else pd.DataFrame()
        df_out["x_fov"]     = self._cell_pts_fov[:, 0]
        df_out["y_fov"]     = self._cell_pts_fov[:, 1]
        df_out["x_section"] = self._cell_pts_section[:, 0]
        df_out["y_section"] = self._cell_pts_section[:, 1]
        cells_path = out / f"{stem}_cells_section.csv"
        df_out.to_csv(cells_path, index=False)

        self._set_status(f"Session saved to {out}")
        QMessageBox.information(
            self, "Session saved",
            f"  {lm_path.name}\n"
            f"  {tf_path.name}\n"
            f"  {cells_path.name}\n\n"
            f"Load {lm_path.name} next time to skip re-clicking landmarks.\n"
            f"Pass {cells_path.name} to Step 2 (Section → Atlas)."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self._status.setText(msg)
