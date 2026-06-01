"""
TPS warp debug script.

Usage:
    uv run python scripts/tps_warp_debug.py \
        --atlas_slice  path/to/*_atlas_slice.tif \
        --section      path/to/section.tif \
        --landmarks    path/to/*_landmarks.csv \
        --out_dir      path/to/output/

--atlas_slice  : the ROTATED 2D atlas slice saved by Phase 1 ("Save & lock rotation").
                 NOT the full 3D atlas volume. Filename ends in _atlas_slice.tif.

Landmarks CSV columns: rot_atlas_x, rot_atlas_y, sec_x, sec_y  (saved by "Save session")

Outputs:
    atlas_warped_to_section.tif   — atlas slice warped into section space
    section_warped_to_atlas.tif   — section warped into atlas slice space
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.interpolate import RBFInterpolator
from skimage.transform import warp as sk_warp


# ── TPS ──────────────────────────────────────────────────────────────────────

class TPSTransform:
    def __init__(self, src_pts, dst_pts, smoothing=0.0):
        self._rbf_x = RBFInterpolator(src_pts, dst_pts[:, 0],
                                       kernel="thin_plate_spline", smoothing=smoothing)
        self._rbf_y = RBFInterpolator(src_pts, dst_pts[:, 1],
                                       kernel="thin_plate_spline", smoothing=smoothing)

    def __call__(self, pts):
        pts = np.atleast_2d(pts)
        return np.column_stack([self._rbf_x(pts), self._rbf_y(pts)])


# ── Load ──────────────────────────────────────────────────────────────────────

def load_image(path):
    arr = tifffile.imread(str(path)).astype(np.float32)
    # Collapse to 2D if needed
    while arr.ndim > 2:
        arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[-1] else arr.mean(axis=-1)
    return arr


def load_landmarks(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "rot_atlas_x" in df.columns:
        ax, ay = "rot_atlas_x", "rot_atlas_y"
    elif "atlas_x" in df.columns:
        ax, ay = "atlas_x", "atlas_y"
    else:
        raise ValueError(f"No atlas columns found. Got: {list(df.columns)}")
    atl_pts = df[[ax, ay]].to_numpy(dtype=float)   # (x,y) = (col,row)
    sec_pts = df[["sec_x", "sec_y"]].to_numpy(dtype=float)
    return atl_pts, sec_pts


# ── Warp ─────────────────────────────────────────────────────────────────────

def warp_atlas_to_section(atlas_sl, section_shape, tps_sec_to_atl, atl_pts, sec_pts):
    """Warp atlas image into section space."""
    print(f"\n[atlas→section]")
    print(f"  atlas shape  : {atlas_sl.shape}")
    print(f"  output shape : {section_shape}")

    # Sanity-check TPS at landmarks
    pred = tps_sec_to_atl(sec_pts)
    for i in range(min(3, len(atl_pts))):
        print(f"  lm{i}: tps_sec_to_atl({sec_pts[i]}) = {pred[i]}  expected {atl_pts[i]}")

    def _inv_map(coords):
        return tps_sec_to_atl(coords)

    warped = sk_warp(atlas_sl, _inv_map,
                     output_shape=section_shape, order=1,
                     mode='constant', preserve_range=True, cval=0).astype(np.float32)

    print(f"  result: min={warped.min():.2f} max={warped.max():.2f} nonzero={np.count_nonzero(warped)}")

    # Verify landmark values
    for i in range(min(3, len(atl_pts))):
        r_sec, c_sec = int(sec_pts[i, 1]), int(sec_pts[i, 0])
        r_atl, c_atl = int(atl_pts[i, 1]), int(atl_pts[i, 0])
        if 0 <= r_sec < warped.shape[0] and 0 <= c_sec < warped.shape[1]:
            print(f"  lm{i}: warped[{r_sec},{c_sec}]={warped[r_sec,c_sec]:.2f}  "
                  f"atlas[{r_atl},{c_atl}]={atlas_sl[r_atl,c_atl]:.2f}  "
                  f"{'✓' if abs(warped[r_sec,c_sec]-atlas_sl[r_atl,c_atl])<5 else '✗ MISMATCH'}")
    return warped


def warp_section_to_atlas(section_arr, atlas_shape, tps_atl_to_sec, atl_pts, sec_pts):
    """Warp section image into atlas space."""
    print(f"\n[section→atlas]")
    print(f"  section shape : {section_arr.shape}")
    print(f"  output shape  : {atlas_shape}")

    def _inv_map(coords):
        return tps_atl_to_sec(coords)

    warped = sk_warp(section_arr, _inv_map,
                     output_shape=atlas_shape, order=1,
                     mode='constant', preserve_range=True, cval=0).astype(np.float32)

    print(f"  result: min={warped.min():.2f} max={warped.max():.2f} nonzero={np.count_nonzero(warped)}")

    for i in range(min(3, len(atl_pts))):
        r_atl, c_atl = int(atl_pts[i, 1]), int(atl_pts[i, 0])
        r_sec, c_sec = int(sec_pts[i, 1]), int(sec_pts[i, 0])
        if 0 <= r_atl < warped.shape[0] and 0 <= c_atl < warped.shape[1]:
            if 0 <= r_sec < section_arr.shape[0] and 0 <= c_sec < section_arr.shape[1]:
                print(f"  lm{i}: warped[{r_atl},{c_atl}]={warped[r_atl,c_atl]:.2f}  "
                      f"section[{r_sec},{c_sec}]={section_arr[r_sec,c_sec]:.2f}  "
                      f"{'✓' if abs(warped[r_atl,c_atl]-section_arr[r_sec,c_sec])<10 else '✗ MISMATCH'}")
    return warped


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="TPS warp debug")
    p.add_argument("--atlas_slice", required=True, help="Rotated 2D atlas slice TIFF (from Phase 1 save, ends in _atlas_slice.tif)")
    p.add_argument("--section",   required=True, help="Section image TIFF")
    p.add_argument("--landmarks", required=True, help="Landmarks CSV")
    p.add_argument("--out_dir",   default=".", help="Output directory")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading images…")
    atlas_sl    = load_image(args.atlas_slice)
    section_arr = load_image(args.section)
    atl_pts, sec_pts = load_landmarks(args.landmarks)

    print(f"Atlas slice   : {atlas_sl.shape}  range [{atlas_sl.min():.1f}, {atlas_sl.max():.1f}]")
    print(f"Section       : {section_arr.shape}  range [{section_arr.min():.1f}, {section_arr.max():.1f}]")
    print(f"Landmarks     : {len(atl_pts)} pairs")
    print(f"  atl_pts[0] (x,y) = {atl_pts[0]}  (col={atl_pts[0,0]:.1f}, row={atl_pts[0,1]:.1f})")
    print(f"  sec_pts[0] (x,y) = {sec_pts[0]}  (col={sec_pts[0,0]:.1f}, row={sec_pts[0,1]:.1f})")

    print("\nFitting TPS…")
    tps_atl_to_sec = TPSTransform(src_pts=atl_pts, dst_pts=sec_pts)
    tps_sec_to_atl = TPSTransform(src_pts=sec_pts, dst_pts=atl_pts)

    # Quick sanity
    pred = tps_atl_to_sec(atl_pts[:2])
    print(f"  tps_atl_to_sec(atl[0]) = {pred[0]}  expected {sec_pts[0]}")
    pred2 = tps_sec_to_atl(sec_pts[:2])
    print(f"  tps_sec_to_atl(sec[0]) = {pred2[0]}  expected {atl_pts[0]}")

    # Warps
    w1 = warp_atlas_to_section(atlas_sl, section_arr.shape[:2],
                                tps_sec_to_atl, atl_pts, sec_pts)
    tifffile.imwrite(str(out / "atlas_warped_to_section.tif"), w1)
    print(f"\n  Saved → {out / 'atlas_warped_to_section.tif'}")

    w2 = warp_section_to_atlas(section_arr, atlas_sl.shape[:2],
                                tps_atl_to_sec, atl_pts, sec_pts)
    tifffile.imwrite(str(out / "section_warped_to_atlas.tif"), w2)
    print(f"  Saved → {out / 'section_warped_to_atlas.tif'}")


if __name__ == "__main__":
    main()
