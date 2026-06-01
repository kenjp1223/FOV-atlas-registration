# Next session — ANTsPy image registration

## Context

The pipeline (Steps 1–3) is functionally complete. The one remaining issue is
the **image warp visualisation** in the Atlas Registration widget (Phase 2).

Currently the code uses a custom TPS (thin-plate spline) via
`scipy.interpolate.RBFInterpolator` fitted from manually placed landmark pairs.
Cell coordinate mapping works correctly. Image warping produces incorrect
output (black or misaligned).

---

## The problem

TPS is not well-suited as a unified image + point transform:
- `skimage.transform.warp` needs an **inverse map** (output → source coords)
- Using the same TPS fit for both forward and inverse directions gives
  inconsistent results
- TPS extrapolates badly outside the convex hull of landmarks → black edges
- Coordinate axis handling (x,y vs row,col) adds fragile boilerplate

---

## Agreed solution: ANTsPy

Replace the TPS image warp with `ants.fit_transform_to_paired_points()`.
This gives one transform object that handles **both** image warping and point
coordinate mapping correctly, with a proper mathematical inverse.

```python
import ants

result = ants.fit_transform_to_paired_points(
    moving_pts,           # atlas landmarks (N, 2) — x,y
    fixed_pts,            # section landmarks (N, 2) — x,y
    transform_type="syn", # "affine", "rigid", "similarity", or "syn"
    domain_image=atlas_ants_img
)

# Image: warp atlas → section space
warped_atlas = ants.apply_transforms(
    fixed=section_ants_img,
    moving=atlas_ants_img,
    transformlist=result["fwdtransforms"]
)

# Image: warp section → atlas space
warped_section = ants.apply_transforms(
    fixed=atlas_ants_img,
    moving=section_ants_img,
    transformlist=result["invtransforms"]
)

# Points: map cell coords section → atlas
cells_atlas = ants.apply_transforms_to_points(
    2, cells_section_df, result["invtransforms"]
)
```

## What to change

1. **Add `antspyx` to `pyproject.toml` dependencies**

2. **`_widget_atlas_registration.py` — `_on_compute_tps()`**
   - Replace `TPSTransform` fit with `ants.fit_transform_to_paired_points()`
   - Store `result["fwdtransforms"]` and `result["invtransforms"]`
   - Keep the landmark pair UI unchanged

3. **`_widget_atlas_registration.py` — `_on_apply_tps()`**
   - Replace `self._tps_inv(cell_pts)` with
     `ants.apply_transforms_to_points(2, df, invtransforms)`

4. **`_widget_atlas_registration.py` — `_on_warp_atlas_to_section()`**
   - Replace the `sk_warp` + custom `_inv_map` block with
     `ants.apply_transforms(fixed=section, moving=atlas, transformlist=fwd)`

5. **`_widget_atlas_registration.py` — `_on_warp_section_to_atlas()`**
   - Replace with `ants.apply_transforms(fixed=atlas, moving=section, transformlist=inv)`

6. **Remove** the `skimage.transform.warp` imports and the TODO block above
   `_on_warp_atlas_to_section`

7. **Remove** `from .registration.tps import TPSTransform` import from the widget

## Files involved

- `src/napari_atlas_registration/_widget_atlas_registration.py` — main changes
- `pyproject.toml` — add `antspyx`
- `src/napari_atlas_registration/registration/tps.py` — can be kept for
  reference or removed

## Notes

- `ants.fit_transform_to_paired_points` requires ANTsPy ≥ 0.3.x
- ANTs points are in `(x, y)` order — same as our stored landmark pairs ✓
- `transform_type="syn"` is most accurate for tissue distortion but slower;
  `"affine"` is a good starting point
- `domain_image` must be an `ants.image_read()` or `ants.from_numpy()` image
  that defines the physical space of the moving image (atlas slice)
