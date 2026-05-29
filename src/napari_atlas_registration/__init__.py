"""napari-atlas-registration — atlas rotation, BigWarp integration, and coordinate mapping."""

__version__ = "0.1.0"


def AtlasSetupWidget(napari_viewer):
    from ._widget_rotation import AtlasSetupWidget as _W
    return _W(napari_viewer)


def InverseWarpWidget(napari_viewer):
    from ._widget_inverse_warp import InverseWarpWidget as _W
    return _W(napari_viewer)
