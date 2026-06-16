"""dfnrec.geometry — Core 3D geometric primitives for DFN reconstruction."""
from dfnrec.geometry.vector import normalize, axial_angle, pca_line_direction
from dfnrec.geometry.plane import svd_plane_fit, robust_svd_plane_fit, plane_plane_intersection
from dfnrec.geometry.clipping import (
    local_uv_transform,
    uv_to_xyz,
    segment_polygon_clip,
    line_circle_intersection_2d,
)
from dfnrec.geometry.disc_trace import predicted_visible_trace, VisibleTraceResult

__all__ = [
    "normalize",
    "axial_angle",
    "pca_line_direction",
    "svd_plane_fit",
    "robust_svd_plane_fit",
    "plane_plane_intersection",
    "local_uv_transform",
    "uv_to_xyz",
    "segment_polygon_clip",
    "line_circle_intersection_2d",
    "predicted_visible_trace",
    "VisibleTraceResult",
]
