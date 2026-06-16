"""Unit tests for dfnrec.geometry (Branch 2: geometry-core)."""
import math
import numpy as np
import pytest

from dfnrec.geometry.vector import (
    normalize,
    axial_angle,
    angle_between,
    pca_line_direction,
    trend_plunge_from_normal,
    normal_from_trend_plunge,
)
from dfnrec.geometry.plane import (
    svd_plane_fit,
    robust_svd_plane_fit,
    plane_plane_intersection,
)
from dfnrec.geometry.clipping import (
    line_circle_intersection_2d,
    segment_polygon_clip,
    local_uv_transform,
    uv_to_xyz,
)
from dfnrec.geometry.disc_trace import predicted_visible_trace


# ======================================================================
# vector.py
# ======================================================================
class TestVector:
    def test_normalize(self):
        v = normalize(np.array([3.0, 4.0, 0.0]))
        assert abs(np.linalg.norm(v) - 1.0) < 1e-12

    def test_normalize_zero_raises(self):
        with pytest.raises(ValueError):
            normalize(np.zeros(3))

    def test_axial_angle_same_direction(self):
        a = np.array([1.0, 0.0, 0.0])
        assert axial_angle(a, a) < 1e-9

    def test_axial_angle_opposite_direction(self):
        # axial: -a and a represent same orientation → angle = 0
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert axial_angle(a, b) < 1e-9

    def test_axial_angle_90_deg(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(axial_angle(a, b) - math.pi / 2) < 1e-9

    def test_pca_line_direction_collinear(self):
        # Points along y-axis
        pts = np.array([[0, -3, 0], [0, -1, 0], [0, 1, 0], [0, 3, 0]], dtype=float)
        d = pca_line_direction(pts)
        assert abs(abs(d[1]) - 1.0) < 1e-9  # should point along y

    def test_trend_plunge_round_trip(self):
        for trend, plunge in [(0, 0), (90, 45), (180, 60), (270, 30)]:
            n = normal_from_trend_plunge(trend, plunge)
            t2, p2 = trend_plunge_from_normal(n)
            assert abs(t2 - trend) < 0.5 or abs(t2 - (trend + 360) % 360) < 0.5
            assert abs(p2 - plunge) < 0.5


# ======================================================================
# plane.py
# ======================================================================
class TestPlane:
    def _make_xy_plane_points(self, n=20, noise=0.0):
        rng = np.random.default_rng(42)
        pts = rng.uniform(-5, 5, (n, 3))
        pts[:, 2] = 0.0  # z=0 plane
        if noise > 0:
            pts[:, 2] += rng.normal(0, noise, n)
        return pts

    def test_svd_xy_plane(self):
        pts = self._make_xy_plane_points()
        res = svd_plane_fit(pts)
        # Normal should be [0,0,±1]
        assert abs(abs(res.normal[2]) - 1.0) < 1e-9
        assert res.rms < 1e-9

    def test_svd_noisy_plane(self):
        pts = self._make_xy_plane_points(n=50, noise=0.02)
        res = svd_plane_fit(pts)
        assert abs(abs(res.normal[2]) - 1.0) < 0.05
        assert res.rms < 0.05

    def test_svd_weighted(self):
        pts = self._make_xy_plane_points()
        w = np.ones(len(pts))
        res1 = svd_plane_fit(pts)
        res2 = svd_plane_fit(pts, w)
        assert np.allclose(abs(res1.normal), abs(res2.normal), atol=1e-6)

    def test_svd_too_few_points_raises(self):
        with pytest.raises(ValueError, match="3 points"):
            svd_plane_fit(np.array([[0, 0, 0], [1, 0, 0]]))

    def test_robust_svd_consistent(self):
        pts = self._make_xy_plane_points(n=30, noise=0.01)
        # Add 2 outliers
        pts_with_outliers = np.vstack([pts, [[0, 0, 5], [0, 0, -5]]])
        res = robust_svd_plane_fit(pts_with_outliers)
        assert abs(abs(res.normal[2]) - 1.0) < 0.1

    def test_plane_plane_intersection_perpendicular(self):
        # n1 = x-plane: n=(1,0,0), d=0
        # n2 = y-plane: n=(0,1,0), d=0
        # intersection should be along z-axis
        res = plane_plane_intersection(
            np.array([1.0, 0.0, 0.0]), 0.0,
            np.array([0.0, 1.0, 0.0]), 0.0,
        )
        assert not res.is_parallel
        # direction should be along z
        assert abs(abs(res.direction[2]) - 1.0) < 1e-9

    def test_plane_plane_intersection_parallel(self):
        n = np.array([1.0, 0.0, 0.0])
        res = plane_plane_intersection(n, 0.0, n, 1.0)
        assert res.is_parallel


# ======================================================================
# clipping.py
# ======================================================================
class TestClipping:
    def test_line_circle_two_intersections(self):
        # Horizontal line through centre of unit circle at origin
        result = line_circle_intersection_2d(
            p=np.array([0.0, 0.0]),
            d=np.array([1.0, 0.0]),
            center=np.array([0.0, 0.0]),
            radius=1.0,
        )
        assert result is not None
        pt1, pt2 = result
        # Endpoints should be at x=±1
        assert abs(min(pt1[0], pt2[0]) - (-1.0)) < 1e-9
        assert abs(max(pt1[0], pt2[0]) - 1.0) < 1e-9

    def test_line_circle_no_intersection(self):
        result = line_circle_intersection_2d(
            p=np.array([0.0, 5.0]),
            d=np.array([1.0, 0.0]),
            center=np.array([0.0, 0.0]),
            radius=1.0,
        )
        assert result is None

    def test_segment_polygon_inside(self):
        # Square window [-2,2]x[-2,2]
        window = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]], dtype=float)
        seg = segment_polygon_clip(np.array([-1.0, 0.0]), np.array([1.0, 0.0]), window)
        assert seg is not None
        p0, p1 = seg
        assert abs(np.linalg.norm(p1 - p0) - 2.0) < 1e-9

    def test_segment_polygon_outside(self):
        window = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]], dtype=float)
        seg = segment_polygon_clip(np.array([-5.0, 0.0]), np.array([-3.0, 0.0]), window)
        assert seg is None

    def test_segment_polygon_partial_clip(self):
        window = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]], dtype=float)
        # Segment from inside to outside
        seg = segment_polygon_clip(np.array([0.0, 0.0]), np.array([5.0, 0.0]), window)
        assert seg is not None
        p0, p1 = seg
        # Clipped end should be at x=2
        assert abs(max(p0[0], p1[0]) - 2.0) < 1e-6

    def test_local_uv_transform_round_trip(self):
        origin = np.array([5.0, 3.0, 1.0])
        axis_u = np.array([0.0, 1.0, 0.0])
        axis_v = np.array([0.0, 0.0, 1.0])
        xyz_to_uv, _uv_to_xyz = local_uv_transform(origin, axis_u, axis_v)

        pts_xyz = np.array([[5.0, 4.0, 2.0], [5.0, 3.5, 1.5]])
        pts_uv = xyz_to_uv(pts_xyz)
        pts_back = _uv_to_xyz(pts_uv)
        assert np.allclose(pts_xyz, pts_back, atol=1e-9)


# ======================================================================
# disc_trace.py
# ======================================================================
class TestDiscTrace:
    """Test the predicted_visible_trace function."""

    def _make_face(self, x_pos=0.0, window_half=3.0):
        """A face at x=x_pos, normal=[1,0,0], window = square in y-z."""
        return dict(
            face_origin_xyz=np.array([x_pos, 0.0, 0.0]),
            face_normal_xyz=np.array([1.0, 0.0, 0.0]),
            face_axis_u_xyz=np.array([0.0, 1.0, 0.0]),
            face_axis_v_xyz=np.array([0.0, 0.0, 1.0]),
            observation_window_uv=np.array([
                [-window_half, -window_half],
                [window_half, -window_half],
                [window_half, window_half],
                [-window_half, window_half],
            ]),
        )

    def test_vertical_disc_through_face_centre(self):
        """Vertical disc (normal=[0,1,0]) centred on face, radius=2."""
        face = self._make_face(x_pos=0.0)
        result = predicted_visible_trace(
            center_xyz=np.array([0.0, 0.0, 0.0]),
            normal_xyz=np.array([0.0, 1.0, 0.0]),  # vertical disc, trace = z-direction
            radius_m=2.0,
            **face,
        )
        assert result.intersects_face
        assert result.chord_exists
        assert abs(result.full_chord_length - 4.0) < 1e-9  # 2*radius (centre passes through)
        assert result.visible_length > 0.0
        assert result.fully_inside_window  # chord < window

    def test_disc_parallel_to_face(self):
        """Disc plane parallel to face → no intersection."""
        face = self._make_face(x_pos=0.0)
        result = predicted_visible_trace(
            center_xyz=np.array([0.0, 0.0, 0.0]),
            normal_xyz=np.array([1.0, 0.0, 0.0]),  # same as face normal → parallel
            radius_m=2.0,
            **face,
        )
        assert not result.intersects_face

    def test_disc_too_far_from_face(self):
        """Disc centre is farther from face than radius → no chord."""
        face = self._make_face(x_pos=0.0)
        result = predicted_visible_trace(
            center_xyz=np.array([5.0, 0.0, 0.0]),  # 5 m behind face
            normal_xyz=np.array([0.0, 1.0, 0.0]),
            radius_m=2.0,
            **face,
        )
        assert result.intersects_face
        assert not result.chord_exists

    def test_chord_clipped_by_window(self):
        """Disc that would have a 10 m chord, window only 3 m wide."""
        face = self._make_face(x_pos=0.0, window_half=1.5)  # window ±1.5 m
        result = predicted_visible_trace(
            center_xyz=np.array([0.0, 0.0, 0.0]),
            normal_xyz=np.array([0.0, 1.0, 0.0]),
            radius_m=5.0,  # chord = 10 m >> window
            **face,
        )
        assert result.chord_exists
        assert result.clipped_by_window
        assert result.visible_length < result.full_chord_length
        assert result.visible_length <= 3.0 + 1e-9

    def test_disc_outside_window(self):
        """Disc chord exists but lies entirely outside window."""
        face = self._make_face(x_pos=0.0, window_half=1.0)
        result = predicted_visible_trace(
            center_xyz=np.array([0.0, 10.0, 0.0]),  # centre far in y
            normal_xyz=np.array([0.0, 0.0, 1.0]),   # chord is along y
            radius_m=2.0,
            **face,
        )
        # Chord is at y=8..12, window is y ∈ [-1,1] → chord outside window
        assert result.chord_exists
        assert result.visible_length == 0.0

    def test_endpoint_xyz_is_on_face_plane(self):
        """Chord endpoints should lie on the face plane (x=0)."""
        face = self._make_face(x_pos=0.0)
        result = predicted_visible_trace(
            center_xyz=np.array([0.0, 0.0, 0.0]),
            normal_xyz=np.array([0.0, 1.0, 0.0]),
            radius_m=2.0,
            **face,
        )
        assert result.chord_p0_xyz is not None
        assert abs(result.chord_p0_xyz[0]) < 1e-6  # x ≈ 0
        assert abs(result.chord_p1_xyz[0]) < 1e-6
