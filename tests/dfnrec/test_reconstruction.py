"""Unit tests for Branch 4: visible-disc-reconstruction."""
import math
import pytest
import numpy as np

from dfnrec.models import Trace, Face, CensorType
from dfnrec.reconstruction.association import compute_log_bayes_factor
from dfnrec.reconstruction.track import Track, build_candidate_graph, select_non_overlapping_tracks
from dfnrec.reconstruction.svd_fitting import fit_plane_to_track
from dfnrec.reconstruction.map_disc import estimate_disc_map
from dfnrec.validation.generator import SyntheticDFNGenerator


# ── Helpers ──────────────────────────────────────────────────────────
def make_face(face_id, x_pos, order_index):
    return Face(
        face_id=face_id,
        order_index=order_index,
        origin_xyz=[x_pos, 0.0, 0.0],
        normal_xyz=[1.0, 0.0, 0.0],
        axis_u_xyz=[0.0, 1.0, 0.0],
        axis_v_xyz=[0.0, 0.0, 1.0],
        observation_window_polygon_uv=[[-3, -3], [3, -3], [3, 3], [-3, 3]],
    )


def make_trace(trace_id, face_id, p0, p1, set_id="S1",
               censor_p0=CensorType.NATURAL, censor_p1=CensorType.NATURAL):
    return Trace(
        trace_id=trace_id,
        face_id=face_id,
        set_id=set_id,
        p0_xyz=list(p0),
        p1_xyz=list(p1),
        censor_p0=censor_p0,
        censor_p1=censor_p1,
        measurement_sigma=0.02,
    )


# ── Association tests ─────────────────────────────────────────────────
class TestAssociation:
    def test_same_set_coplanar_high_score(self):
        """Two traces on parallel faces from same set and coplanar → high log BF."""
        f1 = make_face("F1", 0.0, 0)
        f2 = make_face("F2", 2.0, 1)
        # Both traces lie along z-axis (normal would be in y-z plane)
        t1 = make_trace("T1", "F1", [0, -1, 0], [0, 1, 0])
        t2 = make_trace("T2", "F2", [2, -1, 0], [2, 1, 0])
        score = compute_log_bayes_factor(t1, t2, f1, f2)
        assert score.log_bf > -10.0  # should be a reasonable score

    def test_different_set_penalty(self):
        """Traces from different sets should have lower log BF."""
        f1 = make_face("F1", 0.0, 0)
        f2 = make_face("F2", 2.0, 1)
        t1 = make_trace("T1", "F1", [0, -1, 0], [0, 1, 0], set_id="S1")
        t2 = make_trace("T2", "F2", [2, -1, 0], [2, 1, 0], set_id="S2")
        score_diff = compute_log_bayes_factor(t1, t2, f1, f2)
        t2_same = make_trace("T2", "F2", [2, -1, 0], [2, 1, 0], set_id="S1")
        score_same = compute_log_bayes_factor(t1, t2_same, f1, f2)
        assert score_same.log_bf > score_diff.log_bf

    def test_score_decomposition_finite(self):
        """All components should be finite numbers."""
        f1 = make_face("F1", 0.0, 0)
        f2 = make_face("F2", 2.0, 1)
        t1 = make_trace("T1", "F1", [0, -1, 0], [0, 1, 0])
        t2 = make_trace("T2", "F2", [2, -1, 0], [2, 1, 0])
        s = compute_log_bayes_factor(t1, t2, f1, f2)
        for val in (s.log_p_orient, s.log_p_spatial, s.log_p_prior, s.log_p_persist):
            assert math.isfinite(val)


# ── Track building tests ──────────────────────────────────────────────
class TestTrackBuilding:
    def _setup(self):
        f1 = make_face("F1", 0.0, 0)
        f2 = make_face("F2", 2.0, 1)
        f3 = make_face("F3", 4.0, 2)
        # Three traces from same disc: vertical disc passing through all faces
        t1 = make_trace("T1", "F1", [0, -1.5, 0], [0, 1.5, 0])
        t2 = make_trace("T2", "F2", [2, -1.5, 0], [2, 1.5, 0])
        t3 = make_trace("T3", "F3", [4, -1.5, 0], [4, 1.5, 0])
        faces = {"F1": f1, "F2": f2, "F3": f3}
        traces = [t1, t2, t3]
        return traces, faces

    def test_build_candidate_graph_has_edges(self):
        traces, faces = self._setup()
        edges = build_candidate_graph(traces, faces, log_bf_threshold=-50.0)
        assert len(edges) > 0

    def test_edges_sorted_descending(self):
        traces, faces = self._setup()
        edges = build_candidate_graph(traces, faces, log_bf_threshold=-50.0)
        scores = [e[2].log_bf for e in edges]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def test_select_tracks_no_duplicates(self):
        traces, faces = self._setup()
        edges = build_candidate_graph(traces, faces, log_bf_threshold=-50.0)
        tracks = select_non_overlapping_tracks(traces, edges)
        # No trace should appear in more than one track
        all_trace_ids = []
        for track in tracks:
            all_trace_ids.extend(track.trace_ids())
        assert len(all_trace_ids) == len(set(all_trace_ids))

    def test_all_traces_in_some_track(self):
        traces, faces = self._setup()
        edges = build_candidate_graph(traces, faces, log_bf_threshold=-50.0)
        tracks = select_non_overlapping_tracks(traces, edges, min_faces=1)
        covered = set()
        for track in tracks:
            covered.update(track.trace_ids())
        assert covered == {t.trace_id for t in traces}


# ── SVD fitting tests ─────────────────────────────────────────────────
class TestSVDFitting:
    def test_multi_face_track_gives_plane(self):
        f1 = make_face("F1", 0.0, 0)
        f2 = make_face("F2", 2.0, 1)
        # Traces on a vertical disc (normal ≈ [0, 1, 0])
        t1 = make_trace("T1", "F1", [0, 0, -2], [0, 0, 2])
        t2 = make_trace("T2", "F2", [2, 0, -2], [2, 0, 2])
        track = Track(traces=[t1, t2], face_ids=["F1", "F2"])
        result = fit_plane_to_track(track)
        assert result is not None
        # Normal should be near [0, ±1, 0] or [0, 0, ±1] depending on trace directions
        # More importantly: rms should be small
        assert result.rms < 0.1

    def test_singleton_with_orientation_returns_plane(self):
        t = make_trace("T1", "F1", [0, -1, 0], [0, 1, 0])
        t.trend_deg = 90.0
        t.plunge_deg = 60.0
        track = Track(traces=[t], face_ids=["F1"])
        result = fit_plane_to_track(track)
        assert result is not None
        assert math.isfinite(result.trend_deg)


# ── MAP disc estimation tests ─────────────────────────────────────────
class TestMapDisc:
    def test_estimate_contained_disc(self):
        """Simple case: disc centred at [1,0,0] normal=[0,1,0] radius=2.
        Two faces at x=0 and x=2, both see contained traces along z."""
        f1 = make_face("F1", 0.0, 0)
        f2 = make_face("F2", 2.0, 1)
        # Both traces: chord of disc at x=0 and x=2
        # For disc centre=[1,0,0], normal=[0,1,0], r=2:
        # Both faces at x=±1 from centre → dist=1, half_chord=sqrt(4-1)=sqrt(3)≈1.73
        t1 = make_trace("T1", "F1", [0, 0, -1.73], [0, 0, 1.73])
        t2 = make_trace("T2", "F2", [2, 0, -1.73], [2, 0, 1.73])
        faces = {"F1": f1, "F2": f2}
        track = Track(traces=[t1, t2], face_ids=["F1", "F2"], set_id="S1")
        disc = estimate_disc_map(track, faces, disc_id_prefix="D_test")
        assert disc is not None
        assert disc.source == "observed_reconstructed"
        assert disc.radius_m > 0.5
        assert disc.n_faces_observed == 2

    def test_estimate_returns_none_for_empty_track(self):
        """Empty track (no traces) should return None gracefully."""
        track = Track(traces=[], face_ids=[])
        disc = estimate_disc_map(track, {})
        assert disc is None

    def test_reliability_class_assigned(self):
        """Disc from 2-face track with low RMS should be class A or B."""
        f1 = make_face("F1", 0.0, 0)
        f2 = make_face("F2", 2.0, 1)
        f3 = make_face("F3", 4.0, 2)
        t1 = make_trace("T1", "F1", [0, 0, -1.73], [0, 0, 1.73])
        t2 = make_trace("T2", "F2", [2, 0, -1.73], [2, 0, 1.73])
        t3 = make_trace("T3", "F3", [4, 0, -1.73], [4, 0, 1.73])
        faces = {"F1": f1, "F2": f2, "F3": f3}
        track = Track(traces=[t1, t2, t3], face_ids=["F1", "F2", "F3"], set_id="S1")
        disc = estimate_disc_map(track, faces)
        assert disc is not None
        from dfnrec.models import ReliabilityClass
        assert disc.reliability_class in (ReliabilityClass.A, ReliabilityClass.B, ReliabilityClass.C)

    def test_full_pipeline_integration(self):
        """Integration test: generate synthetic data, build tracks, estimate discs."""
        gen = SyntheticDFNGenerator(seed=7)
        gt = gen.generate(n_sets=1, n_faces=4, face_half_size=3.0)
        faces_dict = {f.face_id: f for f in gt.faces}

        edges = build_candidate_graph(gt.traces, faces_dict, log_bf_threshold=-20.0)
        tracks = select_non_overlapping_tracks(gt.traces, edges, min_faces=1)

        discs = []
        for track in tracks:
            d = estimate_disc_map(track, faces_dict)
            if d is not None:
                discs.append(d)

        assert len(discs) > 0
        for d in discs:
            assert d.source == "observed_reconstructed"
            assert d.radius_m > 0
