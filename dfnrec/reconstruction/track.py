"""Track-building from candidate trace associations.

A 'track' is an ordered list of traces (at most one per face) that are
hypothesised to belong to the same fracture disc.

Algorithm
---------
1. Build candidate graph: nodes = traces, edges = log BF ≥ threshold.
2. Greedy track selection: extend tracks in order of descending log BF,
   subject to the constraint that each trace appears in at most one track.
3. Filter tracks by global RMS plane residual.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from dfnrec.models import Trace, Face
from dfnrec.reconstruction.association import compute_log_bayes_factor, AssociationScore
from dfnrec.geometry.plane import svd_plane_fit


@dataclass
class Track:
    """An ordered collection of traces hypothesised to belong to one disc."""
    traces: List[Trace] = field(default_factory=list)
    face_ids: List[str] = field(default_factory=list)
    total_log_bf: float = 0.0
    plane_fit_rms: Optional[float] = None
    set_id: Optional[str] = None

    def n_faces(self) -> int:
        return len(set(self.face_ids))

    def trace_ids(self) -> List[str]:
        return [t.trace_id for t in self.traces]

    def all_endpoints_xyz(self) -> np.ndarray:
        """All trace endpoints as (2N, 3) array."""
        pts = []
        for t in self.traces:
            pts.append(t.p0_xyz)
            pts.append(t.p1_xyz)
        return np.asarray(pts, dtype=float)


def build_candidate_graph(
    traces: List[Trace],
    faces: Dict[str, Face],
    log_bf_threshold: float = -5.0,
    max_face_gap: int = 3,
) -> List[Tuple[int, int, AssociationScore]]:
    """Compute all candidate trace-pair edges with log BF above threshold.

    Parameters
    ----------
    traces : list of Trace
    faces : dict mapping face_id → Face
    log_bf_threshold : float
        Minimum log BF to include an edge.
    max_face_gap : int
        Only pair traces whose face order_index differs by at most this.

    Returns
    -------
    edges : list of (i, j, AssociationScore) sorted by descending log_bf.
    """
    edges: List[Tuple[int, int, AssociationScore]] = []
    n = len(traces)
    face_order = {fid: f.order_index for fid, f in faces.items()}

    for i in range(n):
        ti = traces[i]
        fi = faces.get(ti.face_id)
        if fi is None:
            continue
        for j in range(i + 1, n):
            tj = traces[j]
            fj = faces.get(tj.face_id)
            if fj is None:
                continue
            # Skip same face
            if ti.face_id == tj.face_id:
                continue
            # Face gap filter
            oi = face_order.get(ti.face_id, 0)
            oj = face_order.get(tj.face_id, 0)
            if abs(oi - oj) > max_face_gap:
                continue
            score = compute_log_bayes_factor(ti, tj, fi, fj)
            if score.log_bf >= log_bf_threshold:
                edges.append((i, j, score))

    edges.sort(key=lambda e: e[2].log_bf, reverse=True)
    return edges


def select_non_overlapping_tracks(
    traces: List[Trace],
    edges: List[Tuple[int, int, AssociationScore]],
    rms_threshold: float = 0.20,
    min_faces: int = 1,
) -> List[Track]:
    """Greedily select non-overlapping tracks from candidate graph edges.

    Each trace can appear in at most one track. Tracks are grown by greedily
    adding edges in descending log BF order, skipping edges that would add
    an already-used trace or a trace from an already-represented face.

    Parameters
    ----------
    traces : list of Trace
    edges : sorted (descending log BF) list of (i, j, score)
    rms_threshold : float
        Tracks with plane-fit RMS > this value are discarded [m].
    min_faces : int
        Discard tracks with fewer than this many distinct faces.

    Returns
    -------
    list of Track (accepted tracks).
    """
    used: Set[int] = set()
    trace_to_track: Dict[int, int] = {}  # trace_index → track_index
    tracks: List[Track] = []

    for i, j, score in edges:
        if i in used and j in used:
            continue

        ti_idx_used = i in used
        tj_idx_used = j in used

        if not ti_idx_used and not tj_idx_used:
            # Start new track
            track = Track(
                traces=[traces[i], traces[j]],
                face_ids=[traces[i].face_id, traces[j].face_id],
                total_log_bf=score.log_bf,
                set_id=traces[i].set_id if traces[i].set_id == traces[j].set_id else None,
            )
            t_idx = len(tracks)
            tracks.append(track)
            used.add(i)
            used.add(j)
            trace_to_track[i] = t_idx
            trace_to_track[j] = t_idx
        elif ti_idx_used and not tj_idx_used:
            t_idx = trace_to_track[i]
            track = tracks[t_idx]
            # No duplicate faces
            if traces[j].face_id not in track.face_ids:
                track.traces.append(traces[j])
                track.face_ids.append(traces[j].face_id)
                track.total_log_bf += score.log_bf
                used.add(j)
                trace_to_track[j] = t_idx
        elif tj_idx_used and not ti_idx_used:
            t_idx = trace_to_track[j]
            track = tracks[t_idx]
            if traces[i].face_id not in track.face_ids:
                track.traces.append(traces[i])
                track.face_ids.append(traces[i].face_id)
                track.total_log_bf += score.log_bf
                used.add(i)
                trace_to_track[i] = t_idx

    # Add singleton tracks for unmatched traces
    for k, trace in enumerate(traces):
        if k not in used:
            tracks.append(Track(
                traces=[trace],
                face_ids=[trace.face_id],
                total_log_bf=0.0,
                set_id=trace.set_id,
            ))

    # Compute plane fit RMS and filter
    accepted = []
    for track in tracks:
        pts = track.all_endpoints_xyz()
        if len(pts) >= 3:
            try:
                res = svd_plane_fit(pts)
                track.plane_fit_rms = res.rms
            except Exception:
                track.plane_fit_rms = float("inf")
        else:
            track.plane_fit_rms = 0.0  # single trace: no meaningful RMS

        if track.n_faces() < min_faces:
            continue
        if track.plane_fit_rms is not None and track.plane_fit_rms > rms_threshold:
            continue
        accepted.append(track)

    return accepted
