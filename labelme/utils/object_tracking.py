from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


class _Kalman2D:
    """
    Constant-velocity Kalman filter in BEV (x, y, vx, vy).
    Mirrors the lightweight tracker used in EarlyBird for BEV tracks.
    """

    def __init__(self, process_noise: float = 1.0, meas_noise: float = 0.5, dt: float = 1.0):
        self.dt = dt
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        q = process_noise
        r = meas_noise
        self.Q = np.eye(4, dtype=np.float32) * q
        self.R = np.eye(2, dtype=np.float32) * r

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.zeros((4,), dtype=np.float32)
        mean[0:2] = measurement
        cov = np.eye(4, dtype=np.float32)
        return mean, cov

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + self.Q
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        S = self.H @ cov @ self.H.T + self.R
        K = cov @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - (self.H @ mean)
        mean = mean + K @ innovation
        cov = (np.eye(4, dtype=np.float32) - K @ self.H) @ cov
        return mean, cov

    def gating_distance(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray) -> float:
        S = self.H @ cov @ self.H.T + self.R
        diff = measurement - (self.H @ mean)
        return float(diff.T @ np.linalg.inv(S) @ diff)


class _Track:
    def __init__(self, track_id: int, mean: np.ndarray, cov: np.ndarray):
        self.id = track_id
        self.mean = mean
        self.cov = cov
        self.age = 1
        self.time_since_update = 0
        self.hits = 1

    def predict(self, kf: _Kalman2D):
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf: _Kalman2D, measurement: np.ndarray):
        self.mean, self.cov = kf.update(self.mean, self.cov, measurement)
        self.time_since_update = 0
        self.hits += 1

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.mean[0]), float(self.mean[1])


class EarlyBirdLikeTracker:
    """
    Simplified EarlyBird tracker in BEV space:
    - Constant velocity Kalman filter
    - Hungarian association with Mahalanobis gating
    - Track lifecycle: max_age frames without update, min_hits to confirm
    """

    def __init__(
        self,
        max_distance: float = 9.0,
        max_age: int = 10,
        min_hits: int = 2,
        process_noise: float = 1.0,
        meas_noise: float = 0.5,
        dt: float = 1.0,
    ):
        self.max_distance = max_distance  # gating distance (Mahalanobis squared)
        self.max_age = max_age
        self.min_hits = min_hits
        self.kf = _Kalman2D(process_noise=process_noise, meas_noise=meas_noise, dt=dt)
        self.tracks: List[_Track] = []
        self._next_id = 0

    def _associate(self, detections: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self.tracks or len(detections) == 0:
            return [], list(range(len(self.tracks))), list(range(len(detections)))

        cost = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                cost[t_idx, d_idx] = self.kf.gating_distance(track.mean, track.cov, det)

        row, col = linear_sum_assignment(cost)
        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row, col):
            if cost[r, c] <= self.max_distance:
                matches.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matches, list(unmatched_tracks), list(unmatched_dets)

    def update(self, detections_xy: np.ndarray) -> List[Tuple[int, int]]:
        """
        Args:
            detections_xy: array of shape [N, 2] containing (x, y) in BEV / world meters
        Returns:
            List of (det_idx, track_id) for the current frame (only confirmed assignments).
        """
        # Predict existing tracks
        for track in self.tracks:
            track.predict(self.kf)

        matches, unmatched_tracks, unmatched_dets = self._associate(detections_xy)

        # Store track_id for each match before updating/removing tracks
        # This is needed because tracks may be removed, changing indices
        match_track_ids = {}  # {d_idx: track_id}
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            match_track_ids[d_idx] = track.id
            track.update(self.kf, detections_xy[d_idx])

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            mean, cov = self.kf.initiate(detections_xy[d_idx])
            self.tracks.append(_Track(self._next_id, mean, cov))
            self._next_id += 1

        # Remove stale tracks
        alive_tracks: List[_Track] = []
        for track in self.tracks:
            if track.time_since_update <= self.max_age:
                alive_tracks.append(track)
        self.tracks = alive_tracks

        # Build output assignments (only for tracks that are "confirmed")
        # Use track_id to find tracks after removal, since indices may have changed
        det_to_track: List[Tuple[int, int]] = []
        for d_idx, track_id in match_track_ids.items():
            # Find track by ID (since indices may have changed after removal)
            track = None
            for t in self.tracks:
                if t.id == track_id:
                    track = t
                    break
            if track is not None and (track.hits >= self.min_hits or track.time_since_update == 0):
                det_to_track.append((d_idx, track_id))
        return det_to_track


class ObjectTracking:
    """
    Drop-in replacement that runs an EarlyBird-style BEV Kalman + Hungarian tracker
    on per-frame (x, y) detections. Input remains list_of_labels per frame.
    
    Note: Uses actual detection points (footpoints) from clusters, not computed centroids.
    Each point in list_of_labels should be a representative footpoint from a cluster.
    """

    def __init__(
        self,
        list_of_labels: List[List[Tuple[float, float]]],
        max_distance: float = 9.0,
        max_age: int = 10,
        min_hits: int = 2,
        process_noise: float = 1.0,
        meas_noise: float = 0.5,
        dt: float = 1.0,
    ):
        self.list_of_labels = list_of_labels
        self.global_id: Dict[Tuple[int, int], int] = {}
        self.tracker = EarlyBirdLikeTracker(
            max_distance=max_distance,
            max_age=max_age,
            min_hits=min_hits,
            process_noise=process_noise,
            meas_noise=meas_noise,
            dt=dt,
        )

    def assign_global_id(self) -> Dict[Tuple[int, int], int]:
        """
        Run the EarlyBird-like tracker and return mapping (frame_idx, obj_idx) -> global_id.
        """
        self.global_id = {}
        for frame_idx, detections in enumerate(self.list_of_labels):
            if len(detections) == 0:
                # Even if no dets, still age tracks; prediction is done inside update.
                detections_xy = np.empty((0, 2), dtype=np.float32)
            else:
                detections_xy = np.array(detections, dtype=np.float32)

            assignments = self.tracker.update(detections_xy)
            for det_idx, track_id in assignments:
                self.global_id[(frame_idx, det_idx)] = track_id

        return self.global_id

    def get_global_id(self, frame_idx: int, obj_idx: int) -> int:
        return self.global_id.get((frame_idx, obj_idx), -1)