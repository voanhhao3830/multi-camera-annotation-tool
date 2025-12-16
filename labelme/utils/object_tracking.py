import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment
import trackpy as tp


class ObjectTracking:
    def __init__(self, list_of_labels: list[list[tuple[float, float]]], 
                 max_distance: float = 50.0, search_range: float = None,
                 use_hungarian: bool = True, velocity_smoothing: float = 0.3):
        """
        Args:
            list_of_labels: List of frames, each frame is list of (x, y) coordinates
            max_distance: Maximum distance for matching between frames (meters)
            search_range: Search range for trackpy (if not using Hungarian)
            use_hungarian: If True, use Hungarian algorithm for stable matching (recommended)
            velocity_smoothing: Smoothing factor for velocity prediction (0.0 = no smoothing, 1.0 = full smoothing)
        """
        self.list_of_labels = list_of_labels
        self.global_id = {}  # {(frame_idx, obj_idx): global_id}
        self.max_distance = max_distance
        self.search_range = search_range if search_range is not None else max_distance
        self.use_hungarian = use_hungarian
        self.velocity_smoothing = velocity_smoothing
        
        # Track states: {track_id: {'pos': (x, y), 'vel': (vx, vy), 'last_frame': frame_idx}}
        self.track_states = {}

    def _prepare_dataframe(self) -> pd.DataFrame:
        data = []
        for frame_idx, frame_objects in enumerate(self.list_of_labels):
            for obj_idx, (x, y) in enumerate(frame_objects):
                data.append({
                    'frame': frame_idx,
                    'x': float(x),
                    'y': float(y),
                    'obj_idx': obj_idx
                })
        return pd.DataFrame(data)

    def _predict_position(self, track_id: int, current_frame: int) -> Optional[Tuple[float, float]]:
        """Predict next position based on velocity"""
        if track_id not in self.track_states:
            return None
        
        state = self.track_states[track_id]
        if state['pos'] is None:
            return None
        
        pos = state['pos']
        vel = state.get('vel', (0.0, 0.0))
        last_frame = state.get('last_frame', current_frame - 1)
        
        # Predict position based on velocity
        frames_diff = current_frame - last_frame
        if frames_diff > 0 and frames_diff <= 5:  # Only predict for reasonable frame gaps
            predicted_x = pos[0] + vel[0] * frames_diff
            predicted_y = pos[1] + vel[1] * frames_diff
            return (predicted_x, predicted_y)
        
        return pos

    def _update_velocity(self, track_id: int, new_pos: Tuple[float, float], current_frame: int):
        """Update velocity estimate with smoothing"""
        if track_id not in self.track_states:
            return
        
        state = self.track_states[track_id]
        old_pos = state.get('pos')
        old_vel = state.get('vel', (0.0, 0.0))
        last_frame = state.get('last_frame', current_frame - 1)
        
        if old_pos is not None and current_frame > last_frame:
            frames_diff = current_frame - last_frame
            if frames_diff > 0:
                # Calculate instantaneous velocity
                inst_vel = (
                    (new_pos[0] - old_pos[0]) / frames_diff,
                    (new_pos[1] - old_pos[1]) / frames_diff
                )
                # Smooth velocity: v_new = (1-α) * v_inst + α * v_old
                smoothed_vel = (
                    (1 - self.velocity_smoothing) * inst_vel[0] + self.velocity_smoothing * old_vel[0],
                    (1 - self.velocity_smoothing) * inst_vel[1] + self.velocity_smoothing * old_vel[1]
                )
                state['vel'] = smoothed_vel
        
        state['pos'] = new_pos
        state['last_frame'] = current_frame

    def _hungarian_tracking(self) -> Dict[Tuple[int, int], int]:
        """Track using Hungarian algorithm for stable matching"""
        if not self.list_of_labels:
            return {}
        
        temporal_map: Dict[Tuple[int, int], int] = {}
        next_track_id = 0
        
        # Initialize with first frame
        if len(self.list_of_labels) > 0:
            first_dets = self.list_of_labels[0]
            for det_idx, pos in enumerate(first_dets):
                track_id = next_track_id
                next_track_id += 1
                self.track_states[track_id] = {
                    'pos': pos,
                    'vel': (0.0, 0.0),
                    'last_frame': 0
                }
                temporal_map[(0, det_idx)] = track_id
        
        # Process subsequent frames
        for frame_idx in range(1, len(self.list_of_labels)):
            dets = self.list_of_labels[frame_idx]
            if not dets:
                # No detections, keep track states but don't update
                continue
            
            # Get active tracks (tracks that appeared recently)
            active_tracks = []
            active_track_ids = []
            for track_id, state in self.track_states.items():
                last_frame = state.get('last_frame', frame_idx - 1)
                # Consider track active if it appeared within last 3 frames
                if frame_idx - last_frame <= 3:
                    predicted_pos = self._predict_position(track_id, frame_idx)
                    if predicted_pos is not None:
                        active_tracks.append(predicted_pos)
                        active_track_ids.append(track_id)
            
            if not active_tracks:
                # No active tracks, assign new IDs to all detections
                for det_idx, pos in enumerate(dets):
                    track_id = next_track_id
                    next_track_id += 1
                    self.track_states[track_id] = {
                        'pos': pos,
                        'vel': (0.0, 0.0),
                        'last_frame': frame_idx
                    }
                    temporal_map[(frame_idx, det_idx)] = track_id
                continue
            
            # Build cost matrix: tracks x detections
            cost = np.full((len(active_tracks), len(dets)), fill_value=self.max_distance * 10, dtype=np.float32)
            for t_idx, track_pos in enumerate(active_tracks):
                for d_idx, det_pos in enumerate(dets):
                    dx = track_pos[0] - det_pos[0]
                    dy = track_pos[1] - det_pos[1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    cost[t_idx, d_idx] = dist
            
            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            
            # Assign matches within max_distance
            assigned_dets = set()
            for t_idx, d_idx in zip(row_ind, col_ind):
                if cost[t_idx, d_idx] <= self.max_distance:
                    track_id = active_track_ids[t_idx]
                    det_pos = dets[d_idx]
                    self._update_velocity(track_id, det_pos, frame_idx)
                    temporal_map[(frame_idx, d_idx)] = track_id
                    assigned_dets.add(d_idx)
            
            # Unassigned detections: create new tracks
            for det_idx, det_pos in enumerate(dets):
                if det_idx not in assigned_dets:
                    track_id = next_track_id
                    next_track_id += 1
                    self.track_states[track_id] = {
                        'pos': det_pos,
                        'vel': (0.0, 0.0),
                        'last_frame': frame_idx
                    }
                    temporal_map[(frame_idx, det_idx)] = track_id
        
        return temporal_map

    def assign_global_id(self) -> Dict[Tuple[int, int], int]:
        if not self.list_of_labels:
            return {}
        
        if self.use_hungarian:
            # Use Hungarian algorithm for stable tracking
            self.global_id = self._hungarian_tracking()
        else:
            # Use trackpy (fallback)
            df = self._prepare_dataframe()
            
            if df.empty:
                return {}
            
            try:
                linked = tp.link(df, search_range=self.search_range, memory=0)
            except Exception as e:
                print(f"Error in trackpy.link: {e}")
                linked = tp.link_df(df, search_range=self.search_range, memory=0)
            
            self.global_id = {}
            particle_to_global = {}
            next_global_id = 0
            
            for _, row in linked.iterrows():
                frame_idx = int(row['frame'])
                obj_idx = int(row['obj_idx'])
                particle_id = int(row['particle'])
                
                if particle_id not in particle_to_global:
                    particle_to_global[particle_id] = next_global_id
                    next_global_id += 1
                
                global_id = particle_to_global[particle_id]
                self.global_id[(frame_idx, obj_idx)] = global_id
        
        return self.global_id

    def get_global_id(self, frame_idx: int, obj_idx: int) -> int:
        return self.global_id.get((frame_idx, obj_idx), -1)