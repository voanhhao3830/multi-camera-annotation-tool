from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from .multicamera_matching import MultiCameraTracking
from .object_tracking import ObjectTracking


class FixedIDTracker:
    """
    Offline tracker with fixed number of objects.
    Uses Hungarian assignment on BEV centroids frame-to-frame to minimize ID switches.
    """

    def __init__(self, list_of_labels: List[List[Tuple[float, float]]], expected_objects: int, max_distance: float):
        self.list_of_labels = list_of_labels
        self.expected_objects = expected_objects
        self.max_distance = max_distance
        self.track_states = {}  # track_id -> {'pos': (x,y) or None}

    def _init_tracks(self, detections: List[Tuple[float, float]]):
        # Assign IDs 0..expected_objects-1 based on first frame detections
        mapping = {}
        num_init = min(len(detections), self.expected_objects)
        for det_idx in range(num_init):
            self.track_states[det_idx] = {'pos': detections[det_idx]}
            mapping[det_idx] = det_idx
        # If fewer detections than expected, create empty tracks
        for tid in range(num_init, self.expected_objects):
            self.track_states[tid] = {'pos': None}
        return mapping

    def _build_cost(self, detections: List[Tuple[float, float]]):
        # Cost matrix shape: tracks x detections
        if not detections:
            return None
        cost = np.full((self.expected_objects, len(detections)), fill_value=self.max_distance * 10, dtype=np.float32)
        for tid in range(self.expected_objects):
            tpos = self.track_states[tid]['pos']
            if tpos is None:
                continue
            for did, dpos in enumerate(detections):
                dx = tpos[0] - dpos[0]
                dy = tpos[1] - dpos[1]
                dist = np.sqrt(dx * dx + dy * dy)
                cost[tid, did] = dist
        return cost

    def assign_ids(self) -> Dict[Tuple[int, int], int]:
        """
        Returns mapping {(frame_idx, obj_idx): global_id}
        obj_idx corresponds to index in self.list_of_labels[frame_idx]
        """
        if self.expected_objects is None or self.expected_objects <= 0:
            return {}

        temporal_map: Dict[Tuple[int, int], int] = {}

        if not self.list_of_labels:
            return temporal_map

        # Initialize with first frame
        first_dets = self.list_of_labels[0] if len(self.list_of_labels) > 0 else []
        init_map = self._init_tracks(first_dets)
        for det_idx, gid in init_map.items():
            temporal_map[(0, det_idx)] = gid

        # Process subsequent frames
        for f_idx in range(1, len(self.list_of_labels)):
            dets = self.list_of_labels[f_idx]
            if not dets:
                # No detections, keep track states
                continue

            cost = self._build_cost(dets)
            if cost is None:
                continue

            row_ind, col_ind = linear_sum_assignment(cost)

            # Assign matches within max_distance
            assigned_dets = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= self.max_distance:
                    self.track_states[r]['pos'] = dets[c]
                    temporal_map[(f_idx, c)] = r
                    assigned_dets.add(c)

            # Unassigned tracks keep previous position
            # Unassigned detections: assign to nearest available track with None pos
            for det_idx, det in enumerate(dets):
                if det_idx in assigned_dets:
                    continue
                # find a free track (pos None)
                free_tid = None
                for tid in range(self.expected_objects):
                    if tid not in temporal_map.values() and self.track_states[tid]['pos'] is None:
                        free_tid = tid
                        break
                if free_tid is not None:
                    self.track_states[free_tid]['pos'] = det
                    temporal_map[(f_idx, det_idx)] = free_tid
                # else drop this detection

        return temporal_map


class PreprocessLabel:
    """
    Combine multiview matching and temporal tracking to assign global IDs
    to objects across multiple cameras and frames.
    
    Process:
    1. For each frame: match objects across cameras (multiview matching)
    2. Track matched objects across frames (temporal tracking)
    3. Output: final global_id for each (frame_idx, camera_name, local_id)
    """
    
    def __init__(self, calib_folder: str, max_distance_multiview: float = 2.0, 
                 max_distance_tracking: float = 50.0, n_clusters: int = None):
        """
        Args:
            calib_folder: Path to calibration folder containing intrinsic/extrinsic
            max_distance_multiview: Maximum distance in BEV space for multiview clustering (meters) - deprecated, not used with KMeans
            max_distance_tracking: Maximum distance in BEV space for temporal tracking (meters)
            n_clusters: Number of clusters for KMeans = fixed number of objects in the scene.
                       For offline tracking, this should be set to the exact number of objects.
                       KMeans will create exactly n_clusters clusters (some may be empty if fewer detections).
                       If None, will be auto-determined from number of detections (not recommended for offline tracking)
        """
        self.calib_folder = calib_folder
        self.max_distance_multiview = max_distance_multiview
        self.max_distance_tracking = max_distance_tracking
        self.n_clusters = n_clusters
        
        # Initialize multiview matcher
        self.multiview_matcher = MultiCameraTracking(
            calib_folder=calib_folder,
            max_distance=max_distance_multiview,
            n_clusters=n_clusters
        )
        
        # Final mapping: (frame_idx, camera_name, local_id) -> final_global_id
        self.final_global_id_map = {}
        
    def preprocess(self, multi_camera_detections: Dict[str, List[List[List[float]]]]) -> Dict[Tuple[int, str, int], int]:
        """
        Process multi-camera detections across frames to assign final global IDs
        
        Args:
            multi_camera_detections: Dict mapping camera_name -> list of frames,
                                   each frame is list of bboxes [x_min, y_min, x_max, y_max]
            
        Returns:
            Dict mapping (frame_idx, camera_name, local_id) -> final_global_id
        """
        if not multi_camera_detections:
            return {}
        
        # Get number of frames from first camera
        camera_names = list(multi_camera_detections.keys())
        num_frames = len(multi_camera_detections[camera_names[0]])
        
        # Step 1: Multiview matching for each frame
        # frame_multiview_map: {frame_idx: {(camera_name, local_id): frame_global_id}}
        frame_multiview_maps = {}
        # frame_bev_coords: {frame_idx: {frame_global_id: (x, y)}} - centroid of cluster
        frame_bev_coords = {}
        
        print(f"Step 1: Multiview matching for {num_frames} frames...")
        for frame_idx in range(num_frames):
            # Get detections for this frame
            frame_detections = {}
            for camera_name in camera_names:
                if frame_idx < len(multi_camera_detections[camera_name]):
                    frame_detections[camera_name] = multi_camera_detections[camera_name][frame_idx]
                else:
                    frame_detections[camera_name] = []
            
            # Match multiview for this frame
            multiview_map = self.multiview_matcher.match_multiview(frame_detections)
            frame_multiview_maps[frame_idx] = multiview_map
            
            # Collect BEV coordinates for each frame_global_id (centroid of cluster)
            frame_global_id_coords = {}  # {frame_global_id: (x, y)}
            
            # Group by frame_global_id to compute centroids
            frame_global_id_points = {}  # {frame_global_id: [(x, y), ...]}
            
            for (camera_name, local_id), frame_global_id in multiview_map.items():
                bev_coord = self.multiview_matcher.get_bev_coords(camera_name, local_id)
                if bev_coord is not None:
                    x, y = bev_coord
                    if np.isfinite(x) and np.isfinite(y):
                        if frame_global_id not in frame_global_id_points:
                            frame_global_id_points[frame_global_id] = []
                        frame_global_id_points[frame_global_id].append((x, y))
            
            # Compute centroids for each frame_global_id
            for frame_global_id, points in frame_global_id_points.items():
                if points:
                    points_array = np.array(points)
                    centroid = np.mean(points_array, axis=0)
                    # Round centroid to ensure consistency (~1 cm precision)
                    centroid_rounded = (round(float(centroid[0]), 2), round(float(centroid[1]), 2))
                    frame_global_id_coords[frame_global_id] = centroid_rounded
            
            # For offline tracking: ensure we have exactly n_clusters clusters.
            # With KMeans, n_clusters is fixed and KMeans always creates exactly n_clusters clusters.
            # However, after computing and rounding centroids, there may be fewer clusters due to duplicates.
            # In rare cases there may appear to be more clusters; we guard against that here.
            # If there are more than n_clusters clusters, merge the closest ones.
            # If there are fewer than n_clusters clusters, make sure cluster IDs are still numbered from 0 to n_clusters-1.
            if self.n_clusters is not None:
                num_current_clusters = len(frame_global_id_coords)
                
                # KMeans always creates exactly n_clusters clusters, but after centroid rounding
                # we may end up with fewer due to duplicates (or more in edge cases).
                if num_current_clusters > self.n_clusters:
                    if frame_idx == 0:
                        print(f"  Frame {frame_idx}: Found {num_current_clusters} clusters, merging to {self.n_clusters}...")
                    
                    # Merge closest clusters until we have exactly n_clusters
                    merge_count = 0
                    while len(frame_global_id_coords) > self.n_clusters:
                        # Find the two closest centroids
                        coords_list = list(frame_global_id_coords.items())
                        min_dist = float('inf')
                        merge_pair = None
                        
                        for i in range(len(coords_list)):
                            for j in range(i + 1, len(coords_list)):
                                id1, coord1 = coords_list[i]
                                id2, coord2 = coords_list[j]
                                dist = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
                                if dist < min_dist:
                                    min_dist = dist
                                    merge_pair = (id1, id2)
                        
                        # Merge the closest pair
                        if merge_pair:
                            id1, id2 = merge_pair

                        # Before merging, check constraint: the merged cluster
                        # must not contain two detections from the same camera.
                            cams_in_id1 = set()
                            cams_in_id2 = set()
                            for (cam_name, local_id), frame_global_id in multiview_map.items():
                                if frame_global_id == id1:
                                    cams_in_id1.add(cam_name)
                                elif frame_global_id == id2:
                                    cams_in_id2.add(cam_name)

                            # If the two clusters share any camera, merging would violate the constraint -> skip
                            if cams_in_id1 & cams_in_id2:
                                # No valid pairs remain that satisfy the camera-uniqueness constraint
                                break

                            # Merge all detections from id2 into id1
                            for (cam_name, local_id), frame_global_id in list(multiview_map.items()):
                                if frame_global_id == id2:
                                    multiview_map[(cam_name, local_id)] = id1
                            
                            # Update centroids: remove id2, keep id1
                            # Recompute centroid for id1 with all points from both clusters
                            merged_points = []
                            for (cam_name, local_id), fgid in multiview_map.items():
                                if fgid == id1:
                                    bev_coord = self.multiview_matcher.get_bev_coords(cam_name, local_id)
                                    if bev_coord is not None:
                                        x, y = bev_coord
                                        if np.isfinite(x) and np.isfinite(y):
                                            merged_points.append((x, y))
                            
                            if merged_points:
                                points_array = np.array(merged_points)
                                centroid = np.mean(points_array, axis=0)
                                # Round centroid again to ensure consistency
                                centroid_rounded = (round(float(centroid[0]), 2), round(float(centroid[1]), 2))
                                frame_global_id_coords[id1] = centroid_rounded
                            
                            del frame_global_id_coords[id2]
                            frame_multiview_maps[frame_idx] = multiview_map
                            merge_count += 1
                        else:
                            break  # No more pairs to merge
                    
                    if frame_idx == 0 and merge_count > 0:
                        print(f"  Merged {merge_count} clusters in frame {frame_idx}")
                
                # Ensure cluster IDs are numbered from 0 to n_clusters-1.
                # Remap frame_global_id to keep IDs contiguous from 0 up to n_clusters-1.
                # If there are fewer clusters than n_clusters, some IDs will be unused,
                # which is acceptable for temporal tracking.
                if len(frame_global_id_coords) > 0:
                    # Create mapping from old frame_global_id to new (0 .. num_actual_clusters-1)
                    old_ids = sorted(frame_global_id_coords.keys())
                    num_actual_clusters = len(old_ids)
                    # Remap so that IDs range from 0 to num_actual_clusters-1.
                    # If num_actual_clusters < n_clusters, some IDs will remain unused.
                    id_remap = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
                    
                    # Remap trong multiview_map
                    new_multiview_map = {}
                    for (cam_name, local_id), old_frame_global_id in multiview_map.items():
                        new_frame_global_id = id_remap.get(old_frame_global_id, old_frame_global_id)
                        new_multiview_map[(cam_name, local_id)] = new_frame_global_id
                    multiview_map = new_multiview_map
                    
                    # Remap trong frame_global_id_coords
                    new_frame_global_id_coords = {}
                    for old_id, coord in frame_global_id_coords.items():
                        new_id = id_remap.get(old_id, old_id)
                        new_frame_global_id_coords[new_id] = coord
                    frame_global_id_coords = new_frame_global_id_coords
                    
                    frame_multiview_maps[frame_idx] = multiview_map
                    
                    # Log a warning if we ended up with fewer clusters than n_clusters
                    if num_actual_clusters < self.n_clusters and frame_idx == 0:
                        print(f"  Frame {frame_idx}: Found {num_actual_clusters} clusters (expected {self.n_clusters})")
            
            frame_bev_coords[frame_idx] = frame_global_id_coords
            
            if (frame_idx + 1) % 10 == 0:
                print(f"  Processed {frame_idx + 1}/{num_frames} frames...")
        
        print(f"Multiview matching completed. Found {sum(len(m) for m in frame_multiview_maps.values())} mappings.")
        
        # Step 2: Temporal tracking of frame_global_ids across frames
        print("Step 2: Temporal tracking across frames...")
        
        # Prepare data for ObjectTracking: list_of_labels[frame_idx] = [(x, y), ...]
        # We need to map frame_global_id to index in the list
        # For offline tracking with fixed number of objects, ensure consistent ordering
        list_of_labels = []  # List of frames, each frame is list of (x, y) coordinates
        frame_global_id_to_idx = []  # For each frame: {frame_global_id: idx_in_list}
        
        # Get expected number of objects from n_clusters
        expected_objects = self.n_clusters
        
        for frame_idx in range(num_frames):
            coords = frame_bev_coords.get(frame_idx, {})
            # Sort by frame_global_id to ensure consistent ordering
            sorted_ids = sorted(coords.keys())
            if sorted_ids:
                frame_labels = [coords[fid] for fid in sorted_ids]
            else:
                # Empty frame - no objects detected
                frame_labels = []
            
            # For offline tracking: if we have fewer objects than expected, 
            # trackpy will handle it (some objects might not be detected in this frame)
            list_of_labels.append(frame_labels)
            
            # Create mapping: frame_global_id -> idx
            frame_id_map = {fid: idx for idx, fid in enumerate(sorted_ids)}
            frame_global_id_to_idx.append(frame_id_map)
        
        # Use FixedIDTracker to ensure fixed number of IDs and minimal ID switches
        if expected_objects is not None and expected_objects > 0:
            tracker = FixedIDTracker(
                list_of_labels=list_of_labels,
                expected_objects=expected_objects,
                max_distance=self.max_distance_tracking
            )
            temporal_global_id_map = tracker.assign_ids()
        else:
            object_tracker = ObjectTracking(
                list_of_labels=list_of_labels,
                max_distance=self.max_distance_tracking
            )
            temporal_global_id_map = object_tracker.assign_global_id()
        # temporal_global_id_map: {(frame_idx, obj_idx): final_global_id}
        
        # Verify ID consistency: check if we have the expected number of unique IDs
        unique_ids = set(temporal_global_id_map.values())
        if expected_objects is not None:
            if len(unique_ids) > expected_objects:
                print(f"WARNING: Found {len(unique_ids)} unique IDs but expected {expected_objects} objects")
            elif len(unique_ids) < expected_objects:
                print(f"INFO: Found {len(unique_ids)} unique IDs (expected {expected_objects}, some objects may not appear in all frames)")
        
        print(f"Temporal tracking completed. Found {len(temporal_global_id_map)} temporal mappings with {len(unique_ids)} unique IDs.")
        
        # Step 3: Build final mapping: (frame_idx, camera_name, local_id) -> final_global_id
        print("Step 3: Building final global ID mapping...")
        
        self.final_global_id_map = {}
        missing_ids_count = 0
        
        for frame_idx in range(num_frames):
            multiview_map = frame_multiview_maps[frame_idx]
            frame_id_to_idx = frame_global_id_to_idx[frame_idx]
            
            for (camera_name, local_id), frame_global_id in multiview_map.items():
                # Get index in the list for this frame_global_id
                obj_idx = frame_id_to_idx.get(frame_global_id)
                if obj_idx is not None:
                    # Get final_global_id from temporal tracking
                    final_global_id = temporal_global_id_map.get((frame_idx, obj_idx), -1)
                    if final_global_id >= 0:
                        self.final_global_id_map[(frame_idx, camera_name, local_id)] = final_global_id
                    else:
                        # obj_idx not in temporal tracking - assign frame_global_id as fallback
                        # This can happen if the object was not tracked across frames
                        self.final_global_id_map[(frame_idx, camera_name, local_id)] = frame_global_id
                        missing_ids_count += 1
                else:
                    # frame_global_id not in frame_id_to_idx - this means no BEV coord was computed
                    # Use frame_global_id directly as final_global_id (fallback)
                    self.final_global_id_map[(frame_idx, camera_name, local_id)] = frame_global_id
                    missing_ids_count += 1
        
        if missing_ids_count > 0:
            print(f"WARNING: {missing_ids_count} boxes were assigned fallback IDs (not in temporal tracking)")
        print(f"Final mapping completed. Total mappings: {len(self.final_global_id_map)}")
        
        return self.final_global_id_map
    
    def get_global_id(self, frame_idx: int, camera_name: str, local_id: int) -> int:
        """Get final global ID for a local object"""
        return self.final_global_id_map.get((frame_idx, camera_name, local_id), -1)
    
    def get_all_global_ids(self) -> Dict[Tuple[int, str, int], int]:
        """Get all final global ID mappings"""
        return self.final_global_id_map.copy()
