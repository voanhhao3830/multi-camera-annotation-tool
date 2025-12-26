from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from .project_bev import BBoxToBEVConverter
from ultralytics import YOLO


class MultiCameraTracking:
    """Multi-camera object mapping (multiview matching only, no temporal tracking)"""
    
    def __init__(self, calib_folder: str, model_name: str = None, model_path: str = None, 
                 data_path: str = None, max_distance: float = 2.0, n_clusters: int = None,
                 intrinsic_scale: float = 0.25, translation_scale: float = 100.0,
                 bev_x: int = None, bev_y: int = None, bev_bounds: list = None):
        """
        Args:
            calib_folder: Path to calibration folder containing intrinsic/extrinsic
            model_name: Model name for detection (optional)
            model_path: Path to model file (optional)
            data_path: Path to data (optional)
            max_distance: Maximum distance in BEV space for clustering objects (meters) - deprecated, not used with KMeans
            n_clusters: Number of clusters for KMeans. If None, will be auto-determined from number of detections
            intrinsic_scale: Scale factor for intrinsic matrix (default 0.25 = 1/4)
            translation_scale: Scale factor for translation vector (default 100.0)
            bev_x: BEV grid X dimension (default from constants)
            bev_y: BEV grid Y dimension (default from constants)
            bev_bounds: BEV bounds [XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX] (default from constants)
        """
        from labelme.utils.constants import DEFAULT_INTRINSIC_SCALE, DEFAULT_TRANSLATION_SCALE
        from labelme.utils.constants import DEFAULT_BEV_X, DEFAULT_BEV_Y, DEFAULT_BEV_BOUNDS
        
        self.calib_folder = calib_folder
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path
        self.max_distance = max_distance
        self.n_clusters = n_clusters
        self.model = None
        
        # Use provided scale factors or defaults from constants
        intrinsic_scale = intrinsic_scale if intrinsic_scale is not None else DEFAULT_INTRINSIC_SCALE
        translation_scale = translation_scale if translation_scale is not None else DEFAULT_TRANSLATION_SCALE
        bev_x = bev_x if bev_x is not None else DEFAULT_BEV_X
        bev_y = bev_y if bev_y is not None else DEFAULT_BEV_Y
        bev_bounds = bev_bounds if bev_bounds is not None else DEFAULT_BEV_BOUNDS.copy()
        
        # Initialize BEV converter with scale factors
        self.bev_converter = BBoxToBEVConverter(
            calib_folder,
            intrinsic_scale=intrinsic_scale,
            translation_scale=translation_scale,
            bev_x=bev_x,
            bev_y=bev_y,
            bev_bounds=bev_bounds
        )
        
        # Mapping results (for single frame)
        self.global_id_map = {}  # {(camera_name, local_id): global_id}
        self.local_id_map = {}   # {(camera_name, global_id): local_id}
        self.bev_coords = {}     # {(camera_name, local_id): (x, y)}
        
    def load_model(self):
        """Load detection model"""
        if self.model_name == "yolo":
            self.model = YOLO(self.model_path)
        else:
            raise ValueError(f"Model {self.model_name} not supported")

    def load_data(self):
        """Load data - to be implemented based on data format"""
        pass
    
    def project_bev(self, detections: Dict[str, List[List[float]]]) -> Dict[str, List[Optional[Tuple[float, float]]]]:
        """
        Project bounding boxes from all cameras to BEV coordinates
        
        Args:
            detections: Dict mapping camera_name -> list of bboxes [x_min, y_min, x_max, y_max]
            
        Returns:
            Dict mapping camera_name -> list of BEV coordinates (x, y) or None
        """
        bev_results = {}
        
        for camera_name, bboxes in detections.items():
            bev_coords_list = []
            for bbox in bboxes:
                result = self.bev_converter.bbox_to_bev(camera_name, bbox, z_world=0.0)
                if result['success']:
                    bev_coords_list.append(result['bev_coords'])
                else:
                    bev_coords_list.append(None)
            bev_results[camera_name] = bev_coords_list
        
        return bev_results
    
    def _cluster_multiview_detections(
            self,
            bev_coords: List[Tuple[float, float, str, int]],
            prev_centroids: Optional[np.ndarray] = None,
            max_centroid_movement: float = 5.0
        ) -> Dict[int, List[Tuple[str, int]]]:
        """
        Cluster BEV coordinates with a hard constraint:
        - **Ràng buộc quan trọng**: Một cluster **không được** chứa 2 detections đến từ **cùng một camera**.
        - Dùng KMeans để lấy các centroid ban đầu, sau đó gán lại greedily theo khoảng cách,
          đảm bảo mỗi (cluster, camera) xuất hiện tối đa 1 lần.
        - Nếu một detection không thể được gán vào bất kỳ cluster KMeans nào mà vẫn giữ được ràng buộc,
          nó sẽ được đưa vào một cluster mới (singleton cluster), vẫn đảm bảo ràng buộc theo camera.
        
        Args:
            bev_coords: List of (x, y, camera_name, local_id)
            prev_centroids: Previous frame's cluster centers for warm start (shape: [n_clusters, 2])
            max_centroid_movement: Maximum allowed movement of cluster centers in meters (for stability)
        """
        if not bev_coords:
            return {}

        # Prepare data
        all_coords = [(x, y) for x, y, _, _ in bev_coords]
        coord_to_camera = [(cam, det_idx) for _, _, cam, det_idx in bev_coords]
        coords_array = np.array(all_coords, dtype=np.float32)

        n_points = len(coords_array)
        all_cams = {cam for cam, _ in coord_to_camera}
        size_limit = max(1, len(all_cams))  # maximum detections per cluster is #cams

        # Determine initial KMeans cluster count
        if self.n_clusters is None:
            # Rough estimate: each object appears in approximately 3 cameras
            estimated_clusters = max(1, n_points // 3)
            n_clusters = min(estimated_clusters, n_points)
        else:
            n_clusters = min(max(1, self.n_clusters), n_points)

        if n_clusters <= 0:
            return {}

        # KMeans to get centroids with warm start from previous frame
        if prev_centroids is not None and len(prev_centroids) == n_clusters:
            # Use previous centroids as initialization
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, init=prev_centroids, random_state=42)
            kmeans.fit(coords_array)
            centroids = kmeans.cluster_centers_
            
            # Apply constraint: limit centroid movement distance
            for i in range(len(centroids)):
                if i < len(prev_centroids):
                    dist = np.linalg.norm(centroids[i] - prev_centroids[i])
                    if dist > max_centroid_movement:
                        # If moved too far, keep old position and adjust slightly
                        direction = (centroids[i] - prev_centroids[i]) / dist
                        centroids[i] = prev_centroids[i] + direction * max_centroid_movement
        else:
            # No previous centroids or count mismatch, use random init
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            kmeans.fit(coords_array)
            centroids = kmeans.cluster_centers_

        # Calculate distances to centroids
        dists = np.linalg.norm(coords_array[:, None, :] - centroids[None, :, :], axis=2)

        # Detection order: closest to centroid first
        nearest_centroid_dist = np.min(dists, axis=1)
        detection_order = list(np.argsort(nearest_centroid_dist))

        clusters: Dict[int, List[int]] = {i: [] for i in range(n_clusters)}
        cluster_cameras: Dict[int, set] = {i: set() for i in range(n_clusters)}
        unassigned: List[int] = []

        # Assign: nearest cluster, NO same camera, not exceeding size_limit
        for idx in detection_order:
            cam_name, _ = coord_to_camera[idx]
            centroid_candidates = list(np.argsort(dists[idx]))
            assigned = False
            for c in centroid_candidates:
                if len(clusters[c]) >= size_limit:
                    continue
                if cam_name in cluster_cameras[c]:
                    continue  # constraint: 1 cluster cannot contain 2 objects from same camera
                clusters[c].append(idx)
                cluster_cameras[c].add(cam_name)
                assigned = True
                break

            if not assigned:
                unassigned.append(idx)

        # For unassigned detections, create new cluster (singleton) to maintain camera constraint
        next_cluster_id = n_clusters
        for idx in unassigned:
            cam_name, _ = coord_to_camera[idx]
            clusters[next_cluster_id] = [idx]
            cluster_cameras[next_cluster_id] = {cam_name}
            next_cluster_id += 1

        # Build final result: {cluster_id: [(camera_name, local_id), ...]}
        result: Dict[int, List[Tuple[str, int]]] = {}
        for cid, point_indices in clusters.items():
            if not point_indices:
                continue
            members: List[Tuple[str, int]] = []
            for idx in point_indices:
                cam_name, det_idx = coord_to_camera[idx]
                members.append((cam_name, det_idx))
            result[cid] = members

        return result

    
    def match_multiview(self, detections: Dict[str, List[List[float]]], 
                       prev_centroids: Optional[np.ndarray] = None,
                       max_centroid_movement: float = 5.0) -> Dict[Tuple[str, int], int]:
        """
        Match objects across multiple cameras in a single frame using K-means clustering
        
        Args:
            detections: Dict mapping camera_name -> list of bboxes [x_min, y_min, x_max, y_max]
            prev_centroids: Previous frame's cluster centers for warm start (shape: [n_clusters, 2])
            max_centroid_movement: Maximum allowed movement of cluster centers in meters (for stability)
            
        Returns:
            Dict mapping (camera_name, local_id) -> global_id
        """
        if not detections:
            return {}
        
        # Reset mappings for new frame
        self.global_id_map = {}
        self.local_id_map = {}
        self.bev_coords = {}
        
        # Project to BEV
        bev_results = self.project_bev(detections)
        
        # Collect valid BEV coordinates
        bev_coords_list = []  # List of (x, y, camera_name, local_id)
        invalid_detections = []  # List of (camera_name, local_id) with invalid BEV
        
        for camera_name, bev_coords in bev_results.items():
            for local_id, bev_coord in enumerate(bev_coords):
                if bev_coord is not None:
                    x, y = bev_coord
                    # Check for NaN or Inf values
                    if np.isfinite(x) and np.isfinite(y):
                        bev_coords_list.append((x, y, camera_name, local_id))
                        # Store for later lookup
                        self.bev_coords[(camera_name, local_id)] = (x, y)
                    else:
                        # Invalid BEV coordinate
                        invalid_detections.append((camera_name, local_id))
                else:
                    # BEV projection failed
                    invalid_detections.append((camera_name, local_id))
        
        # Cluster BEV coordinates
        if bev_coords_list:
            clusters = self._cluster_multiview_detections(bev_coords_list, prev_centroids, max_centroid_movement)
            next_global_id = 0

            # Assign consecutive global_id to each cluster
            for cluster_id in sorted(clusters.keys()):
                members = clusters[cluster_id]
                if not members:
                    continue
                global_id = next_global_id
                next_global_id += 1
                for camera_name, local_id in members:
                    self.global_id_map[(camera_name, local_id)] = global_id
                    self.local_id_map[(camera_name, global_id)] = local_id
        else:
            next_global_id = 0
        
        # Assign separate IDs to invalid detections (those with failed BEV projection)
        # Each invalid detection gets its own unique ID after reserved cluster IDs
        for camera_name, local_id in invalid_detections:
            self.global_id_map[(camera_name, local_id)] = next_global_id
            next_global_id += 1

        if not self.global_id_map:
            # Don't print warning here - it will be logged at higher level if needed
            return {}
        
        return self.global_id_map
    
    def get_global_id(self, camera_name: str, local_id: int) -> int:
        """Get global ID for a local object"""
        return self.global_id_map.get((camera_name, local_id), -1)
    
    def get_local_id(self, camera_name: str, global_id: int) -> Optional[int]:
        """Get local ID for a global object in a specific camera"""
        return self.local_id_map.get((camera_name, global_id))
    
    def get_bev_coords(self, camera_name: str, local_id: int) -> Optional[Tuple[float, float]]:
        """Get BEV coordinates for an object"""
        return self.bev_coords.get((camera_name, local_id))
    
    def get_cluster_centroids(self) -> Optional[np.ndarray]:
        """
        Get current frame's cluster centroids after clustering.
        Returns centroids as numpy array of shape [n_clusters, 2] or None if not available.
        """
        if not self.global_id_map:
            return None
        
        # Group BEV coordinates by global_id
        global_id_points = {}  # {global_id: [(x, y), ...]}
        for (camera_name, local_id), global_id in self.global_id_map.items():
            bev_coord = self.get_bev_coords(camera_name, local_id)
            if bev_coord is not None:
                x, y = bev_coord
                if np.isfinite(x) and np.isfinite(y):
                    if global_id not in global_id_points:
                        global_id_points[global_id] = []
                    global_id_points[global_id].append((x, y))
        
        if not global_id_points:
            return None
        
        # Compute centroids for each global_id
        centroids = []
        sorted_ids = sorted(global_id_points.keys())
        for global_id in sorted_ids:
            points = global_id_points[global_id]
            if points:
                points_array = np.array(points)
                centroid = np.mean(points_array, axis=0)
                centroids.append(centroid)
        
        if not centroids:
            return None
        
        return np.array(centroids, dtype=np.float32)
    
    def map_to_local_ids(self, camera_name: str, global_ids: List[int]) -> Dict[int, int]:
        """
        Map global IDs to local IDs for a specific camera
        
        Args:
            camera_name: Camera name
            global_ids: List of global IDs to map
            
        Returns:
            Dict mapping global_id -> local_id
        """
        result = {}
        for global_id in global_ids:
            local_id = self.get_local_id(camera_name, global_id)
            if local_id is not None:
                result[global_id] = local_id
        return result