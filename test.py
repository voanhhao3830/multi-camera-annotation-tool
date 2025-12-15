"""
Test script for MultiCameraTracking with visualization
Similar to visualize_clustering.py but using MultiCameraTracking
"""
import os
import sys
import glob
import math
import json
import shutil
import tempfile
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from labelme.utils.multicamera_matching import MultiCameraTracking
from labelme.utils.calibration import CameraCalibration

# Configuration - Update these paths
DATA_ROOT = r"D:\MultiCameraLabelTool\aicv\aicv"
IMAGE_SUBSETS_FOLDER = os.path.join(DATA_ROOT, "Image_subsets")  # contains subfolders "1"..."6"
ANNOTATIONS_FOLDER = os.path.join(DATA_ROOT, "annotations_positions")  # per-frame JSON boxes
CALIBRATIONS_ROOT = os.path.join(DATA_ROOT, "calibrations")  # contains CameraX/calibrations/...
CAMERAS = [f"Camera{i}" for i in range(1, 7)]  # matches calibration file names

# Mapping viewNum in annotations to camera names and image folders
# Note: viewNum might be 0-indexed or 1-indexed, so we support both
VIEW_TO_CAMERA = {}
# 1-indexed (viewNum 1-6 -> Camera1-Camera6)
for i in range(1, 7):
    VIEW_TO_CAMERA[i] = f"Camera{i}"
# 0-indexed (viewNum 0-5 -> Camera1-Camera6) - in case annotations use 0-based indexing
for i in range(6):
    VIEW_TO_CAMERA[i] = f"Camera{i+1}"

CAMERA_TO_VIEW_FOLDER = {f"Camera{i}": str(i) for i in range(1, 7)}

# Structure expected:
# DATA_ROOT/
#   Image_subsets/
#     1/00000.jpg ...
#     2/...
#     ...
#   calibrations/
#     Camera1/calibrations/intrinsic_original/intr_Camera1.xml
#     Camera1/calibrations/extrinsic/extr_Camera1.xml
#   annotations_positions/
#     00000.json  (list of people with per-view boxes)

# BEV visualization parameters
BEV_SIZE = 1000  # Increased for better visibility
BEV_SCALE = 150  # Default scale (will be auto-adjusted based on data)
BEV_ORIGIN = (BEV_SIZE // 2, BEV_SIZE // 2)

# Detection and tracking parameters
CONFIDENCE_THRESHOLD = 0.5
MAX_DISTANCE = 1  # Maximum distance in BEV space for matching (meters)
AVERAGE_OBJECT_HEIGHT = 10
MIN_DIST = 20.0
MAX_DIST = 40.0
NUM_CLUSTERS = 10  # Number of clusters for K-means (None = auto-detect from data, or set to specific number)

CAMERA_COLOR_PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def world_to_bev_pixel(x, y, origin, scale):
    """Convert world coordinates to BEV pixel coordinates"""
    px = int(origin[0] + x * scale)
    py = int(origin[1] - y * scale)
    return px, py


def draw_bev_grid(img, origin, scale, step=1.0):
    """Draw grid on BEV image"""
    h, w = img.shape[:2]
    max_range = max(w, h) // scale

    for i in range(-int(max_range), int(max_range) + 1):
        x = int(origin[0] + i * step * scale)
        if 0 <= x < w:
            c = (150, 150, 150) if i % 5 == 0 else (100, 100, 100)
            cv2.line(img, (x, 0), (x, h), c, 1)
        y = int(origin[1] - i * step * scale)
        if 0 <= y < h:
            c = (150, 150, 150) if i % 5 == 0 else (100, 100, 100)
            cv2.line(img, (0, y), (w, y), c, 1)

    cv2.line(img, (origin[0], 0), (origin[0], h), (0, 255, 0), 2)
    cv2.line(img, (0, origin[1]), (w, origin[1]), (255, 0, 0), 2)
    cv2.circle(img, origin, 5, (0, 0, 255), -1)


def project_bbox_bottom_to_bev(tracker: MultiCameraTracking, camera_name: str, bbox: List[float]) -> Optional[Tuple[float, float]]:
    """
    Project the bbox bottom-center to world (ground) coordinates.
    Uses the calibration already loaded in tracker without the extra heuristics
    in BBoxToBEVConverter that can scatter per-camera points.
    """
    try:
        bev_projector = tracker.bev_converter.bev_projector
        if camera_name not in bev_projector.cameras:
            return None

        # Bottom-center of the box (assumed to touch ground)
        bottom_center = ((bbox[0] + bbox[2]) * 0.5, bbox[3])
        world_point = bev_projector.image_to_world(camera_name, bottom_center, z_world=0.0)
        if world_point is None or len(world_point) < 2:
            return None

        x, y = float(world_point[0]), float(world_point[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return x, y
    except Exception as e:
        print(f"  WARNING: BEV projection failed for {camera_name}: {e}")
        return None


def cluster_bev_points(tracker: MultiCameraTracking, bev_points: List[Tuple[float, float, str, int]], 
                       num_clusters: Optional[int] = None) -> Dict[int, List[Tuple[str, int]]]:
    """
    Cluster BEV coordinates using K-means with fixed number of clusters.
    
    Args:
        tracker: MultiCameraTracking instance (not used but kept for compatibility)
        bev_points: List of (x, y, camera_name, local_id)
        num_clusters: Number of clusters (if None, will be determined from data)
    
    Returns:
        Dict mapping cluster_id -> list of (camera_name, local_id)
    """
    if not bev_points:
        return {}
    
    from sklearn.cluster import KMeans
    
    # Extract only xy coordinates
    coords = np.array([[x, y] for x, y, _, _ in bev_points], dtype=np.float32)
    
    # Determine number of clusters
    if num_clusters is None:
        # Auto-detect: use number of unique persons in annotations if available
        # For now, use number of points (each point is a detection)
        n_points = len(bev_points)
        # Estimate: assume each person appears in multiple cameras
        # Rough estimate: divide by average cameras per person (e.g., 2-3)
        estimated_clusters = max(1, n_points // 3)
        num_clusters = estimated_clusters
        print(f"  Auto-detected {num_clusters} clusters from {n_points} detections")
    else:
        num_clusters = min(num_clusters, len(bev_points))  # Can't have more clusters than points
        print(f"  Using {num_clusters} clusters for {len(bev_points)} detections")
    
    if num_clusters <= 0:
        return {}
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    
    # Group by cluster
    clusters: Dict[int, List[Tuple[str, int]]] = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        _, _, cam_name, local_id = bev_points[idx]
        clusters[label].append((cam_name, local_id))
    
    return clusters


def load_ground_truth_detections(frame_idx: int, annotations_folder: str, cameras: List[str]) -> Tuple[Dict[str, List[List[float]]], Dict[Tuple[str, int], int]]:
    """
    Load ground-truth boxes from annotations_positions instead of running detection.
    
    Returns:
        detections: Dict mapping camera -> list of bboxes
        person_id_map: Dict mapping (camera, local_id) -> personID from JSON
    """
    detections: Dict[str, List[List[float]]] = {cam: [] for cam in cameras}
    person_id_map: Dict[Tuple[str, int], int] = {}  # (camera, local_id) -> personID
    
    ann_path = os.path.join(annotations_folder, f"{frame_idx:05d}.json")
    if not os.path.exists(ann_path):
        print(f"  WARNING: Annotation missing for frame {frame_idx}: {ann_path}")
        return detections, person_id_map

    with open(ann_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  WARNING: Failed to parse annotation for frame {frame_idx}: {e}")
            return detections, person_id_map

    if not isinstance(data, list):
        print(f"  WARNING: Annotation format error for frame {frame_idx}: expected list, got {type(data)}")
        return detections, person_id_map

    total_persons = len(data)
    total_views_found = 0
    total_views_valid = 0
    skipped_invalid_bbox = 0
    skipped_no_camera = 0

    for person_idx, person in enumerate(data):
        person_id = person.get("personID", person_idx)  # Use personID if available, else use index
        views = person.get("views", [])
        if not views:
            continue
        
        for view in views:
            view_num = view.get("viewNum")
            
            # Try to convert to int if it's a string
            if isinstance(view_num, str):
                try:
                    view_num = int(view_num)
                except ValueError:
                    pass
            
            cam_name = VIEW_TO_CAMERA.get(view_num)
            
            if cam_name is None:
                skipped_no_camera += 1
                if view_num is not None:
                    print(f"    [DEBUG] Unknown viewNum: {view_num} (type: {type(view_num)})")
                continue
            
            if cam_name not in detections:
                skipped_no_camera += 1
                continue

            xmin, ymin, xmax, ymax = (
                view.get("xmin", -1),
                view.get("ymin", -1),
                view.get("xmax", -1),
                view.get("ymax", -1),
            )
            
            total_views_found += 1
            
            # Skip invalid placeholders
            if min(xmin, ymin, xmax, ymax) < 0:
                skipped_invalid_bbox += 1
                continue
            
            # Validate bbox dimensions
            if xmax <= xmin or ymax <= ymin:
                skipped_invalid_bbox += 1
                continue
            
            # Add bbox and map to personID
            local_id = len(detections[cam_name])  # Current index will be the local_id
            detections[cam_name].append([float(xmin), float(ymin), float(xmax), float(ymax)])
            person_id_map[(cam_name, local_id)] = person_id
            total_views_valid += 1

    # Debug summary
    total_loaded = sum(len(d) for d in detections.values())
    print(f"[DEBUG Frame {frame_idx}] Loaded: {total_persons} persons, {total_views_found} views found, "
          f"{total_views_valid} valid bboxes loaded, {skipped_invalid_bbox} invalid bboxes, "
          f"{skipped_no_camera} views with no camera mapping")
    
    # Print per-camera counts
    for cam in cameras:
        count = len(detections[cam])
        if count > 0:
            print(f"  {cam}: {count} bboxes")

    return detections, person_id_map


def draw_bbox_on_image(image, bbox, obj_id, color, conf):
    """Draw bounding box on image"""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    label = f"ID:{obj_id} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
    cv2.putText(image, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_object_on_bev(img, bev_coords, global_id, camera_colors, cameras, origin, scale):
    """Draw object on BEV image"""
    # Draw points for each camera
    points_drawn = 0
    for cam_name, coord in zip(cameras, bev_coords):
        if coord is not None:
            x, y = coord
            px, py = world_to_bev_pixel(x, y, origin, scale)
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                color = camera_colors.get(cam_name, (255, 255, 255))
                cv2.circle(img, (px, py), 6, color, -1)
                cv2.circle(img, (px, py), 8, (255, 255, 255), 1)
                points_drawn += 1
            else:
                # Debug: print out-of-bounds points
                print(f"[DEBUG] Point out of bounds: ({x:.2f}, {y:.2f}) -> pixel ({px}, {py}), bounds: [0-{img.shape[1]}, 0-{img.shape[0]}]")
    
    # Calculate centroid
    valid_coords = [c for c in bev_coords if c is not None]
    if valid_coords:
        cx = np.mean([c[0] for c in valid_coords])
        cy = np.mean([c[1] for c in valid_coords])
        cpx, cpy = world_to_bev_pixel(cx, cy, origin, scale)
        if 0 <= cpx < img.shape[1] and 0 <= cpy < img.shape[0]:
            s = 15
            cv2.line(img, (cpx - s, cpy - s), (cpx + s, cpy + s), (255, 255, 255), 3)
            cv2.line(img, (cpx + s, cpy - s), (cpx - s, cpy + s), (255, 255, 255), 3)
            cv2.putText(img, f"ID:{global_id}", (cpx + 20, cpy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"{len(valid_coords)}cam", (cpx + 20, cpy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            print(f"[DEBUG] Centroid out of bounds: ({cx:.2f}, {cy:.2f}) -> pixel ({cpx}, {cpy})")
    
    if points_drawn == 0 and valid_coords:
        print(f"[WARNING] Object ID:{global_id} has {len(valid_coords)} coords but none drawn!")


def create_camera_grid(annotated_images, cameras, frame_idx=None):
    """Create grid of camera views (supports variable camera count)."""
    h, w = 360, 480
    tiles = []
    for cam in cameras:
        img = annotated_images.get(cam)
        if img is not None:
            tile = cv2.resize(img, (w, h))
        else:
            tile = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(tile, f"{cam} - No Image", (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(tile)

    # Pad to multiples of 3 for grid layout
    while len(tiles) % 3 != 0:
        tiles.append(np.zeros((h, w, 3), dtype=np.uint8))

    rows = []
    for i in range(0, len(tiles), 3):
        rows.append(np.hstack(tiles[i:i + 3]))
    cam_grid = np.vstack(rows) if rows else np.zeros((h, w, 3), dtype=np.uint8)
    
    # Add frame number to top-left corner
    if frame_idx is not None:
        cv2.putText(cam_grid, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return cam_grid


def create_bev_display(bev_img, matched_objects, object_colors, cameras, camera_colors):
    """Create BEV display with legend panel."""
    # Resize BEV to a reasonable size
    bev_height = 800
    bev_width = 800
    bev_resized = cv2.resize(bev_img, (bev_width, bev_height))
    
    # Create legend panel
    legend = create_legend_panel(matched_objects, object_colors, cameras, camera_colors, 300, bev_height)
    
    # Combine BEV and legend
    final = np.hstack([bev_resized, legend])
    return final


def create_legend_panel(matched_objects, object_colors, cameras, camera_colors, width, height):
    """Create legend panel"""
    panel = np.ones((height, width, 3), dtype=np.uint8) * 30
    y = 30
    
    # Camera colors
    cv2.putText(panel, "Camera Colors (BEV):", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 25
    for cam, color in camera_colors.items():
        cv2.circle(panel, (30, y - 5), 8, color, -1)
        cv2.putText(panel, cam, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25

    y += 20
    cv2.putText(panel, "Object Colors:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 25

    max_display = min(8, len(matched_objects))
    for i in range(max_display):
        obj_id, _ = matched_objects[i]
        color = object_colors.get(obj_id, (255, 255, 255))
        cv2.rectangle(panel, (20, y - 12), (40, y + 2), color, -1)
        cv2.putText(panel, f"ID:{obj_id}", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 20

    if len(matched_objects) > max_display:
        cv2.putText(panel, f"... +{len(matched_objects) - max_display} more", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    y = height - 100
    cv2.line(panel, (10, y), (width - 10, y), (100, 100, 100), 1)
    y += 20
    cv2.putText(panel, "STATISTICS", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 25
    stats = [f"Total objects: {len(matched_objects)}"]
    for t in stats:
        cv2.putText(panel, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 18

    return panel


def process_frame(frame_idx, image_files, cameras, annotations_folder, tracker):
    """Process a single frame for multiview matching"""
    images = {}
    detections = {}

    # Load images
    for cam in cameras:
        files = image_files.get(cam, [])
        if frame_idx < len(files):
            path = files[frame_idx]
            images[cam] = cv2.imread(path)
        else:
            images[cam] = None

    # Load GT detections (no model inference)
    detections, person_id_map = load_ground_truth_detections(frame_idx, annotations_folder, cameras)

    # Reset mapping buffers for this frame
    tracker.global_id_map = {}
    tracker.local_id_map = {}
    tracker.bev_coords = {}

    # Project all detections to BEV using bottom-center intersection
    bev_results = {cam: [] for cam in cameras}
    bev_points = []  # (x, y, cam, local_id)
    
    total_detections = 0
    successful_projections = 0

    for cam in cameras:
        bboxes = detections.get(cam, [])
        total_detections += len(bboxes)
        for local_id, bbox in enumerate(bboxes):
            coord = project_bbox_bottom_to_bev(tracker, cam, bbox)
            bev_results[cam].append(coord)
            if coord is not None:
                x, y = coord
                if np.isfinite(x) and np.isfinite(y):
                    bev_points.append((coord[0], coord[1], cam, local_id))
                    tracker.bev_coords[(cam, local_id)] = coord
                    successful_projections += 1
                else:
                    print(f"[WARNING] Invalid BEV coord for {cam} bbox {local_id}: ({x}, {y})")
            # Removed verbose debug - only print summary
    
    print(f"[DEBUG Frame {frame_idx}] Total detections: {total_detections}, Successful projections: {successful_projections}, BEV points: {len(bev_points)}")

    # Use personID from annotations to assign global IDs instead of clustering
    # This ensures that bboxes from the same person (same personID) get the same global_id
    person_to_global_id = {}  # personID -> global_id
    next_global_id = 0
    
    # First pass: assign global IDs based on personID for ALL detections (not just those with BEV projection)
    # This ensures every bbox gets a global_id based on its personID
    for cam_name in cameras:
        for local_id in range(len(detections.get(cam_name, []))):
            person_id = person_id_map.get((cam_name, local_id))
            if person_id is not None:
                if person_id not in person_to_global_id:
                    person_to_global_id[person_id] = next_global_id
                    next_global_id += 1
                global_id = person_to_global_id[person_id]
                tracker.global_id_map[(cam_name, local_id)] = global_id
                # Note: local_id_map can have multiple local_ids for same global_id (same person in different cameras)
                if (cam_name, global_id) not in tracker.local_id_map:
                    tracker.local_id_map[(cam_name, global_id)] = local_id
    
    # Debug: print personID to global_id mapping and per-camera counts
    print(f"[DEBUG Frame {frame_idx}] PersonID to GlobalID mapping: {dict(sorted(person_to_global_id.items()))}")
    for cam in cameras:
        bbox_count = len(detections.get(cam, []))
        mapped_count = sum(1 for lid in range(bbox_count) if (cam, lid) in tracker.global_id_map)
        print(f"  {cam}: {bbox_count} bboxes, {mapped_count} mapped to global_id")
    
    # For points without personID (shouldn't happen with proper annotations), use clustering as fallback
    unassigned_bboxes = [(cam, lid) for cam in cameras 
                         for lid in range(len(detections.get(cam, [])))
                         if (cam, lid) not in tracker.global_id_map]
    
    if unassigned_bboxes:
        print(f"[WARNING] {len(unassigned_bboxes)} bboxes without personID, using clustering fallback")
        # Get BEV points for unassigned bboxes
        unassigned_bev_points = [(x, y, cam, lid) for x, y, cam, lid in bev_points 
                                 if (cam, lid) in unassigned_bboxes]
        if unassigned_bev_points:
            clusters = cluster_bev_points(tracker, unassigned_bev_points, num_clusters=None)
            for _, members in clusters.items():
                global_id = next_global_id
                next_global_id += 1
                for cam_name, local_id in members:
                    if (cam_name, local_id) not in tracker.global_id_map:
                        tracker.global_id_map[(cam_name, local_id)] = global_id
                        tracker.local_id_map[(cam_name, global_id)] = local_id

    # Aggregate frame objects for visualization
    frame_objects = {}  # {global_id: {bev_coords: [...], cameras: [...], local_ids: {...}}}
    camera_colors = {cam: CAMERA_COLOR_PALETTE[i % len(CAMERA_COLOR_PALETTE)] for i, cam in enumerate(cameras)}

    for cam in cameras:
        bev_coords_list = bev_results.get(cam, [])
        for local_id, bev_coord in enumerate(bev_coords_list):
            if bev_coord is not None:
                global_id = tracker.get_global_id(cam, local_id)
                if global_id >= 0:
                    if global_id not in frame_objects:
                        frame_objects[global_id] = {
                            'bev_coords': [],
                            'cameras': [],
                            'local_ids': {}
                        }
                    frame_objects[global_id]['bev_coords'].append(bev_coord)
                    frame_objects[global_id]['cameras'].append(cam)
                    frame_objects[global_id]['local_ids'][cam] = local_id

    # Collect all BEV coordinates to calculate bounds
    all_bev_coords = []
    for obj_data in frame_objects.values():
        all_bev_coords.extend([c for c in obj_data['bev_coords'] if c is not None])
    
    # Calculate dynamic scale and origin based on actual data
    if all_bev_coords:
        all_x = [c[0] for c in all_bev_coords]
        all_y = [c[1] for c in all_bev_coords]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Add padding (10% on each side)
        range_x = max_x - min_x
        range_y = max_y - min_y
        if range_x == 0:
            range_x = 10.0
        if range_y == 0:
            range_y = 10.0
        
        padding_x = range_x * 0.1
        padding_y = range_y * 0.1
        min_x -= padding_x
        max_x += padding_x
        min_y -= padding_y
        max_y += padding_y
        
        # Calculate scale to fit in BEV_SIZE
        scale_x = (BEV_SIZE * 0.9) / (max_x - min_x) if (max_x - min_x) > 0 else BEV_SCALE
        scale_y = (BEV_SIZE * 0.9) / (max_y - min_y) if (max_y - min_y) > 0 else BEV_SCALE
        dynamic_scale = min(scale_x, scale_y)
        
        # Calculate origin (center of the range)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        origin_x = int(BEV_SIZE // 2 - center_x * dynamic_scale)
        origin_y = int(BEV_SIZE // 2 + center_y * dynamic_scale)  # Note: + because y is flipped
        
        print(f"[DEBUG] BEV bounds: x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")
        print(f"[DEBUG] Using scale={dynamic_scale:.2f}, origin=({origin_x}, {origin_y})")
    else:
        # No points, use defaults
        dynamic_scale = BEV_SCALE
        origin_x, origin_y = BEV_ORIGIN
        print(f"[WARNING] No BEV coordinates found! Using default scale={dynamic_scale}, origin={BEV_ORIGIN}")
    
    # Create BEV image
    bev_img = np.ones((BEV_SIZE, BEV_SIZE, 3), dtype=np.uint8) * 30
    draw_bev_grid(bev_img, (origin_x, origin_y), dynamic_scale)
    
    # Generate colors for objects
    object_colors = {}
    for global_id in frame_objects.keys():
        hue = int((global_id * 60) % 180)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        object_colors[global_id] = tuple(map(int, color))
    
    # Draw objects on BEV
    for global_id, obj_data in frame_objects.items():
        draw_object_on_bev(
            bev_img, 
            obj_data['bev_coords'], 
            global_id, 
            camera_colors, 
            obj_data['cameras'],
            (origin_x, origin_y), 
            dynamic_scale
        )
    
    # Annotate camera images
    annotated = {}
    for cam, img in images.items():
        if img is None:
            annotated[cam] = None
            continue
        out = img.copy()
        bboxes_list = detections.get(cam, [])
        # Draw ALL bboxes from detections, not just those with BEV projection
        for local_id in range(len(bboxes_list)):
            global_id = tracker.get_global_id(cam, local_id)
            if global_id >= 0:
                color = object_colors.get(global_id, (255, 255, 255))
                bbox = bboxes_list[local_id]
                conf = 1.0  # Ground-truth box confidence placeholder
                draw_bbox_on_image(out, bbox, global_id, color, conf)
            else:
                # Draw bbox without global_id (shouldn't happen, but for debugging)
                bbox = bboxes_list[local_id]
                cv2.rectangle(out, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (128, 128, 128), 2)
                cv2.putText(out, f"NoID:{local_id}", (int(bbox[0]), int(bbox[1]) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        annotated[cam] = out
    
    cv2.putText(bev_img, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    matched_objects = sorted(frame_objects.items())
    return bev_img, annotated, matched_objects, object_colors, camera_colors


def load_calibrations_from_views(calibrations_root, cameras, image_subsets_folder=None):
    """
    Load calibrations from DATA_ROOT/calibrations/CameraX/calibrations/... into a temp folder.
    Automatically scales calibration to match actual image size.
    """
    temp_calib_folder = tempfile.mkdtemp(prefix="multicam_calib_")
    intrinsic_folder = os.path.join(temp_calib_folder, "intrinsic")
    extrinsic_folder = os.path.join(temp_calib_folder, "extrinsic")
    os.makedirs(intrinsic_folder, exist_ok=True)
    os.makedirs(extrinsic_folder, exist_ok=True)
    
    print("Loading calibrations from calibrations folder...")
    
    # Get image sizes for each camera if image folder is provided
    image_sizes = {}
    if image_subsets_folder:
        for cam in cameras:
            folder_name = CAMERA_TO_VIEW_FOLDER.get(cam, cam)
            frames_folder = os.path.join(image_subsets_folder, folder_name)
            if os.path.exists(frames_folder):
                # Load first frame to get image size
                files = sorted(glob.glob(os.path.join(frames_folder, "*.png")) + 
                              glob.glob(os.path.join(frames_folder, "*.jpg")) +
                              glob.glob(os.path.join(frames_folder, "*.jpeg")))
                if files:
                    img = cv2.imread(files[0])
                    if img is not None:
                        img_h, img_w = img.shape[:2]
                        image_sizes[cam] = (img_h, img_w)
                        print(f"  {cam}: Image size {img_w}x{img_h}")
    
    for cam in cameras:
        cam_root = os.path.join(calibrations_root, cam, "calibrations")
        if not os.path.exists(cam_root):
            print(f"  WARNING: Calibrations folder not found for {cam}: {cam_root}")
            continue

        intrinsic_subfolder = None
        for subfolder_name in ["intrinsic_original", "intrinsic_optimal", "intrinsic"]:
            test_path = os.path.join(cam_root, subfolder_name)
            if os.path.exists(test_path):
                intrinsic_subfolder = test_path
                break

        extrinsic_subfolder = os.path.join(cam_root, "extrinsic")

        if not intrinsic_subfolder or not os.path.exists(extrinsic_subfolder):
            print(f"  WARNING: Intrinsic or extrinsic folder not found for {cam}")
            continue

        intrinsic_files = glob.glob(os.path.join(intrinsic_subfolder, f"intr_{cam}.xml"))
        extrinsic_files = glob.glob(os.path.join(extrinsic_subfolder, f"extr_{cam}.xml"))

        if not intrinsic_files or not extrinsic_files:
            all_intrinsic = glob.glob(os.path.join(intrinsic_subfolder, "*.xml"))
            all_extrinsic = glob.glob(os.path.join(extrinsic_subfolder, "*.xml"))

            if all_intrinsic and all_extrinsic:
                intrinsic_files = [all_intrinsic[0]]
                extrinsic_files = [all_extrinsic[0]]
                print(f"  Using first found XML files for {cam}")
            else:
                print(f"  WARNING: No calibration XML files found for {cam}")
                continue

        # Load calibration
        calib = CameraCalibration.load_from_xml_files_scaled(intrinsic_files[0], extrinsic_files[0], intrinsic_scale=1.0 / 4.0, translation_scale=100.0)
        if calib is None:
            print(f"  WARNING: Failed to load calibration for {cam}")
            continue
        
        # Scale calibration if image size is known
        if cam in image_sizes:
            img_h, img_w = image_sizes[cam]
            K = calib.intrinsic
            cx_orig = K[0, 2]
            cy_orig = K[1, 2]
            fx_orig = K[0, 0]
            fy_orig = K[1, 1]
            
            # Check if scaling is needed
            if cx_orig > img_w * 0.8 or cy_orig > img_h * 0.8:
                # Calibration appears to be for larger image
                orig_w_est = max(cx_orig * 2, img_w)
                orig_h_est = max(cy_orig * 2, img_h)
                
                scale_w = img_w / orig_w_est
                scale_h = img_h / orig_h_est
                scale = (scale_w + scale_h) / 2.0
                
                if 0.1 <= scale <= 2.0:
                    print(f"  {cam}: Scaling calibration by {scale:.4f} (orig: fx={fx_orig:.1f}, cx={cx_orig:.1f}, img={img_w}x{img_h})")
                    scaled_intrinsic = CameraCalibration._scale_intrinsic(K, scale)
                    calib = CameraCalibration(scaled_intrinsic, calib.extrinsic, calib.distortion)
                else:
                    print(f"  {cam}: Scale {scale:.4f} seems unreasonable, using calibration as-is")
        
        # Save scaled calibration to temp folder
        # Write scaled intrinsic to XML if calibration was scaled
        import xml.etree.ElementTree as ET
        
        dest_intr = os.path.join(intrinsic_folder, f"intr_{cam}.xml")
        dest_extr = os.path.join(extrinsic_folder, f"extr_{cam}.xml")
        
        if cam in image_sizes:
            img_h, img_w = image_sizes[cam]
            # Load original calibration (no scaling) using scaled method with scale=1.0
            K_orig = CameraCalibration.load_from_xml_files_scaled(
                intrinsic_files[0], 
                extrinsic_files[0], 
                intrinsic_scale=1.0, 
                translation_scale=1.0
            ).intrinsic
            cx_orig = K_orig[0, 2]
            cy_orig = K_orig[1, 2]
            
            # Check if we scaled
            if cx_orig > img_w * 0.8 or cy_orig > img_h * 0.8:
                # Write scaled intrinsic to XML
                tree = ET.parse(intrinsic_files[0])
                root = tree.getroot()
                
                # Find and update camera_matrix
                camera_matrix_elem = root.find('camera_matrix')
                if camera_matrix_elem is not None:
                    data_elem = camera_matrix_elem.find('data')
                    if data_elem is not None:
                        # Update with scaled intrinsic (row-major order)
                        scaled_data = ' '.join([f"{calib.intrinsic[i, j]:.10f}" for i in range(3) for j in range(3)])
                        data_elem.text = scaled_data
                
                tree.write(dest_intr, encoding='utf-8', xml_declaration=True)
            else:
                # No scaling needed, just copy
                shutil.copy2(intrinsic_files[0], dest_intr)
        else:
            # No image size info, just copy
            shutil.copy2(intrinsic_files[0], dest_intr)
        
        shutil.copy2(extrinsic_files[0], dest_extr)
        print(f"  Loaded (and scaled if needed) calibration for {cam}")

    return temp_calib_folder


def main():
    """Main function - Test multiview matching only"""
    print("MULTI-CAMERA MULTIVIEW MATCHING VISUALIZATION")
    print("="*50)
    
    # Check paths
    if not os.path.exists(IMAGE_SUBSETS_FOLDER):
        print(f"ERROR: Image folder not found: {IMAGE_SUBSETS_FOLDER}")
        print("Please update IMAGE_SUBSETS_FOLDER in the script")
        return
    if not os.path.exists(ANNOTATIONS_FOLDER):
        print(f"ERROR: Annotations folder not found: {ANNOTATIONS_FOLDER}")
        print("Please update ANNOTATIONS_FOLDER in the script")
        return
    if not os.path.exists(CALIBRATIONS_ROOT):
        print(f"ERROR: Calibrations folder not found: {CALIBRATIONS_ROOT}")
        print("Please update CALIBRATIONS_ROOT in the script")
        return
    
    # Load calibrations from Camera folders (with scaling based on actual image sizes)
    temp_calib_folder = load_calibrations_from_views(CALIBRATIONS_ROOT, CAMERAS, IMAGE_SUBSETS_FOLDER)
    
    try:
        # Initialize tracker
        print("Initializing MultiCameraTracking...")
        tracker = MultiCameraTracking(
            calib_folder=temp_calib_folder,
            max_distance=MAX_DISTANCE
        )
        # Ensure BEV converter uses our height/dist settings
        tracker.bev_converter.average_object_height = AVERAGE_OBJECT_HEIGHT
        tracker.bev_converter.min_dist = MIN_DIST
        tracker.bev_converter.max_dist = MAX_DIST
        
        # Load image files from Image_subsets/<id> folders
        print("\nLoading image files...")
        image_files = {}
        for cam in CAMERAS:
            folder_name = CAMERA_TO_VIEW_FOLDER.get(cam, cam)
            frames_folder = os.path.join(IMAGE_SUBSETS_FOLDER, folder_name)
            if not os.path.exists(frames_folder):
                print(f"  WARNING: Frames folder not found for {cam}: {frames_folder}")
                image_files[cam] = []
                continue
            
            files = sorted(glob.glob(os.path.join(frames_folder, "*.png")) + 
                          glob.glob(os.path.join(frames_folder, "*.jpg")) +
                          glob.glob(os.path.join(frames_folder, "*.jpeg")))
            image_files[cam] = files
            print(f"  {cam}: {len(files)} frames")
        
        max_frames_images = max(len(f) for f in image_files.values()) if image_files else 0
        annotation_files = sorted(glob.glob(os.path.join(ANNOTATIONS_FOLDER, "*.json")))
        max_frames_annotations = len(annotation_files)
        max_frames = min(max_frames_images, max_frames_annotations)
        print(f"Total frames (images vs annotations): {max_frames_images} / {max_frames_annotations}")
        print(f"Using {max_frames} frames for visualization")
        
        if max_frames == 0:
            print("ERROR: No images found!")
            return
        
        # Visualization loop - test multiview matching for each frame independently
        print("\n" + "="*50)
        print("Starting multiview matching visualization")
        print("(press 'q' to quit, 'n' next, 'p' previous, ' ' pause)")
        print("="*50 + "\n")
        
        idx = 0
        paused = False
        while True:
            if idx < 0:
                idx = 0
            if idx >= max_frames:
                idx = max_frames - 1
            
            # Process frame: load GT, match multiview, visualize
            bev_img, annotated_images, matched_objects, object_colors, camera_colors = process_frame(
                idx, image_files, CAMERAS, ANNOTATIONS_FOLDER, tracker
            )
            
            # Create separate displays
            camera_grid = create_camera_grid(annotated_images, CAMERAS, frame_idx=idx)
            bev_display = create_bev_display(bev_img, matched_objects, object_colors, CAMERAS, camera_colors)
            
            # Show in separate windows
            cv2.imshow("Camera Grid Views", camera_grid)
            cv2.imshow("BEV View", bev_display)
            
            key = cv2.waitKey(0 if paused else 100) & 0xFF
            if key == ord('q'):
                break
            if key == ord('n'):
                idx += 1
                paused = True
            elif key == ord('p'):
                idx -= 1
                paused = True
            elif key == ord(' '):
                paused = not paused
            elif not paused:
                idx += 1
                if idx >= max_frames:
                    idx = 0
        
        cv2.destroyAllWindows()
        print("Visualization completed!")
    
    finally:
        # Clean up temporary calibration folder
        if 'temp_calib_folder' in locals() and os.path.exists(temp_calib_folder):
            shutil.rmtree(temp_calib_folder)
            print(f"\nCleaned up temporary calibration folder")


if __name__ == '__main__':
    main()
