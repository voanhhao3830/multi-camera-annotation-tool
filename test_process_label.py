import os
import json
import cv2
import numpy as np
from math import ceil, sqrt
import tkinter as tk
import tempfile
import shutil
import glob
from labelme.utils.preprocess_label import PreprocessLabel

# ==============================================
# CONFIG
# ==============================================
DATA_ROOT = r"D:\MultiCameraLabelTool\aicv\multi-camera-annotation-tool\data\aicv"
IMAGE_SUBSETS_FOLDER = os.path.join(DATA_ROOT, "Image_subsets")
ANNOTATIONS_FOLDER = os.path.join(DATA_ROOT, "detection_positions")
CALIBRATIONS_ROOT = os.path.join(DATA_ROOT, "calibrations")

# Get screen size and calculate appropriate image size
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# Original image size (from annotations)
ORIGINAL_IMG_WIDTH = 960
ORIGINAL_IMG_HEIGHT = 540

# auto detect cameras (folder names like "1", "2", "3", ...)
CAMERA_FOLDERS = sorted([d for d in os.listdir(IMAGE_SUBSETS_FOLDER)
                          if os.path.isdir(os.path.join(IMAGE_SUBSETS_FOLDER, d))])
print("Detected camera folders:", CAMERA_FOLDERS)

# Mapping: viewNum -> camera_name and camera_name -> folder
VIEW_TO_CAMERA = {i: f"Camera{i}" for i in range(1, 7)}
CAMERA_TO_FOLDER = {f"Camera{i}": str(i) for i in range(1, 7)}
FOLDER_TO_CAMERA = {str(i): f"Camera{i}" for i in range(1, 7)}


def load_calibrations_to_temp(calibrations_root, cameras, image_subsets_folder):
    """
    Load calibrations from DATA_ROOT/calibrations/CameraX/calibrations/... into a temp folder.
    Returns path to temp calibration folder.
    """
    temp_calib_folder = tempfile.mkdtemp(prefix="multicam_calib_")
    intrinsic_folder = os.path.join(temp_calib_folder, "intrinsic")
    extrinsic_folder = os.path.join(temp_calib_folder, "extrinsic")
    os.makedirs(intrinsic_folder, exist_ok=True)
    os.makedirs(extrinsic_folder, exist_ok=True)
    
    print("Loading calibrations...")
    
    # Get image sizes
    image_sizes = {}
    for cam_name in cameras:
        folder_name = CAMERA_TO_FOLDER.get(cam_name, cam_name)
        frames_folder = os.path.join(image_subsets_folder, folder_name)
        if os.path.exists(frames_folder):
            files = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))
            if files:
                img = cv2.imread(files[0])
                if img is not None:
                    img_h, img_w = img.shape[:2]
                    image_sizes[cam_name] = (img_h, img_w)
    
    for cam_name in cameras:
        cam_root = os.path.join(calibrations_root, cam_name, "calibrations")
        if not os.path.exists(cam_root):
            print(f"  WARNING: Calibrations not found for {cam_name}")
            continue

        # Find intrinsic subfolder
        intrinsic_subfolder = None
        for subfolder_name in ["intrinsic_original", "intrinsic_optimal", "intrinsic"]:
            test_path = os.path.join(cam_root, subfolder_name)
            if os.path.exists(test_path):
                intrinsic_subfolder = test_path
                break

        extrinsic_subfolder = os.path.join(cam_root, "extrinsic")

        if not intrinsic_subfolder or not os.path.exists(extrinsic_subfolder):
            print(f"  WARNING: Intrinsic or extrinsic folder not found for {cam_name}")
            continue

        intrinsic_files = glob.glob(os.path.join(intrinsic_subfolder, f"intr_{cam_name}.xml"))
        extrinsic_files = glob.glob(os.path.join(extrinsic_subfolder, f"extr_{cam_name}.xml"))

        if not intrinsic_files or not extrinsic_files:
            all_intrinsic = glob.glob(os.path.join(intrinsic_subfolder, "*.xml"))
            all_extrinsic = glob.glob(os.path.join(extrinsic_subfolder, "*.xml"))
            if all_intrinsic and all_extrinsic:
                intrinsic_files = [all_intrinsic[0]]
                extrinsic_files = [all_extrinsic[0]]
            else:
                print(f"  WARNING: No calibration XML files found for {cam_name}")
                continue

        # Copy files to temp folder
        dest_intr = os.path.join(intrinsic_folder, f"intr_{cam_name}.xml")
        dest_extr = os.path.join(extrinsic_folder, f"extr_{cam_name}.xml")
        shutil.copy2(intrinsic_files[0], dest_intr)
        shutil.copy2(extrinsic_files[0], dest_extr)
        print(f"  Loaded calibration for {cam_name}")

    return temp_calib_folder


def load_all_annotations(annotations_folder, frame_count):
    """Load all annotations and convert to format for PreprocessLabel"""
    all_annotations = []
    for frame_idx in range(frame_count):
        annot_path = os.path.join(annotations_folder, f"{frame_idx:05d}.json")
        if os.path.exists(annot_path):
            with open(annot_path, "r") as f:
                data = json.load(f)
                all_annotations.append(data)
        else:
            all_annotations.append([])
    return all_annotations


def convert_annotations_to_detections(all_annotations, cameras):
    """
    Convert annotations to format: Dict[camera_name, List[frames, List[bboxes]]]
    Each bbox is [x_min, y_min, x_max, y_max]
    Returns: (multi_camera_detections, annotation_to_local_id_map)
    annotation_to_local_id_map: {(frame_idx, cam_name, person_idx, view_idx): local_id}
    """
    # Initialize structure: camera_name -> list of frames, each frame is list of bboxes
    multi_camera_detections = {cam: [] for cam in cameras}
    annotation_to_local_id_map = {}  # Map from annotation to local_id
    
    num_frames = len(all_annotations)
    
    for frame_idx in range(num_frames):
        frame_data = all_annotations[frame_idx]
        
        # Initialize bboxes for each camera in this frame
        frame_bboxes = {cam: [] for cam in cameras}
        local_id_counters = {cam: 0 for cam in cameras}  # Track local_id per camera
        
        # Process each person in this frame
        for person_idx, person in enumerate(frame_data):
            for view_idx, view in enumerate(person.get("views", [])):
                if view.get("xmin") == -1:
                    continue
                
                view_num = view.get("viewNum")
                cam_name = VIEW_TO_CAMERA.get(view_num)
                
                if cam_name and cam_name in frame_bboxes:
                    xmin = int(view["xmin"])
                    ymin = int(view["ymin"])
                    xmax = int(view["xmax"])
                    ymax = int(view["ymax"])
                    frame_bboxes[cam_name].append([xmin, ymin, xmax, ymax])
                    
                    # Map annotation to local_id
                    local_id = local_id_counters[cam_name]
                    annotation_to_local_id_map[(frame_idx, cam_name, person_idx, view_idx)] = local_id
                    local_id_counters[cam_name] += 1
        
        # Add frame bboxes to each camera
        for cam_name in cameras:
            multi_camera_detections[cam_name].append(frame_bboxes.get(cam_name, []))
    
    return multi_camera_detections, annotation_to_local_id_map


def get_color_for_id(global_id):
    """Generate a color for a global ID"""
    np.random.seed(global_id)
    color = np.random.randint(0, 255, 3).tolist()
    return tuple(map(int, color))


def _build_bev_display_config(preprocessor, target_size=800, grid_step_world=50):
    """
    Precompute BEV display scaling once so all frames share the same BEV size and ratio.
    """
    bev_projector = preprocessor.multiview_matcher.bev_converter.bev_projector
    bev_w = getattr(bev_projector, "bev_x", target_size)
    bev_h = getattr(bev_projector, "bev_y", target_size)
    bounds = getattr(bev_projector, "bev_bounds", [0, bev_w, 0, bev_h])

    xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
    world_w = max(float(xmax - xmin), 1.0)
    world_h = max(float(ymax - ymin), 1.0)

    aspect = world_w / world_h
    if aspect >= 1.0:
        canvas_w = int(target_size)
        canvas_h = int(round(target_size / aspect))
    else:
        canvas_h = int(target_size)
        canvas_w = int(round(target_size * aspect))

    scale_x = canvas_w / world_w
    scale_y = canvas_h / world_h
    offset_x = -xmin * scale_x
    offset_y = -ymin * scale_y

    return {
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "bounds": (xmin, xmax, ymin, ymax),
        "grid_step_world": grid_step_world
    }


def create_bev_visualization(frame_idx, data, global_id_map, annotation_to_local_id_map, 
                             preprocessor, camera_names, bev_display_cfg):
    """
    Create BEV visualization showing clusters and object positions.
    All frames use the same BEV canvas size and scaling.
    """
    cfg = bev_display_cfg
    canvas_w, canvas_h = cfg["canvas_w"], cfg["canvas_h"]
    scale_x, scale_y = cfg["scale_x"], cfg["scale_y"]
    offset_x, offset_y = cfg["offset_x"], cfg["offset_y"]
    xmin, xmax, ymin, ymax = cfg["bounds"]
    grid_step_world = cfg["grid_step_world"]

    # Create BEV canvas with fixed size
    bev_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 40  # Dark gray background

    def world_to_canvas(x, y):
        bev_x = int(x * scale_x + offset_x)
        bev_y = int(y * scale_y + offset_y)
        return bev_x, bev_y

    # Draw grid based on world coordinates to keep spacing consistent
    grid_start_x = int(np.floor(xmin / grid_step_world) * grid_step_world)
    grid_end_x = int(np.ceil(xmax / grid_step_world) * grid_step_world)
    grid_start_y = int(np.floor(ymin / grid_step_world) * grid_step_world)
    grid_end_y = int(np.ceil(ymax / grid_step_world) * grid_step_world)

    for gx in range(grid_start_x, grid_end_x + 1, grid_step_world):
        cx, _ = world_to_canvas(gx, ymin)
        cv2.line(bev_canvas, (cx, 0), (cx, canvas_h), (60, 60, 60), 1)
    for gy in range(grid_start_y, grid_end_y + 1, grid_step_world):
        _, cy = world_to_canvas(xmin, gy)
        cv2.line(bev_canvas, (0, cy), (canvas_w, cy), (60, 60, 60), 1)

    # Draw center lines at world center
    center_world_x = (xmin + xmax) / 2.0
    center_world_y = (ymin + ymax) / 2.0
    center_x, center_y = world_to_canvas(center_world_x, center_world_y)
    cv2.line(bev_canvas, (0, center_y), (canvas_w, center_y), (255, 100, 100), 2)  # Horizontal
    cv2.line(bev_canvas, (center_x, 0), (center_x, canvas_h), (100, 255, 100), 2)  # Vertical
    cv2.circle(bev_canvas, (center_x, center_y), 5, (0, 0, 255), -1)  # Center point
    
    # Collect BEV points for this frame
    bev_points = {}  # {global_id: [(x, y, camera_name), ...]}
    all_bev_coords = []
    
    # Get frame detections
    frame_detections = {}
    for cam_name in camera_names:
        cam_folder = CAMERA_TO_FOLDER.get(cam_name, "")
        if cam_folder and cam_folder in CAMERA_FOLDERS:
            # Get detections for this frame from annotations
            frame_detections[cam_name] = []
            for person_idx, person in enumerate(data):
                for view_idx, view in enumerate(person.get("views", [])):
                    if view.get("xmin") == -1:
                        continue
                    view_num = view.get("viewNum")
                    if VIEW_TO_CAMERA.get(view_num) == cam_name:
                        xmin = int(view["xmin"])
                        ymin = int(view["ymin"])
                        xmax = int(view["xmax"])
                        ymax = int(view["ymax"])
                        frame_detections[cam_name].append([xmin, ymin, xmax, ymax])
    
    # Project to BEV using preprocessor's multiview matcher
    bev_results = preprocessor.multiview_matcher.project_bev(frame_detections)
    
    # Collect points and map to global IDs
    for cam_name, bev_coords_list in bev_results.items():
        for local_id, bev_coord in enumerate(bev_coords_list):
            if bev_coord is not None:
                x, y = bev_coord
                if np.isfinite(x) and np.isfinite(y):
                    # Get global_id
                    global_id = global_id_map.get((frame_idx, cam_name, local_id), -1)
                    if global_id >= 0:
                        if global_id not in bev_points:
                            bev_points[global_id] = []
                        bev_points[global_id].append((x, y, cam_name))
                        all_bev_coords.append((x, y, global_id, cam_name))
    
    if all_bev_coords:
        # Camera colors for BEV visualization (dynamically assigned)
        camera_colors = {}
        color_list = [
            (0, 0, 255),     # Red (BGR)
            (0, 255, 0),     # Green (BGR)
            (255, 0, 0),     # Blue (BGR)
            (255, 255, 0),   # Cyan (BGR)
            (255, 0, 255),   # Magenta (BGR)
            (0, 255, 255),   # Yellow (BGR)
        ]
        for idx, cam_name in enumerate(camera_names):
            camera_colors[cam_name] = color_list[idx % len(color_list)]

        # First, draw all BEV points as colored dots (one dot per point)
        for global_id, points in bev_points.items():
            if not points:
                continue
            
            for x, y, cam_name in points:
                bev_x, bev_y = world_to_canvas(x, y)
                
                if 0 <= bev_x < canvas_w and 0 <= bev_y < canvas_h:
                    cam_color = camera_colors.get(cam_name, (255, 255, 255))
                    dot_radius = 5
                    cv2.circle(bev_canvas, (bev_x, bev_y), dot_radius, cam_color, -1)
                    cv2.circle(bev_canvas, (bev_x, bev_y), dot_radius, (255, 255, 255), 1)  # White border
        
        # Then, draw X mark and label at centroid for each cluster
        for global_id, points in bev_points.items():
            if not points:
                continue
            
            unique_cameras = list(set([cam_name for _, _, cam_name in points]))
            num_cams = len(unique_cameras)
            
            centroid_x = np.mean([x for x, _, _ in points])
            centroid_y = np.mean([y for _, y, _ in points])
            centroid_bev_x, centroid_bev_y = world_to_canvas(centroid_x, centroid_y)
            
            if 0 <= centroid_bev_x < canvas_w and 0 <= centroid_bev_y < canvas_h:
                x_size = 12  # Size of X mark
                cv2.line(bev_canvas, 
                        (centroid_bev_x - x_size, centroid_bev_y - x_size), 
                        (centroid_bev_x + x_size, centroid_bev_y + x_size), 
                        (255, 255, 255), 2)  # White X
                cv2.line(bev_canvas, 
                        (centroid_bev_x - x_size, centroid_bev_y + x_size), 
                        (centroid_bev_x + x_size, centroid_bev_y - x_size), 
                        (255, 255, 255), 2)  # White X
                
                label = f"ID:{global_id} {num_cams}cam"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(bev_canvas, (centroid_bev_x - text_width // 2 - 4, centroid_bev_y + x_size + 10),
                             (centroid_bev_x + text_width // 2 + 4, centroid_bev_y + x_size + 10 + text_height + 4), (0, 0, 0), -1)
                cv2.putText(bev_canvas, label, (centroid_bev_x - text_width // 2, centroid_bev_y + x_size + 10 + text_height),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw legend
    camera_colors = {
        "Camera1": (255, 0, 0),    # Blue (BGR)
        "Camera2": (0, 255, 0),     # Green (BGR)
        "Camera3": (0, 0, 255),     # Red (BGR)
        "Camera4": (255, 255, 0),  # Cyan (BGR)
        "Camera5": (255, 0, 255),  # Magenta (BGR)
        "Camera6": (0, 255, 255),  # Yellow (BGR)
    }
    
    legend_x = max(canvas_w - 200, 10)
    legend_y = 30
    cv2.putText(bev_canvas, "Camera Colors (BEV):", (legend_x, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    legend_y += 25
    
    for i, (cam_name, cam_color) in enumerate(list(camera_colors.items())):
        view_num = i + 1
        cv2.circle(bev_canvas, (legend_x - 10, legend_y), 8, cam_color, -1)
        cv2.putText(bev_canvas, f"View {view_num}", (legend_x + 5, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 25
    
    # Draw object colors legend
    legend_y += 10
    cv2.putText(bev_canvas, "Object Colors:", (legend_x, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    legend_y += 25
    
    # Show first 6 global IDs
    unique_global_ids = sorted(set([gid for _, _, gid, _ in all_bev_coords]))[:6]
    for global_id in unique_global_ids:
        color = get_color_for_id(global_id)
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.rectangle(bev_canvas, (legend_x - 10, legend_y - 8), 
                     (legend_x + 5, legend_y + 8), color_bgr, -1)
        cv2.putText(bev_canvas, f"ID:{global_id}", (legend_x + 10, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 25
    
    # Draw statistics
    stats_y = canvas_h - 80
    cv2.putText(bev_canvas, "STATISTICS", (legend_x, stats_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    stats_y += 25
    total_objects = len(bev_points)
    cv2.putText(bev_canvas, f"Total Objects: {total_objects}", (legend_x, stats_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return bev_canvas


def create_bev_points_only_visualization(frame_idx, data, global_id_map, annotation_to_local_id_map, 
                                         preprocessor, camera_names, bev_display_cfg):
    """
    Create a BEV visualization that only shows projected points (no cluster centroids).
    Useful for debugging pure multiview geometry without clustering overlay.
    """
    cfg = bev_display_cfg
    canvas_w, canvas_h = cfg["canvas_w"], cfg["canvas_h"]
    scale_x, scale_y = cfg["scale_x"], cfg["scale_y"]
    offset_x, offset_y = cfg["offset_x"], cfg["offset_y"]
    xmin, xmax, ymin, ymax = cfg["bounds"]
    grid_step_world = cfg["grid_step_world"]

    # Create BEV canvas with fixed size
    bev_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 40  # Dark gray background

    def world_to_canvas(x, y):
        bev_x = int(x * scale_x + offset_x)
        bev_y = int(y * scale_y + offset_y)
        return bev_x, bev_y

    # Draw grid based on world coordinates to keep spacing consistent
    grid_start_x = int(np.floor(xmin / grid_step_world) * grid_step_world)
    grid_end_x = int(np.ceil(xmax / grid_step_world) * grid_step_world)
    grid_start_y = int(np.floor(ymin / grid_step_world) * grid_step_world)
    grid_end_y = int(np.ceil(ymax / grid_step_world) * grid_step_world)

    for gx in range(grid_start_x, grid_end_x + 1, grid_step_world):
        cx, _ = world_to_canvas(gx, ymin)
        cv2.line(bev_canvas, (cx, 0), (cx, canvas_h), (60, 60, 60), 1)
    for gy in range(grid_start_y, grid_end_y + 1, grid_step_world):
        _, cy = world_to_canvas(xmin, gy)
        cv2.line(bev_canvas, (0, cy), (canvas_w, cy), (60, 60, 60), 1)

    # Draw center lines at world center
    center_world_x = (xmin + xmax) / 2.0
    center_world_y = (ymin + ymax) / 2.0
    center_x, center_y = world_to_canvas(center_world_x, center_world_y)
    cv2.line(bev_canvas, (0, center_y), (canvas_w, center_y), (255, 100, 100), 2)  # Horizontal
    cv2.line(bev_canvas, (center_x, 0), (center_x, canvas_h), (100, 255, 100), 2)  # Vertical
    cv2.circle(bev_canvas, (center_x, center_y), 5, (0, 0, 255), -1)  # Center point

    # Collect BEV points for this frame
    bev_points = {}  # {global_id: [(x, y, camera_name), ...]}

    # Get frame detections
    frame_detections = {}
    for cam_name in camera_names:
        cam_folder = CAMERA_TO_FOLDER.get(cam_name, "")
        if cam_folder and cam_folder in CAMERA_FOLDERS:
            # Get detections for this frame from annotations
            frame_detections[cam_name] = []
            for person_idx, person in enumerate(data):
                for view_idx, view in enumerate(person.get("views", [])):
                    if view.get("xmin") == -1:
                        continue
                    view_num = view.get("viewNum")
                    if VIEW_TO_CAMERA.get(view_num) == cam_name:
                        xmin_v = int(view["xmin"])
                        ymin_v = int(view["ymin"])
                        xmax_v = int(view["xmax"])
                        ymax_v = int(view["ymax"])
                        frame_detections[cam_name].append([xmin_v, ymin_v, xmax_v, ymax_v])

    # Project to BEV using preprocessor's multiview matcher
    bev_results = preprocessor.multiview_matcher.project_bev(frame_detections)

    # Collect points and map to global IDs
    for cam_name, bev_coords_list in bev_results.items():
        for local_id, bev_coord in enumerate(bev_coords_list):
            if bev_coord is not None:
                x, y = bev_coord
                if np.isfinite(x) and np.isfinite(y):
                    # Get global_id
                    global_id = global_id_map.get((frame_idx, cam_name, local_id), -1)
                    if global_id >= 0:
                        if global_id not in bev_points:
                            bev_points[global_id] = []
                        bev_points[global_id].append((x, y, cam_name))

    if bev_points:
        # Camera colors for BEV visualization (dynamically assigned)
        camera_colors = {}
        color_list = [
            (0, 0, 255),     # Red (BGR)
            (0, 255, 0),     # Green (BGR)
            (255, 0, 0),     # Blue (BGR)
            (255, 255, 0),   # Cyan (BGR)
            (255, 0, 255),   # Magenta (BGR)
            (0, 255, 255),   # Yellow (BGR)
        ]
        for idx, cam_name in enumerate(camera_names):
            camera_colors[cam_name] = color_list[idx % len(color_list)]

        # Draw all BEV points as colored dots (one dot per point), no centroid / label
        for global_id, points in bev_points.items():
            if not points:
                continue
            for x, y, cam_name in points:
                bev_x, bev_y = world_to_canvas(x, y)
                if 0 <= bev_x < canvas_w and 0 <= bev_y < canvas_h:
                    cam_color = camera_colors.get(cam_name, (255, 255, 255))
                    dot_radius = 5
                    cv2.circle(bev_canvas, (bev_x, bev_y), dot_radius, cam_color, -1)
                    cv2.circle(bev_canvas, (bev_x, bev_y), dot_radius, (255, 255, 255), 1)  # White border

    return bev_canvas

def compute_layout(n, prefer_cols=True):
    """
    Compute layout with preference for more columns than rows (landscape).
    prefer_cols=True: prioritize more columns (landscape layout)
    """
    if prefer_cols:
        # Start with more columns for landscape layout
        # For n cameras, try to get cols >= rows
        cols = ceil(sqrt(n) * 1.3)  # Prefer more columns
        rows = ceil(n / cols)
        
        # Ensure cols >= rows (landscape)
        if cols < rows:
            cols = rows
            rows = ceil(n / cols)
        
        # Fine-tune: if we have too many empty cells, reduce cols slightly
        while (cols - 1) * rows >= n and cols > rows:
            cols -= 1
        
        return rows, cols
    else:
        rows = ceil(sqrt(n))
        cols = ceil(n / rows)
        return rows, cols


# ==============================================
# PREPROCESS: Load calibrations and run tracking
# ==============================================
print("=" * 60)
print("Step 1: Loading calibrations...")
print("=" * 60)

# Get camera names from detected folders
camera_names = [FOLDER_TO_CAMERA.get(folder, f"Camera{folder}") for folder in CAMERA_FOLDERS 
                if folder.isdigit() and int(folder) in range(1, 7)]
print(f"Camera names: {camera_names}")

# Create temp calibration folder
temp_calib_folder = load_calibrations_to_temp(
    CALIBRATIONS_ROOT, 
    camera_names,
    IMAGE_SUBSETS_FOLDER
)

print("\n" + "=" * 60)
print("Step 2: Loading all annotations...")
print("=" * 60)
total_frames = len(os.listdir(ANNOTATIONS_FOLDER))
MAX_FRAMES = 200  # Limit to 100 frames
frame_count = min(total_frames, MAX_FRAMES)
all_annotations = load_all_annotations(ANNOTATIONS_FOLDER, frame_count)
print(f"Loaded {len(all_annotations)} frames (out of {total_frames} total)")

print("\n" + "=" * 60)
print("Step 3: Converting annotations to detection format...")
print("=" * 60)
multi_camera_detections, annotation_to_local_id_map = convert_annotations_to_detections(all_annotations, camera_names)
print(f"Converted detections for {len(camera_names)} cameras, {frame_count} frames")

print("\n" + "=" * 60)
print("Step 4: Running PreprocessLabel (multiview matching + temporal tracking)...")
print("=" * 60)

# Set number of clusters for KMeans (None = auto-detect, or set a specific number)
N_CLUSTERS = 10  # Set to None for auto-detect, or set to a number like 5, 10, etc.

preprocessor = PreprocessLabel(
    calib_folder=temp_calib_folder,
    max_distance_multiview=2.0,
    max_distance_tracking=50.0,
    n_clusters=N_CLUSTERS
)
global_id_map = preprocessor.preprocess(multi_camera_detections)
print(f"Mapping completed! Total mappings: {len(global_id_map)}")

# Precompute BEV display config once so all frames share the same canvas and scaling
bev_display_cfg = _build_bev_display_config(preprocessor, target_size=800, grid_step_world=50)

# ==============================================
# MAIN LOOP: Visualization
# ==============================================
print("\n" + "=" * 60)
print("Step 5: Starting visualization...")
print("=" * 60)

# Helper functions to check if key is arrow key or navigation key
def is_prev_frame_key(key_code, key_raw):
    """Check if key is previous frame (A, Left Arrow)"""
    if key_raw == -1:
        return False
    # Check for A key or Left Arrow key
    # Left arrow codes: 81 (0x51) or 2 (when & 0xFF)
    return (key_code == ord('a') or key_code == ord('A') or 
            key_raw == 81 or key_code == 2 or 
            (key_raw >= 224 and key_raw <= 255 and key_code == 2))  # Extended key codes

def is_next_frame_key(key_code, key_raw):
    """Check if key is next frame (D, Right Arrow)"""
    if key_raw == -1:
        return False
    # Check for D key or Right Arrow key
    # Right arrow codes: 83 (0x53) or 3 (when & 0xFF)
    return (key_code == ord('d') or key_code == ord('D') or 
            key_raw == 83 or key_code == 3 or 
            (key_raw >= 224 and key_raw <= 255 and key_code == 3))  # Extended key codes

frame_idx = 0
running = True

while running:
    # Wrap around frame index
    if frame_idx >= frame_count:
        frame_idx = 0
    elif frame_idx < 0:
        frame_idx = frame_count - 1

    # --- load annotation ---
    data = all_annotations[frame_idx]
    if not data:
        frame_idx += 1
        continue

    # --- load images ---
    cam_imgs = {}
    for cam_folder in CAMERA_FOLDERS:
        img_path = os.path.join(IMAGE_SUBSETS_FOLDER, cam_folder, f"{frame_idx:05d}.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            cam_imgs[cam_folder] = img

    if len(cam_imgs) == 0:
        continue

    # --- build layout ---
    cam_folders = list(cam_imgs.keys())
    C = len(cam_folders)
    ROWS, COLS = compute_layout(C, prefer_cols=True)

    # Calculate image size to fit screen (use 95% of screen)
    available_width = int(screen_width * 0.95)
    available_height = int(screen_height * 0.95)
    
    # Calculate max size per camera based on both width and height
    max_cam_width = available_width // COLS
    max_cam_height = available_height // ROWS
    
    # Choose the smaller one to ensure fit, maintaining aspect ratio
    CAM_WIDTH = min(max_cam_width, int(max_cam_height * ORIGINAL_IMG_WIDTH / ORIGINAL_IMG_HEIGHT))
    CAM_HEIGHT = int(CAM_WIDTH * ORIGINAL_IMG_HEIGHT / ORIGINAL_IMG_WIDTH)
    
    # Double check: if height still too large, scale down
    if CAM_HEIGHT * ROWS > available_height:
        CAM_HEIGHT = available_height // ROWS
        CAM_WIDTH = int(CAM_HEIGHT * ORIGINAL_IMG_WIDTH / ORIGINAL_IMG_HEIGHT)
    
    IMG_SIZE = (CAM_WIDTH, CAM_HEIGHT)

    # Scale factor for bounding boxes
    scale_x = CAM_WIDTH / ORIGINAL_IMG_WIDTH
    scale_y = CAM_HEIGHT / ORIGINAL_IMG_HEIGHT

    # Resize all images
    for cam in cam_imgs:
        cam_imgs[cam] = cv2.resize(cam_imgs[cam], IMG_SIZE)

    CANVAS_W = COLS * IMG_SIZE[0]
    CANVAS_H = ROWS * IMG_SIZE[1]

    big_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # store top-left pixel of each camera tile
    tile_pos = {}

    k = 0
    for r in range(ROWS):
        for c in range(COLS):
            if k < C:
                cam_folder = cam_folders[k]
                x0 = c * IMG_SIZE[0]
                y0 = r * IMG_SIZE[1]
                big_canvas[y0:y0+IMG_SIZE[1], x0:x0+IMG_SIZE[0]] = cam_imgs[cam_folder]
                tile_pos[cam_folder] = (x0, y0)
                k += 1


    # --- draw bounding boxes on global canvas with global IDs ---
    for person_idx, person in enumerate(data):
        for view_idx, view in enumerate(person.get("views", [])):
            if view.get("xmin") == -1:
                continue

            view_num = view.get("viewNum")
            cam_name = VIEW_TO_CAMERA.get(view_num)
            cam_folder = str(view_num)
            
            if cam_folder not in tile_pos or cam_name is None:
                continue

            # Get local_id from annotation_to_local_id_map
            local_id = annotation_to_local_id_map.get((frame_idx, cam_name, person_idx, view_idx), -1)
            
            if local_id < 0:
                continue

            # Get global_id from mapping
            global_id = global_id_map.get((frame_idx, cam_name, local_id), -1)
            
            x_off, y_off = tile_pos[cam_folder]

            # Scale bounding box coordinates from original image size to resized size
            xmin = x_off + int(view["xmin"] * scale_x)
            ymin = y_off + int(view["ymin"] * scale_y)
            xmax = x_off + int(view["xmax"] * scale_x)
            ymax = y_off + int(view["ymax"] * scale_y)
        
            # Use color based on global_id
            if global_id >= 0:
                color = get_color_for_id(global_id)
                thickness = 3
                # Draw rectangle
                cv2.rectangle(big_canvas, (xmin, ymin), (xmax, ymax), color, thickness)
                # Draw global_id label
                label = f"ID:{global_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(big_canvas, (xmin, ymin - label_size[1] - 5), 
                             (xmin + label_size[0], ymin), color, -1)
                cv2.putText(big_canvas, label, (xmin, ymin - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # No global_id assigned - draw in gray
                cv2.rectangle(big_canvas, (xmin, ymin), (xmax, ymax), (128, 128, 128), 2)

    # --- show ---
    # Resize window to fit screen if needed
    display_canvas = big_canvas.copy()
    if CANVAS_W > screen_width or CANVAS_H > screen_height:
        scale_w = screen_width / CANVAS_W
        scale_h = screen_height / CANVAS_H
        display_scale = min(scale_w, scale_h) * 0.95
        display_w = int(CANVAS_W * display_scale)
        display_h = int(CANVAS_H * display_scale)
        display_canvas = cv2.resize(big_canvas, (display_w, display_h))
    
    # Add control instructions on canvas
    info_text = [
        "SPACE: Pause | A/D or ←/→: Prev/Next | Q: Quit"
    ]
    y_offset = 30
    for i, text in enumerate(info_text):
        # Draw black background for text
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(display_canvas, (5, y_offset + i * 25 - text_height - 5), 
                     (5 + text_width + 10, y_offset + i * 25 + 5), (0, 0, 0), -1)
        cv2.putText(display_canvas, text, (10, y_offset + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # --- Create BEV visualizations ---
    bev_canvas = create_bev_visualization(
        frame_idx, data, global_id_map, annotation_to_local_id_map,
        preprocessor, camera_names, bev_display_cfg
    )
    bev_points_canvas = create_bev_points_only_visualization(
        frame_idx, data, global_id_map, annotation_to_local_id_map,
        preprocessor, camera_names, bev_display_cfg
    )
    
    # --- show windows ---
    cv2.namedWindow("Multi-Camera View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi-Camera View", display_canvas.shape[1], display_canvas.shape[0])
    cv2.imshow("Multi-Camera View", display_canvas)
    
    cv2.namedWindow("BEV Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BEV Map", bev_canvas.shape[1], bev_canvas.shape[0])
    cv2.imshow("BEV Map", bev_canvas)

    cv2.namedWindow("BEV Points Only", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BEV Points Only", bev_points_canvas.shape[1], bev_points_canvas.shape[0])
    cv2.imshow("BEV Points Only", bev_points_canvas)
    
    # Handle keyboard input
    key_raw = cv2.waitKey(30)  # Get full key code
    key = key_raw & 0xFF  # Get ASCII part
    
    if key == ord('q') or key == ord('Q'):
        running = False
        break
    elif key == ord(' '):  # Space bar - pause/resume
        paused = True
        key_pause_raw = None
        key_pause = None
        while paused:
            key_pause_raw = cv2.waitKey(0)  # Get full key code
            key_pause = key_pause_raw & 0xFF  # Get ASCII part
            
            if key_pause == ord(' '):  # Space again to resume
                paused = False
            elif key_pause == ord('q') or key_pause == ord('Q'):
                paused = False
                running = False
                break
            elif is_prev_frame_key(key_pause, key_pause_raw):  # Previous frame
                frame_idx = (frame_idx - 1) % frame_count
                paused = False
                break
            elif is_next_frame_key(key_pause, key_pause_raw):  # Next frame
                frame_idx = (frame_idx + 1) % frame_count
                paused = False
                break
        if not running:
            break
    elif is_prev_frame_key(key, key_raw):  # Previous frame
        frame_idx = (frame_idx - 1) % frame_count
    elif is_next_frame_key(key, key_raw):  # Next frame
        frame_idx = (frame_idx + 1) % frame_count
    else:
        # Auto-advance to next frame when no navigation key is pressed
        if key_raw == -1:  # No key pressed
            frame_idx = (frame_idx + 1) % frame_count

cv2.destroyAllWindows()
