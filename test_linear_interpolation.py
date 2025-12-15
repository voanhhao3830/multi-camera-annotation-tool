import os
import json
import cv2
import numpy as np
from math import ceil, sqrt
import tkinter as tk
import tempfile
import shutil
import glob
from labelme.utils.linear_interpolation import vector_lerp, get_dynamic_projection_point, estimate_distance_from_camera
from labelme.utils.project_bev import BBoxToBEVConverter

# ==============================================
# CONFIG
# ==============================================
DATA_ROOT = r"D:\MultiCameraLabelTool\aicv\multi-camera-annotation-tool\data\aicv"
IMAGE_SUBSETS_FOLDER = os.path.join(DATA_ROOT, "Image_subsets")
ANNOTATIONS_FOLDER = os.path.join(DATA_ROOT, "annotations_positions")
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


def load_annotation(annotations_folder, frame_idx):
    """Load annotation for a specific frame"""
    annot_path = os.path.join(annotations_folder, f"{frame_idx:05d}.json")
    if os.path.exists(annot_path):
        with open(annot_path, "r") as f:
            data = json.load(f)
            return data
    return []


def compute_layout(n, prefer_cols=True):
    """
    Compute layout with preference for more columns than rows (landscape).
    prefer_cols=True: prioritize more columns (landscape layout)
    """
    if prefer_cols:
        # Start with more columns for landscape layout
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


def interpolate_bbox(bbox0, bbox1, t):
    """Interpolate bounding box between frame0 and frame1"""
    if bbox0 is None or bbox1 is None:
        return None
    
    xmin0, ymin0, xmax0, ymax0 = bbox0
    xmin1, ymin1, xmax1, ymax1 = bbox1
    
    xmin_interp = vector_lerp(xmin0, xmin1, t)
    ymin_interp = vector_lerp(ymin0, ymin1, t)
    xmax_interp = vector_lerp(xmax0, xmax1, t)
    ymax_interp = vector_lerp(ymax0, ymax1, t)
    
    return [xmin_interp, ymin_interp, xmax_interp, ymax_interp]


def get_projection_point_for_bbox(bbox, camera_name, bev_converter):
    """Get projection point for a bounding box"""
    calib_info = bev_converter._get_calibration(camera_name)
    bbox_bottom_px = ((bbox[0] + bbox[2]) / 2, bbox[3])
    camera_extrinsics = {
        "R": calib_info["R"],
        "t": calib_info["tvec"]
    }
    camera_intrinsics = calib_info["camera_matrix"]
    distance = estimate_distance_from_camera(
        bbox_bottom_px, 
        camera_extrinsics, 
        camera_intrinsics, 
        ground_z=0
    )
    projection_point, _ = get_dynamic_projection_point(
        bbox, 
        distance, 
        calib_info.get('rvec'), 
        min_dist=1.0, 
        max_dist=10.0
    )
    return projection_point


# ==============================================
# MAIN
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

# Initialize BEV converter
bev_converter = BBoxToBEVConverter(temp_calib_folder)

print("\n" + "=" * 60)
print("Step 2: Loading frame 0 and frame 1 annotations...")
print("=" * 60)

FRAME_0 = 0
FRAME_1 = 1
# INTERPOLATION_T (t): linear interpolation factor, value from 0.0 to 1.0
# - t = 0.0: result = frame 0 (start point)
# - t = 1.0: result = frame 1 (end point)
# - t = 0.5: result = middle point (50% between frame 0 and frame 1)
# - t = 0.3: result closer to frame 0 (30% from frame 0 to frame 1)
# - t = 0.7: result closer to frame 1 (70% from frame 0 to frame 1)
INTERPOLATION_T = 1  # Change this value to test different interpolation levels

annotation_0 = load_annotation(ANNOTATIONS_FOLDER, FRAME_0)
annotation_1 = load_annotation(ANNOTATIONS_FOLDER, FRAME_1)

print(f"Frame {FRAME_0}: {len(annotation_0)} objects")
print(f"Frame {FRAME_1}: {len(annotation_1)} objects")

# Load images for frame 0
cam_imgs = {}
for cam_folder in CAMERA_FOLDERS:
    img_path = os.path.join(IMAGE_SUBSETS_FOLDER, cam_folder, f"{FRAME_0:05d}.jpg")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        cam_imgs[cam_folder] = img

if len(cam_imgs) == 0:
    print("ERROR: No images found!")
    exit(1)

# Build layout
cam_folders = list(cam_imgs.keys())
C = len(cam_folders)
ROWS, COLS = compute_layout(C, prefer_cols=True)

# Calculate image size to fit screen
available_width = int(screen_width * 0.95)
available_height = int(screen_height * 0.95)

max_cam_width = available_width // COLS
max_cam_height = available_height // ROWS

CAM_WIDTH = min(max_cam_width, int(max_cam_height * ORIGINAL_IMG_WIDTH / ORIGINAL_IMG_HEIGHT))
CAM_HEIGHT = int(CAM_WIDTH * ORIGINAL_IMG_HEIGHT / ORIGINAL_IMG_WIDTH)

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

# Store top-left pixel of each camera tile
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

# Draw bounding boxes and interpolated points
print("\n" + "=" * 60)
print("Step 3: Drawing bounding boxes and interpolated points...")
print("=" * 60)

# Build a map: (personID, viewNum) -> bbox for frame 1
frame1_map = {}
for person_idx, person in enumerate(annotation_1):
    person_id = person.get("personID", person_idx)
    for view_idx, view in enumerate(person.get("views", [])):
        if view.get("xmin") == -1:
            continue
        view_num = view.get("viewNum")
        frame1_map[(person_id, view_num)] = [view["xmin"], view["ymin"], view["xmax"], view["ymax"]]

interpolated_count = 0

# Process frame 0 annotations
for person_idx, person in enumerate(annotation_0):
    person_id = person.get("personID", person_idx)
    
    for view_idx, view in enumerate(person.get("views", [])):
        if view.get("xmin") == -1:
            continue

        view_num = view.get("viewNum")
        cam_name = VIEW_TO_CAMERA.get(view_num)
        cam_folder = str(view_num)
        
        if cam_folder not in tile_pos or cam_name is None:
            continue

        x_off, y_off = tile_pos[cam_folder]

        # Get bbox from frame 0
        bbox0 = [view["xmin"], view["ymin"], view["xmax"], view["ymax"]]
        
        # Scale bounding box coordinates
        xmin = x_off + int(bbox0[0] * scale_x)
        ymin = y_off + int(bbox0[1] * scale_y)
        xmax = x_off + int(bbox0[2] * scale_x)
        ymax = y_off + int(bbox0[3] * scale_y)
        
        # Draw bounding box from frame 0 (green)
        cv2.rectangle(big_canvas, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Calculate center and bottom points of bbox0 for visualization
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        bottom_x = center_x  # Same x as center
        bottom_y = ymax
        
        # Draw center point (blue)
        cv2.circle(big_canvas, (center_x, center_y), 5, (255, 0, 0), -1)  # Blue circle
        cv2.circle(big_canvas, (center_x, center_y), 5, (255, 255, 255), 1)  # White border
        
        # Draw bottom center point (cyan)
        cv2.circle(big_canvas, (bottom_x, bottom_y), 5, (255, 255, 0), -1)  # Cyan circle
        cv2.circle(big_canvas, (bottom_x, bottom_y), 5, (255, 255, 255), 1)  # White border
        
        # Draw line from center to bottom (yellow dashed line)
        cv2.line(big_canvas, (center_x, center_y), (bottom_x, bottom_y), (0, 255, 255), 1)
        
        # Try to find corresponding bbox in frame 1 using personID and viewNum
        bbox1 = frame1_map.get((person_id, view_num))
        
        # If we have both bboxes, interpolate and show interpolated point
        if bbox1 is not None:
            # Interpolate bbox
            bbox_interp = interpolate_bbox(bbox0, bbox1, INTERPOLATION_T)
            
            # Get projection point for interpolated bbox
            # This point is computed by interpolating between center and bottom of the bbox
            try:
                proj_point_interp = get_projection_point_for_bbox(bbox_interp, cam_name, bev_converter)
                
                if proj_point_interp is not None and np.isfinite(proj_point_interp).all():
                    # Scale projection point
                    proj_x = x_off + int(proj_point_interp[0] * scale_x)
                    proj_y = y_off + int(proj_point_interp[1] * scale_y)
                    
                    # Make sure point is within image bounds
                    if 0 <= proj_x < CANVAS_W and 0 <= proj_y < CANVAS_H:
                        # Draw interpolated projection point (red circle with cross).
                        # This is the ground projection point, located between center and bottom.
                        cv2.circle(big_canvas, (proj_x, proj_y), 10, (0, 0, 255), -1)  # Red filled circle
                        cv2.circle(big_canvas, (proj_x, proj_y), 10, (255, 255, 255), 2)  # White border
                        # Draw cross
                        cv2.line(big_canvas, (proj_x - 8, proj_y), (proj_x + 8, proj_y), (255, 255, 255), 2)
                        cv2.line(big_canvas, (proj_x, proj_y - 8), (proj_x, proj_y + 8), (255, 255, 255), 2)
                        
                        # Draw line from center to projection point (red dashed)
                        cv2.line(big_canvas, (center_x, center_y), (proj_x, proj_y), (0, 0, 255), 1)
                        # Draw line from bottom to projection point (red dashed)
                        cv2.line(big_canvas, (bottom_x, bottom_y), (proj_x, proj_y), (0, 0, 255), 1)
                        
                        interpolated_count += 1
                        
                        # Label
                        label = f"{INTERPOLATION_T:.1f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        # Draw background for label
                        cv2.rectangle(big_canvas, (proj_x - label_size[0] // 2 - 2, proj_y - 25),
                                     (proj_x + label_size[0] // 2 + 2, proj_y - 10), (0, 0, 0), -1)
                        cv2.putText(big_canvas, label, (proj_x - label_size[0] // 2, proj_y - 12),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e:
                print(f"Error calculating projection point for {cam_name}, person {person_id}: {e}")

print(f"Interpolated {interpolated_count} points successfully")

# Add legend
legend_y = 30
legend_texts = [
    ("Green Box", (0, 255, 0), f"Frame {FRAME_0} bbox"),
    ("Blue Point", (255, 0, 0), "Center of bbox"),
    ("Cyan Point", (255, 255, 0), "Bottom center of bbox"),
    ("Yellow Line", (0, 255, 255), "Center -> Bottom"),
    ("Red Point", (0, 0, 255), f"Projection point (t={INTERPOLATION_T:.2f})"),
    ("Red Lines", (0, 0, 255), "Center/Bottom -> Projection"),
    ("Info", (255, 255, 255), f"Frame {FRAME_0} -> Frame {FRAME_1}, {interpolated_count} points")
]

for i, (text, color, desc) in enumerate(legend_texts):
    y_pos = legend_y + i * 25
    if i == 0:
        # Draw colored box
        cv2.rectangle(big_canvas, (10, y_pos - 8), (30, y_pos + 8), color, -1)
    elif i == 1 or i == 2:
        # Draw colored point (center or bottom)
        cv2.circle(big_canvas, (20, y_pos), 5, color, -1)
        cv2.circle(big_canvas, (20, y_pos), 5, (255, 255, 255), 1)
    elif i == 3:
        # Draw line (center to bottom)
        cv2.line(big_canvas, (10, y_pos), (30, y_pos), color, 2)
    elif i == 4:
        # Draw projection point (red circle with cross)
        cv2.circle(big_canvas, (20, y_pos), 10, color, -1)
        cv2.circle(big_canvas, (20, y_pos), 10, (255, 255, 255), 2)
        cv2.line(big_canvas, (12, y_pos), (28, y_pos), (255, 255, 255), 2)
        cv2.line(big_canvas, (20, y_pos - 8), (20, y_pos + 8), (255, 255, 255), 2)
    elif i == 5:
        # Draw lines (center/bottom to projection)
        cv2.line(big_canvas, (10, y_pos - 3), (30, y_pos + 3), color, 1)
    # Draw text with background
    text_size, _ = cv2.getTextSize(f"{text}: {desc}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(big_canvas, (35, y_pos - text_size[1] - 5), 
                 (35 + text_size[0] + 10, y_pos + 5), (0, 0, 0), -1)
    cv2.putText(big_canvas, f"{text}: {desc}", (40, y_pos + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Add title
title = f"Linear Interpolation Test - Frame {FRAME_0} (t={INTERPOLATION_T:.2f})"
subtitle = "Projection point: interpolated between center and bottom of the bbox"
title_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
subtitle_size, _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
cv2.putText(big_canvas, title, (CANVAS_W // 2 - title_size[0] // 2, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(big_canvas, subtitle, (CANVAS_W // 2 - subtitle_size[0] // 2, 55),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# Show window
cv2.namedWindow("Linear Interpolation Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Linear Interpolation Test", min(CANVAS_W, screen_width), min(CANVAS_H, screen_height))
cv2.imshow("Linear Interpolation Test", big_canvas)

print("\n" + "=" * 60)
print("Displaying results. Press any key to exit...")
print("=" * 60)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Cleanup temp folder
try:
    shutil.rmtree(temp_calib_folder)
    print(f"\nCleaned up temp calibration folder: {temp_calib_folder}")
except:
    pass

print("Done!")
