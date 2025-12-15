"""
Quick viewer to show the first frame of each camera in a grid.
Click any camera cell to project the clicked ground point to all other cameras.

Expected dataset layout (edit the paths below to match yours):
DATA_ROOT/
  Image_subsets/
    1/00000.jpg, 00001.jpg, ...
    2/...
  calibrations/
    Camera1/calibrations/intrinsic_original/intr_Camera1.xml
    Camera1/calibrations/extrinsic/extr_Camera1.xml
    ...
"""

import os
import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

from labelme.utils.calibration import CameraCalibration

# ----- User editable paths -----
DATA_ROOT = r"D:\MultiCameraLabelTool\aicv\aicv"
IMAGE_ROOT = os.path.join(DATA_ROOT, "Image_subsets")
CALIB_ROOT = os.path.join(DATA_ROOT, "calibrations")
CAMERAS: List[str] = [f"Camera{i}" for i in range(1, 7)]
CAMERA_TO_VIEW_FOLDER: Dict[str, str] = {f"Camera{i}": str(i) for i in range(1, 7)}

# Grid settings
GRID_COLS = 3
CELL_WIDTH = 640


def load_first_frame(cam: str) -> np.ndarray:
    """Load the first frame for a camera."""
    folder = CAMERA_TO_VIEW_FOLDER.get(cam, cam)
    view_dir = os.path.join(IMAGE_ROOT, folder)
    if not os.path.isdir(view_dir):
        raise FileNotFoundError(f"Missing image folder for {cam}: {view_dir}")

    # Try a fixed name first, then fall back to the lexicographically first image.
    first_candidates = [os.path.join(view_dir, "00000.jpg"), os.path.join(view_dir, "00000.png")]
    for fp in first_candidates:
        if os.path.isfile(fp):
            img = cv2.imread(fp)
            if img is not None:
                return img

    images = sorted([f for f in os.listdir(view_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if not images:
        raise FileNotFoundError(f"No images found in {view_dir}")

    img_path = os.path.join(view_dir, images[0])
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image {img_path}")
    return img


def load_calibration(cam: str, image_size: Tuple[int, int]) -> CameraCalibration:
    """
    Load intrinsic/extrinsic from default calibration layout.
    Automatically scales calibration to match actual image size.
    """
    cam_dir = os.path.join(CALIB_ROOT, cam, "calibrations")
    intr_path = os.path.join(cam_dir, "intrinsic_original", f"intr_{cam}.xml")
    extr_path = os.path.join(cam_dir, "extrinsic", f"extr_{cam}.xml")
    
    # Load base calibration
    # Translation scale: multiply by 100
    # Intrinsic scale: divide by 4
    calib = CameraCalibration.load_from_xml_files_scaled(
        intr_path, 
        extr_path,
        intrinsic_scale=1.0 / 4.0,  # Divide intrinsic by 4
        translation_scale=100.0      # Multiply translation by 100
    )
    if calib is None:
        raise FileNotFoundError(f"Cannot load calibration for {cam}: {intr_path} / {extr_path}")
    
    # Check if scaling is needed based on principal point (cx, cy)
    img_h, img_w = image_size
    K = calib.intrinsic
    cx_orig = K[0, 2]
    cy_orig = K[1, 2]
    fx_orig = K[0, 0]
    fy_orig = K[1, 1]
    
    print(f"[CALIB] {cam}:")
    print(f"  Image size: {img_w}x{img_h}")
    print(f"  Original intrinsic: fx={fx_orig:.1f}, fy={fy_orig:.1f}, cx={cx_orig:.1f}, cy={cy_orig:.1f}")
    
    # If principal point is way outside image bounds, calibration is for different size
    # Estimate scale: if cx is roughly at center, original width ≈ 2*cx
    if cx_orig > img_w * 0.8 or cy_orig > img_h * 0.8:
        # Calibration appears to be for larger image
        # Estimate original size from principal point (assuming it's near center)
        orig_w_est = max(cx_orig * 2, img_w)
        orig_h_est = max(cy_orig * 2, img_h)
        
        # Calculate scale factor
        scale_w = img_w / orig_w_est
        scale_h = img_h / orig_h_est
        scale = (scale_w + scale_h) / 2.0
        
        # Sanity check: scale should be reasonable
        if 0.1 <= scale <= 2.0:
            print(f"  Scaling by {scale:.4f} (estimated orig size: {orig_w_est:.0f}x{orig_h_est:.0f})")
            scaled_intrinsic = CameraCalibration._scale_intrinsic(K, scale)
            calib = CameraCalibration(scaled_intrinsic, calib.extrinsic, calib.distortion)
            print(f"  Scaled intrinsic: fx={scaled_intrinsic[0,0]:.1f}, fy={scaled_intrinsic[1,1]:.1f}, "
                  f"cx={scaled_intrinsic[0,2]:.1f}, cy={scaled_intrinsic[1,2]:.1f}")
        else:
            print(f"  Scale {scale:.4f} seems unreasonable, using calibration as-is")
    else:
        print(f"  Calibration seems to match image size, using as-is")
    
    return calib


def compute_grid_layout(images: Dict[str, np.ndarray]) -> Tuple[List[Dict], int, int]:
    """Pre-compute grid cell layout and sizes."""
    entries: List[Dict] = []
    for cam, img in images.items():
        h, w = img.shape[:2]
        scale = CELL_WIDTH / float(w)
        resized_h = int(h * scale)
        entries.append(
            {
                "cam": cam,
                "orig_h": h,
                "orig_w": w,
                "scale": scale,
                "resized_h": resized_h,
            }
        )

    cols = GRID_COLS
    rows = math.ceil(len(entries) / cols)
    row_heights = [0] * rows
    for idx, entry in enumerate(entries):
        row_heights[idx // cols] = max(row_heights[idx // cols], entry["resized_h"])

    y_offsets: List[int] = []
    running_y = 0
    for rh in row_heights:
        y_offsets.append(running_y)
        running_y += rh

    grid_w = cols * CELL_WIDTH
    grid_h = running_y

    for idx, entry in enumerate(entries):
        row = idx // cols
        col = idx % cols
        entry["x0"] = col * CELL_WIDTH
        entry["y0"] = y_offsets[row]
        entry["cell_h"] = row_heights[row]
    return entries, grid_w, grid_h


def render_grid(images: Dict[str, np.ndarray], layout: List[Dict], grid_w: int, grid_h: int,
                points: Dict[str, List[Tuple[int, int, Tuple[int, int, int]]]]) -> np.ndarray:
    """Compose the grid image with optional projected points."""
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for entry in layout:
        cam = entry["cam"]
        img = images[cam].copy()

        # Draw projected points for this camera
        for x, y, color in points.get(cam, []):
            cv2.circle(img, (int(x), int(y)), 8, color, 2)
            cv2.circle(img, (int(x), int(y)), 4, color, -1)

        resized = cv2.resize(img, (CELL_WIDTH, entry["resized_h"]))
        x0, y0 = entry["x0"], entry["y0"]
        canvas[y0:y0 + entry["resized_h"], x0:x0 + CELL_WIDTH] = resized
        cv2.putText(
            canvas,
            cam,
            (x0 + 10, y0 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def find_cell(layout: List[Dict], x: int, y: int) -> Tuple[str, Dict]:
    """Locate which camera cell was clicked."""
    for entry in layout:
        x0, y0 = entry["x0"], entry["y0"]
        if x0 <= x < x0 + CELL_WIDTH and y0 <= y < y0 + entry["resized_h"]:
            return entry["cam"], entry
    return "", {}


def main():
    # Load images and calibrations
    images: Dict[str, np.ndarray] = {}
    calibrations: Dict[str, CameraCalibration] = {}
    for cam in CAMERAS:
        try:
            img = load_first_frame(cam)
            images[cam] = img
            img_h, img_w = img.shape[:2]
            calibrations[cam] = load_calibration(cam, (img_h, img_w))
            print(f"Loaded {cam} - Image: {img_w}x{img_h}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Skip {cam}: {exc}")

    if not images:
        print("No images loaded. Please check paths.")
        return

    layout, grid_w, grid_h = compute_grid_layout(images)
    points: Dict[str, List[Tuple[int, int, Tuple[int, int, int]]]] = {}
    window_name = "Multi-view grid (click to project)"

    def on_mouse(event, mx, my, _flags, _userdata):
        nonlocal points
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        cam_name, cell = find_cell(layout, mx, my)
        if not cam_name:
            print(f"[DEBUG] Click at grid ({mx}, {my}) - not in any cell")
            return

        print(f"\n{'='*60}")
        print(f"[DEBUG] Click detected on {cam_name}")
        print(f"[DEBUG] Grid click coordinates: ({mx}, {my})")
        print(f"[DEBUG] Cell bounds: x0={cell['x0']}, y0={cell['y0']}, scale={cell['scale']:.4f}")

        # Map click to original pixel coordinates
        orig_x = (mx - cell["x0"]) / cell["scale"]
        orig_y = (my - cell["y0"]) / cell["scale"]
        print(f"[DEBUG] Original image coordinates: ({orig_x:.2f}, {orig_y:.2f})")
        print(f"[DEBUG] Original image size: {cell['orig_w']}x{cell['orig_h']}")

        src_calib = calibrations.get(cam_name)
        if src_calib is None:
            print(f"[ERROR] No calibration for {cam_name}")
            return

        print(f"[DEBUG] Projecting 2D -> MEM (ground plane)...")
        mem_pt = src_calib.project_2d_to_mem(np.array([[orig_x, orig_y]], dtype=np.float32))
        print(f"[DEBUG] MEM coordinates: {mem_pt}")
        if np.any(np.isnan(mem_pt)):
            print("[ERROR] Projection to ground failed (NaN).")
            return

        # Collect projected points for all cameras
        new_points: Dict[str, List[Tuple[int, int, Tuple[int, int, int]]]] = {}
        print(f"\n[DEBUG] Projecting MEM -> 2D for all cameras:")
        for tgt_cam, tgt_calib in calibrations.items():
            img_h, img_w = images[tgt_cam].shape[:2]
            print(f"\n  [{tgt_cam}] Image size: {img_w}x{img_h}")
            
            proj = tgt_calib.project_mem_to_2d(mem_pt)
            print(f"  [{tgt_cam}] Projection result: {proj}")
            
            if proj is None:
                print(f"  [{tgt_cam}] SKIP: proj is None")
                continue
            if np.any(np.isnan(proj)):
                print(f"  [{tgt_cam}] SKIP: contains NaN")
                continue
                
            px, py = float(proj[0, 0]), float(proj[0, 1])
            print(f"  [{tgt_cam}] Pixel coordinates: ({px:.2f}, {py:.2f})")
            print(f"  [{tgt_cam}] Bounds check: 0 <= {px:.2f} < {img_w} and 0 <= {py:.2f} < {img_h}")
            
            if 0 <= px < img_w and 0 <= py < img_h:
                color = (0, 0, 255) if tgt_cam == cam_name else (0, 255, 0)
                new_points.setdefault(tgt_cam, []).append((int(px), int(py), color))
                print(f"  [{tgt_cam}] ✓ ADDED point at ({int(px)}, {int(py)})")
            else:
                print(f"  [{tgt_cam}] ✗ SKIP: out of bounds")
        
        print(f"\n[DEBUG] Total points added: {sum(len(v) for v in new_points.values())}")
        print(f"{'='*60}\n")
        points = new_points

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        grid_img = render_grid(images, layout, grid_w, grid_h, points)
        cv2.imshow(window_name, grid_img)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
