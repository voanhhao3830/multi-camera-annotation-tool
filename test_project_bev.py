"""
Quick BEV demo:
- Load frame 0 of every camera and show them in a single grid window.
- Draw a bounding box on any camera view (click-drag with left mouse).
- The bbox is converted to a projection point via linear interpolation and
  projected to BEV; the BEV point shows up in a separate window.

Dataset layout expected (adapt paths below if needed):
DATA_ROOT/
  Image_subsets/
    1/00000.jpg ...
    2/...
  calibrations/
    Camera1/calibrations/intrinsic_original/intr_Camera1.xml
    Camera1/calibrations/extrinsic/extr_Camera1.xml
    ...
"""

import glob
import os
import shutil
import tempfile
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from labelme.utils.constants import DEFAULT_BEV_X, DEFAULT_BEV_Y
from labelme.utils.project_bev import BBoxToBEVConverter

# ------------------------------------------------------------
# Paths and basic config
# ------------------------------------------------------------
DATA_ROOT = r"D:\MultiCameraLabelTool\aicv\multi-camera-annotation-tool\data\aicv"
IMAGE_SUBSETS_FOLDER = os.path.join(DATA_ROOT, "Image_subsets")
CALIBRATIONS_ROOT = os.path.join(DATA_ROOT, "calibrations")
FRAME_IDX = 0  # which frame to load from each camera
MODEL_PATH = r"D:\MultiCameraLabelTool\chicken_model.pt"
CONF_THRESH = 0.5
model = YOLO(MODEL_PATH)
# Camera folder discovery (expects folders named "1", "2", ...)
CAMERA_FOLDERS = sorted(
    [
        d
        for d in os.listdir(IMAGE_SUBSETS_FOLDER)
        if os.path.isdir(os.path.join(IMAGE_SUBSETS_FOLDER, d))
    ]
)
VIEW_TO_CAMERA = {i: f"Camera{i}" for i in range(1, 7)}
CAMERA_TO_FOLDER = {f"Camera{i}": str(i) for i in range(1, 7)}
FOLDER_TO_CAMERA = {str(i): f"Camera{i}" for i in range(1, 7)}

# Grid rendering settings
GRID_COLS = 3
CELL_WIDTH = 640
CAMPAL = {
    "Camera1": (0, 255, 0),      # green
    "Camera2": (0, 200, 255),    # yellow-ish
    "Camera3": (255, 0, 0),      # blue (BGR)
    "Camera4": (255, 0, 255),    # magenta
    "Camera5": (0, 165, 255),    # orange
    "Camera6": (255, 255, 0),    # cyan
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_calibrations_to_temp(
    calibrations_root: str, cameras: List[str], image_subsets_folder: str
) -> str:
    """
    Copy calibration files into a temp folder laid out as:
    temp/intrinsic/intr_<cam>.xml and temp/extrinsic/extr_<cam>.xml
    so BBoxToBEVConverter can consume them directly.
    """
    temp_calib_folder = tempfile.mkdtemp(prefix="multicam_calib_")
    intrinsic_folder = os.path.join(temp_calib_folder, "intrinsic")
    extrinsic_folder = os.path.join(temp_calib_folder, "extrinsic")
    os.makedirs(intrinsic_folder, exist_ok=True)
    os.makedirs(extrinsic_folder, exist_ok=True)

    for cam_name in cameras:
        cam_root = os.path.join(calibrations_root, cam_name, "calibrations")
        if not os.path.exists(cam_root):
            print(f"[WARN] Missing calibration root for {cam_name}: {cam_root}")
            continue

        # Pick an intrinsic folder
        intrinsic_sub = None
        for sub in ["intrinsic_original", "intrinsic_optimal", "intrinsic"]:
            candidate = os.path.join(cam_root, sub)
            if os.path.exists(candidate):
                intrinsic_sub = candidate
                break

        extrinsic_sub = os.path.join(cam_root, "extrinsic")
        if not intrinsic_sub or not os.path.exists(extrinsic_sub):
            print(f"[WARN] Missing intrinsic/extrinsic for {cam_name}")
            continue

        intr_files = glob.glob(os.path.join(intrinsic_sub, f"intr_{cam_name}.xml"))
        extr_files = glob.glob(os.path.join(extrinsic_sub, f"extr_{cam_name}.xml"))
        if not intr_files:
            all_intr = glob.glob(os.path.join(intrinsic_sub, "*.xml"))
            if all_intr:
                intr_files = [all_intr[0]]
        if not extr_files:
            all_extr = glob.glob(os.path.join(extrinsic_sub, "*.xml"))
            if all_extr:
                extr_files = [all_extr[0]]

        if not intr_files or not extr_files:
            print(f"[WARN] No XML calibration found for {cam_name}")
            continue

        shutil.copy2(intr_files[0], os.path.join(intrinsic_folder, f"intr_{cam_name}.xml"))
        shutil.copy2(extr_files[0], os.path.join(extrinsic_folder, f"extr_{cam_name}.xml"))
        print(f"[CALIB] Loaded {cam_name}")

    return temp_calib_folder


def load_first_frame(folder: str) -> np.ndarray:
    """Load the first image in a camera folder."""
    frame_candidates = [
        os.path.join(IMAGE_SUBSETS_FOLDER, folder, f"{FRAME_IDX:05d}.jpg"),
        os.path.join(IMAGE_SUBSETS_FOLDER, folder, f"{FRAME_IDX:05d}.png"),
    ]
    for fp in frame_candidates:
        if os.path.isfile(fp):
            img = cv2.imread(fp)
            if img is not None:
                return img

    # fallback: lexicographically first image
    files = sorted(
        [
            f
            for f in os.listdir(os.path.join(IMAGE_SUBSETS_FOLDER, folder))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    if not files:
        raise FileNotFoundError(f"No images found in folder {folder}")
    img = cv2.imread(os.path.join(IMAGE_SUBSETS_FOLDER, folder, files[0]))
    if img is None:
        raise FileNotFoundError(f"Cannot read first image in folder {folder}")
    return img


def run_yolo_on_image(img: np.ndarray) -> List[Tuple[float, float, float, float, float, int]]:
    """
    Run YOLO on a single image.
    Returns list of (x1, y1, x2, y2, conf, cls).
    """
    results = model(img, verbose=False)
    if not results:
        return []
    dets: List[Tuple[float, float, float, float, float, int]] = []
    boxes = results[0].boxes
    if boxes is None:
        return dets
    for b in boxes:
        conf = float(b.conf.item())
        if conf < CONF_THRESH:
            continue
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        cls_id = int(b.cls.item()) if b.cls is not None else -1
        dets.append((x1, y1, x2, y2, conf, cls_id))
    return dets


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
    rows = int(np.ceil(len(entries) / cols))
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


def find_cell(layout: List[Dict], x: int, y: int) -> Tuple[str, Dict]:
    """Locate which camera cell was clicked."""
    for entry in layout:
        x0, y0 = entry["x0"], entry["y0"]
        if x0 <= x < x0 + CELL_WIDTH and y0 <= y < y0 + entry["resized_h"]:
            return entry["cam"], entry
    return "", {}


def render_grid(
    images: Dict[str, np.ndarray],
    layout: List[Dict],
    grid_w: int,
    grid_h: int,
    boxes: Dict[str, List[Dict]],
    live_box: Dict,
) -> np.ndarray:
    """Render grid with drawn boxes."""
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for entry in layout:
        cam = entry["cam"]
        img = cv2.resize(images[cam], (CELL_WIDTH, entry["resized_h"]))
        x0, y0 = entry["x0"], entry["y0"]
        canvas[y0 : y0 + entry["resized_h"], x0 : x0 + CELL_WIDTH] = img
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

        # Draw finalized boxes with thin stroke and id
        for rec in boxes.get(cam, []):
            (bx0, by0, bx1, by1) = rec["disp_bbox"]
            color = rec["color"]
            det_id = rec.get("id")
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), color, 1)
            if det_id is not None:
                cv2.putText(
                    canvas,
                    f"{det_id}",
                    (bx0 + 4, by0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        # Draw live box if applicable
        if live_box and live_box.get("cam") == cam and live_box.get("active"):
            (bx0, by0) = live_box["start_disp"]
            (bx1, by1) = live_box["cur_disp"]
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (0, 255, 255), 1)

    return canvas


def render_bev(bev_points: List[Dict]) -> np.ndarray:
    """Render BEV canvas with projected points."""
    bev = np.zeros((DEFAULT_BEV_Y, DEFAULT_BEV_X, 3), dtype=np.uint8)

    # Light grid to help spatially
    grid_step = 50
    for x in range(0, DEFAULT_BEV_X, grid_step):
        cv2.line(bev, (x, 0), (x, DEFAULT_BEV_Y), (40, 40, 40), 1)
    for y in range(0, DEFAULT_BEV_Y, grid_step):
        cv2.line(bev, (0, y), (DEFAULT_BEV_X, y), (40, 40, 40), 1)

    for p in bev_points:
        bx, by = int(p["mem"][0]), int(p["mem"][1])
        color = p["color"]
        det_id = p.get("id")
        cv2.circle(bev, (bx, by), 3, color, -1)
        cv2.circle(bev, (bx, by), 5, color, 1)  # thin outline
        if det_id is not None:
            cv2.putText(
                bev,
                f"{det_id}",
                (bx + 4, by - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        bev,
        "BEV (mem coords)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return bev


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("=" * 60)
    print("Loading images...")
    print("=" * 60)

    images: Dict[str, np.ndarray] = {}
    camera_names: List[str] = []
    for folder in CAMERA_FOLDERS:
        cam_name = FOLDER_TO_CAMERA.get(folder, f"Camera{folder}")
        try:
            img = load_first_frame(folder)
            images[cam_name] = img
            camera_names.append(cam_name)
            print(f"Loaded {cam_name} from folder {folder} -> {img.shape[1]}x{img.shape[0]}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Skip folder {folder}: {exc}")

    if not images:
        print("No images loaded. Check DATA_ROOT and folders.")
        return

    print("\n" + "=" * 60)
    print("Preparing calibrations...")
    print("=" * 60)
    temp_calib = load_calibrations_to_temp(CALIBRATIONS_ROOT, camera_names, IMAGE_SUBSETS_FOLDER)
    bev_converter = BBoxToBEVConverter(temp_calib)

    layout, grid_w, grid_h = compute_grid_layout(images)
    cell_by_cam = {entry["cam"]: entry for entry in layout}
    boxes: Dict[str, List[Dict]] = {}
    bev_points: List[Dict] = []
    live_box = {"active": False}

    det_counter = 0

    def add_bbox_and_project(cam: str, img_bbox: Tuple[float, float, float, float]):
        """Store bbox (disp/img) and project to BEV."""
        cam_color = CAMPAL.get(cam, (0, 255, 0))
        cell = cell_by_cam.get(cam)
        if cell is None:
            return
        scale = cell["scale"]
        disp_bbox = (
            int(img_bbox[0] * scale + cell["x0"]),
            int(img_bbox[1] * scale + cell["y0"]),
            int(img_bbox[2] * scale + cell["x0"]),
            int(img_bbox[3] * scale + cell["y0"]),
        )
        boxes.setdefault(cam, []).append(
            {"disp_bbox": disp_bbox, "img_bbox": img_bbox, "color": cam_color, "id": det_counter}
        )
        try:
            result = bev_converter.bbox_to_bev(cam, img_bbox, z_world=0.0)
            if result.get("success") and result.get("bev_coords") is not None:
                bev_pt = result["bev_coords"]
                bev_points.append({"cam": cam, "mem": bev_pt, "color": cam_color, "id": det_counter})
                print(f"[BEV] {cam} id={det_counter} -> mem {bev_pt}")
            else:
                print(f"[WARN] Projection failed for {cam}: {result}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Projection error for {cam}: {exc}")

    # Auto-detect bboxes using YOLO for each camera image
    print("\n" + "=" * 60)
    print("Running YOLO on first frame of each camera...")
    print("=" * 60)
    for cam, img in images.items():
        dets = run_yolo_on_image(img)
        print(f"[YOLO] {cam}: {len(dets)} detections (conf>={CONF_THRESH})")
        for (x1, y1, x2, y2, conf, cls_id) in dets:
            add_bbox_and_project(cam, (x1, y1, x2, y2))
            det_counter += 1

    def on_mouse(event, mx, my, _flags, _userdata):
        nonlocal live_box, boxes, bev_points
        if event == cv2.EVENT_LBUTTONDOWN:
            cam, cell = find_cell(layout, mx, my)
            if not cam:
                return
            live_box = {
                "active": True,
                "cam": cam,
                "cell": cell,
                "start_disp": (mx, my),
                "cur_disp": (mx, my),
            }
        elif event == cv2.EVENT_MOUSEMOVE and live_box.get("active"):
            live_box["cur_disp"] = (mx, my)
        elif event == cv2.EVENT_LBUTTONUP and live_box.get("active"):
            cam = live_box["cam"]
            cell = live_box["cell"]
            x0, y0 = live_box["start_disp"]
            x1, y1 = mx, my
            live_box["active"] = False

            # Normalize display bbox
            disp_bbox = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            if disp_bbox[2] - disp_bbox[0] < 5 or disp_bbox[3] - disp_bbox[1] < 5:
                print("[INFO] Box too small, ignored.")
                return

            # Map back to original image coordinates
            scale = cell["scale"]
            orig_x0 = (disp_bbox[0] - cell["x0"]) / scale
            orig_y0 = (disp_bbox[1] - cell["y0"]) / scale
            orig_x1 = (disp_bbox[2] - cell["x0"]) / scale
            orig_y1 = (disp_bbox[3] - cell["y0"]) / scale
            img_bbox = (
                max(0.0, min(orig_x0, orig_x1)),
                max(0.0, min(orig_y0, orig_y1)),
                max(0.0, max(orig_x0, orig_x1)),
                max(0.0, max(orig_y0, orig_y1)),
            )

            cam_color = CAMPAL.get(cam, (0, 255, 0))
            boxes.setdefault(cam, []).append({"disp_bbox": disp_bbox, "img_bbox": img_bbox, "color": cam_color})
            print(f"[BOX] {cam} bbox img coords: {img_bbox}")

            # Project to BEV
            try:
                result = bev_converter.bbox_to_bev(cam, img_bbox, z_world=0.0)
                if result.get("success") and result.get("bev_coords") is not None:
                    bev_pt = result["bev_coords"]
                    bev_points.append({"cam": cam, "mem": bev_pt, "color": cam_color})
                    print(f"[BEV] {cam} -> mem {bev_pt}")
                else:
                    print(f"[WARN] Projection failed for {cam}: {result}")
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Projection error for {cam}: {exc}")

    grid_window = "Multi-camera views (draw bbox)"
    bev_window = "BEV projection"
    cv2.namedWindow(grid_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(bev_window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(grid_window, on_mouse)

    try:
        while True:
            grid_img = render_grid(images, layout, grid_w, grid_h, boxes, live_box)
            bev_img = render_bev(bev_points)

            cv2.imshow(grid_window, grid_img)
            cv2.imshow(bev_window, bev_img)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                boxes.clear()
                bev_points.clear()
                print("[INFO] Cleared overlays.")
    finally:
        cv2.destroyAllWindows()
        if os.path.isdir(temp_calib):
            shutil.rmtree(temp_calib, ignore_errors=True)


if __name__ == "__main__":
    main()