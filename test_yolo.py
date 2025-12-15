from ultralytics import YOLO
import os
import cv2
import numpy as np
from math import ceil, sqrt
import natsort
import tkinter as tk
import json

model_path = r"D:\MultiCameraLabelTool\chicken_model.pt"

DATA_ROOT = r"D:\MultiCameraLabelTool\aicv\multi-camera-annotation-tool\data\aicv"
IMAGE_SUBSETS_FOLDER = os.path.join(DATA_ROOT, "Image_subsets")

model = YOLO(model_path)

# Number of frames to process
NUM_FRAMES = 200


def scan_aicv_camera_data(root_dir: str, frame_index: int = 0):
    """
    Scan AICV folder structure and return camera data for a specific frame.
    
    Structure:
        root_dir/
            Image_subsets/
                1/  (images for camera 1)
                2/  (images for camera 2)
                ...
    """
    camera_data = []
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    image_subsets_dir = os.path.join(root_dir, "Image_subsets")
    
    if not os.path.exists(image_subsets_dir):
        print(f"Image_subsets folder not found: {image_subsets_dir}")
        return camera_data
    
    # Scan numbered folders in Image_subsets
    subdirs = sorted([d for d in os.listdir(image_subsets_dir) 
                      if os.path.isdir(os.path.join(image_subsets_dir, d))])
    
    for folder_num in subdirs:
        folder_path = os.path.join(image_subsets_dir, folder_num)
        
        # Find all images in this folder
        frame_files = []
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in extensions):
                frame_files.append(os.path.join(folder_path, file))
        
        frame_files = natsort.os_sorted(frame_files)
        
        if frame_index < len(frame_files):
            image_path = frame_files[frame_index]
            camera_id = f"Camera{folder_num}"
            
            camera_data.append({
                "camera_id": camera_id,
                "image_path": image_path,
            })
    
    return camera_data


def draw_yolo_detections(image, results):
    """Draw bounding boxes from YOLO results on the image."""
    annotated = image.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            # Color for bounding box (BGR)
            color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label (confidence)
            label = f"{conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated


def create_camera_grid(images_dict, max_window_size=None):
    """
    Create a grid layout from a dictionary of camera images with an optimal size.
    
    Args:
        images_dict: {camera_id: image}
        max_window_size: (max_width, max_height) - window size limit, None for auto-detect
    """
    if not images_dict:
        return None
    
    num_cameras = len(images_dict)
    
    # Get screen size if max_window_size is not provided
    if max_window_size is None:
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        # Leave a small margin around the window
        max_window_size = (screen_width - 100, screen_height - 100)
    
    max_width, max_height = max_window_size
    
    # Use the first image size to compute aspect ratio
    first_img = list(images_dict.values())[0]
    img_h, img_w = first_img.shape[:2]
    aspect_ratio = img_w / img_h
    
    # Compute optimal number of rows and columns
    cols = ceil(sqrt(num_cameras))
    rows = ceil(num_cameras / cols)
    
    # Compute cell size based on rows/cols and max window size,
    # while preserving original image aspect ratio.
    cell_width = min(max_width // cols, int((max_height // rows) * aspect_ratio))
    cell_height = int(cell_width / aspect_ratio)
    
    # Make sure the grid does not exceed the screen size
    if cols * cell_width > max_width:
        cell_width = max_width // cols
        cell_height = int(cell_width / aspect_ratio)
    
    if rows * cell_height > max_height:
        cell_height = max_height // rows
        cell_width = int(cell_height * aspect_ratio)
    
    # Create the big canvas
    canvas_width = cols * cell_width
    canvas_height = rows * cell_height
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < num_cameras:
                camera_id = list(images_dict.keys())[idx]
                img = images_dict[camera_id]
                
                # Resize image while preserving aspect ratio
                img_resized = cv2.resize(img, (cell_width, cell_height), interpolation=cv2.INTER_LINEAR)
                
                # Position on the canvas
                x0 = c * cell_width
                y0 = r * cell_height
                
                # Place the image onto the canvas
                canvas[y0:y0+cell_height, x0:x0+cell_width] = img_resized
                
                # Draw camera name with background for readability
                label = camera_id
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # Background cho text
                cv2.rectangle(canvas, (x0 + 5, y0 + 5), (x0 + tw + 10, y0 + th + baseline + 10), 
                             (0, 0, 0), -1)
                # Text
                cv2.putText(canvas, label, (x0 + 8, y0 + th + 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                idx += 1
    
    return canvas


def camera_id_to_view_num(camera_id):
    """Convert Camera1, Camera2, ... to viewNum 1, 2, ..."""
    try:
        # Extract number from "Camera1", "Camera2", etc.
        num_str = camera_id.replace("Camera", "")
        return int(num_str)
    except:
        return None


def extract_detections_from_results(results):
    """Extract bounding boxes from YOLO results"""
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            detections.append({
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "confidence": conf,
                "class": cls
            })
    return detections


def save_detections_to_json(frame_idx, detections_dict, camera_data, output_folder):
    """
    Save detections in the `annotations_positions` format.
    
    Args:
        detections_dict: {camera_id: list of detections}
        camera_data: list of camera data to know which cameras are available
    
    Output format: list of person objects, each with personID, ground_points, views.
    """
    # Create folder if needed
    os.makedirs(output_folder, exist_ok=True)
    
    # Build mapping camera_id -> viewNum from all cameras
    camera_to_view = {}
    for cam_data in camera_data:
        camera_id = cam_data["camera_id"]
        view_num = camera_id_to_view_num(camera_id)
        if view_num:
            camera_to_view[camera_id] = view_num
    
    # Find maximum camera index to create enough views
    max_view_num = max(camera_to_view.values()) if camera_to_view else 0
    
    if max_view_num == 0:
        print(f"  Warning: No valid cameras found for frame {frame_idx}")
        return None
    
    # Create list of persons.
    # Each detection becomes a separate person.
    persons = []
    person_id = 0
    
    # Group detections by camera
    for camera_id, detections in detections_dict.items():
        view_num = camera_to_view.get(camera_id)
        if view_num is None:
            continue
        
        for detection in detections:
            # Create a new person for each detection
            person = {
                "personID": person_id,
                "ground_points": [-1, -1],  # No ground point from YOLO
                "views": []
            }
            
            # Initialize all views with -1
            for v in range(1, max_view_num + 1):
                person["views"].append({
                    "viewNum": v,
                    "xmin": -1,
                    "ymin": -1,
                    "xmax": -1,
                    "ymax": -1
                })
            
            # Update the view for this specific camera
            person["views"][view_num - 1] = {
                "viewNum": view_num,
                "xmin": detection["xmin"],
                "ymin": detection["ymin"],
                "xmax": detection["xmax"],
                "ymax": detection["ymax"]
            }
            
            persons.append(person)
            person_id += 1
    
    # Save JSON file
    output_file = os.path.join(output_folder, f"{frame_idx:05d}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(persons, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved {len(persons)} detections to {output_file}")
    return output_file


def process_frame(frame_idx, save_detections=True):
    """
    Process a single frame: load images from all cameras, run YOLO, and return annotated images.
    
    Returns:
        annotated_images: dict {camera_id: annotated_image}
        detections_dict: dict {camera_id: list of detections}
        camera_data: list of camera data
    """
    print(f"Processing frame {frame_idx}...")
    
    # Scan camera data for this frame
    camera_data = scan_aicv_camera_data(DATA_ROOT, frame_idx)
    
    if not camera_data:
        print(f"No camera data found for frame {frame_idx}")
        return None, None, None
    
    annotated_images = {}
    detections_dict = {}
    
    # Process each camera
    for cam_data in camera_data:
        camera_id = cam_data["camera_id"]
        image_path = cam_data["image_path"]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # Run YOLO detection
        results = model(image, verbose=False, conf=0.5)
        
        # Extract detections
        detections = extract_detections_from_results(results)
        detections_dict[camera_id] = detections
        
        # Draw detections onto the image
        annotated = draw_yolo_detections(image, results)
        annotated_images[camera_id] = annotated
    
    return annotated_images, detections_dict, camera_data


def main():
    """Main function: process the first NUM_FRAMES frames and visualize results."""
    print(f"Starting YOLO detection on first {NUM_FRAMES} frames...")
    print(f"Data root: {DATA_ROOT}")
    
    # Create detection_positions folder
    detection_folder = os.path.join(DATA_ROOT, "detection_positions")
    print(f"Output folder: {detection_folder}")
    
    for frame_idx in range(NUM_FRAMES):
        # Process frame
        annotated_images, detections_dict, camera_data = process_frame(frame_idx, save_detections=True)
        
        if annotated_images is None or not annotated_images:
            print(f"Skipping frame {frame_idx} - no valid images")
            continue
        
        # Save detections to JSON
        if detections_dict and camera_data:
            save_detections_to_json(frame_idx, detections_dict, camera_data, detection_folder)
        
        # Create grid layout with optimal size
        grid_image = create_camera_grid(annotated_images)
        
        if grid_image is None:
            continue
        
        # Add frame information on the image
        cv2.putText(grid_image, f"Frame: {frame_idx}/{NUM_FRAMES-1}", 
                   (10, grid_image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("Multi-Camera YOLO Detection", grid_image)
        
        # Press 'q' to quit, any other key to continue
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stopped by user")
            break
    
    cv2.destroyAllWindows()
    print(f"Done! Detection results saved to: {detection_folder}")


if __name__ == "__main__":
    main()
