"""
YOLO detection utilities for multi-camera annotation preprocessing.
"""

import os
import os.path as osp
import glob
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from loguru import logger


def camera_id_to_view_num(camera_id: str) -> Optional[int]:
    """Convert camera_id to viewNum (Camera1 -> 1, Camera2 -> 2, etc.)"""
    try:
        return int(camera_id.replace("Camera", ""))
    except:
        return None


def convert_to_native(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj


def run_yolo_detection(
    model_path: str,
    image_subsets_folder: str,
    annotations_folder: str,
    camera_folders: List[str],
    frame_count: int,
    start_frame: int,
    conf_threshold: float = 0.5,
    progress_callback: Optional[Callable[[int], None]] = None
) -> int:
    """
    Run YOLO detection on multi-camera images and save annotations.
    
    Args:
        model_path: Path to YOLO model file
        image_subsets_folder: Path to Image_subsets folder
        annotations_folder: Path to annotations_positions folder
        camera_folders: List of camera folder names (e.g., ["1", "2", "3"])
        frame_count: Number of frames to process
        start_frame: Starting frame index (absolute)
        conf_threshold: Confidence threshold for YOLO detection
        progress_callback: Optional callback function(progress_value) for progress updates
        
    Returns:
        Number of frames with detections created
    """
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError as e:
        logger.error(f"Failed to import YOLO dependencies: {e}")
        raise ImportError("ultralytics and cv2 are required for YOLO detection")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    
    logger.info(f"Loading YOLO model from {model_path}")
    yolo_model = YOLO(model_path)
    
    detections_created = 0
    
    # Process each frame
    for frame_idx in range(frame_count):
        actual_frame_idx = start_frame + frame_idx
        
        # Update progress
        if progress_callback and frame_idx % 10 == 0:
            # YOLO detection: 45-70 (25% range)
            progress_value = 45 + int(25 * frame_idx / frame_count)
            progress_callback(progress_value)
        
        frame_detections_dict = {}
        camera_data_for_frame = []
        
        # Load images from all cameras for this frame
        for cam_folder in camera_folders:
            cam_path = os.path.join(image_subsets_folder, cam_folder)
            if not os.path.isdir(cam_path):
                continue
                
            image_files = sorted(
                glob.glob(os.path.join(cam_path, "*.jpg")) + 
                glob.glob(os.path.join(cam_path, "*.png")) + 
                glob.glob(os.path.join(cam_path, "*.jpeg"))
            )
            
            if actual_frame_idx < len(image_files):
                image_path = image_files[actual_frame_idx]
                camera_id = f"Camera{cam_folder}"
                
                # Load image and run YOLO
                image = cv2.imread(image_path)
                if image is not None:
                    results = yolo_model(image, verbose=False, conf=conf_threshold)
                    
                    # Extract detections
                    detections = []
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append({
                                "xmin": int(x1),
                                "ymin": int(y1),
                                "xmax": int(x2),
                                "ymax": int(y2),
                                "confidence": float(box.conf[0].cpu().numpy()),
                                "class": int(box.cls[0].cpu().numpy())
                            })
                    
                    frame_detections_dict[camera_id] = detections
                    camera_data_for_frame.append({
                        "camera_id": camera_id,
                        "image_path": image_path
                    })
        
        # Save detections to annotation file
        if frame_detections_dict:
            # Build camera to viewNum mapping
            camera_to_view = {}
            for cam_data in camera_data_for_frame:
                camera_id = cam_data["camera_id"]
                view_num = camera_id_to_view_num(camera_id)
                if view_num:
                    camera_to_view[camera_id] = view_num
            
            max_view_num = max(camera_to_view.values()) if camera_to_view else 0
            
            if max_view_num > 0:
                persons = []
                person_id = 0
                
                # Create a person for each detection
                for camera_id, detections in frame_detections_dict.items():
                    view_num = camera_to_view.get(camera_id)
                    if view_num is None:
                        continue
                    
                    for detection in detections:
                        person = {
                            "personID": person_id,
                            "ground_points": [-1, -1],
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
                        
                        # Update the view for this camera
                        person["views"][view_num - 1] = {
                            "viewNum": view_num,
                            "xmin": detection["xmin"],
                            "ymin": detection["ymin"],
                            "xmax": detection["xmax"],
                            "ymax": detection["ymax"]
                        }
                        
                        persons.append(person)
                        person_id += 1
                
                # Save to annotation file
                # Use actual_frame_idx (absolute index) for file naming
                frame_id = f"{actual_frame_idx:05d}"
                annot_file = os.path.join(annotations_folder, f"{frame_id}.json")
                
                # Convert numpy types to Python native types for JSON serialization
                persons_serializable = convert_to_native(persons)
                
                with open(annot_file, 'w') as f:
                    json.dump(persons_serializable, f, indent=4)
                
                if len(persons) > 0:
                    detections_created += 1
    
    logger.info(f"Created detections for {detections_created} frames using YOLO")
    return detections_created


def run_yolo_detection_return_dict(
    model_path: str,
    image_subsets_folder: str,
    camera_folders: List[str],
    frame_count: int,
    start_frame: int = 0,
    conf_threshold: float = 0.5,
    progress_callback: Optional[Callable[[int], None]] = None
) -> Tuple[Dict[str, List[List[List[float]]]], List[Dict[str, np.ndarray]]]:
    """
    Run YOLO detection with 1 model for all cameras and return detections dict.
    Uses the exact same logic from run_yolo_detection but returns detections instead of saving files.
    
    Args:
        model_path: Path to YOLO model file
        image_subsets_folder: Path to Image_subsets folder
        camera_folders: List of camera folder names (e.g., ["1", "2", "3"])
        frame_count: Number of frames to process
        start_frame: Starting frame index (absolute, default 0)
        conf_threshold: Confidence threshold for YOLO detection
        progress_callback: Optional callback function(progress_value) for progress updates
        
    Returns:
        Tuple of:
        - multi_camera_detections: Dict[str, List[List[List[float]]]] - {camera_name: [frame0_bboxes, frame1_bboxes, ...]}
          where each bbox is [x1, y1, x2, y2]
        - frame_images: List[Dict[str, np.ndarray]] - [frame0_images, frame1_images, ...]
          where each frame is {camera_name: image}
    """
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError as e:
        logger.error(f"Failed to import YOLO dependencies: {e}")
        raise ImportError("ultralytics and cv2 are required for YOLO detection")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    
    logger.info(f"Loading YOLO model from {model_path}")
    yolo_model = YOLO(model_path)
    
    # Build camera_name list from camera_folders
    camera_names = [f"Camera{cam_folder}" for cam_folder in camera_folders]
    
    # Initialize output structures
    multi_camera_detections: Dict[str, List[List[List[float]]]] = {
        cam: [] for cam in camera_names
    }
    frame_images: List[Dict[str, np.ndarray]] = [dict() for _ in range(frame_count)]
    
    # Process each frame - using EXACT same logic from run_yolo_detection
    for frame_idx in range(frame_count):
        actual_frame_idx = start_frame + frame_idx
        
        # Update progress
        if progress_callback and frame_idx % 10 == 0:
            progress_value = int(100 * frame_idx / frame_count)
            progress_callback(progress_value)
        
        # Load images from all cameras for this frame - EXACT same as run_yolo_detection
        for cam_folder in camera_folders:
            cam_path = os.path.join(image_subsets_folder, cam_folder)
            if not os.path.isdir(cam_path):
                continue
                
            image_files = sorted(
                glob.glob(os.path.join(cam_path, "*.jpg")) + 
                glob.glob(os.path.join(cam_path, "*.png")) + 
                glob.glob(os.path.join(cam_path, "*.jpeg"))
            )
            
            if actual_frame_idx < len(image_files):
                image_path = image_files[actual_frame_idx]
                camera_id = f"Camera{cam_folder}"
                
                # Load image and run YOLO - EXACT same as run_yolo_detection
                image = cv2.imread(image_path)
                if image is not None:
                    # Run YOLO detection - EXACT same as run_yolo_detection
                    results = yolo_model(image, verbose=False, conf=conf_threshold)
                    
                    # Extract detections - EXACT same as run_yolo_detection
                    detections = []
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append({
                                "xmin": int(x1),
                                "ymin": int(y1),
                                "xmax": int(x2),
                                "ymax": int(y2),
                                "confidence": float(box.conf[0].cpu().numpy()),
                                "class": int(box.cls[0].cpu().numpy())
                            })
                    
                    # Convert detections format from dict to list of [x1,y1,x2,y2]
                    bboxes: List[List[float]] = []
                    for det in detections:
                        bboxes.append([float(det["xmin"]), float(det["ymin"]), 
                                     float(det["xmax"]), float(det["ymax"])])
                    
                    multi_camera_detections[camera_id].append(bboxes)
                    frame_images[frame_idx][camera_id] = image
                else:
                    multi_camera_detections[camera_id].append([])
            else:
                # If no image for this frame, add empty list
                camera_id = f"Camera{cam_folder}"
                if camera_id not in multi_camera_detections:
                    multi_camera_detections[camera_id] = []
                multi_camera_detections[camera_id].append([])
    
    logger.info(f"Completed YOLO detection for {frame_count} frames")
    return multi_camera_detections, frame_images

