"""Utilities for loading and using camera calibrations for 3D projection"""

import json
import os.path as osp
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any, List
import numpy as np
from loguru import logger

try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("OpenCV not available, calibration loading may fail")


class CameraCalibration:
    """Represents camera calibration data"""
    
    def __init__(self, intrinsic: np.ndarray, extrinsic: np.ndarray, distortion: Optional[np.ndarray] = None):
        """
        Initialize camera calibration
        
        Args:
            intrinsic: 3x3 camera intrinsic matrix
            extrinsic: 4x4 camera extrinsic matrix (world to camera)
            distortion: distortion coefficients (optional)
        """
        self.intrinsic = np.array(intrinsic, dtype=np.float32)
        self.extrinsic = np.array(extrinsic, dtype=np.float32)
        self.distortion = np.array(distortion, dtype=np.float32) if distortion is not None else np.zeros(5, dtype=np.float32)
        
        # Validate matrices
        if self.intrinsic.shape != (3, 3):
            raise ValueError(f"Intrinsic matrix must be 3x3, got {self.intrinsic.shape}")
        if self.extrinsic.shape != (4, 4):
            raise ValueError(f"Extrinsic matrix must be 4x4, got {self.extrinsic.shape}")
    
    @classmethod
    def load_from_xml_files(cls, intrinsic_path: str, extrinsic_path: str) -> Optional['CameraCalibration']:
        """Load calibration from OpenCV XML files"""
        try:
            # Load intrinsic
            intrinsic = cls._load_intrinsic_xml(intrinsic_path)
            if intrinsic is None:
                return None
            
            # Load extrinsic
            extrinsic = cls._load_extrinsic_xml(extrinsic_path)
            if extrinsic is None:
                return None
            
            # Load distortion (from intrinsic file)
            distortion = cls._load_distortion_xml(intrinsic_path)
            
            return cls(intrinsic, extrinsic, distortion)
        except Exception as e:
            logger.error(f"Failed to load calibration from XML files: {e}")
            return None
    
    @classmethod
    def load_from_xml_files_scaled(
        cls, 
        intrinsic_path: str, 
        extrinsic_path: str,
        intrinsic_scale: float = 1.0,
        translation_scale: float = 1.0
    ) -> Optional['CameraCalibration']:
        """
        Load calibration from OpenCV XML files with scaling applied.
        
        Args:
            intrinsic_path: Path to intrinsic XML file
            extrinsic_path: Path to extrinsic XML file
            intrinsic_scale: Scale factor for intrinsic matrix (fx, fy, cx, cy)
            translation_scale: Scale factor for translation vector in extrinsic
            
        Returns:
            CameraCalibration with scaled parameters
        """
        try:
            # Load intrinsic
            intrinsic = cls._load_intrinsic_xml(intrinsic_path)
            if intrinsic is None:
                return None
            
            # Load extrinsic
            extrinsic = cls._load_extrinsic_xml(extrinsic_path)
            if extrinsic is None:
                return None
            
            # Load distortion (from intrinsic file)
            distortion = cls._load_distortion_xml(intrinsic_path)
            
            # Apply intrinsic scale
            scaled_intrinsic = cls._scale_intrinsic(intrinsic, intrinsic_scale)
            
            # Apply translation scale
            scaled_extrinsic = cls._scale_extrinsic_translation(extrinsic, translation_scale)
            
            return cls(scaled_intrinsic, scaled_extrinsic, distortion)
        except Exception as e:
            logger.error(f"Failed to load scaled calibration from XML files: {e}")
            return None
    
    @staticmethod
    def _scale_intrinsic(K: np.ndarray, scale: float) -> np.ndarray:
        """Scale intrinsic matrix by given factor"""
        K_scaled = K.copy()
        K_scaled[0, 0] *= scale  # fx
        K_scaled[1, 1] *= scale  # fy
        K_scaled[0, 2] *= scale  # cx
        K_scaled[1, 2] *= scale  # cy
        return K_scaled
    
    @staticmethod
    def _scale_extrinsic_translation(extrinsic: np.ndarray, scale: float) -> np.ndarray:
        """Scale translation vector in extrinsic matrix"""
        scaled = extrinsic.copy()
        scaled[:3, 3] *= scale  # Scale translation vector
        return scaled
    
    @staticmethod
    def _load_intrinsic_xml(filepath: str) -> Optional[np.ndarray]:
        """Load intrinsic matrix from OpenCV XML file"""
        if not osp.exists(filepath):
            logger.warning(f"Intrinsic file not found: {filepath}")
            return None
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Find camera_matrix
            camera_matrix_elem = root.find('camera_matrix')
            if camera_matrix_elem is None:
                logger.error(f"No camera_matrix found in {filepath}")
                return None
            
            # Parse matrix data
            rows = int(camera_matrix_elem.find('rows').text)
            cols = int(camera_matrix_elem.find('cols').text)
            data_str = camera_matrix_elem.find('data').text
            data = np.array([float(x) for x in data_str.split()], dtype=np.float32)
            
            # Reshape to matrix
            matrix = data.reshape(rows, cols)
            return matrix
        except Exception as e:
            logger.error(f"Failed to parse intrinsic XML {filepath}: {e}")
            return None
    
    @staticmethod
    def _load_distortion_xml(filepath: str) -> Optional[np.ndarray]:
        """Load distortion coefficients from OpenCV XML file"""
        if not osp.exists(filepath):
            return None
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Find distortion_coefficients
            dist_elem = root.find('distortion_coefficients')
            if dist_elem is None:
                return np.zeros(5, dtype=np.float32)
            
            # Parse distortion data
            rows = int(dist_elem.find('rows').text)
            data_str = dist_elem.find('data').text
            data = np.array([float(x) for x in data_str.split()], dtype=np.float32)
            
            return data
        except Exception as e:
            logger.warning(f"Failed to parse distortion from {filepath}: {e}")
            return np.zeros(5, dtype=np.float32)
    
    @staticmethod
    def _load_extrinsic_xml(filepath: str) -> Optional[np.ndarray]:
        """Load extrinsic matrix from OpenCV XML file (rvec + tvec -> 4x4 matrix)"""
        if not osp.exists(filepath):
            logger.warning(f"Extrinsic file not found: {filepath}")
            return None
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Find rvec and tvec
            rvec_elem = root.find('rvec')
            tvec_elem = root.find('tvec')
            
            if rvec_elem is None or tvec_elem is None:
                logger.error(f"Missing rvec or tvec in {filepath}")
                return None
            
            # Parse rvec (rotation vector)
            rvec_data_str = rvec_elem.find('data').text
            rvec = np.array([float(x) for x in rvec_data_str.split()], dtype=np.float32)
            
            # Parse tvec (translation vector)
            tvec_data_str = tvec_elem.find('data').text
            tvec = np.array([float(x) for x in tvec_data_str.split()], dtype=np.float32)
            
            # Convert rotation vector to rotation matrix using Rodrigues
            if cv2 is None:
                logger.error("OpenCV is required for Rodrigues conversion")
                return None
            R, _ = cv2.Rodrigues(rvec)
            
            # Build 4x4 extrinsic matrix (world to camera)
            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = tvec
            
            return extrinsic
        except Exception as e:
            logger.error(f"Failed to parse extrinsic XML {filepath}: {e}")
            return None
    
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['CameraCalibration']:
        """Load calibration from JSON file (legacy support)"""
        if not osp.exists(filepath):
            logger.warning(f"Calibration file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            intrinsic = np.array(data.get("intrinsic", np.eye(3)), dtype=np.float32)
            extrinsic = np.array(data.get("extrinsic", np.eye(4)), dtype=np.float32)
            distortion = np.array(data.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float32)
            
            return cls(intrinsic, extrinsic, distortion)
        except Exception as e:
            logger.error(f"Failed to load calibration from {filepath}: {e}")
            return None
    
    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates using OpenCV model.
        
        Args:
            points_3d: Nx3 array of 3D points in world coordinates
            
        Returns:
            Nx2 array of 2D image coordinates, NaN for points behind camera.
        """
        if points_3d.shape[1] != 3:
            raise ValueError(f"points_3d must be Nx3, got {points_3d.shape}")
        
        # Ensure float32
        points_3d = np.asarray(points_3d, dtype=np.float32)
        
        # Compute camera coordinates to know which points are in front of camera
        n_points = points_3d.shape[0]
        points_3d_homogeneous = np.hstack([points_3d, np.ones((n_points, 1), dtype=np.float32)])
        points_cam = (self.extrinsic @ points_3d_homogeneous.T).T
        valid_mask = points_cam[:, 2] > 0  # z > 0 in camera frame
        # Use OpenCV's projectPoints for consistency with calibration files
        R = self.extrinsic[:3, :3]
        t = self.extrinsic[:3, 3]
        rvec, _ = cv2.Rodrigues(R)
        image_points, _ = cv2.projectPoints(
            points_3d,
            rvec,
            t,
            self.intrinsic,
            self.distortion,
        )
        result = image_points.reshape(-1, 2)

        
        # Mark invalid points as NaN
        result = result.astype(np.float32)
        result[~valid_mask] = np.nan
        
        return result
    
    def project_3d_box_to_2d(
        self,
        center_3d: np.ndarray,
        size_3d: np.ndarray,
        rotation: float = 0.0,
    ) -> Optional[np.ndarray]:
        """
        Project a 3D bounding box to 2D image coordinates.

        This follows the same convention as `anylabeling`'s Box3D:
        - center_3d: [x, y, z] where x,y are ground-plane coordinates, z is bottom height.
        - size_3d: [width, height, depth] where:
            * width  (w) is lateral size on ground (y-axis),
            * depth  (d) is longitudinal size on ground (x-axis),
            * height (h) is vertical size (z-axis).
        - rotation: rotation around the vertical (z) axis in radians.

        Returns:
            8x2 array of 2D corner coordinates, or None if box is not visible.
        """
        w, h, d = size_3d  # width (y), height (z), depth/length (x)
        cx, cy, cz = center_3d

        # 1) Build 2D rectangle on ground plane (x,y) with length=d, width=w
        half_w = w / 2.0
        half_l = d / 2.0

        # Local 2D corners before rotation (same order as anylabeling Box3D.get_corners_2d)
        corners_local = np.array(
            [
                [-half_l, -half_w],
                [half_l, -half_w],
                [half_l, half_w],
                [-half_l, half_w],
            ],
            dtype=np.float32,
        )

        # Apply rotation around z-axis (in ground plane)
        if rotation != 0.0:
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            rot_2d = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
            corners_rot = (rot_2d @ corners_local.T).T
        else:
            corners_rot = corners_local

        # Translate to world (x,y), bottom face at z = cz
        bottom = np.zeros((4, 3), dtype=np.float32)
        bottom[:, 0] = corners_rot[:, 0] + cx  # x
        bottom[:, 1] = corners_rot[:, 1] + cy  # y
        bottom[:, 2] = cz                      # z

        # Top face at z = cz + h
        top = bottom.copy()
        top[:, 2] = cz + h

        corners_3d = np.vstack([bottom, top])  # 8x3

        # Project to 2D
        corners_2d = self.project_3d_to_2d(corners_3d)

        # If all projected points are invalid, box is not visible
        if np.all(np.isnan(corners_2d)):
            return None

        return corners_2d
    
    def get_2d_bbox_from_3d_box(self, center_3d: np.ndarray, size_3d: np.ndarray,
                                rotation: float = 0.0) -> Optional[np.ndarray]:
        """
        Get 2D bounding box from 3D box projection
        
        Args:
            center_3d: [x, y, z] center of box in world coordinates
            size_3d: [width, height, depth] size of box
            rotation: rotation angle around z-axis (in radians)
            
        Returns:
            [x_min, y_min, x_max, y_max] or None if box is not visible
        """
        corners_2d = self.project_3d_box_to_2d(center_3d, size_3d, rotation)
        if corners_2d is None:
            return None
        
        # Filter out NaN values
        valid_corners = corners_2d[~np.isnan(corners_2d).any(axis=1)]
        if len(valid_corners) == 0:
            return None
        
        x_min = np.min(valid_corners[:, 0])
        y_min = np.min(valid_corners[:, 1])
        x_max = np.max(valid_corners[:, 0])
        y_max = np.max(valid_corners[:, 1])
        
        return np.array([x_min, y_min, x_max, y_max])


def generate_bev_from_cameras(
    camera_calibrations: Dict[str, CameraCalibration],
    camera_data: List[Dict[str, Any]],
    bev_width: float,
    bev_height: float,
    resolution: float = 0.2,
) -> Optional[np.ndarray]:
    """
    Generate a simple BEV (top-down) image from multiple cameras.

    The ground plane is assumed to be z=0 in world coordinates and the BEV
    covers [-bev_width/2, bev_width/2] x [-bev_height/2, bev_height/2].

    Args:
        camera_calibrations: mapping camera_id -> CameraCalibration.
        camera_data: list of dicts with at least keys: 'camera_id', 'image_path'.
        bev_width: BEV width in meters.
        bev_height: BEV height in meters.
        resolution: meters per pixel (smaller -> higher resolution).

    Returns:
        HxWx3 uint8 RGB image, or None if generation fails.
    """
    try:
        import cv2  # type: ignore[import]
    except Exception:
        logger.warning("OpenCV not available, cannot generate BEV overlay")
        return None

    if not camera_calibrations or not camera_data:
        return None

    # Determine BEV grid size in pixels
    bev_width = float(bev_width)
    bev_height = float(bev_height)
    resolution = float(max(resolution, 1e-3))

    n_x = max(1, int(bev_width / resolution))
    n_y = max(1, int(bev_height / resolution))

    # World coordinates on ground plane (z=0), centered at origin
    xs = np.linspace(-bev_width / 2.0, bev_width / 2.0, n_x, dtype=np.float32)
    ys = np.linspace(-bev_height / 2.0, bev_height / 2.0, n_y, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X, dtype=np.float32)
    points_world = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N,3)

    accum = np.zeros((points_world.shape[0], 3), dtype=np.float32)
    counts = np.zeros(points_world.shape[0], dtype=np.int32)

    for cam in camera_data:
        camera_id = cam.get("camera_id")
        image_path = cam.get("image_path")
        if not camera_id or not image_path:
            continue

        calib = camera_calibrations.get(camera_id)
        if calib is None:
            continue

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Failed to read image for BEV: {image_path}")
            continue

        h, w = img.shape[:2]

        # Project ground points into this camera
        # print("generate canvas")  # REMOVED - debug statement
        pts_2d = calib.project_3d_to_2d(points_world)  # (N,2) with NaNs for invalid
        valid = ~np.isnan(pts_2d).any(axis=1)
        if not np.any(valid):
            continue

        u = pts_2d[valid, 0]
        v = pts_2d[valid, 1]

        inside = (u >= 0) & (u <= (w - 1)) & (v >= 0) & (v <= (h - 1))
        if not np.any(inside):
            continue

        idx = np.where(valid)[0][inside]
        if len(idx) == 0:
            continue

        u_inside = u[inside].astype(np.int32)
        v_inside = v[inside].astype(np.int32)

        # Sample pixels (nearest neighbour)
        # cv2.imread returns BGR format, convert to RGB
        sampled_bgr = img[v_inside, u_inside].astype(np.float32)  # (N, 3) BGR
        # Convert BGR to RGB by reversing channels
        sampled_rgb = sampled_bgr[:, ::-1]  # Reverse last dimension: BGR -> RGB

        accum[idx] += sampled_rgb
        counts[idx] += 1

    valid_points = counts > 0
    if not np.any(valid_points):
        logger.warning("No valid BEV pixels generated from cameras")
        return None

    bev = np.zeros((points_world.shape[0], 3), dtype=np.uint8)
    bev[valid_points] = (accum[valid_points] / counts[valid_points, None]).astype(
        np.uint8
    )

    bev = bev.reshape(n_y, n_x, 3)

    # Post-process for better visibility: enhance contrast while preserving original colors
    try:
        # Convert to LAB color space for better contrast enhancement
        # LAB preserves color information better than RGB when enhancing brightness/contrast
        bev_lab = cv2.cvtColor(bev, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(bev_lab)
        
        # Apply CLAHE to L channel only (lightness) to enhance contrast while preserving color channels
        # Use moderate clipLimit to avoid over-enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        # Slightly increase brightness to make it more visible
        # Scale L channel moderately to preserve natural look
        l_enhanced = np.clip(l_enhanced.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        
        # Merge back with original color channels (a and b) to preserve original colors
        bev_lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        bev_rgb = cv2.cvtColor(bev_lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # Apply slight saturation boost to make colors more vibrant (optional)
        # Convert to HSV, slightly boost saturation, convert back
        bev_hsv = cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = cv2.split(bev_hsv)
        # Boost saturation slightly (1.15x) to make colors pop while keeping original hue
        s_enhanced = np.clip(s * 1.15, 0, 255).astype(np.uint8)
        bev_hsv_enhanced = cv2.merge([h.astype(np.uint8), s_enhanced, v.astype(np.uint8)])
        bev_rgb = cv2.cvtColor(bev_hsv_enhanced, cv2.COLOR_HSV2RGB)
        
        return bev_rgb
    except Exception as e:
        logger.warning(f"Failed to enhance BEV image contrast, using raw mosaic: {e}")
        return bev
        