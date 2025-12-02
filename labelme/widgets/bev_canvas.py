"""BEV (Bird's Eye View) canvas for 3D box placement"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPointF, Qt
from loguru import logger

from labelme.shape import Shape


class BEVCanvas(QtWidgets.QWidget):
    """Canvas for BEV view to place 3D boxes"""
    
    boxPlaced = QtCore.pyqtSignal(float, float, float, float, float, float)  # x, y, z, w, h, d
    boxSelected = QtCore.pyqtSignal(int)  # box index
    boxMoved = QtCore.pyqtSignal(int, float, float, float)  # box index, new x, new y, new z
    boxSizeChanged = QtCore.pyqtSignal(int, float, float, float)  # box index, new w, new h, new d
    boxDeleted = QtCore.pyqtSignal(int, str, object)  # box index, label, group_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard focus
        
        # BEV view settings
        self.bev_scale = 1.0  # pixels per meter
        self.bev_offset_x = 0.0  # offset in meters
        self.bev_offset_y = 0.0  # offset in meters
        self.bev_width = 50.0  # meters
        self.bev_height = 50.0  # meters
        
        # 3D boxes: list of (center, size, rotation, label, group_id)
        self.boxes_3d: list[tuple] = []
        self.selected_box_idx: Optional[int] = None
        self.hovered_box_idx: Optional[int] = None
        
        # Drawing state
        self.drawing_box = False
        self.drawing_start_pos: Optional[QPointF] = None
        # Default 3D box size in meters: width, height, depth
        self.current_box_size = [50.0, 160.0, 50.0]
        
        # Movement step size (meters)
        # These will typically be overridden by the control in the BEV dock,
        # but set sensible defaults here as well.
        self.move_step = 15.0
        self.move_step_large = 150.0  # Large step size (with Shift)
        
        # Camera overlay data
        self.camera_calibrations: dict = {}  # camera_id -> calibration
        self.camera_data: list = []  # List of camera data dicts
        # Whether to draw camera positions/FOV on top of BEV background
        self.show_camera_overlay: bool = True
        
        # Colors - make them more transparent so BEV overlay is clearly visible
        self.grid_color = QtGui.QColor(200, 200, 200, 60)  # Very light grid
        self.box_color = QtGui.QColor(0, 255, 0)
        self.selected_box_color = QtGui.QColor(255, 0, 0)
        self.hovered_box_color = QtGui.QColor(255, 255, 0)
        self.camera_color = QtGui.QColor(0, 150, 255)  # Blue for cameras
        # Used only if we ever re-enable FOV fill; for now we draw outline only
        self.camera_fov_color = QtGui.QColor(0, 150, 255, 16)

        # Optional BEV background image (e.g. stitched top-down view)
        self.background_image: QtGui.QImage | None = None
        # Use full opacity so BEV overlay is clearly visible
        self.background_alpha: float = 1.0
    
    def setMoveStep(self, step: float):
        """Set movement step size"""
        self.move_step = step
        self.move_step_large = step * 10.0

    def setBackgroundImage(self, image: QtGui.QImage | str, alpha: float = 0.8) -> None:
        """Set BEV background image (top-down overlay) and optional opacity."""
        if isinstance(image, str):
            img = QtGui.QImage(image)
            if img.isNull():
                logger.warning(f"Failed to load BEV background image: {image}")
                return
            self.background_image = img
        else:
            self.background_image = image

        # Clamp alpha between 0 and 1
        self.background_alpha = float(max(0.0, min(alpha, 1.0)))
        self.update()
    
    def setCameraOverlay(self, camera_calibrations: dict, camera_data: list):
        """Set camera calibrations and data for overlay"""
        self.camera_calibrations = camera_calibrations
        self.camera_data = camera_data
        self.update()
    
    def setBEVParams(self, width: float, height: float, scale: float = 1.0):
        """Set BEV view parameters"""
        self.bev_width = width
        self.bev_height = height
        self.bev_scale = scale
        self.update()
    
    def addBox3D(self, x: float, y: float, z: float, w: float, h: float, d: float,
                 label: str = "", group_id: Optional[int] = None, rotation: float = 0.0):
        """Add a 3D box"""
        self.boxes_3d.append((np.array([x, y, z]), np.array([w, h, d]), rotation, label, group_id))
        self.update()
    
    def clearBoxes(self):
        """Clear all 3D boxes"""
        self.boxes_3d = []
        self.selected_box_idx = None
        self.hovered_box_idx = None
        self.update()
    
    def _world_to_screen(self, x: float, y: float) -> QPointF:
        """Convert world coordinates to screen coordinates"""
        screen_x = (x - self.bev_offset_x) * self.bev_scale + self.width() / 2
        screen_y = (y - self.bev_offset_y) * self.bev_scale + self.height() / 2
        return QPointF(screen_x, screen_y)
    
    def _screen_to_world(self, screen_pos: QPointF) -> tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        x = (screen_pos.x() - self.width() / 2) / self.bev_scale + self.bev_offset_x
        y = (screen_pos.y() - self.height() / 2) / self.bev_scale + self.bev_offset_y
        return x, y
    
    def _get_box_at_position(self, pos: QPointF) -> Optional[int]:
        """Get box index at given screen position"""
        for idx, (center, size, rotation, _, _) in enumerate(self.boxes_3d):
            screen_center = self._world_to_screen(center[0], center[1])
            w_screen = size[0] * self.bev_scale  # width
            d_screen = size[2] * self.bev_scale  # depth
            
            # Simple rectangle check (ignoring rotation for now)
            dx = pos.x() - screen_center.x()
            dy = pos.y() - screen_center.y()
            
            if abs(dx) <= w_screen / 2 and abs(dy) <= d_screen / 2:
                return idx
        
        return None
    
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint BEV view"""
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Draw background first (BEV overlay if available, otherwise flat gray)
        if self.background_image is not None and not self.background_image.isNull():
            target_rect = self.rect()
            pixmap = QtGui.QPixmap.fromImage(self.background_image)
            scaled = pixmap.scaled(
                target_rect.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            src_x = max(0, (scaled.width() - target_rect.width()) // 2)
            src_y = max(0, (scaled.height() - target_rect.height()) // 2)
            src_rect = QtCore.QRect(src_x, src_y, target_rect.width(), target_rect.height())

            p.save()
            p.setOpacity(self.background_alpha)
            p.drawPixmap(target_rect, scaled, src_rect)
            p.restore()
        else:
            # Fallback: flat dark background (should not be hit if BEV overlay is set)
            p.fillRect(self.rect(), QtGui.QColor(30, 30, 30))
        
        # Draw grid - only within visible widget bounds to avoid overflow.
        # Keep it very subtle so BEV overlay is clearly visible.
        p.setPen(QtGui.QPen(self.grid_color, 1))
        grid_spacing = 5.0  # meters
        grid_spacing_screen = grid_spacing * self.bev_scale
        
        widget_width = self.width()
        widget_height = self.height()
        center_x = widget_width / 2
        center_y = widget_height / 2
        
        # Only draw grid within widget bounds with small margin
        margin = 100  # pixels
        visible_left = -margin
        visible_right = widget_width + margin
        visible_top = -margin
        visible_bottom = widget_height + margin
        
        # Helper function to safely clamp to int32 range
        def safe_int(value):
            return max(-2147483648, min(int(value), 2147483647))
        
        # Vertical lines - only draw lines that intersect visible area
        if grid_spacing_screen > 0:
            # Find first grid line that's within or before visible area
            y_start_world = (visible_top - center_y) / self.bev_scale + self.bev_offset_y
            y_end_world = (visible_bottom - center_y) / self.bev_scale + self.bev_offset_y
            
            # Start from a grid line near the visible area
            grid_y_world = (y_start_world // grid_spacing) * grid_spacing
            max_iterations = int((y_end_world - y_start_world) / grid_spacing) + 10
            
            iteration = 0
            while grid_y_world <= y_end_world and iteration < max_iterations:
                y_screen = center_y + (grid_y_world - self.bev_offset_y) * self.bev_scale
                
                # Only draw if within visible bounds
                if visible_top <= y_screen <= visible_bottom:
                    y_int = safe_int(y_screen)
                    start_x_int = safe_int(visible_left)
                    end_x_int = safe_int(visible_right)
                    
                    if abs(start_x_int) < 2000000000 and abs(end_x_int) < 2000000000:
                        p.drawLine(start_x_int, y_int, end_x_int, y_int)
                
                grid_y_world += grid_spacing
                iteration += 1
        
        # Horizontal lines - only draw lines that intersect visible area
        if grid_spacing_screen > 0:
            # Find first grid line that's within or before visible area
            x_start_world = (visible_left - center_x) / self.bev_scale + self.bev_offset_x
            x_end_world = (visible_right - center_x) / self.bev_scale + self.bev_offset_x
            
            # Start from a grid line near the visible area
            grid_x_world = (x_start_world // grid_spacing) * grid_spacing
            max_iterations = int((x_end_world - x_start_world) / grid_spacing) + 10
            
            iteration = 0
            while grid_x_world <= x_end_world and iteration < max_iterations:
                x_screen = center_x + (grid_x_world - self.bev_offset_x) * self.bev_scale
                
                # Only draw if within visible bounds
                if visible_left <= x_screen <= visible_right:
                    x_int = safe_int(x_screen)
                    start_y_int = safe_int(visible_top)
                    end_y_int = safe_int(visible_bottom)
                    
                    if abs(start_y_int) < 2000000000 and abs(end_y_int) < 2000000000:
                        p.drawLine(x_int, start_y_int, x_int, end_y_int)
                
                grid_x_world += grid_spacing
                iteration += 1
        
        # Draw origin with coordinate labels
        origin = self._world_to_screen(0, 0)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2))
        p.drawLine(int(origin.x() - 10), int(origin.y()), int(origin.x() + 10), int(origin.y()))
        p.drawLine(int(origin.x()), int(origin.y() - 10), int(origin.x()), int(origin.y() + 10))
        
        # Draw coordinate labels at origin
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
        p.drawText(int(origin.x() + 12), int(origin.y() - 12), "Origin (0, 0)")
        
        # Draw coordinate axes labels
        p.drawText(int(origin.x() + 12), int(origin.y() + 5), "X →")
        p.drawText(int(origin.x() - 25), int(origin.y() - 12), "↑ Y")
        
        # Draw camera overlays (positions and FOV)
        if self.show_camera_overlay and self.camera_calibrations and self.camera_data:
            for cam_data in self.camera_data:
                camera_id = cam_data.get("camera_id")
                calibration = self.camera_calibrations.get(camera_id)
                
                if not calibration:
                    continue
                
                # Get camera position in world space
                R = calibration.extrinsic[:3, :3]
                t = calibration.extrinsic[:3, 3]
                camera_pos_world = -R.T @ t
                
                # Draw camera position
                cam_screen = self._world_to_screen(camera_pos_world[0], camera_pos_world[1])
                p.setPen(QtGui.QPen(self.camera_color, 3))
                p.setBrush(QtGui.QBrush(self.camera_color, Qt.SolidPattern))
                p.drawEllipse(int(cam_screen.x() - 5), int(cam_screen.y() - 5), 10, 10)
                
                # # Draw camera label
                # p.setPen(QtGui.QPen(self.camera_color, 1))
                # p.drawText(int(cam_screen.x() + 8), int(cam_screen.y() - 8), camera_id)
                
                # Draw camera FOV (project image corners to ground)
                try:
                    image_path = cam_data.get("image_path")
                    if image_path:
                        image = QtGui.QImage(image_path)
                        if not image.isNull():
                            img_width = image.width()
                            img_height = image.height()
                        else:
                            fx = calibration.intrinsic[0, 0]
                            cx = calibration.intrinsic[0, 2]
                            img_width = int(cx * 2)
                            img_height = int(calibration.intrinsic[1, 2] * 2)
                    else:
                        fx = calibration.intrinsic[0, 0]
                        cx = calibration.intrinsic[0, 2]
                        img_width = int(cx * 2)
                        img_height = int(calibration.intrinsic[1, 2] * 2)
                    
                    fx = calibration.intrinsic[0, 0]
                    fy = calibration.intrinsic[1, 1]
                    cx = calibration.intrinsic[0, 2]
                    cy = calibration.intrinsic[1, 2]
                    
                    # Project image corners to ground plane
                    corners_2d = [
                        (0, 0),  # top-left
                        (img_width, 0),  # top-right
                        (img_width, img_height),  # bottom-right
                        (0, img_height),  # bottom-left
                    ]
                    
                    fov_points = []
                    for u, v in corners_2d:
                        x_norm = (u - cx) / fx
                        y_norm = (v - cy) / fy
                        ray_dir_cam = np.array([x_norm, y_norm, 1.0])
                        ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)
                        ray_dir_world = R.T @ ray_dir_cam
                        
                        if abs(ray_dir_world[2]) > 1e-6:
                            t_intersect = -camera_pos_world[2] / ray_dir_world[2]
                            if t_intersect > 0:
                                point_world = camera_pos_world + t_intersect * ray_dir_world
                                fov_screen = self._world_to_screen(point_world[0], point_world[1])
                                fov_points.append(fov_screen)
                    
                    # Draw FOV polygon: OUTLINE ONLY so BEV overlay is fully visible
                    if len(fov_points) >= 3:
                        polygon = QtGui.QPolygonF(fov_points)
                        p.setPen(QtGui.QPen(self.camera_color, 1, Qt.DashLine))
                        p.setBrush(QtCore.Qt.NoBrush)
                        p.drawPolygon(polygon)
                except Exception as e:
                    logger.debug(f"Failed to draw camera FOV for {camera_id}: {e}")
        
        # Draw 3D boxes (top-down view, showing w and d)
        for idx, (center, size, rotation, label, group_id) in enumerate(self.boxes_3d):
            screen_center = self._world_to_screen(center[0], center[1])
            # Use w (width) and d (depth) for BEV display
            w_screen = size[0] * self.bev_scale  # width
            d_screen = size[2] * self.bev_scale  # depth
            
            # Determine color
            if idx == self.selected_box_idx:
                color = self.selected_box_color
            elif idx == self.hovered_box_idx:
                color = self.hovered_box_color
            else:
                color = self.box_color
            
            p.setPen(QtGui.QPen(color, 2))
            p.setBrush(QtGui.QBrush(color, Qt.SolidPattern))
            
            # Draw rectangle (top-down view: w x d, small size)
            rect = QtCore.QRectF(
                screen_center.x() - w_screen / 2,
                screen_center.y() - d_screen / 2,
                w_screen,
                d_screen
            )
            p.drawRect(rect)
            
            # Draw label if available
            if label:
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
                p.drawText(rect, Qt.AlignCenter, label)
        
        # Draw current drawing box
        if self.drawing_box and self.drawing_start_pos:
            current_pos = self.drawing_start_pos
            x, y = self._screen_to_world(current_pos)
            center_x_world, center_y_world = self._screen_to_world(self.drawing_start_pos)
            
            # Draw preview box (top-down view: w x d)
            screen_center = self._world_to_screen(center_x_world, center_y_world)
            w_screen = self.current_box_size[0] * self.bev_scale  # width
            d_screen = self.current_box_size[2] * self.bev_scale  # depth
            
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2, Qt.DashLine))
            p.setBrush(QtCore.Qt.NoBrush)
            rect = QtCore.QRectF(
                screen_center.x() - w_screen / 2,
                screen_center.y() - d_screen / 2,
                w_screen,
                d_screen
            )
            p.drawRect(rect)
        
        p.end()
    
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press"""
        pos = event.localPos()
        
        if event.button() == Qt.LeftButton:
            # Check if clicking on existing box
            box_idx = self._get_box_at_position(pos)
            if box_idx is not None:
                self.selected_box_idx = box_idx
                self.boxSelected.emit(box_idx)
                self.update()
            else:
                # Start drawing new box
                self.drawing_box = True
                self.drawing_start_pos = pos
                self.update()
        elif event.button() == Qt.RightButton:
            # Right-click to edit box
            box_idx = self._get_box_at_position(pos)
            if box_idx is not None:
                self.selected_box_idx = box_idx
                self.boxSelected.emit(box_idx)
                self.update()
    
    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle double-click to edit box size"""
        if event.button() == Qt.LeftButton:
            pos = event.localPos()
            box_idx = self._get_box_at_position(pos)
            if box_idx is not None:
                self.selected_box_idx = box_idx
                self.boxSelected.emit(box_idx)
                # Emit signal to open edit dialog
                center, size, rotation, label, group_id = self.boxes_3d[box_idx]
                self.boxSizeChanged.emit(box_idx, size[0], size[1], size[2])
                self.update()
    
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move"""
        pos = event.localPos()
        
        # Update hovered box
        box_idx = self._get_box_at_position(pos)
        if box_idx != self.hovered_box_idx:
            self.hovered_box_idx = box_idx
            self.update()
        
        if self.drawing_box:
            self.update()
        
        # Update tooltip with world coordinates
        x, y = self._screen_to_world(pos)
        self.setToolTip(f"X: {x:.2f}m, Y: {y:.2f}m")
    
    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release"""
        if event.button() == Qt.LeftButton and self.drawing_box:
            pos = event.localPos()
            x, y = self._screen_to_world(pos)
            
            # Emit signal with box parameters
            # z will be set by dialog, using default for now
            self.boxPlaced.emit(
                x, y, 0.0,  # x, y, z (z will be updated by dialog)
                self.current_box_size[0],  # w
                self.current_box_size[1],  # h
                self.current_box_size[2]   # d
            )
            
            self.drawing_box = False
            self.drawing_start_pos = None
            self.update()
    
    def setCurrentBoxSize(self, w: float, h: float, d: float):
        """Set current box size for drawing"""
        self.current_box_size = [w, h, d]
    
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle keyboard input for moving boxes"""
        # Delete selected box with Delete/Backspace
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.selected_box_idx is not None and 0 <= self.selected_box_idx < len(self.boxes_3d):
                center, size, rotation, label, group_id = self.boxes_3d[self.selected_box_idx]
                # Emit signal so MainWindow can remove all projected 2D boxes
                self.boxDeleted.emit(self.selected_box_idx, label, group_id)
                # Remove from BEV
                del self.boxes_3d[self.selected_box_idx]
                self.selected_box_idx = None
                self.hovered_box_idx = None
                self.update()
            else:
                super().keyPressEvent(event)
            return

        if self.selected_box_idx is None or self.selected_box_idx >= len(self.boxes_3d):
            super().keyPressEvent(event)
            return
        
        # Get current box
        center, size, rotation, label, group_id = self.boxes_3d[self.selected_box_idx]
        
        # Check for modifier keys
        is_shift = event.modifiers() & Qt.ShiftModifier
        step = self.move_step_large if is_shift else self.move_step
        
        new_center = center.copy()
        moved = False
        
        if event.key() == Qt.Key_Left:
            new_center[0] -= step
            moved = True
        elif event.key() == Qt.Key_Right:
            new_center[0] += step
            moved = True
        elif event.key() == Qt.Key_Up:
            new_center[1] -= step
            moved = True
        elif event.key() == Qt.Key_Down:
            new_center[1] += step
            moved = True
        elif event.key() == Qt.Key_PageUp:
            new_center[2] += step
            moved = True
        elif event.key() == Qt.Key_PageDown:
            new_center[2] -= step
            moved = True
        
        if moved:
            # Update box position
            self.boxes_3d[self.selected_box_idx] = (new_center, size, rotation, label, group_id)
            
            # Emit signal to update projection
            self.boxMoved.emit(self.selected_box_idx, new_center[0], new_center[1], new_center[2])
            
            self.update()
        else:
            super().keyPressEvent(event)
    
    def getBox3D(self, box_idx: int) -> Optional[tuple]:
        """Get 3D box data by index"""
        if 0 <= box_idx < len(self.boxes_3d):
            return self.boxes_3d[box_idx]
        return None
    
    def updateBox3D(self, box_idx: int, x: float, y: float, z: float, 
                   w: float = None, h: float = None, d: float = None):
        """Update 3D box position and/or size"""
        if 0 <= box_idx < len(self.boxes_3d):
            center, size, rotation, label, group_id = self.boxes_3d[box_idx]
            new_center = np.array([x, y, z])
            new_size = size.copy()
            if w is not None:
                new_size[0] = w
            if h is not None:
                new_size[1] = h
            if d is not None:
                new_size[2] = d
            self.boxes_3d[box_idx] = (new_center, new_size, rotation, label, group_id)
            self.update()

