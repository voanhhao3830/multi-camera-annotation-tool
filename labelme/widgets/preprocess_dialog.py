from __future__ import annotations

import os
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QFileDialog, QMessageBox, QGroupBox, QFormLayout


class PreprocessDialog(QDialog):
    def __init__(self, parent=None, max_frames=0):
        super().__init__(parent)
        self.setWindowTitle("Preprocessing Settings")
        self.setMinimumWidth(500)
        self.max_frames = max_frames
        
        layout = QVBoxLayout()
        
        # Model selection group
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()
        
        self.model_path_edit = QtWidgets.QLineEdit()
        self.model_path_edit.setPlaceholderText("Select model file (optional)")
        model_browse_btn = QPushButton("Browse...")
        model_browse_btn.clicked.connect(self._browse_model)
        
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(model_browse_btn)
        
        model_layout.addRow("Model Path:", model_path_layout)
        
        self.conf_threshold_spinbox = QDoubleSpinBox()
        self.conf_threshold_spinbox.setMinimum(0.0)
        self.conf_threshold_spinbox.setMaximum(1.0)
        self.conf_threshold_spinbox.setValue(0.5)
        self.conf_threshold_spinbox.setDecimals(2)
        self.conf_threshold_spinbox.setSingleStep(0.05)
        self.conf_threshold_spinbox.setToolTip("Confidence threshold for model detection (0.0 - 1.0)")
        model_layout.addRow("Confidence Threshold:", self.conf_threshold_spinbox)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Frame range group
        frame_group = QGroupBox("Frame Range (Optional)")
        frame_layout = QFormLayout()
        
        self.start_frame_spinbox = QSpinBox()
        self.start_frame_spinbox.setMinimum(0)
        self.start_frame_spinbox.setMaximum(999999)
        self.start_frame_spinbox.setValue(0)
        frame_layout.addRow("Start Frame:", self.start_frame_spinbox)
        
        self.end_frame_spinbox = QSpinBox()
        self.end_frame_spinbox.setMinimum(0)
        self.end_frame_spinbox.setMaximum(999999)
        # Set default to max_frames - 1 (last frame index) if available, otherwise 0
        default_end_frame = max(0, max_frames - 1) if max_frames > 0 else 0
        self.end_frame_spinbox.setValue(default_end_frame)
        frame_layout.addRow("End Frame:", self.end_frame_spinbox)
        
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Run Preprocessing")
        self.ok_button.setDefault(True)
        self.cancel_button = QPushButton("Cancel")
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Connect signals
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
    
    def _browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.pt *.pth *.onnx);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def get_settings(self):
        """Get preprocessing settings"""
        return {
            "model_path": self.model_path_edit.text() if self.model_path_edit.text() else None,
            "conf_threshold": self.conf_threshold_spinbox.value(),
            "start_frame": self.start_frame_spinbox.value(),  # Always return value (0 for start)
            "end_frame": self.end_frame_spinbox.value(),  # Always return value (will be clamped to actual frame count)
        }
