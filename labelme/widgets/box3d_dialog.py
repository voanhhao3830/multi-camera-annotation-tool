"""Dialog for entering 3D box parameters"""

from typing import Optional

from PyQt5 import QtCore
from PyQt5 import QtWidgets


class Box3DDialog(QtWidgets.QDialog):
    """Dialog for entering 3D box parameters"""
    
    def __init__(self, parent=None, x: float = 0.0, y: float = 0.0, z: float = 0.0,
                 w: float = 50.0, h: float = 50.0, d: float = 50.0,
                 label: str = "object", group_id: Optional[int] = None, action: str = "walking"):
        super().__init__(parent)
        self.setWindowTitle("3D Box Parameters")
        self.setModal(True)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Position
        pos_group = QtWidgets.QGroupBox("Position (meters)")
        pos_layout = QtWidgets.QGridLayout()
        
        pos_layout.addWidget(QtWidgets.QLabel("X:"), 0, 0)
        self.edit_x = QtWidgets.QDoubleSpinBox()
        self.edit_x.setRange(-999999.0, 999999.0)
        self.edit_x.setValue(x)
        self.edit_x.setDecimals(2)
        pos_layout.addWidget(self.edit_x, 0, 1)
        
        pos_layout.addWidget(QtWidgets.QLabel("Y:"), 0, 2)
        self.edit_y = QtWidgets.QDoubleSpinBox()
        self.edit_y.setRange(-999999.0, 999999.0)
        self.edit_y.setValue(y)
        self.edit_y.setDecimals(2)
        pos_layout.addWidget(self.edit_y, 0, 3)
        
        pos_layout.addWidget(QtWidgets.QLabel("Z:"), 1, 0)
        self.edit_z = QtWidgets.QDoubleSpinBox()
        self.edit_z.setRange(-999999.0, 999999.0)
        self.edit_z.setValue(z)
        self.edit_z.setDecimals(2)
        pos_layout.addWidget(self.edit_z, 1, 1)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Size
        size_group = QtWidgets.QGroupBox("Size (meters)")
        size_layout = QtWidgets.QGridLayout()
        
        size_layout.addWidget(QtWidgets.QLabel("Width (W):"), 0, 0)
        self.edit_w = QtWidgets.QDoubleSpinBox()
        self.edit_w.setRange(0.1, 999999.0)
        self.edit_w.setValue(w)
        self.edit_w.setDecimals(2)
        self.edit_w.setSingleStep(0.1)
        size_layout.addWidget(self.edit_w, 0, 1)
        
        size_layout.addWidget(QtWidgets.QLabel("Height (H):"), 0, 2)
        self.edit_h = QtWidgets.QDoubleSpinBox()
        self.edit_h.setRange(0.1, 999999.0)
        self.edit_h.setValue(h)
        self.edit_h.setDecimals(2)
        self.edit_h.setSingleStep(0.1)
        size_layout.addWidget(self.edit_h, 0, 3)
        
        size_layout.addWidget(QtWidgets.QLabel("Depth (D):"), 1, 0)
        self.edit_d = QtWidgets.QDoubleSpinBox()
        self.edit_d.setRange(0.1, 999999.0)
        self.edit_d.setValue(d)
        self.edit_d.setDecimals(2)
        self.edit_d.setSingleStep(0.1)
        size_layout.addWidget(self.edit_d, 1, 1)
        
        size_group.setLayout(size_layout)
        layout.addWidget(size_group)
        
        # Label and Group ID
        label_group = QtWidgets.QGroupBox("Label Information")
        label_layout = QtWidgets.QGridLayout()
        
        label_layout.addWidget(QtWidgets.QLabel("Label:"), 0, 0)
        self.edit_label = QtWidgets.QLineEdit()
        self.edit_label.setText(label)
        self.edit_label.setPlaceholderText("Enter label")
        label_layout.addWidget(self.edit_label, 0, 1)
        
        label_layout.addWidget(QtWidgets.QLabel("Group ID:"), 1, 0)
        self.edit_group_id = QtWidgets.QSpinBox()
        self.edit_group_id.setRange(0, 999999)
        if group_id is not None:
            self.edit_group_id.setValue(group_id)
        self.edit_group_id.setSpecialValueText("None")
        label_layout.addWidget(self.edit_group_id, 1, 1)
        
        label_layout.addWidget(QtWidgets.QLabel("Action:"), 2, 0)
        self.edit_action = QtWidgets.QComboBox()
        self.edit_action.addItems(["walking", "eating", "sitting", "standing"])
        # Set current action
        action_index = self.edit_action.findText(action)
        if action_index >= 0:
            self.edit_action.setCurrentIndex(action_index)
        label_layout.addWidget(self.edit_action, 2, 1)
        
        label_group.setLayout(label_layout)
        layout.addWidget(label_group)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def getValues(self) -> tuple[float, float, float, float, float, float, str, Optional[int], str]:
        """Get all values including action"""
        group_id = self.edit_group_id.value() if self.edit_group_id.value() > 0 else None
        return (
            self.edit_x.value(),
            self.edit_y.value(),
            self.edit_z.value(),
            self.edit_w.value(),
            self.edit_h.value(),
            self.edit_d.value(),
            self.edit_label.text(),
            group_id,
            self.edit_action.currentText()
        )

