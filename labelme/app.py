from __future__ import annotations

import enum
import functools
import html
import json
import math
import os
import os.path as osp
import re
import types
import typing
from typing import Optional
import webbrowser

import imgviz
import natsort
import numpy as np
# import osam
from loguru import logger
from numpy.typing import NDArray
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from labelme import __appname__
from labelme import __version__
from labelme._automation import bbox_from_text
from labelme._label_file import LabelFile
from labelme._label_file import LabelFileError
from labelme._label_file import ShapeDict
from labelme.config import get_config
from labelme.shape import Shape
# from labelme.widgets import AiPromptWidget
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import Canvas
from labelme.widgets import FileDialogPreview
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidget
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import StatusStats
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ZoomWidget
# from labelme.widgets import download_ai_model
from labelme.widgets.multi_camera_canvas import MultiCameraCanvas
from labelme.widgets.bev_canvas import BEVCanvas
from labelme.widgets.box3d_dialog import Box3DDialog
from labelme.utils.calibration import CameraCalibration, generate_bev_from_cameras
from labelme.utils.constants import DEFAULT_BOX_SIZE, DEFAULT_BOX_WIDTH, DEFAULT_BOX_HEIGHT, DEFAULT_BOX_DEPTH
from labelme.utils.metadata import save_metadata, load_metadata, get_bev_grid_from_metadata, get_box_size_from_metadata
from . import utils

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".

# handle high-dpi scaling issue
# https://leomoon.com/journal/python/high-dpi-scaling-in-pyqt5
if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


LABEL_COLORMAP: NDArray[np.uint8] = imgviz.label_colormap()


class _ZoomMode(enum.Enum):
    FIT_WINDOW = enum.auto()
    FIT_WIDTH = enum.auto()
    MANUAL_ZOOM = enum.auto()


class MainWindow(QtWidgets.QMainWindow):
    filename: str | None
    _config: dict
    _is_changed: bool = False
    _copied_shapes: list[Shape]
    _zoom_mode: _ZoomMode
    _zoom_values: dict[str, tuple[_ZoomMode, int]]
    _brightness_contrast_values: dict[str, tuple[int | None, int | None]]
    _prev_opened_dir: str | None
    _other_data: dict | None

    # NB: this tells Mypy etc. that `actions` here
    #     is a different type cf. the parent class
    #     (where it is Callable[[QWidget], list[QAction]]).
    actions: types.SimpleNamespace  # type: ignore[assignment]

    def __init__(
        self,
        config: dict | None = None,
        filename: str | None = None,
        output: str | None = None,
        output_file: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        if output is not None:
            logger.warning("argument output is deprecated, use output_file instead")
            if output_file is None:
                output_file = output
        del output

        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]

        super().__init__()
        self.setWindowTitle(__appname__)

        self._copied_shapes = []

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
        )

        self.labelList = LabelListWidget()
        self._prev_opened_dir = None
        self._aicv_root_dir = None  # Root directory for AICV folder structure
        
        # Import default values from constants
        try:
            from labelme.utils.constants import (
                DEFAULT_INTRINSIC_SCALE, DEFAULT_TRANSLATION_SCALE,
                DEFAULT_BEV_X, DEFAULT_BEV_Y, DEFAULT_BEV_BOUNDS
            )
            self._intrinsic_scale_factor = DEFAULT_INTRINSIC_SCALE
            self._translation_scale_factor = DEFAULT_TRANSLATION_SCALE
            self._bev_x = DEFAULT_BEV_X
            self._bev_y = DEFAULT_BEV_Y
            self._bev_bounds = DEFAULT_BEV_BOUNDS.copy()
        except ImportError:
            # Fallback to hardcoded defaults
            self._intrinsic_scale_factor = 0.25
            self._translation_scale_factor = 100.0
            self._bev_x = 1200
            self._bev_y = 800
            self._bev_bounds = [0, 600, 0, 400, 0, 200]

        # Ground Point List (replaces flags widget)
        self.ground_point_dock = QtWidgets.QDockWidget(self.tr("Ground Points (BEV)"), self)
        self.ground_point_dock.setObjectName("GroundPoints")
        self.ground_point_widget = QtWidgets.QListWidget()
        self.ground_point_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ground_point_widget.customContextMenuRequested.connect(self._ground_point_context_menu)
        self.ground_point_widget.itemEntered.connect(self._ground_point_hover)
        self.ground_point_widget.itemSelectionChanged.connect(self._ground_point_selection_changed)
        self.ground_point_widget.itemDoubleClicked.connect(self._ground_point_double_clicked)
        self.ground_point_widget.setMouseTracking(True)
        self.ground_point_widget.installEventFilter(self)  # For Delete key handling
        self.ground_point_dock.setWidget(self.ground_point_widget)
        
        # Keep flag_widget reference for compatibility but point to ground point widget
        self.flag_dock = self.ground_point_dock
        self.flag_widget = self.ground_point_widget

        self.labelList.itemSelectionChanged.connect(self._label_selection_changed)
        self.labelList.itemDoubleClicked.connect(self._edit_label)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Polygon Labels"), self)
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr("Select label to start annotating for it. Press 'Esc' to deselect.")
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                self.uniqLabelList.add_label_item(
                    label=label, color=self._get_rgb_by_label(label=label)
                )
        self.label_dock = QtWidgets.QDockWidget(self.tr("Label List"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(self.uniqLabelList)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)

        # Initialize with regular canvas, will switch to multi-camera if needed
        self.is_multi_camera_mode = False
        self.multi_camera_canvas: MultiCameraCanvas | None = None
        
        self.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
        self.canvas.zoomRequest.connect(self._zoom_requested)
        self.canvas.mouseMoved.connect(self._update_status_stats)
        self.canvas.statusUpdated.connect(lambda text: self.status_left.setText(text))

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scrollArea)

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)

        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config["shortcuts"]
        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            icon=None,
            tip=self.tr("Quit application"),
        )
        open_ = action(
            self.tr("&Open\n"),
            self._open_file_with_dialog,
            shortcuts["open"],
            icon="folder-open.svg",
            tip=self.tr("Open image or label file"),
        )
        opendir = action(
            self.tr("Open Dir"),
            self._open_dir_with_dialog,
            shortcuts["open_dir"],
            icon="folder-open.svg",
            tip=self.tr("Open Dir"),
        )
        openNextImg = action(
            self.tr("&Next Image"),
            self._open_next_image,
            shortcuts["open_next"],
            icon="arrow-fat-right.svg",
            tip=self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openPrevImg = action(
            self.tr("&Prev Image"),
            self._open_prev_image,
            shortcuts["open_prev"],
            icon="arrow-fat-left.svg",
            tip=self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        save = action(
            self.tr("&Save\n"),
            self.saveFile,
            shortcuts["save"],
            icon="floppy-disk.svg",
            tip=self.tr("Save labels to file"),
            enabled=False,
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            shortcuts["save_as"],
            icon="floppy-disk.svg",
            tip=self.tr("Save labels to a different file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            icon="file-x.svg",
            tip=self.tr("Delete current label file"),
            enabled=False,
        )

        changeOutputDir = action(
            self.tr("&Change Output Dir"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="folders.svg",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self.actions.saveAuto.setChecked(x),
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config["auto_save"])

        saveWithImageData = action(
            text=self.tr("Save With Image Data"),
            slot=self.enableSaveImageWithData,
            tip=self.tr("Save image data in label file"),
            checkable=True,
            checked=self._config["store_data"],
        )

        close = action(
            self.tr("&Close"),
            self.closeFile,
            shortcuts["close"],
            icon="x-circle.svg",
            tip=self.tr("Close current file"),
        )

        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep previous annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        # createMode = action(
        #     self.tr("Create Polygons"),
        #     lambda: self._switch_canvas_mode(edit=False, createMode="polygon"),
        #     shortcuts["create_polygon"],
        #     "polygon.svg",
        #     self.tr("Start drawing polygons"),
        #     enabled=False,
        # )
        createRectangleMode = action(
            self.tr("Create Rectangle"),
            lambda: self._switch_canvas_mode(edit=False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "rectangle.svg",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        # createCircleMode = action(
        #     self.tr("Create Circle"),
        #     lambda: self._switch_canvas_mode(edit=False, createMode="circle"),
        #     shortcuts["create_circle"],
        #     "circle.svg",
        #     self.tr("Start drawing circles"),
        #     enabled=False,
        # )
        # createLineMode = action(
        #     self.tr("Create Line"),
        #     lambda: self._switch_canvas_mode(edit=False, createMode="line"),
        #     shortcuts["create_line"],
        #     "line-segment.svg",
        #     self.tr("Start drawing lines"),
        #     enabled=False,
        # )
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self._switch_canvas_mode(edit=False, createMode="point"),
            shortcuts["create_point"],
            icon="circles-four.svg",
            tip=self.tr("Start drawing points"),
            enabled=False,
        )
        # createLineStripMode = action(
        #     self.tr("Create LineStrip"),
        #     lambda: self._switch_canvas_mode(edit=False, createMode="linestrip"),
        #     shortcuts["create_linestrip"],
        #     "line-segments.svg",
        #     self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
        #     enabled=False,
        # )
        # createAiPolygonMode = action(
        #     self.tr("Create AI-Polygon"),
        #     lambda: self._switch_canvas_mode(edit=False, createMode="ai_polygon"),
        #     None,
        #     "ai-polygon.svg",
        #     self.tr("Start drawing ai_polygon. Ctrl+LeftClick ends creation."),
        #     enabled=False,
        # )
        # createAiMaskMode = action(
        #     self.tr("Create AI-Mask"),
        #     lambda: self._switch_canvas_mode(edit=False, createMode="ai_mask"),
        #     None,
        #     "ai-mask.svg",
        #     self.tr("Start drawing ai_mask. Ctrl+LeftClick ends creation."),
        #     enabled=False,
        # )
        editMode = action(
            self.tr("Edit Polygons"),
            lambda: self._switch_canvas_mode(edit=True),
            shortcuts["edit_polygon"],
            icon="note-pencil.svg",
            tip=self.tr("Move and edit the selected polygons"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            icon="trash.svg",
            tip=self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            shortcuts["duplicate_polygon"],
            icon="copy.svg",
            tip=self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Polygons"),
            self.copySelectedShape,
            shortcuts["copy_polygon"],
            "copy_clipboard",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Polygons"),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=False,
        )
        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            icon="arrow-u-up-left.svg",
            tip=self.tr("Undo last drawn point"),
            enabled=False,
        )
        removePoint = action(
            text=self.tr("Remove Selected Point"),
            slot=self.removeSelectedPoint,
            shortcut=shortcuts["remove_selected_point"],
            icon="trash.svg",
            tip=self.tr("Remove selected point from polygon"),
            enabled=False,
        )

        undo = action(
            self.tr("Undo\n"),
            self.undoShapeEdit,
            shortcuts["undo"],
            icon="arrow-u-up-left.svg",
            tip=self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            shortcuts["hide_all_polygons"],
            icon="eye.svg",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            shortcuts["show_all_polygons"],
            icon="eye.svg",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )
        toggleAll = action(
            self.tr("&Toggle\nPolygons"),
            functools.partial(self.togglePolygons, None),
            shortcuts["toggle_all_polygons"],
            icon="eye.svg",
            tip=self.tr("Toggle all polygons"),
            enabled=False,
        )

        help = action(
            self.tr("&Tutorial"),
            self.tutorial,
            icon="question.svg",
            tip=self.tr("Show tutorial page"),
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoomBoxLayout = QtWidgets.QVBoxLayout()
        zoomLabel = QtWidgets.QLabel(self.tr("Zoom"))
        zoomLabel.setAlignment(Qt.AlignCenter)
        zoomBoxLayout.addWidget(zoomLabel)
        zoomBoxLayout.addWidget(self.zoomWidget)
        zoom.setDefaultWidget(QtWidgets.QWidget())
        zoom.defaultWidget().setLayout(zoomBoxLayout)
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(f"{shortcuts['zoom_in']},{shortcuts['zoom_out']}"),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            self.tr("Zoom &In"),
            lambda _: self._add_zoom(increment=1.1),
            shortcuts["zoom_in"],
            icon="magnifying-glass-minus.svg",
            tip=self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            lambda _: self._add_zoom(increment=0.9),
            shortcuts["zoom_out"],
            icon="magnifying-glass-plus.svg",
            tip=self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            self._set_zoom_to_original,
            shortcuts["zoom_to_original"],
            icon="image-square.svg",
            tip=self.tr("Zoom to original size"),
            enabled=False,
        )
        keepPrevScale = action(
            self.tr("&Keep Previous Scale"),
            self.enableKeepPrevScale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            icon="frame-corners.svg",
            tip=self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            icon="frame-arrows-horizontal.svg",
            tip=self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightnessContrast = action(
            self.tr("&Brightness Contrast"),
            self.brightnessContrast,
            None,
            "brightness-contrast.svg",
            self.tr("Adjust brightness and contrast"),
            enabled=False,
        )
        self._zoom_mode = _ZoomMode.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            _ZoomMode.FIT_WINDOW: self.scaleFitWindow,
            _ZoomMode.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            _ZoomMode.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("&Edit Label"),
            self._edit_label,
            shortcuts["edit_label"],
            icon="note-pencil.svg",
            tip=self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.setFillDrawing,
            None,
            icon="paint-bucket.svg",
            tip=self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        if self._config["canvas"]["fill_drawing"]:
            fill_drawing.trigger()

        # Label list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = types.SimpleNamespace(
            about=action(
                text=f"&About {__appname__}",
                slot=functools.partial(
                    QMessageBox.about,
                    self,
                    f"About {__appname__}",
                    f"""
<h3>{__appname__}</h3>
<p>Image Polygonal Annotation with Python</p>
<p>Version: {__version__}</p>
<p>Author: Kentaro Wada</p>
<p>
    <a href="https://labelme.io">Homepage</a> |
    <a href="https://labelme.io/docs">Documentation</a> |
    <a href="https://labelme.io/docs/troubleshoot">Troubleshooting</a>
</p>
<p>
    <a href="https://github.com/wkentaro/labelme">GitHub</a> |
    <a href="https://x.com/labelmeai">Twitter/X</a>
</p>
""",
                ),
            ),
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save,
            saveAs=saveAs,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            toggle_keep_prev_brightness_contrast=action(
                text=self.tr("Keep Previous Brightness/Contrast"),
                slot=lambda: self._config.__setitem__(
                    "keep_prev_brightness_contrast",
                    not self._config["keep_prev_brightness_contrast"],
                ),
                checkable=True,
                checked=self._config["keep_prev_brightness_contrast"],
            ),
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            copy=copy,
            paste=paste,
            undoLastPoint=undoLastPoint,
            undo=undo,
            removePoint=removePoint,
            # createMode=createMode,
            editMode=editMode,
            createRectangleMode=createRectangleMode,
            # createCircleMode=createCircleMode,
            # createLineMode=createLineMode,
            createPointMode=createPointMode,
            # createLineStripMode=createLineStripMode,
            # createAiPolygonMode=createAiPolygonMode,
            # createAiMaskMode=createAiMaskMode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,
        )
        self.on_shapes_present_actions = (saveAs, hideAll, showAll, toggleAll)

        self.draw_actions: list[tuple[str, QtWidgets.QAction]] = [
            # ("polygon", createMode),
            ("rectangle", createRectangleMode),
            # ("circle", createCircleMode),
            # ("point", createPointMode),
            # ("line", createLineMode),
            # ("linestrip", createLineStripMode),
            # ("ai_polygon", createAiPolygonMode),
            # ("ai_mask", createAiMaskMode),
        ]

        # Group zoom controls into a list for easier toggling.
        self.zoom_actions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.on_load_active_actions = (
            close,
            # createMode,
            createRectangleMode,
            # createCircleMode,
            # createLineMode,
            createPointMode,
            # createLineStripMode,
            # createAiPolygonMode,
            # createAiMaskMode,
            brightnessContrast,
        )
        # menu shown at right click
        self.context_menu_actions = (
            *[draw_action for _, draw_action in self.draw_actions],
            editMode,
            edit,
            duplicate,
            copy,
            paste,
            delete,
            undo,
            undoLastPoint,
            removePoint,
        )
        # XXX: need to add some actions here to activate the shortcut
        self.edit_menu_actions = (
            edit,
            duplicate,
            copy,
            paste,
            delete,
            None,
            undo,
            undoLastPoint,
            None,
            removePoint,
            None,
            toggle_keep_prev_mode,
        )

        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        self.menus = types.SimpleNamespace(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            labelList=labelMenu,
        )

        utils.addActions(
            self.menus.file,
            (
                open_,
                openNextImg,
                openPrevImg,
                opendir,
                self.menus.recentFiles,
                save,
                saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                close,
                deleteFile,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help, self.actions.about))
        utils.addActions(
            self.menus.view,
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                hideAll,
                showAll,
                toggleAll,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                keepPrevScale,
                None,
                fitWindow,
                fitWidth,
                None,
                brightnessContrast,
                self.actions.toggle_keep_prev_brightness_contrast,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.context_menu_actions)
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )

        # selectAiModel = QtWidgets.QWidgetAction(self)
        # selectAiModel.setDefaultWidget(QtWidgets.QWidget())
        # selectAiModel.defaultWidget().setLayout(QtWidgets.QVBoxLayout())
        # #
        # selectAiModelLabel = QtWidgets.QLabel(self.tr("AI Mask Model"))
        # selectAiModelLabel.setAlignment(QtCore.Qt.AlignCenter)
        # selectAiModel.defaultWidget().layout().addWidget(selectAiModelLabel)
        # #
        # self._selectAiModelComboBox = QtWidgets.QComboBox()
        # selectAiModel.defaultWidget().layout().addWidget(self._selectAiModelComboBox)
        # MODEL_NAMES: list[tuple[str, str]] = [
        #     ("efficientsam:10m", "EfficientSam (speed)"),
        #     ("efficientsam:latest", "EfficientSam (accuracy)"),
        #     ("sam:100m", "Sam (speed)"),
        #     ("sam:300m", "Sam (balanced)"),
        #     ("sam:latest", "Sam (accuracy)"),
        #     ("sam2:small", "Sam2 (speed)"),
        #     ("sam2:latest", "Sam2 (balanced)"),
        #     ("sam2:large", "Sam2 (accuracy)"),
        # ]
        # for model_name, model_ui_name in MODEL_NAMES:
        #     self._selectAiModelComboBox.addItem(model_ui_name, userData=model_name)
        # model_ui_names: list[str] = [model_ui_name for _, model_ui_name in MODEL_NAMES]
        # if self._config["ai"]["default"] in model_ui_names:
        #     model_index = model_ui_names.index(self._config["ai"]["default"])
        # else:
        #     logger.warning(
        #         "Default AI model is not found: %r",
        #         self._config["ai"]["default"],
        #     )
        #     model_index = 0
        # self._selectAiModelComboBox.currentIndexChanged.connect(
        #     lambda index: self.canvas.set_ai_model_name(
        #         model_name=self._selectAiModelComboBox.itemData(index)
        #     )
        # )
        # self._selectAiModelComboBox.setCurrentIndex(model_index)

        # self._ai_prompt_widget: AiPromptWidget = AiPromptWidget(
        #     on_submit=self._submit_ai_prompt, parent=self
        # )
        # ai_prompt_action = QtWidgets.QWidgetAction(self)
        # ai_prompt_action.setDefaultWidget(self._ai_prompt_widget)

        self.addToolBar(
            Qt.TopToolBarArea,
            ToolBar(
                title="Tools",
                actions=[
                    open_,
                    opendir,
                    openPrevImg,
                    openNextImg,
                    save,
                    deleteFile,
                    None,
                    editMode,
                    duplicate,
                    delete,
                    undo,
                    brightnessContrast,
                    None,
                    fitWindow,
                    zoom,
                    # None,
                    # selectAiModel,
                    # None,
                    # ai_prompt_action,
                ],
                font_base=self.font(),
            ),
        )
        self.addToolBar(
            Qt.LeftToolBarArea,
            ToolBar(
                title="CreateShapeTools",
                actions=[a for _, a in self.draw_actions],
                orientation=Qt.Vertical,
                button_style=Qt.ToolButtonTextUnderIcon,
                font_base=self.font(),
            ),
        )
        
        # Calibration scale factors toolbar
        self._calib_toolbar = QtWidgets.QToolBar("BEV Settings")
        self._calib_toolbar.setObjectName("CalibrationToolbar")
        
        # Intrinsic scale factor spinbox
        calib_label1 = QtWidgets.QLabel(" K: ")
        self._intrinsic_scale_spinbox = QtWidgets.QDoubleSpinBox()
        self._intrinsic_scale_spinbox.setRange(0.01, 2.0)
        self._intrinsic_scale_spinbox.setSingleStep(0.05)
        self._intrinsic_scale_spinbox.setDecimals(3)
        self._intrinsic_scale_spinbox.setValue(self._intrinsic_scale_factor)
        self._intrinsic_scale_spinbox.setToolTip("Scale factor for K and optimalK (default 0.25)")
        
        # Translation scale factor spinbox
        calib_label2 = QtWidgets.QLabel(" T: ")
        self._translation_scale_spinbox = QtWidgets.QDoubleSpinBox()
        self._translation_scale_spinbox.setRange(0.1, 1000.0)
        self._translation_scale_spinbox.setSingleStep(10.0)
        self._translation_scale_spinbox.setDecimals(1)
        self._translation_scale_spinbox.setValue(self._translation_scale_factor)
        self._translation_scale_spinbox.setToolTip("Scale factor for translation (default 100)")
        
        # BEV X dimension spinbox
        bev_label_x = QtWidgets.QLabel(" X: ")
        self._bev_x_spinbox = QtWidgets.QSpinBox()
        self._bev_x_spinbox.setRange(100, 4000)
        self._bev_x_spinbox.setSingleStep(100)
        self._bev_x_spinbox.setValue(self._bev_x)
        self._bev_x_spinbox.setToolTip("BEV grid X dimension (default 1200)")
        
        # BEV Y dimension spinbox
        bev_label_y = QtWidgets.QLabel(" Y: ")
        self._bev_y_spinbox = QtWidgets.QSpinBox()
        self._bev_y_spinbox.setRange(100, 4000)
        self._bev_y_spinbox.setSingleStep(100)
        self._bev_y_spinbox.setValue(self._bev_y)
        self._bev_y_spinbox.setToolTip("BEV grid Y dimension (default 800)")
        
        # Bounds: XMAX, YMAX
        bounds_label_xmax = QtWidgets.QLabel(" XMAX: ")
        self._bev_xmax_spinbox = QtWidgets.QSpinBox()
        self._bev_xmax_spinbox.setRange(100, 2000)
        self._bev_xmax_spinbox.setSingleStep(50)
        self._bev_xmax_spinbox.setValue(self._bev_bounds[1])
        self._bev_xmax_spinbox.setToolTip("BEV bounds XMAX (default 600)")
        
        bounds_label_ymax = QtWidgets.QLabel(" YMAX: ")
        self._bev_ymax_spinbox = QtWidgets.QSpinBox()
        self._bev_ymax_spinbox.setRange(100, 2000)
        self._bev_ymax_spinbox.setSingleStep(50)
        self._bev_ymax_spinbox.setValue(self._bev_bounds[3])
        self._bev_ymax_spinbox.setToolTip("BEV bounds YMAX (default 400)")
        
        # Apply button
        apply_calib_btn = QtWidgets.QPushButton("Apply")
        apply_calib_btn.setToolTip("Apply settings and regenerate BEV")
        apply_calib_btn.clicked.connect(self._apply_calibration_scales)
        
        self._calib_toolbar.addWidget(calib_label1)
        self._calib_toolbar.addWidget(self._intrinsic_scale_spinbox)
        self._calib_toolbar.addWidget(calib_label2)
        self._calib_toolbar.addWidget(self._translation_scale_spinbox)
        self._calib_toolbar.addSeparator()
        self._calib_toolbar.addWidget(bev_label_x)
        self._calib_toolbar.addWidget(self._bev_x_spinbox)
        self._calib_toolbar.addWidget(bev_label_y)
        self._calib_toolbar.addWidget(self._bev_y_spinbox)
        self._calib_toolbar.addSeparator()
        self._calib_toolbar.addWidget(bounds_label_xmax)
        self._calib_toolbar.addWidget(self._bev_xmax_spinbox)
        self._calib_toolbar.addWidget(bounds_label_ymax)
        self._calib_toolbar.addWidget(self._bev_ymax_spinbox)
        self._calib_toolbar.addSeparator()
        self._calib_toolbar.addWidget(apply_calib_btn)
        
        self.addToolBar(Qt.TopToolBarArea, self._calib_toolbar)

        self.status_left = QtWidgets.QLabel(self.tr("%s started.") % __appname__)
        self.status_right = StatusStats()
        self.statusBar().addWidget(self.status_left, 1)
        self.statusBar().addWidget(self.status_right, 0)
        self.statusBar().show()

        if output_file is not None and self._config["auto_save"]:
            logger.warning(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.labelFile: LabelFile | None = None
        self.imagePath: str | None = None
        self.recentFiles: list[str] = []
        self.maxRecent = 7
        self._other_data = None
        self.zoom_level = 100
        self.fit_window = False
        self._zoom_values = {}
        self._brightness_contrast_values = {}
        self.scroll_values = {  # type: ignore[var-annotated]
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value
        
        # Multi-camera state
        self.multi_camera_data: list[dict] = []  # List of camera data dicts
        self.multi_camera_canvas: MultiCameraCanvas | None = None
        self.is_multi_camera_mode = False
        self.bev_canvas: BEVCanvas | None = None
        self.camera_calibrations: dict[str, CameraCalibration] = {}  # camera_id -> calibration

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(900, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry'])
        self.restoreState(state)

        if filename:
            if osp.isdir(filename):
                self._import_images_from_dir(root_dir=filename)
                self._open_next_image()
            else:
                self._load_file(filename=filename)
        else:
            self.filename = None

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self._paint_canvas)

        self.populateModeActions()

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    # Support Functions

    def noShapes(self):
        if self.is_multi_camera_mode and self.multi_camera_canvas:
            # Check if any cell has shapes
            for cell in self.multi_camera_canvas.camera_cells:
                if cell.shapes:
                    return False
            return True
        return not len(self.labelList)

    def populateModeActions(self):
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], self.context_menu_actions)
        self.menus.edit.clear()
        actions = (
            *[draw_action for _, draw_action in self.draw_actions],
            self.actions.editMode,
            *self.edit_menu_actions,
        )
        utils.addActions(self.menus.edit, actions)

    def _get_window_title(self, dirty: bool) -> str:
        window_title: str = __appname__
        if self.imagePath:
            window_title = f"{window_title} - {self.imagePath}"
            if self.fileListWidget.count() and self.fileListWidget.currentItem():
                window_title = (
                    f"{window_title} "
                    f"[{self.fileListWidget.currentRow() + 1}"
                    f"/{self.fileListWidget.count()}]"
                )
        if dirty:
            window_title = f"{window_title}*"
        return window_title

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            assert self.imagePath
            label_file = f"{osp.splitext(self.imagePath)[0]}.json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.saveLabels(label_file)
            return
        self._is_changed = True
        self.actions.save.setEnabled(True)
        self.setWindowTitle(self._get_window_title(dirty=True))

    def setClean(self):
        self._is_changed = False
        self.actions.save.setEnabled(False)
        for _, action in self.draw_actions:
            action.setEnabled(True)
        self.setWindowTitle(self._get_window_title(dirty=False))

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.zoom_actions:
            z.setEnabled(value)
        for action in self.on_load_active_actions:
            action.setEnabled(value)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def show_status_message(self, message, delay=500):
        self.statusBar().showMessage(message, delay)

    # def _submit_ai_prompt(self, _) -> None:
    #     texts = self._ai_prompt_widget.get_text_prompt().split(",")

    #     model_name: str = "yoloworld"
    #     model_type = osam.apis.get_model_type_by_name(model_name)
    #     model_type = typing.cast(type[osam.types.Model], model_type)
    #     if not (_is_already_downloaded := model_type.get_size() is not None):
    #         if not download_ai_model(model_name=model_name, parent=self):
    #             return

    #     boxes, scores, labels = bbox_from_text.get_bboxes_from_texts(
    #         model=model_name,
    #         image=utils.img_qt_to_arr(self.image)[:, :, :3],
    #         texts=texts,
    #     )

    #     # Get shapes from canvas (works for both regular and multi-camera)
    #     canvas_shapes = self.canvas.shapes if hasattr(self.canvas, 'shapes') else []
    #     if self.is_multi_camera_mode and self.multi_camera_canvas:
    #         canvas_shapes = self.multi_camera_canvas.getShapes()
        
    #     for shape in canvas_shapes:
    #         if shape.shape_type != "rectangle" or shape.label not in texts:
    #             continue
    #         box = np.array(
    #             [
    #                 shape.points[0].x(),
    #                 shape.points[0].y(),
    #                 shape.points[1].x(),
    #                 shape.points[1].y(),
    #             ],
    #             dtype=np.float32,
    #         )
    #         boxes = np.r_[boxes, [box]]
    #         scores = np.r_[scores, [1.01]]
    #         labels = np.r_[labels, [texts.index(shape.label)]]

    #     boxes, scores, labels = bbox_from_text.nms_bboxes(
    #         boxes=boxes,
    #         scores=scores,
    #         labels=labels,
    #         iou_threshold=self._ai_prompt_widget.get_iou_threshold(),
    #         score_threshold=self._ai_prompt_widget.get_score_threshold(),
    #         max_num_detections=100,
    #     )

    #     keep = scores != 1.01
    #     boxes = boxes[keep]
    #     scores = scores[keep]
    #     labels = labels[keep]

    #     shape_dicts: list[dict] = bbox_from_text.get_shapes_from_bboxes(
    #         boxes=boxes,
    #         scores=scores,
    #         labels=labels,
    #         texts=texts,
    #     )

    #     shapes: list[Shape] = []
    #     for shape_dict in shape_dicts:
    #         shape = Shape(
    #             label=shape_dict["label"],
    #             shape_type=shape_dict["shape_type"],
    #             description=shape_dict["description"],
    #         )
    #         for point in shape_dict["points"]:
    #             shape.addPoint(QtCore.QPointF(*point))
    #         shapes.append(shape)

    #     self.canvas.storeShapes()
    #     self._load_shapes(shapes, replace=False)
    #     self.setDirty()

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self._other_data = None
        if self.is_multi_camera_mode and self.multi_camera_canvas:
            self.multi_camera_canvas.resetState()
        else:
            self.canvas.resetState()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        if self.is_multi_camera_mode and self.multi_camera_canvas:
            # Reload all shapes from all cells
            all_shapes = self.multi_camera_canvas.getShapes()
            self._load_shapes(all_shapes)
        else:
            self._load_shapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

    def tutorial(self):
        url = "https://github.com/labelmeai/labelme/tree/main/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def _switch_canvas_mode(
        self, edit: bool = True, createMode: str | None = None
    ) -> None:
        self.canvas.setEditing(edit)
        if createMode is not None:
            self.canvas.createMode = createMode
        if edit:
            for _, draw_action in self.draw_actions:
                draw_action.setEnabled(True)
        else:
            for draw_mode, draw_action in self.draw_actions:
                draw_action.setEnabled(createMode != draw_mode)
        self.actions.editMode.setEnabled(not edit)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, f"&{i + 1} {QtCore.QFileInfo(f).fileName()}", self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def validateLabel(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)  # type: ignore[attr-defined,union-attr]
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def _edit_label(self, value=None):
        if not self.canvas.editing():
            return

        items = self.labelList.selectedItems()
        if not items:
            logger.warning("No label is selected, so cannot edit label.")
            return

        shape = items[0].shape()

        if len(items) == 1:
            edit_text = True
            edit_flags = True
            edit_group_id = True
            edit_description = True
        else:
            edit_text = all(item.shape().label == shape.label for item in items[1:])
            edit_flags = all(item.shape().flags == shape.flags for item in items[1:])
            edit_group_id = all(
                item.shape().group_id == shape.group_id for item in items[1:]
            )
            edit_description = all(
                item.shape().description == shape.description for item in items[1:]
            )

        if not edit_text:
            self.labelDialog.edit.setDisabled(True)
            self.labelDialog.labelList.setDisabled(True)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(True)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(True)

        text, flags, group_id, description = self.labelDialog.popUp(
            text=shape.label if edit_text else "",
            flags=shape.flags if edit_flags else None,
            group_id=shape.group_id if edit_group_id else None,
            description=shape.description if edit_description else None,
            flags_disabled=not edit_flags,
        )

        if not edit_text:
            self.labelDialog.edit.setDisabled(False)
            self.labelDialog.labelList.setDisabled(False)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(False)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(False)

        if text is None:
            assert flags is None
            assert group_id is None
            assert description is None
            return

        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        self.canvas.storeShapes()
        for item in items:
            shape: Shape = item.shape()  # type: ignore[no-redef]

            if edit_text:
                shape.label = text
            if edit_flags:
                shape.flags = flags
            if edit_group_id:
                shape.group_id = group_id
            if edit_description:
                shape.description = description

            self._update_shape_color(shape)
            if shape.group_id is None:
                r, g, b = shape.fill_color.getRgb()[:3]
                item.setText(
                    f"{html.escape(shape.label)} "
                    f'<font color="#{r:02x}{g:02x}{b:02x}">●</font>'
                )
            else:
                item.setText(f"{shape.label} ({shape.group_id})")
            self.setDirty()
            if self.uniqLabelList.find_label_item(shape.label) is None:
                self.uniqLabelList.add_label_item(
                    label=shape.label, color=self._get_rgb_by_label(label=shape.label)
                )

    def fileSearchChanged(self):
        self._import_images_from_dir(
            root_dir=self._prev_opened_dir, pattern=self.fileSearch.text()
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self._can_continue():
            return

        if self.is_multi_camera_mode:
            # Extract frame index from item text (e.g., "Frame_0001")
            frame_text = str(item.text())
            try:
                frame_index = int(frame_text.split("_")[1])
                self._load_multi_camera_frame(frame_index)
            except (ValueError, IndexError):
                logger.error(f"Failed to parse frame index from: {frame_text}")
        else:
            currIndex = self.imageList.index(str(item.text()))
            if currIndex < len(self.imageList):
                filename = self.imageList[currIndex]
                if filename:
                    self._load_file(filename)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self.labelList.itemSelectionChanged.disconnect(self._label_selection_changed)
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self.labelList.itemSelectionChanged.connect(self._label_selection_changed)
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected)

    def addLabel(self, shape):
        if shape.group_id is None:
            text = shape.label
        else:
            text = f"{shape.label} ({shape.group_id})"
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if self.uniqLabelList.find_label_item(shape.label) is None:
            self.uniqLabelList.add_label_item(
                label=shape.label, color=self._get_rgb_by_label(label=shape.label)
            )
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.on_shapes_present_actions:
            action.setEnabled(True)

        self._update_shape_color(shape)
        r, g, b = shape.fill_color.getRgb()[:3]
        label_list_item.setText(
            f'{html.escape(text)} <font color="#{r:02x}{g:02x}{b:02x}">●</font>'
        )

    def _update_shape_color(self, shape):
        if shape.group_id is not None:
            r, g, b = self._get_rgb_by_group_id(shape.group_id)
        else:
            r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label: str) -> tuple[int, int, int]:
        if self._config["shape_color"] == "auto":
            item = self.uniqLabelList.find_label_item(label)
            item_index: int = (
                self.uniqLabelList.indexFromItem(item).row()
                if item
                else self.uniqLabelList.count()
            )
            label_id: int = (
                1  # skip black color by default
                + item_index
                + self._config["shift_auto_shape_color"]
            )
            rgb: tuple[int, int, int] = tuple(
                LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)].tolist()
            )
            return rgb
        elif (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            if not (
                len(self._config["label_colors"][label]) == 3
                and all(0 <= c <= 255 for c in self._config["label_colors"][label])
            ):
                raise ValueError(
                    "Color for label must be 0-255 RGB tuple, but got: "
                    f"{self._config['label_colors'][label]}"
                )
            return tuple(self._config["label_colors"][label])
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def _get_rgb_by_group_id(self, group_id: int) -> tuple[int, int, int]:
        color_id: int = (
            1
            + group_id
            + self._config["shift_auto_shape_color"]
        )
        rgb: tuple[int, int, int] = tuple(
            LABEL_COLORMAP[color_id % len(LABEL_COLORMAP)].tolist()
        )
        return rgb

    def remLabels(self, shapes):
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            self.labelList.removeItem(item)

    def _load_shapes(self, shapes: list[Shape], replace: bool = True) -> None:
        self.labelList.itemSelectionChanged.disconnect(self._label_selection_changed)
        shape: Shape
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.labelList.itemSelectionChanged.connect(self._label_selection_changed)
        self.canvas.loadShapes(shapes=shapes, replace=replace)

    def _load_shape_dicts(self, shape_dicts: list[ShapeDict]) -> None:
        shapes: list[Shape] = []
        shape_dict: ShapeDict
        for shape_dict in shape_dicts:
            shape: Shape = Shape(
                label=shape_dict["label"],
                shape_type=shape_dict["shape_type"],
                group_id=shape_dict["group_id"],
                description=shape_dict["description"],
                mask=shape_dict["mask"],
            )
            for x, y in shape_dict["points"]:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if not isinstance(shape.label, str):
                        logger.warning("shape.label is not str: {}", shape.label)
                        continue
                    if re.match(pattern, shape.label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(shape_dict["flags"])
            shape.other_data = shape_dict["other_data"]

            shapes.append(shape)
        self._load_shapes(shapes=shapes)

    def _load_flags(self, flags: dict[str, bool]) -> None:
        self.flag_widget.clear()  # type: ignore[union-attr]
        key: str
        flag: bool
        for key, flag in flags.items():
            item: QtWidgets.QListWidgetItem = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)  # type: ignore[union-attr]

    def saveLabels(self, filename):
        if self.is_multi_camera_mode and self.multi_camera_canvas:
            return self._save_multi_camera_labels()
        
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    description=s.description,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    mask=None
                    if s.mask is None
                    else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        for i in range(self.flag_widget.count()):  # type: ignore[union-attr]
            item = self.flag_widget.item(i)  # type: ignore[union-attr]
            assert item
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            assert self.imagePath
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            # imageData = self.imageData if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=None,  # imageData disabled
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self._other_data,
                flags=flags,
            )
            self.labelFile = lf
            items = self.fileListWidget.findItems(self.imagePath, Qt.MatchExactly)
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def _save_multi_camera_labels(self) -> bool:
        """Save labels for all cameras in multi-camera mode"""
        if not self.multi_camera_canvas or not self.multi_camera_data:
            return False
        
        success = True
        
        # For AICV structure, only save to annotations_positions (skip individual camera JSON files)
        is_aicv = hasattr(self, "_aicv_root_dir") and self._aicv_root_dir is not None
        
        if not is_aicv:
            # Legacy mode: save individual camera JSON files
            for idx, cam_data in enumerate(self.multi_camera_data):
                image_path = cam_data["image_path"]
                label_file = f"{osp.splitext(image_path)[0]}.json"
                if self.output_dir:
                    label_file_without_path = osp.basename(label_file)
                    label_file = osp.join(self.output_dir, label_file_without_path)
                
                # Get shapes for this camera
                cell = self.multi_camera_canvas.camera_cells[idx]
                shapes = cell.shapes
                
                # Format shapes
                def format_shape(s):
                    data = s.other_data.copy() if hasattr(s, 'other_data') else {}
                    data.update(
                        dict(
                            label=s.label,
                            points=[(p.x(), p.y()) for p in s.points],
                            group_id=s.group_id,
                            description=s.description,
                            shape_type=s.shape_type,
                            flags=s.flags if hasattr(s, 'flags') else {},
                            mask=None
                            if not hasattr(s, 'mask') or s.mask is None
                            else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                        )
                    )
                    return data
                
                formatted_shapes = [format_shape(s) for s in shapes]
                
                # Load image data
                image_data = LabelFile.load_image_file(image_path)
                if not image_data:
                    logger.warning(f"Failed to load image: {image_path}")
                    success = False
                    continue
                
                image = QtGui.QImage.fromData(image_data)
                if image.isNull():
                    logger.warning(f"Failed to decode image: {image_path}")
                    success = False
                    continue
                
                # Save label file
                lf = LabelFile()
                try:
                    if osp.dirname(label_file) and not osp.exists(osp.dirname(label_file)):
                        os.makedirs(osp.dirname(label_file))
                    
                    imagePath = osp.relpath(image_path, osp.dirname(label_file))
                    # imageData = image_data if self._config["store_data"] else None
                    
                    flags = {}
                    for i in range(self.flag_widget.count()):  # type: ignore[union-attr]
                        item = self.flag_widget.item(i)  # type: ignore[union-attr]
                        assert item
                        key = item.text()
                        flag = item.checkState() == Qt.Checked
                        flags[key] = flag
                    
                    lf.save(
                        filename=label_file,
                        shapes=formatted_shapes,
                        imagePath=imagePath,
                        imageData=None,  # imageData disabled
                        imageHeight=image.height(),
                        imageWidth=image.width(),
                        otherData=self._other_data,
                        flags=flags,
                    )
                    logger.info(f"Saved labels for camera {cam_data['camera_id']}: {label_file}")
                except LabelFileError as e:
                    logger.error(f"Error saving label file {label_file}: {e}")
                    success = False
        
        # Save global annotation format (always save for both AICV and legacy modes)
        self._save_global_annotations()
        
        return success

    def eventFilter(self, obj, event):
        """Handle Delete key press in ground point widget"""
        if obj == self.ground_point_widget and event.type() == QtCore.QEvent.KeyPress:
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                current_item = self.ground_point_widget.currentItem()
                if current_item:
                    group_id = current_item.data(Qt.UserRole)
                    self._delete_ground_point(group_id)
                    return True
        return super().eventFilter(obj, event)
    
    def _ground_point_double_clicked(self, item):
        """Handle double-click on ground point to delete it"""
        if item:
            group_id = item.data(Qt.UserRole)
            self._delete_ground_point(group_id)
    
    def _ground_point_context_menu(self, point):
        """Show context menu for ground point (BEV point)"""
        menu = QtWidgets.QMenu()
        delete_action = menu.addAction("Delete Ground Point")
        action = menu.exec_(self.ground_point_widget.mapToGlobal(point))
        
        if action == delete_action:
            current_item = self.ground_point_widget.currentItem()
            if current_item:
                group_id = current_item.data(Qt.UserRole)
                self._delete_ground_point(group_id)
    
    def _ground_point_hover(self, item):
        """Highlight bounding boxes when hovering over ground point"""
        if not item or not self.multi_camera_canvas:
            return
        
        group_id = item.data(Qt.UserRole)
        self._highlight_shapes_by_group_id(group_id, highlight=True)
    
    def _ground_point_selection_changed(self):
        """Highlight bounding boxes when ground point is selected"""
        if not self.multi_camera_canvas:
            return
        
        # Clear all highlights first
        for cell in self.multi_camera_canvas.camera_cells:
            for shape in cell.shapes:
                if hasattr(shape, '_bev_highlighted'):
                    shape._bev_highlighted = False
        
        # Highlight selected
        selected_items = self.ground_point_widget.selectedItems()
        if selected_items:
            group_id = selected_items[0].data(Qt.UserRole)
            self._highlight_shapes_by_group_id(group_id, highlight=True)
        
        self.multi_camera_canvas.update()
    
    def _highlight_shapes_by_group_id(self, group_id: int, highlight: bool = True):
        """Highlight all shapes with the given group_id"""
        if not self.multi_camera_canvas:
            return
        
        for cell in self.multi_camera_canvas.camera_cells:
            for shape in cell.shapes:
                if shape.group_id == group_id:
                    shape._bev_highlighted = highlight
        
        self.multi_camera_canvas.update()
    
    def _delete_ground_point(self, group_id: int):
        """Delete ground point and all related bounding boxes"""
        if not self.multi_camera_canvas or not self.bev_canvas:
            return
        
        # Delete point from BEV canvas
        self.bev_canvas.deletePointByGroupId(group_id)
        
        # Delete from all cameras
        for idx, cell in enumerate(self.multi_camera_canvas.camera_cells):
            shapes_to_remove = []
            for shape in cell.shapes:
                if shape.group_id == group_id:
                    shapes_to_remove.append(shape)
            
            for shape in shapes_to_remove:
                cell.shapes.remove(shape)
                self.remLabels([shape])
            
            self.multi_camera_canvas.camera_cells[idx] = cell
        
        # Update displays
        self.bev_canvas.update()
        self.multi_camera_canvas.update()
        self._update_ground_point_list()
        self.setDirty()
    
    def _update_ground_point_list(self):
        """Update the ground point list widget"""
        self.ground_point_widget.clear()
        
        if not self.bev_canvas or not self.bev_canvas.points:
            return
        
        for mem_x, mem_y, label, group_id in self.bev_canvas.points:
            if group_id is None:
                continue
            
            # Convert back to grid coordinates for display
            grid_x, grid_y = self._mem_to_worldgrid(mem_x, mem_y)
            
            # Format: "ID 1: grid(137, 37) - object"
            text = f"ID {group_id}: grid({grid_x}, {grid_y}) - {label}"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(Qt.UserRole, group_id)
            self.ground_point_widget.addItem(item)
    
    def _get_next_group_id(self) -> int:
        """Get next unique group_id"""
        max_id = 0
        
        # Check BEV points
        if self.bev_canvas and self.bev_canvas.points:
            for _, _, _, group_id in self.bev_canvas.points:
                if group_id is not None and group_id > max_id:
                    max_id = group_id
        
        # Check shapes in all cameras
        if self.multi_camera_canvas:
            for cell in self.multi_camera_canvas.camera_cells:
                for shape in cell.shapes:
                    if shape.group_id is not None and shape.group_id > max_id:
                        max_id = shape.group_id
        
        return max_id + 1

    def _get_vox_util(self):
        """Get or create VoxelUtil instance with current BEV parameters."""
        try:
            from labelme.utils.vox_utils import vox
            import torch
            
            X, Y, Z = self._bev_x, self._bev_y, 2
            bounds = self._bev_bounds
            scene_centroid = torch.tensor([0.0, 0.0, 0.0]).reshape([1, 3])
            
            return vox.VoxelUtil(Y, Z, X, scene_centroid=scene_centroid, bounds=bounds)
        except Exception as e:
            logger.warning(f"Failed to create VoxelUtil: {e}")
            return None

    def _worldgrid_to_mem(self, grid_x: float, grid_y: float) -> tuple[float, float]:
        """
        Transform world grid coordinates to memory coordinates for BEV display.
        
        Transformation chain: worldgrid -> worldcoord -> ref -> mem
        Uses: worldgrid2worldcoord @ ref_T_mem (from vox_util)
        
        Args:
            grid_x, grid_y: World grid coordinates from ground_points
            
        Returns:
            (mem_x, mem_y): Memory coordinates for BEV display
        """
        return float(grid_x), float(grid_y)
        # try:
        #     import torch
        #     from labelme.utils.constants import get_worldgrid2worldcoord_torch
            
        #     device = 'cpu'
        #     vox_util = self._get_vox_util()
            
        #     if vox_util is None:
        #         # Fallback to simple identity
        #         return float(grid_x), float(grid_y)
            
        #     X, Y, Z = self._bev_x, self._bev_y, 2
        #     B = 1
            
        #     # Get transformation matrices
        #     worldgrid2worldcoord = get_worldgrid2worldcoord_torch(device)  # 3x3
        #     ref_T_mem = vox_util.get_ref_T_mem(B, Y, Z, X, device=device)  # B x 4 x 4
            
        #     # Extract 3x3 submatrix for 2D (rows [0,1,3], cols [0,1,3])
        #     ref_T_mem_2d = ref_T_mem[0, [0, 1, 3]][:, [0, 1, 3]]  # 3x3
            
        #     # Combined: worldgrid -> worldcoord -> mem
        #     # worldcoord_T_mem = worldgrid2worldcoord @ ref_T_mem_2d
        #     worldcoord_T_mem = torch.matmul(worldgrid2worldcoord, ref_T_mem_2d)  # 3x3
            
        #     # Invert to get mem_T_worldgrid (worldgrid -> mem)
        #     mem_T_worldgrid = torch.inverse(worldcoord_T_mem)  # 3x3
            
        #     # Apply transformation to grid point
        #     grid_pt = torch.tensor([[grid_x, grid_y, 1.0]], dtype=torch.float32, device=device)  # 1x3
        #     mem_pt = torch.matmul(grid_pt, mem_T_worldgrid.T)  # 1x3
            
        #     # Convert from homogeneous
        #     mem_x = (mem_pt[0, 0] / mem_pt[0, 2]).item()
        #     mem_y = (mem_pt[0, 1] / mem_pt[0, 2]).item()
            
        #     return mem_x, mem_y
            
        # except Exception as e:
        #     logger.warning(f"worldgrid_to_mem failed: {e}, using fallback")
        #     return float(grid_x), float(grid_y)

    def _mem_to_worldgrid(self, mem_x: float, mem_y: float) -> tuple[float, float]:
        """
        Transform memory coordinates back to world grid coordinates.
        
        Transformation chain: mem -> ref -> worldcoord -> worldgrid
        Uses: ref_T_mem (from vox_util) @ worldcoord2worldgrid
        
        Args:
            mem_x, mem_y: Memory coordinates from BEV display
            
        Returns:
            (grid_x, grid_y): World grid coordinates for ground_points
        """
        return mem_x, mem_y
        # try:
        #     import torch
        #     from labelme.utils.constants import get_worldgrid2worldcoord_torch
            
        #     device = 'cpu'
        #     vox_util = self._get_vox_util()
            
        #     if vox_util is None:
        #         return int(round(mem_x)), int(round(mem_y))
            
        #     X, Y, Z = self._bev_x, self._bev_y, 2
        #     B = 1
            
        #     # Get transformation matrices
        #     worldgrid2worldcoord = get_worldgrid2worldcoord_torch(device)  # 3x3
        #     ref_T_mem = vox_util.get_ref_T_mem(B, Y, Z, X, device=device)  # B x 4 x 4
            
        #     # Extract 3x3 submatrix for 2D (rows [0,1,3], cols [0,1,3])
        #     ref_T_mem_2d = ref_T_mem[0, [0, 1, 3]][:, [0, 1, 3]]  # 3x3
            
        #     # Combined: worldgrid -> worldcoord -> mem
        #     worldcoord_T_mem = torch.matmul(worldgrid2worldcoord, ref_T_mem_2d)  # 3x3
            
        #     # For mem -> worldgrid, we use worldcoord_T_mem directly
        #     # Apply transformation to mem point
        #     mem_pt = torch.tensor([[mem_x, mem_y, 1.0]], dtype=torch.float32, device=device)  # 1x3
        #     grid_pt = torch.matmul(mem_pt, worldcoord_T_mem.T)  # 1x3
            
        #     # Convert from homogeneous
        #     grid_x = (grid_pt[0, 0] / grid_pt[0, 2]).item()
        #     grid_y = (grid_pt[0, 1] / grid_pt[0, 2]).item()
            
        #     return int(round(grid_x)), int(round(grid_y))
            
        # except Exception as e:
        #     logger.warning(f"mem_to_worldgrid failed: {e}, using fallback")
            # return int(round(mem_x)), int(round(mem_y))

    def _load_global_annotations(self, frame_index: int) -> None:
        """Load global annotations from annotations_positions folder"""
        if not self.multi_camera_canvas or not self.multi_camera_data:
            return
        
        # Find annotation file
        root_dir = None
        if hasattr(self, "_aicv_root_dir") and self._aicv_root_dir:
            root_dir = self._aicv_root_dir
        elif self._prev_opened_dir:
            root_dir = self._prev_opened_dir
        
        if not root_dir:
            return
        
        # Load metadata (constants.json) to get default box sizes
        metadata = load_metadata(root_dir)
        default_box_width, default_box_height, default_box_depth = get_box_size_from_metadata(metadata)
        logger.info(f"Loaded metadata: default box size = ({default_box_width}, {default_box_height}, {default_box_depth})")
        
        # Try different frame_id formats (5-digit is most common)
        annotation_file = None
        for fmt in [f"{frame_index:05d}.json", f"{frame_index:08d}.json", f"{frame_index:04d}.json", f"{frame_index}.json"]:
            path = osp.join(root_dir, "annotations_positions", fmt)
            if osp.exists(path):
                annotation_file = path
                break
        
        if not annotation_file:
            logger.info(f"No annotation file found for frame {frame_index}")
            return
        
        try:
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            logger.info(f"Loading {len(annotations)} annotations from {annotation_file}")
            
            for person in annotations:
                person_id = person.get("personID")
                # Prioritize 'ground_points' (grid coordinates) over 'coordinates' (world coordinates)
                ground_points = person.get("ground_points")
                views = person.get("views", [])
                box_3d = person.get("box_3d")  # Load 3D box metadata if available
                
                if person_id is None:
                    continue
                
                # Add 3D box on BEV if box_3d metadata is available
                if self.bev_canvas and box_3d:
                    try:
                        # Extract 3D box parameters
                        grid_x = box_3d.get("x", -1)
                        grid_y = box_3d.get("y", -1)
                        z = box_3d.get("z", 0)
                        w = box_3d.get("w", default_box_width)
                        h = box_3d.get("h", default_box_height)
                        d = box_3d.get("d", default_box_depth)
                        
                        # Check if coordinates are valid
                        if grid_x >= 0 and grid_y >= 0:
                            # Transform to memory coordinates for BEV display
                            mem_x, mem_y = self._worldgrid_to_mem(grid_x, grid_y)
                            
                            # Create 3D box on BEV canvas
                            # Box format: (center, size, rotation, label, group_id)
                            center = np.array([mem_x, mem_y, z])
                            size = np.array([w, h, d])
                            rotation = 0.0  # Default rotation
                            
                            # Add box to BEV canvas
                            self.bev_canvas.boxes_3d.append((center, size, rotation, "object", person_id))
                            logger.info(f"Restored 3D box for person {person_id}: center=({mem_x:.1f}, {mem_y:.1f}, {z:.1f}), size=({w:.1f}, {h:.1f}, {d:.1f})")
                            
                    except Exception as box_e:
                        logger.warning(f"Failed to load 3D box for person {person_id}: {box_e}")
                        import traceback
                        traceback.print_exc()
                
                # Fall back to point if no 3D box or if box loading failed
                elif self.bev_canvas and ground_points and len(ground_points) >= 2:
                    # ground_points are world grid coordinates (grid_x, grid_y)
                    grid_x = ground_points[0]
                    grid_y = ground_points[1]
                    # Check if ground_points are valid (positive values)
                    if grid_x >= 0 and grid_y >= 0:
                        # Transform to memory coordinates for BEV display
                        mem_x, mem_y = self._worldgrid_to_mem(grid_x, grid_y)
                        # Add point to BEV canvas (not a box)
                        self.bev_canvas.addPoint(mem_x, mem_y, "object", person_id)
                        
                # Add bounding boxes to camera views
                for view in views:
                    view_num = view.get("viewNum", -1)
                    xmin = view.get("xmin", -1)
                    ymin = view.get("ymin", -1)
                    xmax = view.get("xmax", -1)
                    ymax = view.get("ymax", -1)
                    
                    # Skip invalid boxes
                    if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
                        continue
                    
                    # viewNum is 1-indexed in AICV format, convert to 0-indexed
                    camera_idx = view_num - 1
                    
                    if camera_idx < 0 or camera_idx >= len(self.multi_camera_canvas.camera_cells):
                        continue
                    
                    # Create rectangle shape for this camera view
                    shape = Shape(
                        label="object",
                        shape_type="rectangle",
                        group_id=person_id,
                    )
                    shape.addPoint(QtCore.QPointF(xmin, ymin))
                    shape.addPoint(QtCore.QPointF(xmax, ymax))
                    shape.close()
                    
                    # Update shape color based on group_id
                    self._update_shape_color(shape)
                    
                    # Add to the corresponding camera cell
                    cell = self.multi_camera_canvas.camera_cells[camera_idx]
                    cell.shapes.append(shape)
                    self.multi_camera_canvas.camera_cells[camera_idx] = cell
                    
                    # Add to label list
                    self.addLabel(shape)
            
            
            # Update BEV canvas
            if self.bev_canvas:
                self.bev_canvas.update()
            
            logger.info(f"Loaded annotations for frame {frame_index}")
            
        except Exception as e:
            logger.error(f"Failed to load annotations from {annotation_file}: {e}")
            import traceback
            traceback.print_exc()

    def _save_global_annotations(self) -> None:
        """Save global annotations in annotations_positions folder"""
        if not self.multi_camera_canvas or not self.multi_camera_data or not self.filename:
            return
        
        # Extract frame index from filename (e.g., "Frame_0001" -> "00001")
        try:
            if self.filename.startswith("Frame_"):
                frame_index = int(self.filename.split("_")[1])
                frame_id = f"{frame_index:05d}"  # 5-digit format to match AICV
            else:
                frame_id = "00000"
        except (ValueError, IndexError):
            logger.warning(f"Could not parse frame index from {self.filename}")
            return
        
        # Find the root directory
        # For AICV structure: _aicv_root_dir is the root that contains Image_subsets and calibrations
        # For legacy structure: _prev_opened_dir is the parent that contains camera folders
        root_dir = None
        
        if hasattr(self, "_aicv_root_dir") and self._aicv_root_dir:
            # AICV structure: root_dir is the AICV root (contains Image_subsets, calibrations)
            root_dir = self._aicv_root_dir
        elif hasattr(self, "_multi_camera_root_dirs") and self._multi_camera_root_dirs:
            # Legacy: _multi_camera_root_dirs[0] = /wildtrack/Camera1
            # Go up to get the root
            root_dir = osp.dirname(self._multi_camera_root_dirs[0])
        elif self._prev_opened_dir:
            # _prev_opened_dir is the multi-camera root
            root_dir = self._prev_opened_dir
        
        if not root_dir:
            return
        
        # Create annotations_positions folder at same level as Image_subsets
        annotation_dir = osp.join(root_dir, "annotations_positions")
        if not osp.exists(annotation_dir):
            os.makedirs(annotation_dir)
        
        # Aggregate shapes by group_id (personID)
        persons = {}  # group_id -> person data
        
        for camera_idx, cam_data in enumerate(self.multi_camera_data):
            cell = self.multi_camera_canvas.camera_cells[camera_idx]
            
            for shape in cell.shapes:
                if shape.group_id is None:
                    continue
                
                person_id = shape.group_id
                
                # Initialize person if not exists
                if person_id not in persons:
                    persons[person_id] = {
                        "personID": person_id,
                        "ground_points": [-1, -1],  # Will be updated from BEV if available (grid coordinates)
                        "views": []
                    }
                    # Initialize all views with -1 (viewNum is 1-indexed to match AICV format)
                    for view_num in range(1, len(self.multi_camera_data) + 1):
                        persons[person_id]["views"].append({
                            "viewNum": view_num,
                            "xmin": -1,
                            "ymin": -1,
                            "xmax": -1,
                            "ymax": -1
                        })
                
                # Extract bounding box for this view
                if shape.shape_type == "rectangle" and len(shape.points) >= 2:
                    x_coords = [shape.points[0].x(), shape.points[1].x()]
                    y_coords = [shape.points[0].y(), shape.points[1].y()]
                    xmin = int(min(x_coords))
                    ymin = int(min(y_coords))
                    xmax = int(max(x_coords))
                    ymax = int(max(y_coords))
                    
                    # Update the view for this camera (viewNum is 1-indexed)
                    persons[person_id]["views"][camera_idx] = {
                        "viewNum": camera_idx + 1,  # 1-indexed
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax
                    }
        
        # Update ground_points from BEV points if available
        # BEV points are in memory coordinates, need to convert back to world grid
        if self.bev_canvas and self.bev_canvas.points:
            for mem_x, mem_y, label, group_id in self.bev_canvas.points:
                if group_id in persons:
                    # Convert memory coordinates back to world grid coordinates
                    grid_x, grid_y = self._mem_to_worldgrid(mem_x, mem_y)
                    persons[group_id]["ground_points"] = [grid_x, grid_y]
        
        # Update 3D bounding box information from BEV boxes if available
        # Store x, y, z (center position) and w, h, d (box dimensions)
        if self.bev_canvas and self.bev_canvas.boxes_3d:
            for box_data in self.bev_canvas.boxes_3d:
                try:
                    # box_data format: (center, size, rotation, label, group_id)
                    center, size, rotation, label, group_id = box_data
                    
                    if group_id in persons:
                        # Convert memory coordinates to world grid coordinates
                        mem_x, mem_y, mem_z = center[0], center[1], center[2] if len(center) > 2 else 0
                        grid_x, grid_y = self._mem_to_worldgrid(mem_x, mem_y)
                        
                        # Store 3D box metadata
                        persons[group_id]["box_3d"] = {
                            "x": float(grid_x),
                            "y": float(grid_y),
                            "z": float(mem_z),  # z is typically in world units already
                            "w": float(size[0]),  # width
                            "h": float(size[1]),  # height
                            "d": float(size[2]),  # depth
                        }
                        
                        # Also update ground_points to match box center
                        persons[group_id]["ground_points"] = [grid_x, grid_y]
                        
                except Exception as e:
                    logger.warning(f"Failed to process 3D box for group {group_id}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Convert to list format
        annotations = list(persons.values())
        
        # Save to JSON file
        output_file = osp.join(annotation_dir, f"{frame_id}.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(annotations, f, indent=4)
            logger.info(f"Saved global annotations: {output_file}")
            
            # Save metadata (constants.json) at the root directory level
            # This includes BEV grid dimensions and default box sizes
            if self.bev_canvas:
                try:
                    # Get current box size from BEV canvas
                    current_box_size = getattr(self.bev_canvas, 'current_box_size', [DEFAULT_BOX_WIDTH, DEFAULT_BOX_HEIGHT, DEFAULT_BOX_DEPTH])
                    
                    save_metadata(
                        root_dir,
                        bev_x=self.bev_canvas.grid_width,
                        bev_y=self.bev_canvas.grid_height,
                        bev_z=getattr(self.bev_canvas, 'grid_depth', 2),  # Default to 2 if not set
                        box_width=current_box_size[0],
                        box_height=current_box_size[1],
                        box_depth=current_box_size[2],
                    )
                    logger.info(f"Saved metadata to {root_dir}/constants.json")
                except Exception as meta_e:
                    logger.warning(f"Failed to save metadata: {meta_e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            logger.error(f"Failed to save global annotations: {e}")
            import traceback
            traceback.print_exc()

    def duplicateSelectedShape(self):
        self.copySelectedShape()
        self.pasteSelectedShape()

    def pasteSelectedShape(self):
        self._load_shapes(shapes=self._copied_shapes, replace=False)
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def _label_selection_changed(self) -> None:
        if not self.canvas.editing():
            logger.warning("canvas is not editing mode, cannot change label selection")
            return

        selected_shapes: list[Shape] = []
        for item in self.labelList.selectedItems():
            selected_shapes.append(item.shape())
        if selected_shapes:
            self.canvas.selectShapes(selected_shapes)
        else:
            self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    # Callback functions:

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = "object"
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        description = ""
        if self._config["display_label_popup"] or not text:
            previous_text = self.labelDialog.edit.text()
            text, flags, group_id, description = self.labelDialog.popUp(text)
            if not text:
                self.labelDialog.edit.setText(previous_text)

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
        if text:
            self.labelList.clearSelection()
            
            if self.is_multi_camera_mode and self.multi_camera_canvas:
                # Multi-camera mode
                if self.multi_camera_canvas.current and self.multi_camera_canvas.current_cell_index is not None:
                    shape = self.multi_camera_canvas.current
                    shape.label = text
                    shape.flags = flags
                    shape.group_id = group_id
                    shape.description = description
                    
                    # Add shape to current cell
                    cell_idx = self.multi_camera_canvas.current_cell_index
                    cell = self.multi_camera_canvas.camera_cells[cell_idx]
                    
                    # Ensure shape is within cell bounds
                    if self.multi_camera_canvas._is_shape_in_cell_bounds(shape, cell):
                        cell.shapes.append(shape)
                        self.multi_camera_canvas.camera_cells[cell_idx] = cell
                        self.addLabel(shape)
                        self.multi_camera_canvas.current = None
                        self.multi_camera_canvas.current_cell_index = None
                        self.multi_camera_canvas.update()
                    else:
                        logger.warning("Shape is outside cell bounds, discarding")
                        self.multi_camera_canvas.current = None
                        self.multi_camera_canvas.current_cell_index = None
            else:
                # Regular mode
                shape = self.canvas.setLastLabel(text, flags)
                shape.group_id = group_id
                shape.description = description
                self.addLabel(shape)
            
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
        else:
            if self.is_multi_camera_mode and self.multi_camera_canvas:
                self.multi_camera_canvas.current = None
                self.multi_camera_canvas.current_cell_index = None
            else:
                self.canvas.undoLastLine()
                self.canvas.shapesBackups.pop()

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))
        self.scroll_values[orientation][self.filename] = value

    def _set_zoom(self, value: int, pos: QtCore.QPointF | None = None) -> None:
        if self.filename is None:
            logger.warning("filename is None, cannot set zoom")
            return

        if self.is_multi_camera_mode:
            # Multi-camera mode doesn't use zoom in the same way
            self.zoomWidget.setValue(value)
            return

        if pos is None:
            pos = QtCore.QPointF(self.canvas.visibleRegion().boundingRect().center())
        canvas_width_old: int = self.canvas.width()

        self.actions.fitWidth.setChecked(self._zoom_mode == _ZoomMode.FIT_WIDTH)
        self.actions.fitWindow.setChecked(self._zoom_mode == _ZoomMode.FIT_WINDOW)
        self.canvas.enableDragging(
            enabled=value > int(self.scalers[_ZoomMode.FIT_WINDOW]() * 100)
        )
        self.zoomWidget.setValue(value)  # triggers self._paint_canvas
        self._zoom_values[self.filename] = (self._zoom_mode, value)

        canvas_width_new: int = self.canvas.width()
        if canvas_width_old == canvas_width_new:
            return
        canvas_scale_factor = canvas_width_new / canvas_width_old
        x_shift: float = pos.x() * canvas_scale_factor - pos.x()
        y_shift: float = pos.y() * canvas_scale_factor - pos.y()
        self.setScroll(
            Qt.Horizontal,
            self.scrollBars[Qt.Horizontal].value() + x_shift,
        )
        self.setScroll(
            Qt.Vertical,
            self.scrollBars[Qt.Vertical].value() + y_shift,
        )

    def _set_zoom_to_original(self):
        self._zoom_mode = _ZoomMode.MANUAL_ZOOM
        self._set_zoom(value=100)

    def _add_zoom(self, increment: float, pos: QtCore.QPointF | None = None) -> None:
        zoom_value: int
        if increment > 1:
            zoom_value = math.ceil(self.zoomWidget.value() * increment)
        else:
            zoom_value = math.floor(self.zoomWidget.value() * increment)
        self._zoom_mode = _ZoomMode.MANUAL_ZOOM
        self._set_zoom(value=zoom_value, pos=pos)

    def _zoom_requested(self, delta: int, pos: QtCore.QPointF) -> None:
        self._add_zoom(increment=1.1 if delta > 0 else 0.9, pos=pos)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self._zoom_mode = _ZoomMode.FIT_WINDOW if value else _ZoomMode.MANUAL_ZOOM
        self._adjust_scale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self._zoom_mode = _ZoomMode.FIT_WIDTH if value else _ZoomMode.MANUAL_ZOOM
        self._adjust_scale()

    def enableKeepPrevScale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)

    def onNewBrightnessContrast(self, qimage):
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage), clear_shapes=False)

    def brightnessContrast(self, value: bool, is_initial_load: bool = False):
        if self.filename is None:
            logger.warning("filename is None, cannot set brightness/contrast")
            return

        # Multi-camera mode doesn't support brightness/contrast adjustment
        if self.is_multi_camera_mode:
            self.errorMessage(
                self.tr("Not supported"),
                self.tr("Brightness/Contrast adjustment is not available in multi-camera mode."),
            )
            return

        if not hasattr(self, 'imageData') or self.imageData is None:
            logger.warning("imageData is None, cannot set brightness/contrast")
            return

        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData).convert("RGB"),
            self.onNewBrightnessContrast,
            parent=self,
        )

        brightness: int | None
        contrast: int | None
        brightness, contrast = self._brightness_contrast_values.get(
            self.filename, (None, None)
        )
        if is_initial_load:
            prev_filename: str = self.recentFiles[0] if self.recentFiles else ""
            if self._config["keep_prev_brightness_contrast"] and prev_filename:
                brightness, contrast = self._brightness_contrast_values.get(
                    prev_filename, (None, None)
                )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)

        if is_initial_load:
            dialog.onNewValue(None)
        else:
            dialog.exec_()
            brightness = dialog.slider_brightness.value()
            contrast = dialog.slider_contrast.value()

        self._brightness_contrast_values[self.filename] = (brightness, contrast)

    def togglePolygons(self, value):
        flag = value
        for item in self.labelList:
            if value is None:
                flag = item.checkState() == Qt.Unchecked
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)

    def _load_file(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        if filename in self.imageList and (
            self.fileListWidget.currentRow() != self.imageList.index(filename)
        ):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            self.fileListWidget.repaint()
            return

        # Get previous shapes (works for both regular and multi-camera)
        if self.is_multi_camera_mode and self.multi_camera_canvas:
            prev_shapes = (
                self.multi_camera_canvas.getShapes()
                if self._config["keep_prev"]
                or QtWidgets.QApplication.keyboardModifiers()
                == (Qt.ControlModifier | Qt.ShiftModifier)
                else []
            )
        else:
            prev_shapes = (
                self.canvas.shapes
                if self._config["keep_prev"]
                or QtWidgets.QApplication.keyboardModifiers()
                == (Qt.ControlModifier | Qt.ShiftModifier)
                else []
            )
        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False
        # assumes same name, but json extension
        self.show_status_message(self.tr("Loading %s...") % osp.basename(str(filename)))
        label_file = f"{osp.splitext(filename)[0]}.json"
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p><p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.show_status_message(self.tr("Error reading %s") % label_file)
                return False
            assert self.labelFile is not None
            self.imageData = self.labelFile.imageData
            assert self.labelFile.imagePath
            self.imagePath = osp.join(
                osp.dirname(label_file),
                self.labelFile.imagePath,
            )
            self._other_data = self.labelFile.otherData
        else:
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
        assert self.imageData is not None
        image = QtGui.QImage.fromData(self.imageData)

        if image.isNull():
            formats = [
                f"*.{fmt.data().decode()}"
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.show_status_message(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.filename = filename
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        flags = {k: False for k in self._config["flags"] or []}
        if self.labelFile:
            self._load_shape_dicts(shape_dicts=self.labelFile.shapes)
            if self.labelFile.flags is not None:
                flags.update(self.labelFile.flags)
        self._load_flags(flags=flags)
        if prev_shapes and self.noShapes():
            self._load_shapes(shapes=prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self._zoom_values
        if self.filename in self._zoom_values:
            self._zoom_mode = self._zoom_values[self.filename][0]
            self._set_zoom(self._zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self._zoom_mode = _ZoomMode.FIT_WINDOW
            self._adjust_scale()
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        self.brightnessContrast(value=False, is_initial_load=True)
        self._paint_canvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.canvas.setFocus()
        self.show_status_message(self.tr("Loaded %s") % osp.basename(filename))
        logger.debug("loaded file: {!r}", filename)
        return True

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        if (
            self.canvas
            and not self.image.isNull()
            and self._zoom_mode != _ZoomMode.MANUAL_ZOOM
        ):
            self._adjust_scale()
        super().resizeEvent(a0)

    def _paint_canvas(self) -> None:
        if self.is_multi_camera_mode:
            if self.multi_camera_canvas:
                self.multi_camera_canvas.scale = 0.01 * self.zoomWidget.value()
                self.multi_camera_canvas.adjustSize()
                self.multi_camera_canvas.update()
            return
        
        if self.image.isNull():
            logger.warning("image is null, cannot paint canvas")
            return
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def _adjust_scale(self) -> None:
        self._set_zoom(value=int(self.scalers[self._zoom_mode]() * 100))

    def scaleFitWindow(self) -> float:
        EPSILON_TO_HIDE_SCROLLBAR: float = 2.0
        w1: float = self.centralWidget().width() - EPSILON_TO_HIDE_SCROLLBAR
        h1: float = self.centralWidget().height() - EPSILON_TO_HIDE_SCROLLBAR
        a1: float = w1 / h1

        if self.is_multi_camera_mode:
            # For multi-camera, just return a reasonable scale
            return 1.0

        w2: float = self.canvas.pixmap.width()
        h2: float = self.canvas.pixmap.height()
        a2: float = w2 / h2

        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        EPSILON_TO_HIDE_SCROLLBAR: float = 15.0
        w = self.centralWidget().width() - EPSILON_TO_HIDE_SCROLLBAR
        
        if self.is_multi_camera_mode:
            # For multi-camera, just return a reasonable scale
            return 1.0
        
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        # self._config["store_data"] = enabled
        # self.actions.saveWithImageData.setChecked(enabled)
        pass  # imageData saving disabled

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if not self._can_continue():
            a0.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def dragEnterEvent(self, a0: QtGui.QDragEnterEvent) -> None:
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if a0.mimeData().hasUrls():
            items = [i.toLocalFile() for i in a0.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                a0.accept()
        else:
            a0.ignore()

    def dropEvent(self, a0: QtGui.QDropEvent) -> None:
        if not self._can_continue():
            a0.ignore()
            return
        items = [i.toLocalFile() for i in a0.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self._can_continue():
            self._load_file(filename)

    def _open_prev_image(self, _value=False) -> None:
        row_prev: int = self.fileListWidget.currentRow() - 1
        if row_prev < 0:
            logger.debug("there is no prev image")
            return

        logger.debug("setting current row to {:d}", row_prev)
        self.fileListWidget.setCurrentRow(row_prev)
        self.fileListWidget.repaint()

    def _open_next_image(self, _value=False) -> None:
        row_next: int = self.fileListWidget.currentRow() + 1
        if row_next >= self.fileListWidget.count():
            logger.debug("there is no next image")
            return

        logger.debug("setting current row to {:d}", row_next)
        self.fileListWidget.setCurrentRow(row_next)
        self.fileListWidget.repaint()

    def _open_file_with_dialog(self, _value: bool = False) -> None:
        if not self._can_continue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            f"*.{fmt.data().decode()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + [f"*{LabelFile.suffix}"]
        )
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        fileDialog.setWindowFilePath(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]
            if fileName:
                self._load_file(fileName)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()

        current_filename = self.filename
        self._import_images_from_dir(root_dir=self._prev_opened_dir)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(self.imageList.index(current_filename))
            self.fileListWidget.repaint()

    def saveFile(self, _value=False):
        if self.is_multi_camera_mode:
            # Multi-camera mode - save all camera labels
            if self._save_multi_camera_labels():
                self.setClean()
            return
        
        assert not self.image.isNull(), "cannot save empty image"
        if self.labelFile:
            # DL20180323 - overwrite when in directory
            self._saveFile(self.labelFile.filename)
        elif self.output_file:
            self._saveFile(self.output_file)
            self.close()
        else:
            self._saveFile(self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        assert self.filename is not None
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(self, caption, self.output_dir, filters)
        else:
            dlg = QtWidgets.QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            return filename[0]
        return filename

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()

    def closeFile(self, _value=False):
        if not self._can_continue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.fileListWidget.setFocus()
        self.actions.saveAs.setEnabled(False)

    def getLabelFile(self):
        assert self.filename is not None
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = f"{osp.splitext(self.filename)[0]}.json"

        return label_file

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info(f"Label file is removed: {label_file}")

            item = self.fileListWidget.currentItem()
            if item:
                item.setCheckState(Qt.Unchecked)

            self.resetState()

    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def _can_continue(self) -> bool:
        if not self._is_changed:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(self.filename)
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, f"<p><b>{title}</b></p>{message}"
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggleKeepPrevMode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]

    def removeSelectedPoint(self):
        if self.is_multi_camera_mode and self.multi_camera_canvas:
            self.multi_camera_canvas.removeSelectedPoint()
            if self.multi_camera_canvas.hShape and not self.multi_camera_canvas.hShape.points:
                # Remove shape from its cell
                for idx, cell in enumerate(self.multi_camera_canvas.camera_cells):
                    if self.multi_camera_canvas.hShape in cell.shapes:
                        cell.shapes.remove(self.multi_camera_canvas.hShape)
                        self.multi_camera_canvas.camera_cells[idx] = cell
                        self.remLabels([self.multi_camera_canvas.hShape])
                        break
                if self.noShapes():
                    for action in self.on_shapes_present_actions:
                        action.setEnabled(False)
        else:
            self.canvas.removeSelectedPoint()
            if self.canvas.hShape and not self.canvas.hShape.points:
                self.canvas.deleteShape(self.canvas.hShape)
                self.remLabels([self.canvas.hShape])
                if self.noShapes():
                    for action in self.on_shapes_present_actions:
                        action.setEnabled(False)
        self.setDirty()

    def deleteSelectedShape(self):
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        num_selected = len(self.canvas.selectedShapes) if hasattr(self.canvas, 'selectedShapes') else 0
        msg = self.tr(
            "You are about to permanently delete {} polygons, proceed anyway?"
        ).format(num_selected)
        if yes == QtWidgets.QMessageBox.warning(
            self, self.tr("Attention"), msg, yes | no, yes
        ):
            deleted = self.canvas.deleteSelected()
            self.remLabels(deleted)
            self.setDirty()
            if self.noShapes():
                for action in self.on_shapes_present_actions:
                    action.setEnabled(False)

    def copyShape(self):
        if self.is_multi_camera_mode and self.multi_camera_canvas:
            # Multi-camera mode
            if self.multi_camera_canvas.selectedShapesCopy:
                # Add copies to their respective cells
                for shape_copy in self.multi_camera_canvas.selectedShapesCopy:
                    # Find which cell the original shape belongs to
                    for idx, cell in enumerate(self.multi_camera_canvas.camera_cells):
                        if any(s in cell.shapes for s in self.multi_camera_canvas.selectedShapes):
                            # Add copy to same cell
                            new_shape = shape_copy.copy()
                            cell.shapes.append(new_shape)
                            self.multi_camera_canvas.camera_cells[idx] = cell
                            self.addLabel(new_shape)
                            break
                self.multi_camera_canvas.selectedShapesCopy = []
                self.multi_camera_canvas.storeShapes()
                self.multi_camera_canvas.update()
        else:
            # Regular mode
            self.canvas.endMove(copy=True)
            for shape in self.canvas.selectedShapes:
                self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()

    def moveShape(self):
        if self.is_multi_camera_mode and self.multi_camera_canvas:
            # Multi-camera mode - move is already handled in mouseReleaseEvent
            self.multi_camera_canvas.endMove(copy=False)
        else:
            # Regular mode
            self.canvas.endMove(copy=False)
        self.setDirty()

    def _open_dir_with_dialog(self, _value: bool = False) -> None:
        if not self._can_continue():
            return

        defaultOpenDirPath: str
        if getattr(self, "_multi_camera_root_dirs", None):
            # If multi-camera root dirs were used before, reuse the first one as default
            defaultOpenDirPath = self._multi_camera_root_dirs[0]
        elif self._prev_opened_dir and osp.exists(self._prev_opened_dir):
            defaultOpenDirPath = self._prev_opened_dir
        else:
            defaultOpenDirPath = osp.dirname(self.filename) if self.filename else "."

        # Use non-native QFileDialog to allow selecting multiple folders (Ctrl / Shift)
        dlg = QtWidgets.QFileDialog(
            self,
            self.tr("%s - Open Directory") % __appname__,
            defaultOpenDirPath,
        )
        # Important: Directory + DontUseNativeDialog + adjust selectionMode internally
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dlg.setOption(QtWidgets.QFileDialog.DontResolveSymlinks, True)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)

        # Allow selecting multiple folders by forcing internal views to ExtendedSelection
        for view in dlg.findChildren(QtWidgets.QListView) + dlg.findChildren(QtWidgets.QTreeView):
            view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # Get all selected paths and keep only directories
        selected_dirs = [p for p in dlg.selectedFiles() if osp.isdir(p)]

        # If user selects only one folder => keep legacy behavior (single root_dir)
        # If multiple folders are selected => treat them as explicit camera folders
        if not selected_dirs:
            return

        if len(selected_dirs) == 1:
            targetDirPath = selected_dirs[0]
            self._multi_camera_root_dirs = [targetDirPath]
            self._import_images_from_dir(root_dir=targetDirPath)
        else:
            # Store explicit camera folders for multi-camera mode
            self._multi_camera_root_dirs = list(selected_dirs)
            # Use parent directory of the first folder as _prev_opened_dir to keep compatibility
            parent_dir = osp.dirname(self._multi_camera_root_dirs[0])
            self._prev_opened_dir = parent_dir
            # Import as multi-camera using the common parent directory (frame index 0)
            self._import_images_from_dir(root_dir=parent_dir)

        self._open_next_image()

    @property
    def imageList(self) -> list[str]:
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            assert item
            lst.append(item.text())
        return lst

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(tuple(extensions)):
                continue
            label_file = f"{osp.splitext(file)[0]}.json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)

        if len(self.imageList) > 1:
            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)

        self._open_next_image()

    def _import_images_from_dir(
        self, root_dir: str | None, pattern: str | None = None
    ) -> None:
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)

        if not self._can_continue() or not root_dir:
            return

        self._prev_opened_dir = root_dir
        self.filename = None
        self.fileListWidget.clear()

        # Check if this is a multi-camera directory (legacy or AICV structure)
        if _detect_aicv_structure(root_dir):
            # AICV structure: Image_subsets + calibrations
            self.is_multi_camera_mode = True
            self._aicv_root_dir = root_dir  # Store AICV root for calibration lookup
            
            # Load metadata (constants.json) to get BEV grid dimensions and default box sizes
            try:
                metadata = load_metadata(root_dir)
                bev_x, bev_y, bev_z = get_bev_grid_from_metadata(metadata)
                box_w, box_h, box_d = get_box_size_from_metadata(metadata)
                
                logger.info(f"Loaded metadata: BEV grid=({bev_x}, {bev_y}, {bev_z}), box size=({box_w}, {box_h}, {box_d})")
                
                # Apply metadata to BEV canvas if available
                if self.bev_canvas:
                    self.bev_canvas.grid_width = bev_x
                    self.bev_canvas.grid_height = bev_y
                    self.bev_canvas.current_box_size = [box_w, box_h, box_d]
                    self.bev_canvas.update()
                    logger.info(f"Applied metadata to BEV canvas")
                    
            except Exception as meta_e:
                logger.warning(f"Failed to load or apply metadata: {meta_e}")
                import traceback
                traceback.print_exc()
            
            image_subsets_dir = osp.join(root_dir, "Image_subsets")
            subdirs = sorted([d for d in os.listdir(image_subsets_dir) if osp.isdir(osp.join(image_subsets_dir, d))])
            max_frames = 0
            extensions = [
                f".{fmt.data().decode().lower()}"
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            for subdir in subdirs:
                folder_path = osp.join(image_subsets_dir, subdir)
                frame_files = [
                    f for f in os.listdir(folder_path)
                    if f.lower().endswith(tuple(extensions))
                ]
                max_frames = max(max_frames, len(frame_files))
            
            # Add frame indices to file list
            for frame_idx in range(max_frames):
                frame_label = f"Frame_{frame_idx:04d}"
                item = QtWidgets.QListWidgetItem(frame_label)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                # Check if annotation exists in annotations_positions (try different formats)
                annotation_exists = False
                for fmt in [f"{frame_idx:05d}.json", f"{frame_idx:04d}.json", f"{frame_idx:08d}.json"]:
                    test_path = osp.join(root_dir, "annotations_positions", fmt)
                    if osp.exists(test_path):
                        annotation_exists = True
                        break
                item.setCheckState(Qt.Checked if annotation_exists else Qt.Unchecked)
                self.fileListWidget.addItem(item)
        elif _detect_multi_camera_structure(root_dir):
            # Legacy multi-camera structure with frames/calibrations per camera
            self.is_multi_camera_mode = True
            self._aicv_root_dir = None
            # For multi-camera, we list frame indices instead of individual files
            # Find max number of frames
            subdirs = sorted([d for d in os.listdir(root_dir) if osp.isdir(osp.join(root_dir, d))])
            max_frames = 0
            for subdir in subdirs:
                frames_dir = osp.join(root_dir, subdir, "frames")
                if osp.isdir(frames_dir):
                    extensions = [
                        f".{fmt.data().decode().lower()}"
                        for fmt in QtGui.QImageReader.supportedImageFormats()
                    ]
                    frame_files = [
                        f for f in os.listdir(frames_dir)
                        if f.lower().endswith(tuple(extensions))
                    ]
                    max_frames = max(max_frames, len(frame_files))
            
            # Add frame indices to file list
            for frame_idx in range(max_frames):
                frame_label = f"Frame_{frame_idx:04d}"
                item = QtWidgets.QListWidgetItem(frame_label)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                # Check if all cameras have labels for this frame
                all_labeled = True
                for subdir in subdirs:
                    frames_dir = osp.join(root_dir, subdir, "frames")
                    if osp.isdir(frames_dir):
                        frame_files = sorted([
                            f for f in os.listdir(frames_dir)
                            if f.lower().endswith(tuple([
                                f".{fmt.data().decode().lower()}"
                                for fmt in QtGui.QImageReader.supportedImageFormats()
                            ]))
                        ])
                        if frame_idx < len(frame_files):
                            frame_file = frame_files[frame_idx]
                            label_file = f"{osp.splitext(osp.join(frames_dir, frame_file))[0]}.json"
                            if self.output_dir:
                                label_file_without_path = osp.basename(label_file)
                                label_file = osp.join(self.output_dir, label_file_without_path)
                            if not (QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file)):
                                all_labeled = False
                                break
                item.setCheckState(Qt.Checked if all_labeled else Qt.Unchecked)
                self.fileListWidget.addItem(item)
        else:
            # Regular single-image mode
            self.is_multi_camera_mode = False
            filenames = _scan_image_files(root_dir=root_dir)
            if pattern:
                try:
                    filenames = [f for f in filenames if re.search(pattern, f)]
                except re.error:
                    pass
            for filename in filenames:
                label_file = f"{osp.splitext(filename)[0]}.json"
                if self.output_dir:
                    label_file_without_path = osp.basename(label_file)
                    label_file = osp.join(self.output_dir, label_file_without_path)
                item = QtWidgets.QListWidgetItem(filename)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                    item.setCheckState(Qt.Checked)
                else:
                    item.setCheckState(Qt.Unchecked)
                self.fileListWidget.addItem(item)

    def _load_multi_camera_frame(self, frame_index: int) -> None:
        """Load a multi-camera frame"""
        # Prefer using explicit camera folders list when available
        if getattr(self, "_multi_camera_root_dirs", None):
            camera_data = _scan_multi_camera_data(self._multi_camera_root_dirs, frame_index)
        else:
            if not self._prev_opened_dir:
                return
            # Scan camera data for this frame from a single root directory (legacy behavior)
            camera_data = _scan_multi_camera_data(self._prev_opened_dir, frame_index)
        if not camera_data:
            self.errorMessage(
                self.tr("Error loading frame"),
                self.tr("No camera data found for frame %d") % frame_index,
            )
            return
        
        # Switch to multi-camera canvas if not already
        if not self.is_multi_camera_mode or self.multi_camera_canvas is None:
            self._switch_to_multi_camera_canvas()
        
        # Load calibrations
        self.camera_calibrations = {}
        for cam_data in camera_data:
            camera_id = cam_data["camera_id"]
            calibration_path = cam_data.get("calibration_path")
            
            # Try to find XML calibration files
            if calibration_path and osp.isdir(calibration_path):
                # Look for intrinsic and extrinsic XML files
                intrinsic_original_dir = osp.join(calibration_path, "intrinsic_original")
                extrinsic_dir = osp.join(calibration_path, "extrinsic")
                
                if osp.isdir(intrinsic_original_dir) and osp.isdir(extrinsic_dir):
                    # Find XML files
                    intrinsic_files = [f for f in os.listdir(intrinsic_original_dir) if f.endswith('.xml')]
                    extrinsic_files = [f for f in os.listdir(extrinsic_dir) if f.endswith('.xml')]
                    
                    if intrinsic_files and extrinsic_files:
                        intrinsic_path = osp.join(intrinsic_original_dir, intrinsic_files[0])
                        extrinsic_path = osp.join(extrinsic_dir, extrinsic_files[0])
                        
                        # Load calibration with default scale factors
                        calibration = CameraCalibration.load_from_xml_files_scaled(
                            intrinsic_path, 
                            extrinsic_path,
                            intrinsic_scale=self._intrinsic_scale_factor,
                            translation_scale=self._translation_scale_factor
                        )
                        if calibration:
                            self.camera_calibrations[camera_id] = calibration
                            logger.info(f"Loaded calibration for camera {camera_id} (K×{self._intrinsic_scale_factor}, T×{self._translation_scale_factor})")
                        else:
                            logger.warning(f"Failed to load calibration for camera {camera_id}")
                    else:
                        logger.warning(f"Calibration XML files not found for camera {camera_id}")
                else:
                    # Try legacy JSON format
                    json_path = osp.join(calibration_path, "calibration.json")
                    if osp.exists(json_path):
                        calibration = CameraCalibration.load_from_file(json_path)
                        if calibration:
                            self.camera_calibrations[camera_id] = calibration
                            logger.info(f"Loaded calibration (JSON) for camera {camera_id}")
            elif calibration_path and osp.exists(calibration_path):
                # Single file (JSON)
                calibration = CameraCalibration.load_from_file(calibration_path)
                if calibration:
                    self.camera_calibrations[camera_id] = calibration
                    logger.info(f"Loaded calibration for camera {camera_id}")
        
        # Load camera data
        self.multi_camera_canvas.loadMultiCameraData(camera_data)
        self.multi_camera_data = camera_data
        
        # Setup BEV if calibrations are available
        if self.camera_calibrations and not self.bev_canvas:
            self._setup_bev_canvas()
        elif self.bev_canvas and self.camera_calibrations:
            # Update camera overlay if BEV already exists
            self.bev_canvas.setCameraOverlay(self.camera_calibrations, camera_data)
            
            # Regenerate BEV background image when clicking different frame in file list
            try:
                bev_width, bev_height, bev_scale = self._calculate_bev_bounds()
                logger.info("Regenerating BEV overlay for new frame...")
                bev_image = generate_bev_from_cameras(
                    self.camera_calibrations,
                    camera_data,
                    bev_width,
                    bev_height,
                    resolution=max(bev_width, bev_height) / 512.0,
                    bev_x=self._bev_x,
                    bev_y=self._bev_y,
                    bev_bounds=self._bev_bounds,
                )
                if bev_image is not None:
                    h, w, _ = bev_image.shape
                    qimg = QtGui.QImage(
                        bev_image.data,
                        w,
                        h,
                        3 * w,
                        QtGui.QImage.Format_RGB888,
                    ).copy()
                    self.bev_canvas.setBackgroundImage(qimg, alpha=1.0)
                    logger.info("Updated BEV overlay image for new frame")
            except Exception as e:
                logger.warning(f"Failed to update BEV overlay image: {e}")
        
        # Clear BEV boxes and points when loading new frame
        if self.bev_canvas:
            self.bev_canvas.clearBoxes()
            self.bev_canvas.clearPoints()
        
        # Load labels
        self.labelList.clear()
        
        # For AICV structure, load from annotations_positions folder
        is_aicv = hasattr(self, "_aicv_root_dir") and self._aicv_root_dir is not None
        
        if is_aicv:
            # Load global annotations from annotations_positions
            self._load_global_annotations(frame_index)
        else:
            # Legacy: load per-camera JSON files
            for idx, cam_data in enumerate(camera_data):
                image_path = cam_data["image_path"]
                label_file = f"{osp.splitext(image_path)[0]}.json"
                if self.output_dir:
                    label_file_without_path = osp.basename(label_file)
                    label_file = osp.join(self.output_dir, label_file_without_path)
                
                if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                    try:
                        label_file_obj = LabelFile(label_file)
                        shapes = []
                        for shape_dict in label_file_obj.shapes:
                            shape = Shape(
                                label=shape_dict["label"],
                                shape_type=shape_dict["shape_type"],
                                group_id=shape_dict.get("group_id"),
                                description=shape_dict.get("description"),
                            )
                            for x, y in shape_dict["points"]:
                                shape.addPoint(QtCore.QPointF(x, y))
                            shape.close()
                            # Preserve from_bev flag if it exists
                            if "other_data" in shape_dict and "from_bev" in shape_dict["other_data"]:
                                shape.other_data["from_bev"] = shape_dict["other_data"]["from_bev"]
                            shapes.append(shape)
                        
                        # Load shapes into the corresponding cell ONLY
                        if idx < len(self.multi_camera_canvas.camera_cells):
                            cell = self.multi_camera_canvas.camera_cells[idx]
                            # Add shapes to this specific cell
                            for shape in shapes:
                                cell.shapes.append(shape)
                            self.multi_camera_canvas.camera_cells[idx] = cell
                            
                            # Add shapes to label list
                            for shape in shapes:
                                self.addLabel(shape)
                    except LabelFileError as e:
                        logger.warning(f"Failed to load label file {label_file}: {e}")
        
        self.multi_camera_canvas.update()
        self._update_ground_point_list()
        self.setClean()
        self.toggleActions(True)
        self.filename = f"Frame_{frame_index:04d}"
        self.show_status_message(self.tr("Loaded frame %d") % frame_index)

    def _switch_to_multi_camera_canvas(self):
        """Switch from regular canvas to multi-camera canvas"""
        # Store current canvas state
        old_canvas = self.canvas
        
        # Create multi-camera canvas
        self.multi_camera_canvas = MultiCameraCanvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
        
        # Connect signals
        self.multi_camera_canvas.newShape.connect(self.newShape)
        self.multi_camera_canvas.shapeMoved.connect(self.setDirty)
        self.multi_camera_canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.multi_camera_canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.multi_camera_canvas.mouseMoved.connect(self._update_status_stats)
        self.multi_camera_canvas.statusUpdated.connect(lambda text: self.status_left.setText(text))
        
        # Replace canvas in scroll area
        scroll_area = self.centralWidget()
        scroll_area.setWidget(self.multi_camera_canvas)
        
        # Update canvas reference
        self.canvas = self.multi_camera_canvas
    
    def _calculate_bev_bounds(self) -> tuple[float, float, float]:
        """Calculate BEV bounds from camera positions - based on max coordinates of cameras"""
        if not self.camera_calibrations or not self.multi_camera_data:
            return 100.0, 100.0, 50.0  # Default values
        
        # Find min/max coordinates from camera positions
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        for cam_data in self.multi_camera_data:
            camera_id = cam_data["camera_id"]
            calibration = self.camera_calibrations.get(camera_id)
            
            if not calibration:
                continue
            
            # Get camera position in world space
            R = calibration.extrinsic[:3, :3]
            t = calibration.extrinsic[:3, 3]
            camera_pos_world = -R.T @ t
            
            # Update bounds based on camera position
            min_x = min(min_x, camera_pos_world[0])
            max_x = max(max_x, camera_pos_world[0])
            min_y = min(min_y, camera_pos_world[1])
            max_y = max(max_y, camera_pos_world[1])
        
        # Calculate bounds with margin
        if min_x == float('inf'):
            return 100.0, 100.0, 50.0  # Default if no valid points
        
        width = max(max_x - min_x, 10.0) * 1.2  # Add 20% margin
        height = max(max_y - min_y, 10.0) * 1.2
        # Scale to fit nicely in widget (assume widget is ~400px, want ~80% coverage)
        scale = min(400.0 / max(width, height) * 0.8, 100.0)  # Limit max scale
        
        return width, height, scale
    
    def _apply_calibration_scales(self) -> None:
        """Apply calibration scale factors and regenerate BEV"""
        if not self.is_multi_camera_mode or not self.multi_camera_data:
            logger.warning("No multi-camera data loaded")
            return
        
        # Read values from spinboxes
        self._intrinsic_scale_factor = self._intrinsic_scale_spinbox.value()
        self._translation_scale_factor = self._translation_scale_spinbox.value()
        self._bev_x = self._bev_x_spinbox.value()
        self._bev_y = self._bev_y_spinbox.value()
        self._bev_bounds[1] = self._bev_xmax_spinbox.value()  # XMAX
        self._bev_bounds[3] = self._bev_ymax_spinbox.value()  # YMAX
        
        logger.info(f"Applying settings: K={self._intrinsic_scale_factor}, T={self._translation_scale_factor}, "
                    f"BEV X={self._bev_x}, Y={self._bev_y}, XMAX={self._bev_bounds[1]}, YMAX={self._bev_bounds[3]}")
        
        # Reload calibrations with new scale factors
        self._reload_calibrations_with_scales()
        
        # Regenerate BEV
        if self.bev_canvas and self.camera_calibrations:
            self._regenerate_bev()
        
        self.show_status_message(f"Applied: K={self._intrinsic_scale_factor}, T={self._translation_scale_factor}, "
                                  f"Grid={self._bev_x}x{self._bev_y}, Bounds=[{self._bev_bounds[1]},{self._bev_bounds[3]}]")
    
    def _reload_calibrations_with_scales(self) -> None:
        """Reload calibrations with current scale factors"""
        if not self.multi_camera_data:
            return
        
        from labelme.utils.calibration import CameraCalibration
        
        for cam_data in self.multi_camera_data:
            camera_id = cam_data["camera_id"]
            calibration_path = cam_data.get("calibration_path")
            
            if not calibration_path or not osp.isdir(calibration_path):
                continue
            
            # Look for intrinsic and extrinsic XML files
            intrinsic_original_dir = osp.join(calibration_path, "intrinsic_original")
            extrinsic_dir = osp.join(calibration_path, "extrinsic")
            
            if osp.isdir(intrinsic_original_dir) and osp.isdir(extrinsic_dir):
                intrinsic_files = [f for f in os.listdir(intrinsic_original_dir) if f.endswith('.xml')]
                extrinsic_files = [f for f in os.listdir(extrinsic_dir) if f.endswith('.xml')]
                
                if intrinsic_files and extrinsic_files:
                    intrinsic_path = osp.join(intrinsic_original_dir, intrinsic_files[0])
                    extrinsic_path = osp.join(extrinsic_dir, extrinsic_files[0])
                    
                    # Load calibration with scale factors
                    calibration = CameraCalibration.load_from_xml_files_scaled(
                        intrinsic_path, 
                        extrinsic_path,
                        intrinsic_scale=self._intrinsic_scale_factor,
                        translation_scale=self._translation_scale_factor
                    )
                    if calibration:
                        self.camera_calibrations[camera_id] = calibration
                        logger.info(f"Reloaded calibration for {camera_id} with scales")
    
    def _regenerate_bev(self) -> None:
        """Regenerate BEV background image with current calibrations"""
        if not self.bev_canvas or not self.camera_calibrations or not self.multi_camera_data:
            return
        
        try:
            # Use configurable BEV parameters
            bev_width = float(self._bev_bounds[1])  # XMAX
            bev_height = float(self._bev_bounds[3])  # YMAX
            bev_scale = max(bev_width, bev_height) / 512.0
            
            # Update BEV canvas parameters
            self.bev_canvas.setBEVParams(width=bev_width, height=bev_height, scale=bev_scale)
            self.bev_canvas.setCameraOverlay(self.camera_calibrations, self.multi_camera_data)
            
            logger.info(f"Regenerating BEV overlay with grid={self._bev_x}x{self._bev_y}, bounds=[{self._bev_bounds[1]},{self._bev_bounds[3]}]...")
            bev_image = generate_bev_from_cameras(
                self.camera_calibrations,
                self.multi_camera_data,
                bev_width,
                bev_height,
                resolution=bev_scale,
                bev_x=self._bev_x,
                bev_y=self._bev_y,
                bev_bounds=self._bev_bounds,
            )
            if bev_image is not None:
                h, w, _ = bev_image.shape
                qimg = QtGui.QImage(
                    bev_image.data,
                    w,
                    h,
                    3 * w,
                    QtGui.QImage.Format_RGB888,
                ).copy()
                self.bev_canvas.setBackgroundImage(qimg, alpha=1.0)
                logger.info("✅ BEV overlay regenerated successfully")
            else:
                logger.warning("BEV generation returned None")
        except Exception as e:
            logger.error(f"Failed to regenerate BEV: {e}")
            import traceback
            traceback.print_exc()
    
    def _setup_bev_canvas(self):
        """Setup BEV canvas for 3D box placement"""
        if self.bev_canvas:
            return
        
        # Calculate BEV bounds from camera calibrations
        bev_width, bev_height, bev_scale = self._calculate_bev_bounds()
        
        # Create BEV canvas
        self.bev_canvas = BEVCanvas(self)
        self.bev_canvas.setBEVParams(width=bev_width, height=bev_height, scale=bev_scale)
        
        # Set camera overlay data
        if self.camera_calibrations and self.multi_camera_data:
            self.bev_canvas.setCameraOverlay(self.camera_calibrations, self.multi_camera_data)

            # Generate BEV overlay image directly from camera images and calibration.
            # This samples the ground plane (z=0) into a top-down mosaic.
            try:
                bev_image = generate_bev_from_cameras(
                    self.camera_calibrations,
                    self.multi_camera_data,
                    bev_width,
                    bev_height,
                    resolution=max(bev_width, bev_height) / 512.0,
                    bev_x=self._bev_x,
                    bev_y=self._bev_y,
                    bev_bounds=self._bev_bounds,
                )
                if bev_image is not None:
                    h, w, _ = bev_image.shape
                    # Convert numpy RGB array to QImage
                    qimg = QtGui.QImage(
                        bev_image.data,
                        w,
                        h,
                        3 * w,
                        QtGui.QImage.Format_RGB888,
                    ).copy()

                    # Set BEV background to generated overlay
                    self.bev_canvas.setBackgroundImage(qimg, alpha=1.0)
                    logger.info("Generated BEV overlay image from cameras")
            except Exception as e:
                logger.warning(f"Failed to generate BEV overlay image: {e}")
        
        # Connect signals
        self.bev_canvas.boxPlaced.connect(self._on_bev_box_placed)
        self.bev_canvas.boxMoved.connect(self._on_bev_box_moved)
        self.bev_canvas.boxSizeChanged.connect(self._on_bev_box_size_changed)
        self.bev_canvas.boxDeleted.connect(self._on_bev_box_deleted)
        
        # Point signals
        self.bev_canvas.pointPlaced.connect(self._on_bev_point_placed)
        self.bev_canvas.pointSelected.connect(self._on_bev_point_selected)
        self.bev_canvas.pointHovered.connect(self._on_bev_point_hovered)
        self.bev_canvas.pointDoubleClicked.connect(self._on_bev_point_double_clicked)
        self.bev_canvas.pointDeleted.connect(self._on_bev_point_deleted)
        self.bev_canvas.pointMoved.connect(self._on_bev_point_moved)
        
        # Create dock widget for BEV with step size control
        bev_dock = QtWidgets.QDockWidget("BEV View (3D Box Placement)", self)
        bev_widget = QtWidgets.QWidget()
        bev_layout = QtWidgets.QVBoxLayout()
        bev_layout.setContentsMargins(0, 0, 0, 0)
        
        # bev_layout.addLayout(step_layout)
        bev_layout.addWidget(self.bev_canvas)
        bev_widget.setLayout(bev_layout)
        
        bev_dock.setWidget(bev_widget)
        bev_dock.setObjectName("BEVViewDock")  # Fix warning about objectName
        bev_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, bev_dock)
        bev_dock.setVisible(True)
    
    def _on_bev_box_placed(self, x: float, y: float, z: float, w: float, h: float, d: float):
        """Handle box placement from BEV canvas"""
        # Get next unique group_id
        next_group_id = self._get_next_group_id()
        
        # Show dialog to get label and group_id
        dialog = Box3DDialog(
            self,
            x=x, y=y, z=z,
            w=w, h=h, d=d,
            group_id=next_group_id
        )
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            x, y, z, w, h, d, label, group_id = dialog.getValues()
            
            if not label:
                logger.warning("Label is required")
                return
            
            # Ensure group_id is set
            if group_id is None:
                group_id = next_group_id
            # Project 3D box to 2D for each camera
            self._project_3d_box_to_cameras(x, y, z, w, h, d, label, group_id)
            
            # Add to BEV canvas
            if self.bev_canvas:
                self.bev_canvas.addBox3D(x, y, z, w, h, d, label, group_id)
            
            # Update ground point list
            self._update_ground_point_list()
    
    def _on_bev_box_moved(self, box_idx: int, x: float, y: float, z: float):
        """Handle box movement from BEV canvas (via arrow keys)"""
        if not self.bev_canvas:
            return
        
        # Get box data AFTER it's been updated in BEV
        box_data = self.bev_canvas.getBox3D(box_idx)
        if box_data is None:
            return
        
        center, size, rotation, label, group_id = box_data
        
        # Verify the position matches (should be updated already in keyPressEvent)
        # Use the provided x, y, z which should be the new position
        new_center = np.array([x, y, z])
        
        # Find all shapes with matching label and group_id across all cameras
        # and update them based on new 3D position
        if not self.multi_camera_canvas or not self.multi_camera_data:
            return
        
        # Remove old projections (only BEV-projected boxes)
        for idx, cam_data in enumerate(self.multi_camera_data):
            cell = self.multi_camera_canvas.camera_cells[idx]
            shapes_to_remove = []
            for shape in cell.shapes:
                # Only remove boxes that are from BEV and match label/group_id
                if (shape.label == label and shape.group_id == group_id and 
                    shape.other_data.get("from_bev", False)):
                    shapes_to_remove.append(shape)
            
            for shape in shapes_to_remove:
                cell.shapes.remove(shape)
                self.remLabels([shape])
            self.multi_camera_canvas.camera_cells[idx] = cell
        
        # Project new position
        self._project_3d_box_to_cameras(x, y, z, size[0], size[1], size[2], label, group_id)
    
    def _on_bev_box_size_changed(self, box_idx: int, w: float, h: float, d: float):
        """Handle box size change request from BEV canvas (opens edit dialog)"""
        if not self.bev_canvas:
            return
        
        # Get box data
        box_data = self.bev_canvas.getBox3D(box_idx)
        if box_data is None:
            return
        
        center, size, rotation, label, group_id = box_data
        
        # Open edit dialog
        dialog = Box3DDialog(
            self,
            x=center[0], y=center[1], z=center[2],
            w=size[0], h=size[1], d=size[2],
            label=label, group_id=group_id
        )
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            x, y, z, new_w, new_h, new_d, new_label, new_group_id = dialog.getValues()
            
            # Update box in BEV
            self.bev_canvas.updateBox3D(box_idx, x, y, z, new_w, new_h, new_d)
            
            # Update label and group_id if changed
            if new_label != label or new_group_id != group_id:
                # Need to update the box data structure in BEV canvas
                center, size, rotation, _, _ = box_data
                self.bev_canvas.boxes_3d[box_idx] = (np.array([x, y, z]), np.array([new_w, new_h, new_d]), rotation, new_label, new_group_id)
            
            # Remove old projections and re-project with new size
            if not self.multi_camera_canvas or not self.multi_camera_data:
                return
            
            # Remove old projections (only BEV-projected boxes)
            for idx, cam_data in enumerate(self.multi_camera_data):
                cell = self.multi_camera_canvas.camera_cells[idx]
                shapes_to_remove = []
                for shape in cell.shapes:
                    # Only remove boxes that are from BEV and match label/group_id
                    if (shape.label == label and shape.group_id == group_id and 
                        shape.other_data.get("from_bev", False)):
                        shapes_to_remove.append(shape)
                
                for shape in shapes_to_remove:
                    cell.shapes.remove(shape)
                    self.remLabels([shape])
                self.multi_camera_canvas.camera_cells[idx] = cell
            
            # Project with new size and position
            self._project_3d_box_to_cameras(x, y, z, new_w, new_h, new_d, new_label, new_group_id)
    
    def _on_bev_box_deleted(self, box_idx: int, label: str, group_id: Optional[int]):
        """Handle deletion of a box in BEV: remove all its 2D projections"""
        if not self.multi_camera_canvas or not self.multi_camera_data:
            return
        
        # Remove all BEV-projected shapes with matching label/group_id
        for idx, cam_data in enumerate(self.multi_camera_data):
            cell = self.multi_camera_canvas.camera_cells[idx]
            shapes_to_remove = []
            for shape in cell.shapes:
                if (
                    shape.label == label
                    and shape.group_id == group_id
                    and shape.other_data.get("from_bev", False)
                ):
                    shapes_to_remove.append(shape)
            
            for shape in shapes_to_remove:
                cell.shapes.remove(shape)
                self.remLabels([shape])
            self.multi_camera_canvas.camera_cells[idx] = cell
        
        self.multi_camera_canvas.update()
        self._update_ground_point_list()
        self.setDirty()

    def _on_bev_point_placed(self, x: float, y: float, group_id: Optional[int]):
        """Handle point placement from BEV canvas - show dialog for configuration"""
        # Get next unique group_id
        next_group_id = self._get_next_group_id()
        
        # Convert mem coordinates to grid coordinates for display
        grid_x, grid_y = self._mem_to_worldgrid(x, y)
        
        # Show dialog for configuration
        from labelme.widgets.box3d_dialog import Box3DDialog
        dialog = Box3DDialog(
            self,
            x=grid_x, y=grid_y, z=0.0,
            w=DEFAULT_BOX_SIZE, h=DEFAULT_BOX_SIZE, d=DEFAULT_BOX_SIZE,  # Default size
            label="object",
            group_id=next_group_id
        )
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            new_x, new_y, new_z, new_w, new_h, new_d, new_label, new_group_id = dialog.getValues()
            
            # Convert back to mem coordinates
            mem_x, mem_y = self._worldgrid_to_mem(new_x, new_y)
            self._project_3d_box_to_cameras(mem_x, mem_y, new_z, new_w, new_h, new_d, new_label, new_group_id)
            
            # Add the point to BEV canvas
            if self.bev_canvas:
                self.bev_canvas.addPoint(mem_x, mem_y, new_label, new_group_id)
            
            # Update ground point list
            self._update_ground_point_list()
            self.setDirty()
            
            logger.info(f"Point placed at grid({new_x}, {new_y}) with group_id={new_group_id}")

    def _on_bev_point_selected(self, group_id: Optional[int]):
        """Handle point selection - highlight related bounding boxes in camera views"""
        if group_id is not None:
            self._highlight_shapes_by_group_id(group_id, highlight=True)
        else:
            # Clear all highlights
            self._clear_all_shape_highlights()

    def _on_bev_point_hovered(self, group_id: Optional[int]):
        """Handle point hover - highlight related bounding boxes in camera views"""
        # First clear all highlights
        self._clear_all_shape_highlights()
        
        # Then highlight shapes with matching group_id
        if group_id is not None:
            self._highlight_shapes_by_group_id(group_id, highlight=True)

    def _clear_all_shape_highlights(self):
        """Clear all shape highlights in camera views"""
        if not self.multi_camera_canvas:
            return
        
        for cell in self.multi_camera_canvas.camera_cells:
            for shape in cell.shapes:
                if hasattr(shape, '_bev_highlighted'):
                    shape._bev_highlighted = False
        
        self.multi_camera_canvas.update()

    def _on_bev_point_double_clicked(self, point_idx: int, group_id: Optional[int]):
        """Handle double-click on point - open edit dialog"""
        if point_idx < 0 or point_idx >= len(self.bev_canvas.points):
            return
        
        x, y, label, gid = self.bev_canvas.points[point_idx]
        
        # Convert back to grid coordinates for display
        grid_x, grid_y = self._mem_to_worldgrid(x, y)
        
        from labelme.widgets.box3d_dialog import Box3DDialog
        dialog = Box3DDialog(
            self,
            x=grid_x, y=grid_y, z=0.0,
            w=DEFAULT_BOX_SIZE, h=DEFAULT_BOX_SIZE, d=DEFAULT_BOX_SIZE,  # Default size
            label=label,
            group_id=gid
        )
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            new_x, new_y, new_z, new_w, new_h, new_d, new_label, new_group_id = dialog.getValues()
            
            # Convert back to mem coordinates
            mem_x, mem_y = self._worldgrid_to_mem(new_x, new_y)
            
            # Update point
            self.bev_canvas.updatePoint(point_idx, mem_x, mem_y, new_label, new_group_id)
            
            # Update ground point list
            self._update_ground_point_list()
            self.setDirty()

    def _on_bev_point_deleted(self, group_id: Optional[int]):
        """Handle point deletion from BEV - remove all related 2D bounding boxes"""
        if group_id is None:
            return
        
        self._delete_ground_point(group_id)

    def _on_bev_point_moved(self, group_id: Optional[int], new_x: float, new_y: float):
        """Handle point movement from BEV - update all related 2D bounding boxes"""
        if group_id is None or not self.multi_camera_canvas:
            return
        
        # Convert mem coordinates to world grid for logging
        grid_x, grid_y = self._mem_to_worldgrid(new_x, new_y)
        logger.info(f"Point {group_id} moved to grid({grid_x}, {grid_y})")
        
        # Project the new point position to each camera and update bounding boxes
        mem_point = np.array([[new_x, new_y]], dtype=np.float32)
        
        for cam_idx, cam_data in enumerate(self.multi_camera_data):
            camera_id = cam_data.get("camera_id", f"Camera{cam_idx}")
            calibration = self.camera_calibrations.get(camera_id)
            
            if not calibration:
                continue
            
            # Project the new point to this camera
            try:
                projected = calibration.project_mem_to_2d(
                    mem_point,
                    bev_x=self._bev_x,
                    bev_y=self._bev_y,
                    bev_bounds=self._bev_bounds,
                    apply_distortion=True
                )
                
                if np.isnan(projected).any():
                    continue
                
                new_center_x, new_center_y = projected[0]
                
                # Find and update the shape with matching group_id in this camera
                cell = self.multi_camera_canvas.camera_cells[cam_idx]
                for shape in cell.shapes:
                    if shape.group_id == group_id:
                        # Get current bounding box dimensions
                        if len(shape.points) >= 2:
                            old_xmin = min(p.x() for p in shape.points)
                            old_xmax = max(p.x() for p in shape.points)
                            old_ymin = min(p.y() for p in shape.points)
                            old_ymax = max(p.y() for p in shape.points)
                            
                            # Calculate current dimensions
                            width = old_xmax - old_xmin
                            height = old_ymax - old_ymin
                            
                            # Calculate new bounding box centered on projected point
                            # Keep bottom of box at projected point (foot position)
                            new_xmin = new_center_x - width / 2
                            new_xmax = new_center_x + width / 2
                            new_ymax = new_center_y  # Foot at projected point
                            new_ymin = new_center_y - height
                            
                            # Update shape points (for rectangle: top-left, bottom-right)
                            if len(shape.points) == 2:
                                shape.points[0] = QtCore.QPointF(new_xmin, new_ymin)
                                shape.points[1] = QtCore.QPointF(new_xmax, new_ymax)
                            elif len(shape.points) == 4:
                                # Rectangle with 4 corners
                                shape.points[0] = QtCore.QPointF(new_xmin, new_ymin)
                                shape.points[1] = QtCore.QPointF(new_xmax, new_ymin)
                                shape.points[2] = QtCore.QPointF(new_xmax, new_ymax)
                                shape.points[3] = QtCore.QPointF(new_xmin, new_ymax)
                        break
                
            except Exception as e:
                logger.warning(f"Failed to project point to camera {camera_id}: {e}")
        
        # Update displays
        self.multi_camera_canvas.update()
        self._update_ground_point_list()
        self.setDirty()
    
    def _project_3d_box_to_cameras(self, x: float, y: float, z: float, 
                                   w: float, h: float, d: float,
                                   label: str, group_id: Optional[int]):
        """Project 3D box to 2D for all cameras"""
        center_3d = np.array([x, y, z])
        size_3d = np.array([w, h, d])
        
        if not self.multi_camera_canvas or not self.multi_camera_data:
            return
        
        # Project to each camera
        for idx, cam_data in enumerate(self.multi_camera_data):
            camera_id = cam_data["camera_id"]
            calibration = self.camera_calibrations.get(camera_id)
            
            if not calibration:
                logger.warning(f"No calibration for camera {camera_id}, skipping projection")
                continue
            
            # Get 2D corners of the 3D box (8 corners: 4 bottom + 4 top)
            corners_2d = calibration.project_3d_box_to_2d(center_3d, size_3d, rotation=0.0)
            
            if corners_2d is None:
                logger.debug(f"Box not visible in camera {camera_id}")
                continue
            
            # Filter out NaN values (points behind camera)
            valid_corners = corners_2d[~np.isnan(corners_2d).any(axis=1)]
            if len(valid_corners) == 0:
                logger.debug(f"No valid corners for camera {camera_id}")
                continue
            
            # Get bounding box from valid corners
            x_min = np.min(valid_corners[:, 0])
            y_min = np.min(valid_corners[:, 1])
            x_max = np.max(valid_corners[:, 0])
            y_max = np.max(valid_corners[:, 1])
            
            # Check if bbox is within image bounds.
            # If it goes completely or partially outside, we treat it as not visible
            # so that the box disappears instead of sticking to the border.
            cell = self.multi_camera_canvas.camera_cells[idx]
            img_w = cell.image.width()
            img_h = cell.image.height()
            if x_min < 0 or y_min < 0 or x_max > img_w or y_max > img_h:
                logger.debug(f"Box outside image bounds for camera {camera_id}, removing projection")
                # Do not create/update any shape for this camera (box disappears here)
                continue
            
            # If the user manually edited this box on this camera (from_bev == False),
            # stop tracking it from BEV and do not create a new box for this camera.
            skip_camera = False
            for existing in cell.shapes:
                if (
                    existing.label == label
                    and existing.group_id == group_id
                    and not existing.other_data.get("from_bev", False)
                ):
                    skip_camera = True
                    break
            if skip_camera:
                logger.debug(
                    f"Skip projection for camera {camera_id} because manual-edited box exists"
                )
                continue

            # Check if box already exists for this camera (only BEV-projected boxes)
            # Only update if shape exists and is from BEV
            existing_shape = None
            for existing in cell.shapes:
                # Only update if it's a BEV-projected box with matching label/group_id
                if (existing.label == label and existing.group_id == group_id and 
                    existing.other_data.get("from_bev", False)):
                    existing_shape = existing
                    break
            
            if existing_shape is None:
                # Create new rectangle shape from bounding box
                # Use rectangle shape type for simplicity, but ensure correct orientation
                shape = Shape(
                    label=label,
                    shape_type="rectangle",
                    group_id=group_id,
                )
                # Rectangle shape: two points define the rectangle
                # Ensure points are ordered correctly (top-left to bottom-right)
                shape.addPoint(QtCore.QPointF(x_min, y_min))
                shape.addPoint(QtCore.QPointF(x_max, y_max))
                shape.close()
                
                # Mark as BEV-projected box
                shape.other_data["from_bev"] = True
                
                # Add to cell
                cell.shapes.append(shape)
                self.multi_camera_canvas.camera_cells[idx] = cell
                
                # Add to label list
                self.addLabel(shape)
            else:
                # Update existing BEV-projected shape
                # Ensure points are ordered correctly
                existing_shape.points[0] = QtCore.QPointF(x_min, y_min)
                existing_shape.points[1] = QtCore.QPointF(x_max, y_max)
                existing_shape.other_data["from_bev"] = True
                self.multi_camera_canvas.camera_cells[idx] = cell
                self.multi_camera_canvas.update()
        
        self.multi_camera_canvas.update()
        self.setDirty()

    def _update_status_stats(self, mouse_pos: QtCore.QPointF) -> None:
        stats: list[str] = []
        if hasattr(self.canvas, 'mode'):
            stats.append(f"mode={self.canvas.mode.name}")
        stats.append(f"x={mouse_pos.x():6.1f}, y={mouse_pos.y():6.1f}")
        self.status_right.setText(" | ".join(stats))


def _scan_image_files(root_dir: str) -> list[str]:
    extensions: list[str] = [
        f".{fmt.data().decode().lower()}"
        for fmt in QtGui.QImageReader.supportedImageFormats()
    ]

    images: list[str] = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(tuple(extensions)):
                relativePath = os.path.normpath(osp.join(root, file))
                images.append(relativePath)

    logger.debug("found {:d} images in {!r}", len(images), root_dir)
    return natsort.os_sorted(images)


def _detect_multi_camera_structure(root_dir: str) -> bool:
    """Detect if directory has multi-camera structure (subfolders with frames/calibrations)"""
    if not osp.isdir(root_dir):
        return False
    
    # Check if root has subdirectories
    subdirs = [d for d in os.listdir(root_dir) if osp.isdir(osp.join(root_dir, d))]
    if len(subdirs) < 2:
        return False
    
    # Check if subdirectories have 'frames' folder
    has_frames = False
    for subdir in subdirs:
        subdir_path = osp.join(root_dir, subdir)
        frames_path = osp.join(subdir_path, "frames")
        if osp.isdir(frames_path):
            has_frames = True
            break
    
    return has_frames


def _detect_aicv_structure(root_dir: str) -> bool:
    """Detect if directory has AICV folder structure (Image_subsets + calibrations)"""
    if not osp.isdir(root_dir):
        return False
    
    # Check for Image_subsets folder
    image_subsets_dir = osp.join(root_dir, "Image_subsets")
    if not osp.isdir(image_subsets_dir):
        return False
    
    # Check for calibrations folder
    calibrations_dir = osp.join(root_dir, "calibrations")
    if not osp.isdir(calibrations_dir):
        return False
    
    # Check if Image_subsets has numbered subfolders with images
    subdirs = [d for d in os.listdir(image_subsets_dir) if osp.isdir(osp.join(image_subsets_dir, d))]
    if len(subdirs) < 1:
        return False
    
    return True


def _get_aicv_camera_mapping(calibrations_dir: str) -> dict:
    """
    Get mapping from folder number to camera name for AICV structure.
    Returns: dict mapping '1' -> 'Camera1', '2' -> 'Camera2', etc.
    """
    mapping = {}
    if not osp.isdir(calibrations_dir):
        return mapping
    
    for camera_name in sorted(os.listdir(calibrations_dir)):
        camera_path = osp.join(calibrations_dir, camera_name)
        if not osp.isdir(camera_path):
            continue
        # Extract number from camera name (e.g., Camera1 -> 1)
        import re
        match = re.search(r'(\d+)', camera_name)
        if match:
            folder_num = match.group(1)
            mapping[folder_num] = camera_name
    
    return mapping


def _scan_multi_camera_data(root_dirs, frame_index: int = 0) -> list[dict]:
    """
    Scan multi-camera directory structure and return camera data for a specific frame.

    Parameters
    ----------
    root_dirs:
        - Legacy case: a single root folder where each subfolder is one camera.
        - New case: a list of specific camera folders (each element is a path to one camera).
        - AICV case: a folder with `Image_subsets` + `calibrations` structure.
    """
    camera_data = []
    
    extensions = [
        f".{fmt.data().decode().lower()}"
        for fmt in QtGui.QImageReader.supportedImageFormats()
    ]
    
    # Check for AICV structure first
    # Handle both string and single-element list cases
    root_dir_to_check = None
    if isinstance(root_dirs, str):
        root_dir_to_check = root_dirs
    elif isinstance(root_dirs, (list, tuple)) and len(root_dirs) == 1:
        # If it's a list with single element, check if that element is AICV structure
        root_dir_to_check = root_dirs[0]
    
    if root_dir_to_check and _detect_aicv_structure(root_dir_to_check):
        return _scan_aicv_camera_data(root_dir_to_check, frame_index)
    
    # Resolve camera directories list
    if isinstance(root_dirs, (list, tuple)):
        camera_dirs = list(root_dirs)
    else:
        root_dir = root_dirs
        subdirs = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if osp.isdir(osp.join(root_dir, d))
            ]
        )
        camera_dirs = [osp.join(root_dir, d) for d in subdirs]
    
    for cam_dir in camera_dirs:
        camera_id = osp.basename(cam_dir.rstrip(os.sep))
        frames_dir = osp.join(cam_dir, "frames")
        calibrations_dir = osp.join(cam_dir, "calibrations")
        
        if not osp.isdir(frames_dir):
            continue
        
        # Find all images in frames directory
        frame_files = []
        for file in os.listdir(frames_dir):
            if file.lower().endswith(tuple(extensions)):
                frame_files.append(osp.join(frames_dir, file))
        
        frame_files = natsort.os_sorted(frame_files)
        
        if frame_index < len(frame_files):
            image_path = frame_files[frame_index]
            # Calibration path is the calibrations directory itself
            calibration_path = calibrations_dir if osp.isdir(calibrations_dir) else None
            
            camera_data.append({
                "camera_id": camera_id,
                "image_path": image_path,
                "calibration_path": calibration_path,
            })
    
    return camera_data


def _scan_aicv_camera_data(root_dir: str, frame_index: int = 0) -> list[dict]:
    """
    Scan AICV folder structure and return camera data for a specific frame.
    
    Structure:
        root_dir/
            Image_subsets/
                1/  (images for camera 1)
                2/  (images for camera 2)
                ...
            calibrations/
                Camera1/calibrations/
                Camera2/calibrations/
                ...
    """
    camera_data = []
    
    extensions = [
        f".{fmt.data().decode().lower()}"
        for fmt in QtGui.QImageReader.supportedImageFormats()
    ]
    
    image_subsets_dir = osp.join(root_dir, "Image_subsets")
    calibrations_dir = osp.join(root_dir, "calibrations")
    
    # Get camera mapping (folder number -> camera name)
    camera_mapping = _get_aicv_camera_mapping(calibrations_dir)
    
    # Scan numbered folders in Image_subsets
    subdirs = sorted([d for d in os.listdir(image_subsets_dir) if osp.isdir(osp.join(image_subsets_dir, d))])
    
    for folder_num in subdirs:
        folder_path = osp.join(image_subsets_dir, folder_num)
        
        # Find all images in this folder (directly, not in frames subfolder)
        frame_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(tuple(extensions)):
                frame_files.append(osp.join(folder_path, file))
        
        frame_files = natsort.os_sorted(frame_files)
        
        if frame_index < len(frame_files):
            image_path = frame_files[frame_index]
            
            # Get camera name from mapping or create one
            camera_name = camera_mapping.get(folder_num, f"Camera{folder_num}")
            camera_id = f"Camera{folder_num}"
            
            # Calibration path points to centralized calibrations folder
            camera_calib_path = osp.join(calibrations_dir, camera_name, "calibrations")
            calibration_path = camera_calib_path if osp.isdir(camera_calib_path) else None
            
            camera_data.append({
                "camera_id": camera_id,
                "image_path": image_path,
                "calibration_path": calibration_path,
            })
    
    return camera_data
