# BEV Metadata Feature Implementation

## Overview

This document describes the implementation of saving and loading 3D bounding box metadata (x, y, z, w, h, d) along with point annotations in the labelme multi-camera application.

## Changes Made

### 1. Updated Constants (`labelme/utils/constants.py`)

Added explicit constants for default 3D box dimensions:

```python
DEFAULT_BOX_WIDTH = 50.0   # Width of bounding box
DEFAULT_BOX_HEIGHT = 50.0  # Height of bounding box  
DEFAULT_BOX_DEPTH = 50.0   # Depth of bounding box
DEFAULT_BOX_SIZE = 50.0 * 2  # Legacy: kept for backward compatibility
```

### 2. Created Metadata Module (`labelme/utils/metadata.py`)

A new module to handle saving and loading of metadata (constants.json):

**Key Functions:**
- `save_metadata()`: Save BEV grid configuration and default box sizes
- `load_metadata()`: Load metadata with fallback to defaults from constants.py
- `get_bev_grid_from_metadata()`: Extract BEV grid dimensions
- `get_box_size_from_metadata()`: Extract default box size

**Metadata Structure (constants.json):**
```json
{
  "version": "1.0",
  "bev_grid": {
    "x": 600,
    "y": 400,
    "z": 2
  },
  "bev_bounds": [0, 600, 0, 400, 0, 200],
  "default_box_size": {
    "width": 50.0,
    "height": 50.0,
    "depth": 50.0
  },
  "world_grid_transform": {
    "theta": -3,
    "scale": 0.5,
    "tx": -74,
    "ty": -54
  }
}
```

### 3. Enhanced Annotation Saving (`app.py` - `_save_global_annotations()`)

**Extended annotation format to include 3D box metadata:**

```json
{
  "personID": 1,
  "ground_points": [100, 200],
  "box_3d": {
    "x": 100.0,
    "y": 200.0,
    "z": 0.0,
    "w": 50.0,
    "h": 50.0,
    "d": 50.0
  },
  "views": [...]
}
```

**Changes:**
- Now saves 3D bounding box information from `bev_canvas.boxes_3d`
- Stores center position (x, y, z) in world grid coordinates
- Stores box dimensions (w, h, d)
- Automatically saves `constants.json` at root directory level when saving annotations

### 4. Enhanced Annotation Loading (`app.py` - `_load_global_annotations()`)

**Changes:**
- Loads `constants.json` at the beginning to get default box sizes
- Checks for `box_3d` metadata in each person annotation
- If 3D box data exists:
  - Reconstructs the 3D box on BEV canvas
  - Converts world grid coordinates back to memory coordinates
  - Creates box with saved dimensions
- Falls back to simple point annotation if no box data is present
- Logs all loaded box information for debugging

### 5. Metadata Loading on Directory Open (`app.py` - directory loading)

**Changes:**
- When opening an AICV directory structure, automatically loads `constants.json`
- Applies loaded BEV grid dimensions to the BEV canvas
- Applies loaded default box size to the BEV canvas
- Gracefully falls back to defaults if metadata file is missing

## File Locations

**Metadata file location:**
```
/path/to/aicv/
├── Image_subsets/
├── calibrations/
├── annotations_positions/
│   ├── 00001.json  (with box_3d data)
│   ├── 00002.json
│   └── ...
└── constants.json  ← NEW: Metadata file at root level
```

## Usage Flow

### Saving Annotations:
1. User places 3D boxes or points on BEV canvas
2. User saves annotations (Ctrl+S)
3. App saves:
   - `annotations_positions/XXXXX.json` with box_3d metadata for each person
   - `constants.json` with current BEV grid and box size settings

### Loading Annotations:
1. User opens AICV directory
2. App loads `constants.json` and applies settings to BEV canvas
3. User selects a frame
4. App loads `annotations_positions/XXXXX.json`
5. For each person with `box_3d` data:
   - Reconstructs 3D box on BEV canvas
   - Uses saved dimensions (w, h, d)
   - Positions at saved location (x, y, z)

## Benefits

1. **Persistence**: 3D box dimensions and positions are fully preserved
2. **Flexibility**: Each dataset can have its own BEV grid configuration
3. **Backward Compatibility**: Falls back to defaults if metadata is missing
4. **Logging**: Comprehensive logging for debugging and verification
5. **Error Handling**: Try-catch blocks prevent crashes from malformed data

## Testing

Both files compile successfully:
- ✅ `metadata.py` compiles without errors
- ✅ `app.py` compiles without errors

The metadata module includes comprehensive tests in its `__main__` block that can be run when dependencies are available.

## Type Hints and Coding Conventions

All new code follows the user's coding conventions:
- ✅ Type hints added to all function signatures
- ✅ Docstrings for all functions
- ✅ Try-catch blocks with logging and traceback
- ✅ Clear, descriptive variable names
- ✅ Follows PEP 8 style guidelines

## Future Enhancements

Possible future improvements:
1. Add UI controls to edit default box size
2. Support multiple box types with different default sizes
3. Version migration for constants.json format changes
4. Validation of loaded metadata values
5. Export/import of metadata between datasets
