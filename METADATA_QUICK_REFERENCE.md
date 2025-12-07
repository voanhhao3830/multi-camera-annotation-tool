# Quick Reference Guide: BEV Metadata Feature

## For Users

### Saving Annotations with 3D Boxes

1. **Place 3D boxes** on the BEV canvas (Bird's Eye View)
2. **Save** with `Ctrl+S` or File > Save
3. **Result**: 
   - Annotations saved to `annotations_positions/XXXXX.json` with box dimensions
   - Metadata saved to `constants.json` at root level

### Loading Annotations with 3D Boxes

1. **Open** AICV directory (File > Open Dir)
2. **Select** a frame from the list
3. **Result**:
   - 3D boxes automatically restored with correct dimensions
   - BEV grid configured from saved metadata
   - Camera views show 2D bounding boxes

### File Structure

```
/your/aicv/dataset/
├── Image_subsets/
│   ├── 1/
│   ├── 2/
│   └── ...
├── calibrations/
│   ├── Camera1/
│   └── ...
├── annotations_positions/
│   ├── 00001.json  ← Contains box_3d data (x,y,z,w,h,d)
│   ├── 00002.json
│   └── ...
└── constants.json  ← NEW: Metadata file
```

## For Developers

### Accessing Metadata Programmatically

```python
from labelme.utils.metadata import load_metadata, get_box_size_from_metadata

# Load metadata
metadata = load_metadata("/path/to/aicv/root")

# Get default box size
width, height, depth = get_box_size_from_metadata(metadata)

# Get BEV grid dimensions
bev_x = metadata["bev_grid"]["x"]
bev_y = metadata["bev_grid"]["y"]
```

### Annotation Format with 3D Box

```python
{
    "personID": 1,
    "ground_points": [100, 200],  # 2D ground position (world grid)
    "box_3d": {                    # 3D bounding box metadata
        "x": 100.0,                # Center X (world grid)
        "y": 200.0,                # Center Y (world grid)
        "z": 0.0,                  # Center Z (height)
        "w": 50.0,                 # Width
        "h": 50.0,                 # Height
        "d": 50.0                  # Depth
    },
    "views": [                     # 2D boxes per camera
        {
            "viewNum": 1,
            "xmin": 100,
            "ymin": 150,
            "xmax": 200,
            "ymax": 300
        }
    ]
}
```

### Constants.json Format

```json
{
    "version": "1.0",
    "bev_grid": {
        "x": 600,      // Grid width in pixels
        "y": 400,      // Grid height in pixels
        "z": 2         // Number of height levels
    },
    "bev_bounds": [0, 600, 0, 400, 0, 200],  // [xmin, xmax, ymin, ymax, zmin, zmax]
    "default_box_size": {
        "width": 50.0,   // Default box width
        "height": 50.0,  // Default box height
        "depth": 50.0    // Default box depth
    },
    "world_grid_transform": {
        "theta": -3,     // Rotation angle (degrees)
        "scale": 0.5,    // Scale factor
        "tx": -74,       // Translation X
        "ty": -54        // Translation Y
    }
}
```

## Backwards Compatibility

- **Missing constants.json**: Falls back to defaults from `constants.py`
- **Missing box_3d**: Falls back to simple point annotation
- **Old annotation files**: Still work, just won't have 3D box data

## Troubleshooting

### Issue: 3D boxes not loading

**Check:**
1. Is `box_3d` present in annotation JSON?
2. Are coordinates valid (x >= 0, y >= 0)?
3. Check logs for error messages

### Issue: constants.json not created

**Check:**
1. Do you have write permissions to the directory?
2. Is BEV canvas initialized?
3. Check logs for save errors

### Issue: Wrong box size after loading

**Solution:**
- Edit `constants.json` and reload the directory
- Or delete `constants.json` to use defaults

## Logs

All operations are logged. Check the console for:
- `"Saved metadata to..."` - Metadata saved successfully
- `"Loaded metadata from..."` - Metadata loaded successfully
- `"Restored 3D box for person..."` - Box restored from saved data
- Warnings/errors for any issues

## API Reference

### Functions

#### `save_metadata(root_dir, bev_x, bev_y, bev_z, box_width, box_height, box_depth)`
Save metadata to constants.json

**Parameters:**
- `root_dir` (str): Root directory path
- `bev_x` (int): BEV grid X dimension
- `bev_y` (int): BEV grid Y dimension
- `bev_z` (int): BEV grid Z dimension
- `box_width` (float): Default box width
- `box_height` (float): Default box height
- `box_depth` (float): Default box depth

**Returns:** bool - Success status

#### `load_metadata(root_dir)`
Load metadata from constants.json with fallback to defaults

**Parameters:**
- `root_dir` (str): Root directory path

**Returns:** dict - Metadata dictionary

#### `get_bev_grid_from_metadata(metadata)`
Extract BEV grid dimensions

**Returns:** tuple[int, int, int] - (x, y, z)

#### `get_box_size_from_metadata(metadata)`
Extract default box size

**Returns:** tuple[float, float, float] - (width, height, depth)
