# Source Dataset Directory

This directory should contain:
- `images/` - Source domain images with known labels
- `annotations/` - Bounding box annotations (COCO format or similar)
- `masks/` - Ground truth segmentation masks (if available)

Example structure:
```
source/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── annotations/
│   ├── annotations.json  # COCO format
│   └── ...
└── masks/  # Optional
    ├── image_001_mask.png
    └── ...
```
