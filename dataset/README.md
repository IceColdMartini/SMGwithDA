# Dataset Directory Structure

This directory contains the datasets for the SAM-based segmentation pipeline.

## Directory Structure:

```
dataset/
├── source/          # Source domain images (with annotations)
├── target/          # Target domain images (forest environment, unlabeled)
└── annotations/     # Bounding box annotations for source images
```

## Data Format Expected:

### Source Images (`source/`)
- Format: `.jpg`, `.png`, `.jpeg`
- Images with objects that have bounding box annotations
- Used for initial SAM mask generation and domain adaptation training

### Target Images (`target/`)
- Format: `.jpg`, `.png`, `.jpeg` 
- Unlabeled images from the target domain (cluttered forest environment)
- Used for domain adaptation and self-training

### Annotations (`annotations/`)
- Format: `.json`, `.xml`, or `.txt`
- Bounding box coordinates for objects in source images
- Supported formats:
  - COCO JSON format
  - Pascal VOC XML format
  - YOLO format (txt with normalized coordinates)

## Usage:

1. **Place your actual dataset** in the respective directories
2. **Update the data loading paths** in the preprocessing module
3. The pipeline will automatically detect and process the data

## Example Annotation Format (COCO JSON):

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "bbox": [100, 100, 200, 150],
      "category_id": 1
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "object_class"
    }
  ]
}
```

## Notes:

- **Dummy structure**: This is currently a placeholder structure
- **Replace with actual data**: You'll need to replace this with your actual forest environment dataset
- **Bounding boxes required**: Source images must have bounding box annotations for SAM prompting
- **No labels needed for target**: Target domain images don't need any annotations (unsupervised adaptation)
