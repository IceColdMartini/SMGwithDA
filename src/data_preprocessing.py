"""
Data Ingestion and Preprocessing Module

This module handles:
1. Loading source and target domain datasets
2. Image preprocessing (resize, normalize, augment)
3. Annotation parsing (COCO format, custom JSON)
4. Data validation and quality checks
5. Dataset splitting and organization
6. Support for the complex forestry dataset structure
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image, ImageOps
import logging
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import dataset configuration
from dataset_config import DatasetConfig

logger = logging.getLogger(__name__)

@dataclass
class ImageAnnotation:
    """Data class for image annotations."""
    image_id: int
    file_name: str
    width: int
    height: int
    boxes: List[List[float]]  # List of [x, y, w, h] bounding boxes
    category_ids: List[int]
    annotation_ids: List[int]
    domain: str  # 'source' or 'target'

class DataPreprocessor:
    """Handles data ingestion and preprocessing for the segmentation pipeline."""
    
    def __init__(self, 
                 project_root: str = None,
                 target_size: int = 512,
                 preserve_aspect_ratio: bool = True,
                 apply_augmentations: bool = True,
                 source_domain: str = None,
                 target_domain: str = None,
                 validation_domain: str = None):
        """
        Initialize data preprocessor.
        
        Args:
            project_root: Path to project root directory
            target_size: Target image size for processing
            preserve_aspect_ratio: Whether to preserve aspect ratio during resizing
            apply_augmentations: Whether to apply data augmentations
            source_domain: Source domain dataset key (e.g., 'main_dataset.part_1')
            target_domain: Target domain dataset key (e.g., 'testing.simulated')
            validation_domain: Validation domain dataset key
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.target_size = target_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.apply_augmentations = apply_augmentations
        
        # Initialize dataset configuration
        self.dataset_config = DatasetConfig(self.project_root)
        
        # Setup domain configuration
        if source_domain or target_domain:
            self.domain_config = self.dataset_config.setup_domain_adaptation_config(
                source=source_domain, 
                target=target_domain, 
                validation=validation_domain
            )
        else:
            # Use recommended configuration
            self.domain_config = self.dataset_config.setup_domain_adaptation_config()
        
        # Legacy dataset paths (for backward compatibility)
        self.dataset_dir = self.project_root / "dataset"
        self.source_dir = self.dataset_dir / "source"
        self.target_dir = self.dataset_dir / "target"
        
        # ImageNet statistics for normalization
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        # Data storage
        self.source_annotations = []
        self.target_annotations = []
        self.validation_annotations = []
        self.category_info = {}
        
        # Setup transforms
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup image transforms for preprocessing and augmentation."""
        
        # Base preprocessing transforms
        self.base_transform = A.Compose([
            A.Resize(self.target_size, self.target_size, p=1.0),
            A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std, p=1.0),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        
        # Augmentation transforms for source domain
        if self.apply_augmentations:
            self.augment_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.Resize(self.target_size, self.target_size, p=1.0),
                A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std, p=1.0),
                ToTensorV2(p=1.0)
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        else:
            self.augment_transform = self.base_transform
        
        # Transform for target domain (no augmentation)
        self.target_transform = self.base_transform
        
        logger.info(f"Transforms setup complete. Target size: {self.target_size}x{self.target_size}")
    
    def load_coco_annotations(self, annotation_file: Path, domain: str) -> List[ImageAnnotation]:
        """
        Load annotations in COCO format.
        
        Args:
            annotation_file: Path to COCO annotation JSON file
            domain: 'source' or 'target'
            
        Returns:
            List of ImageAnnotation objects
        """
        if not annotation_file.exists():
            logger.warning(f"Annotation file not found: {annotation_file}")
            return []
        
        try:
            with open(annotation_file, 'r') as f:
                coco_data = json.load(f)
            
            # Build image id to info mapping
            image_info = {img['id']: img for img in coco_data.get('images', [])}
            
            # Build category info
            for cat in coco_data.get('categories', []):
                self.category_info[cat['id']] = cat['name']
            
            # Group annotations by image
            image_annotations = {}
            for ann in coco_data.get('annotations', []):
                image_id = ann['image_id']
                if image_id not in image_annotations:
                    image_annotations[image_id] = {
                        'boxes': [],
                        'category_ids': [],
                        'annotation_ids': []
                    }
                
                # Convert bbox from [x, y, w, h] to [x, y, w, h] (already in correct format)
                bbox = ann['bbox']
                image_annotations[image_id]['boxes'].append(bbox)
                image_annotations[image_id]['category_ids'].append(ann.get('category_id', 1))
                image_annotations[image_id]['annotation_ids'].append(ann['id'])
            
            # Create ImageAnnotation objects
            annotations = []
            for image_id, img_info in image_info.items():
                if image_id in image_annotations:
                    ann_data = image_annotations[image_id]
                    annotation = ImageAnnotation(
                        image_id=image_id,
                        file_name=img_info['file_name'],
                        width=img_info['width'],
                        height=img_info['height'],
                        boxes=ann_data['boxes'],
                        category_ids=ann_data['category_ids'],
                        annotation_ids=ann_data['annotation_ids'],
                        domain=domain
                    )
                    annotations.append(annotation)
            
            logger.info(f"Loaded {len(annotations)} images with annotations from {annotation_file}")
            return annotations
            
        except Exception as e:
            logger.error(f"Error loading COCO annotations from {annotation_file}: {e}")
            return []
    
    def load_custom_annotations(self, annotation_file: Path, domain: str) -> List[ImageAnnotation]:
        """
        Load annotations in custom JSON format.
        
        Args:
            annotation_file: Path to custom annotation JSON file
            domain: 'source' or 'target'
            
        Returns:
            List of ImageAnnotation objects
        """
        if not annotation_file.exists():
            logger.warning(f"Annotation file not found: {annotation_file}")
            return []
        
        try:
            with open(annotation_file, 'r') as f:
                custom_data = json.load(f)
            
            annotations = []
            
            # Handle different custom formats
            if isinstance(custom_data, list):
                # Format: [{"image": "img.jpg", "boxes": [[x,y,w,h], ...], "categories": [1,2,...]}, ...]
                for item in custom_data:
                    # Get image dimensions (try to load image or use defaults)
                    img_path = self._get_image_path(item['image'], domain)
                    width, height = self._get_image_dimensions(img_path)
                    
                    annotation = ImageAnnotation(
                        image_id=len(annotations),
                        file_name=item['image'],
                        width=width,
                        height=height,
                        boxes=item.get('boxes', []),
                        category_ids=item.get('categories', [1] * len(item.get('boxes', []))),
                        annotation_ids=list(range(len(item.get('boxes', [])))),
                        domain=domain
                    )
                    annotations.append(annotation)
            
            elif isinstance(custom_data, dict):
                # Format: {"image1.jpg": {"boxes": [...], "categories": [...]}, ...}
                for filename, data in custom_data.items():
                    img_path = self._get_image_path(filename, domain)
                    width, height = self._get_image_dimensions(img_path)
                    
                    annotation = ImageAnnotation(
                        image_id=len(annotations),
                        file_name=filename,
                        width=width,
                        height=height,
                        boxes=data.get('boxes', []),
                        category_ids=data.get('categories', [1] * len(data.get('boxes', []))),
                        annotation_ids=list(range(len(data.get('boxes', [])))),
                        domain=domain
                    )
                    annotations.append(annotation)
            
            logger.info(f"Loaded {len(annotations)} custom annotations from {annotation_file}")
            return annotations
            
        except Exception as e:
            logger.error(f"Error loading custom annotations from {annotation_file}: {e}")
            return []
    
    def _get_image_path(self, filename: str, domain: str) -> Path:
        """Get full path to image file based on domain configuration."""
        # Handle legacy domain names
        if domain == 'source':
            if self.domain_config['source_domain']:
                images_dir = self.dataset_config.get_dataset_path(
                    self.domain_config['source_domain'], 'images'
                )
            else:
                images_dir = self.source_dir / "images"
        elif domain == 'target':
            if self.domain_config['target_domain']:
                images_dir = self.dataset_config.get_dataset_path(
                    self.domain_config['target_domain'], 'images'
                )
            else:
                images_dir = self.target_dir / "images"
        elif domain == 'validation':
            if self.domain_config['validation_domain']:
                images_dir = self.dataset_config.get_dataset_path(
                    self.domain_config['validation_domain'], 'images'
                )
            else:
                images_dir = self.source_dir / "images"  # fallback
        else:
            # Direct dataset key provided
            images_dir = self.dataset_config.get_dataset_path(domain, 'images')
        
        return images_dir / filename
    
    def _get_annotation_path(self, domain: str) -> Path:
        """Get annotation file path based on domain configuration."""
        if domain == 'source':
            if self.domain_config['source_domain']:
                return self.dataset_config.get_dataset_path(
                    self.domain_config['source_domain'], 'annotations'
                )
            else:
                return self.source_dir / "annotations" / "annotations.json"
        elif domain == 'target':
            if self.domain_config['target_domain']:
                return self.dataset_config.get_dataset_path(
                    self.domain_config['target_domain'], 'annotations'
                )
            else:
                return self.target_dir / "annotations" / "annotations.json"
        elif domain == 'validation':
            if self.domain_config['validation_domain']:
                return self.dataset_config.get_dataset_path(
                    self.domain_config['validation_domain'], 'annotations'
                )
            else:
                return self.source_dir / "annotations" / "annotations.json"  # fallback
        else:
            # Direct dataset key provided
            return self.dataset_config.get_dataset_path(domain, 'annotations')
    
    def _get_image_dimensions(self, image_path: Path) -> Tuple[int, int]:
        """Get image dimensions."""
        try:
            if image_path.exists():
                with Image.open(image_path) as img:
                    return img.size  # (width, height)
            else:
                logger.warning(f"Image not found: {image_path}, using default dimensions")
                return (1024, 768)  # Default dimensions
        except Exception as e:
            logger.warning(f"Error getting image dimensions for {image_path}: {e}")
            return (1024, 768)  # Default dimensions
    
    def load_datasets(self, annotation_format: str = 'coco', 
                     use_recommended_config: bool = True) -> Tuple[List[ImageAnnotation], List[ImageAnnotation], List[ImageAnnotation]]:
        """
        Load source, target, and validation datasets.
        
        Args:
            annotation_format: 'coco' or 'custom'
            use_recommended_config: Whether to use automatically recommended dataset configuration
            
        Returns:
            Tuple of (source_annotations, target_annotations, validation_annotations)
        """
        logger.info("Loading datasets with new structure...")
        
        # Log current domain configuration
        logger.info(f"Domain configuration:")
        logger.info(f"  Source: {self.domain_config['source_domain']}")
        logger.info(f"  Target: {self.domain_config['target_domain']}")
        logger.info(f"  Validation: {self.domain_config['validation_domain']}")
        
        # Load source dataset
        source_ann_file = self._get_annotation_path('source')
        if annotation_format == 'coco':
            self.source_annotations = self.load_coco_annotations(source_ann_file, 'source')
        else:
            self.source_annotations = self.load_custom_annotations(source_ann_file, 'source')
        
        # Load target dataset
        target_ann_file = self._get_annotation_path('target')
        if annotation_format == 'coco':
            self.target_annotations = self.load_coco_annotations(target_ann_file, 'target')
        else:
            self.target_annotations = self.load_custom_annotations(target_ann_file, 'target')
        
        # Load validation dataset (if configured)
        self.validation_annotations = []
        if self.domain_config['validation_domain']:
            validation_ann_file = self._get_annotation_path('validation')
            if annotation_format == 'coco':
                self.validation_annotations = self.load_coco_annotations(validation_ann_file, 'validation')
            else:
                self.validation_annotations = self.load_custom_annotations(validation_ann_file, 'validation')
        
        logger.info(f"Dataset loading complete:")
        logger.info(f"  Source images: {len(self.source_annotations)}")
        logger.info(f"  Target images: {len(self.target_annotations)}")
        logger.info(f"  Validation images: {len(self.validation_annotations)}")
        logger.info(f"  Categories: {len(self.category_info)}")
        
        return self.source_annotations, self.target_annotations, self.validation_annotations
    
    def validate_dataset(self, annotations: List[ImageAnnotation], domain: str) -> Dict[str, Any]:
        """
        Validate dataset integrity.
        
        Args:
            annotations: List of annotations to validate
            domain: 'source' or 'target'
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {domain} dataset...")
        
        validation_results = {
            'total_images': len(annotations),
            'valid_images': 0,
            'invalid_images': 0,
            'total_boxes': 0,
            'invalid_boxes': 0,
            'missing_images': [],
            'invalid_annotations': [],
            'image_size_stats': {'min_width': float('inf'), 'max_width': 0, 'min_height': float('inf'), 'max_height': 0}
        }
        
        for ann in annotations:
            image_path = self._get_image_path(ann.file_name, domain)
            
            # Check if image exists
            if not image_path.exists():
                validation_results['invalid_images'] += 1
                validation_results['missing_images'].append(ann.file_name)
                continue
            
            # Check image dimensions
            try:
                width, height = self._get_image_dimensions(image_path)
                validation_results['image_size_stats']['min_width'] = min(validation_results['image_size_stats']['min_width'], width)
                validation_results['image_size_stats']['max_width'] = max(validation_results['image_size_stats']['max_width'], width)
                validation_results['image_size_stats']['min_height'] = min(validation_results['image_size_stats']['min_height'], height)
                validation_results['image_size_stats']['max_height'] = max(validation_results['image_size_stats']['max_height'], height)
            except:
                validation_results['invalid_images'] += 1
                continue
            
            # Validate bounding boxes
            valid_boxes = 0
            for box in ann.boxes:
                if len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
                    x, y, w, h = box
                    if w > 0 and h > 0 and x >= 0 and y >= 0:
                        valid_boxes += 1
                    else:
                        validation_results['invalid_boxes'] += 1
                else:
                    validation_results['invalid_boxes'] += 1
            
            validation_results['total_boxes'] += len(ann.boxes)
            
            if valid_boxes == len(ann.boxes):
                validation_results['valid_images'] += 1
            else:
                validation_results['invalid_annotations'].append(ann.file_name)
        
        # Calculate statistics
        validation_results['validity_rate'] = validation_results['valid_images'] / max(1, validation_results['total_images'])
        
        logger.info(f"{domain.capitalize()} dataset validation results:")
        logger.info(f"  Valid images: {validation_results['valid_images']}/{validation_results['total_images']}")
        logger.info(f"  Validity rate: {validation_results['validity_rate']:.2%}")
        logger.info(f"  Total bounding boxes: {validation_results['total_boxes']}")
        logger.info(f"  Invalid boxes: {validation_results['invalid_boxes']}")
        
        return validation_results
    
    def preprocess_image(self, image_path: Path, transform: A.Compose, boxes: List[List[float]] = None, 
                        category_ids: List[int] = None) -> Dict[str, Any]:
        """
        Preprocess a single image with optional bounding boxes.
        
        Args:
            image_path: Path to image file
            transform: Albumentations transform to apply
            boxes: List of bounding boxes in COCO format [x, y, w, h]
            category_ids: List of category IDs for boxes
            
        Returns:
            Dictionary with processed image and transformed boxes
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Prepare transformation parameters
            transform_params = {'image': image}
            
            if boxes is not None and len(boxes) > 0:
                # Convert boxes to albumentations format if needed
                valid_boxes = []
                valid_category_ids = []
                
                for i, box in enumerate(boxes):
                    if len(box) == 4:
                        x, y, w, h = box
                        if w > 0 and h > 0:
                            valid_boxes.append([x, y, w, h])
                            valid_category_ids.append(category_ids[i] if category_ids else 1)
                
                if valid_boxes:
                    transform_params['bboxes'] = valid_boxes
                    transform_params['category_ids'] = valid_category_ids
            
            # Apply transformation
            transformed = transform(**transform_params)
            
            result = {
                'image': transformed['image'],
                'original_size': (image.shape[1], image.shape[0]),  # (width, height)
                'processed_size': (self.target_size, self.target_size)
            }
            
            if 'bboxes' in transformed:
                result['boxes'] = transformed['bboxes']
                result['category_ids'] = transformed['category_ids']
            else:
                result['boxes'] = []
                result['category_ids'] = []
            
            return result
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def create_data_splits(self, annotations: List[ImageAnnotation], 
                          train_ratio: float = 0.8, val_ratio: float = 0.2) -> Dict[str, List[ImageAnnotation]]:
        """
        Split annotations into train/validation sets.
        
        Args:
            annotations: List of annotations to split
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            
        Returns:
            Dictionary with 'train' and 'val' splits
        """
        if abs(train_ratio + val_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio must equal 1.0")
        
        np.random.seed(42)  # For reproducible splits
        indices = np.random.permutation(len(annotations))
        
        train_size = int(len(annotations) * train_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        splits = {
            'train': [annotations[i] for i in train_indices],
            'val': [annotations[i] for i in val_indices]
        }
        
        logger.info(f"Data split created:")
        logger.info(f"  Train: {len(splits['train'])} images")
        logger.info(f"  Validation: {len(splits['val'])} images")
        
        return splits
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        stats = {
            'source': {
                'num_images': len(self.source_annotations),
                'num_boxes': sum(len(ann.boxes) for ann in self.source_annotations),
                'avg_boxes_per_image': 0,
                'categories': {},
                'domain_key': self.domain_config['source_domain']
            },
            'target': {
                'num_images': len(self.target_annotations),
                'num_boxes': sum(len(ann.boxes) for ann in self.target_annotations),
                'avg_boxes_per_image': 0,
                'categories': {},
                'domain_key': self.domain_config['target_domain']
            },
            'validation': {
                'num_images': len(self.validation_annotations),
                'num_boxes': sum(len(ann.boxes) for ann in self.validation_annotations),
                'avg_boxes_per_image': 0,
                'categories': {},
                'domain_key': self.domain_config['validation_domain']
            },
            'combined': {
                'total_images': len(self.source_annotations) + len(self.target_annotations) + len(self.validation_annotations),
                'total_boxes': 0,
                'categories': set()
            }
        }
        
        # Calculate statistics for each domain
        for domain_name, annotations in [('source', self.source_annotations), 
                                       ('target', self.target_annotations),
                                       ('validation', self.validation_annotations)]:
            if len(annotations) > 0:
                stats[domain_name]['avg_boxes_per_image'] = stats[domain_name]['num_boxes'] / len(annotations)
                
                # Category distribution
                for ann in annotations:
                    for cat_id in ann.category_ids:
                        cat_name = self.category_info.get(cat_id, f"category_{cat_id}")
                        stats[domain_name]['categories'][cat_name] = stats[domain_name]['categories'].get(cat_name, 0) + 1
                        stats['combined']['categories'].add(cat_name)
        
        stats['combined']['total_boxes'] = stats['source']['num_boxes'] + stats['target']['num_boxes'] + stats['validation']['num_boxes']
        stats['combined']['categories'] = list(stats['combined']['categories'])
        
        return stats
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get detailed information about the current dataset configuration."""
        info = {
            'dataset_config': self.domain_config,
            'available_datasets': self.dataset_config.list_available_datasets(),
            'dataset_summary': self.dataset_config.get_dataset_summary(),
            'validation_results': self.dataset_config.validate_dataset_paths(),
            'preprocessing_config': {
                'target_size': self.target_size,
                'preserve_aspect_ratio': self.preserve_aspect_ratio,
                'apply_augmentations': self.apply_augmentations,
                'imagenet_normalization': {
                    'mean': self.imagenet_mean,
                    'std': self.imagenet_std
                }
            }
        }
        return info
    
    def save_preprocessed_sample(self, output_dir: Path, num_samples: int = 5):
        """Save preprocessed sample images for visualization."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample from source domain
        if self.source_annotations:
            source_samples = np.random.choice(self.source_annotations, 
                                            min(num_samples, len(self.source_annotations)), 
                                            replace=False)
            
            for i, ann in enumerate(source_samples):
                image_path = self._get_image_path(ann.file_name, 'source')
                if image_path.exists():
                    try:
                        processed = self.preprocess_image(image_path, self.augment_transform, ann.boxes, ann.category_ids)
                        
                        # Convert tensor back to image for saving
                        image_tensor = processed['image']
                        if isinstance(image_tensor, torch.Tensor):
                            # Denormalize
                            for t, m, s in zip(image_tensor, self.imagenet_mean, self.imagenet_std):
                                t.mul_(s).add_(m)
                            
                            # Convert to numpy and save
                            image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            output_path = output_dir / f"source_sample_{i+1}.jpg"
                            Image.fromarray(image_np).save(output_path)
                            
                            logger.info(f"Saved source sample: {output_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save source sample {i}: {e}")
        
        # Sample from target domain
        if self.target_annotations:
            target_samples = np.random.choice(self.target_annotations,
                                            min(num_samples, len(self.target_annotations)),
                                            replace=False)
            
            for i, ann in enumerate(target_samples):
                image_path = self._get_image_path(ann.file_name, 'target')
                if image_path.exists():
                    try:
                        processed = self.preprocess_image(image_path, self.target_transform, ann.boxes, ann.category_ids)
                        
                        # Convert tensor back to image for saving
                        image_tensor = processed['image']
                        if isinstance(image_tensor, torch.Tensor):
                            # Denormalize
                            for t, m, s in zip(image_tensor, self.imagenet_mean, self.imagenet_std):
                                t.mul_(s).add_(m)
                            
                            # Convert to numpy and save
                            image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            output_path = output_dir / f"target_sample_{i+1}.jpg"
                            Image.fromarray(image_np).save(output_path)
                            
                            logger.info(f"Saved target sample: {output_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save target sample {i}: {e}")

class SegmentationDataset(Dataset):
    """PyTorch Dataset for segmentation with bounding boxes."""
    
    def __init__(self, annotations: List[ImageAnnotation], preprocessor: DataPreprocessor, 
                 domain: str, use_augmentation: bool = True):
        """
        Initialize dataset.
        
        Args:
            annotations: List of ImageAnnotation objects
            preprocessor: DataPreprocessor instance
            domain: 'source' or 'target'
            use_augmentation: Whether to apply augmentations
        """
        self.annotations = annotations
        self.preprocessor = preprocessor
        self.domain = domain
        self.use_augmentation = use_augmentation
        
        # Choose appropriate transform
        if domain == 'source' and use_augmentation:
            self.transform = preprocessor.augment_transform
        else:
            self.transform = preprocessor.target_transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = self.preprocessor._get_image_path(ann.file_name, self.domain)
        
        try:
            processed = self.preprocessor.preprocess_image(
                image_path, self.transform, ann.boxes, ann.category_ids
            )
            
            return {
                'image': processed['image'],
                'boxes': torch.tensor(processed['boxes'], dtype=torch.float32) if processed['boxes'] else torch.empty((0, 4)),
                'category_ids': torch.tensor(processed['category_ids'], dtype=torch.long) if processed['category_ids'] else torch.empty((0,), dtype=torch.long),
                'image_id': ann.image_id,
                'file_name': ann.file_name,
                'domain': self.domain,
                'original_size': processed['original_size'],
                'processed_size': processed['processed_size']
            }
        except Exception as e:
            logger.error(f"Error loading sample {idx} ({ann.file_name}): {e}")
            # Return empty sample on error
            return {
                'image': torch.zeros((3, self.preprocessor.target_size, self.preprocessor.target_size)),
                'boxes': torch.empty((0, 4)),
                'category_ids': torch.empty((0,), dtype=torch.long),
                'image_id': ann.image_id,
                'file_name': ann.file_name,
                'domain': self.domain,
                'original_size': (0, 0),
                'processed_size': (self.preprocessor.target_size, self.preprocessor.target_size)
            }

def create_data_loaders(preprocessor: DataPreprocessor, 
                       batch_size: int = 16, 
                       num_workers: int = 4,
                       train_val_split: bool = True) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for the datasets.
    
    Args:
        preprocessor: DataPreprocessor instance with loaded data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        train_val_split: Whether to create train/val splits for source data
        
    Returns:
        Dictionary of DataLoaders
    """
    data_loaders = {}
    
    # Source domain loaders
    if preprocessor.source_annotations:
        if train_val_split:
            source_splits = preprocessor.create_data_splits(preprocessor.source_annotations)
            
            # Training loader (with augmentations)
            train_dataset = SegmentationDataset(
                source_splits['train'], preprocessor, 'source', use_augmentation=True
            )
            data_loaders['source_train'] = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                collate_fn=collate_fn
            )
            
            # Validation loader (no augmentations)
            val_dataset = SegmentationDataset(
                source_splits['val'], preprocessor, 'source', use_augmentation=False
            )
            data_loaders['source_val'] = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                collate_fn=collate_fn
            )
        else:
            # Single source loader
            source_dataset = SegmentationDataset(
                preprocessor.source_annotations, preprocessor, 'source', use_augmentation=True
            )
            data_loaders['source'] = DataLoader(
                source_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                collate_fn=collate_fn
            )
    
    # Target domain loader
    if preprocessor.target_annotations:
        target_dataset = SegmentationDataset(
            preprocessor.target_annotations, preprocessor, 'target', use_augmentation=False
        )
        data_loaders['target'] = DataLoader(
            target_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    logger.info(f"Created {len(data_loaders)} data loaders with batch size {batch_size}")
    return data_loaders

def collate_fn(batch):
    """Custom collate function for handling variable number of boxes per image."""
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'boxes': [item['boxes'] for item in batch],
        'category_ids': [item['category_ids'] for item in batch],
        'image_ids': [item['image_id'] for item in batch],
        'file_names': [item['file_name'] for item in batch],
        'domains': [item['domain'] for item in batch],
        'original_sizes': [item['original_size'] for item in batch],
        'processed_sizes': [item['processed_size'] for item in batch]
    }

def main():
    """Test data preprocessing functionality with the new dataset structure."""
    print("Testing Data Preprocessing with New Dataset Structure")
    print("=" * 60)
    
    # Initialize preprocessor with automatic dataset discovery
    preprocessor = DataPreprocessor(
        target_size=512, 
        apply_augmentations=True
    )
    
    # Print dataset configuration information
    dataset_info = preprocessor.get_dataset_info()
    print("\n📊 Dataset Configuration:")
    print(f"  Source Domain: {dataset_info['dataset_config']['source_domain']}")
    print(f"  Target Domain: {dataset_info['dataset_config']['target_domain']}")
    print(f"  Validation Domain: {dataset_info['dataset_config']['validation_domain']}")
    
    # Print available datasets
    print("\n📁 Available Datasets:")
    for dataset_key, info in dataset_info['available_datasets'].items():
        status = "✓" if info['has_images'] and info['has_annotations'] else "✗"
        print(f"  {status} {dataset_key}: {info['description']}")
        print(f"    Images: {'✓' if info['has_images'] else '✗'}, Annotations: {'✓' if info['has_annotations'] else '✗'}")
    
    # Load datasets
    print("\n🔄 Loading datasets...")
    try:
        source_ann, target_ann, validation_ann = preprocessor.load_datasets(annotation_format='coco')
        
        # Validate datasets
        print("\n🔍 Validating datasets...")
        if source_ann:
            source_validation = preprocessor.validate_dataset(source_ann, 'source')
            print(f"Source validation: {source_validation['validity_rate']:.1%} valid images")
        
        if target_ann:
            target_validation = preprocessor.validate_dataset(target_ann, 'target')
            print(f"Target validation: {target_validation['validity_rate']:.1%} valid images")
        
        if validation_ann:
            val_validation = preprocessor.validate_dataset(validation_ann, 'validation')
            print(f"Validation validation: {val_validation['validity_rate']:.1%} valid images")
        
        # Get and display statistics
        print("\n📈 Dataset Statistics:")
        stats = preprocessor.get_dataset_statistics()
        for domain, domain_stats in stats.items():
            if domain != 'combined' and domain_stats['num_images'] > 0:
                print(f"  {domain.capitalize()} ({domain_stats['domain_key']}):")
                print(f"    Images: {domain_stats['num_images']}")
                print(f"    Bounding boxes: {domain_stats['num_boxes']}")
                print(f"    Avg boxes per image: {domain_stats['avg_boxes_per_image']:.1f}")
                print(f"    Categories: {list(domain_stats['categories'].keys())}")
        
        print("\n🎉 Data preprocessing test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
