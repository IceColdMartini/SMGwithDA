"""
Dataset Configuration and Path Management

This module handles the specific dataset structure and provides utilities for
managing the complex directory structure of the forestry dataset.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DatasetConfig:
    """Configuration and path management for the forestry dataset."""
    
    def __init__(self, project_root: str = None):
        """
        Initialize dataset configuration.
        
        Args:
            project_root: Path to project root directory
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.dataset_dir = self.project_root / "dataset"
        
        # Dataset structure mapping
        self.dataset_structure = {
            'main_dataset': {
                'part_1': {
                    'path': self.dataset_dir / "Dataset" / "part_1",
                    'images': self.dataset_dir / "Dataset" / "part_1" / "2k_dataset",
                    'annotations': self.dataset_dir / "Dataset" / "part_1" / "coco_annotations.json",
                    'cropped_boxes': self.dataset_dir / "Dataset" / "part_1" / "cropped_boxes",
                    'depth_images': self.dataset_dir / "Dataset" / "part_1" / "depth_images",
                    'yolo_labels': self.dataset_dir / "Dataset" / "part_1" / "yolo_labels",
                    'description': 'Main dataset part 1 with tree annotations'
                },
                'part_2': {
                    'path': self.dataset_dir / "Dataset" / "part_2",
                    'images': self.dataset_dir / "Dataset" / "part_2" / "raw_images",
                    'annotations': self.dataset_dir / "Dataset" / "part_2" / "coco_annotations.json",
                    'depth_images': self.dataset_dir / "Dataset" / "part_2" / "depth_images",
                    'yolo_labels': self.dataset_dir / "Dataset" / "part_2" / "yolo_labels",
                    'description': 'Main dataset part 2 with additional annotations'
                }
            },
            'testing': {
                'simulated': {
                    'path': self.dataset_dir / "Testing" / "simulated_images",
                    'images': self.dataset_dir / "Testing" / "simulated_images" / "images",
                    'annotations': self.dataset_dir / "Testing" / "simulated_images" / "coco_annotations.json",
                    'yolo_labels': self.dataset_dir / "Testing" / "simulated_images" / "yolo_labels",
                    'description': 'Simulated test images for validation'
                }
            },
            'legacy': {
                'source': {
                    'path': self.dataset_dir / "source",
                    'images': self.dataset_dir / "source" / "images",
                    'annotations': self.dataset_dir / "source" / "annotations" / "annotations.json",
                    'description': 'Legacy source domain placeholder'
                },
                'target': {
                    'path': self.dataset_dir / "target",
                    'images': self.dataset_dir / "target" / "images", 
                    'annotations': self.dataset_dir / "target" / "annotations" / "annotations.json",
                    'description': 'Legacy target domain placeholder'
                }
            }
        }
        
        # Default configuration for domain adaptation
        self.default_domain_config = {
            'source_domain': 'main_dataset.part_1',
            'target_domain': 'testing.simulated',
            'validation_domain': 'main_dataset.part_2'
        }
    
    def get_dataset_path(self, dataset_key: str, component: str = 'path') -> Path:
        """
        Get path for a specific dataset component.
        
        Args:
            dataset_key: Key in format 'category.subcategory' (e.g., 'main_dataset.part_1')
            component: Component type ('path', 'images', 'annotations', etc.)
            
        Returns:
            Path to the requested component
        """
        try:
            keys = dataset_key.split('.')
            dataset_info = self.dataset_structure
            
            for key in keys:
                dataset_info = dataset_info[key]
            
            if component in dataset_info:
                return dataset_info[component]
            else:
                logger.warning(f"Component '{component}' not found in {dataset_key}")
                return dataset_info['path']
                
        except KeyError:
            logger.error(f"Dataset key '{dataset_key}' not found in configuration")
            raise ValueError(f"Invalid dataset key: {dataset_key}")
    
    def list_available_datasets(self) -> Dict[str, Dict[str, str]]:
        """
        List all available datasets with descriptions.
        
        Returns:
            Dictionary mapping dataset keys to info
        """
        available = {}
        
        def _extract_datasets(data, prefix=""):
            for key, value in data.items():
                current_key = f"{prefix}.{key}" if prefix else key
                
                if 'description' in value:
                    available[current_key] = {
                        'description': value['description'],
                        'path': str(value['path']),
                        'has_images': value.get('images', Path()).exists() if 'images' in value else False,
                        'has_annotations': value.get('annotations', Path()).exists() if 'annotations' in value else False
                    }
                else:
                    _extract_datasets(value, current_key)
        
        _extract_datasets(self.dataset_structure)
        return available
    
    def validate_dataset_paths(self) -> Dict[str, Dict[str, bool]]:
        """
        Validate that all configured paths exist.
        
        Note: Legacy datasets (legacy.source, legacy.target) are placeholders 
        and are expected to have missing image files. Only main_dataset and 
        testing datasets contain actual forestry data.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        available_datasets = self.list_available_datasets()
        
        for dataset_key, info in available_datasets.items():
            dataset_path = self.get_dataset_path(dataset_key, 'path')
            images_path = self.get_dataset_path(dataset_key, 'images')
            annotations_path = self.get_dataset_path(dataset_key, 'annotations')
            
            validation_results[dataset_key] = {
                'path_exists': dataset_path.exists(),
                'images_exist': images_path.exists() if images_path != dataset_path else False,
                'annotations_exist': annotations_path.exists() if annotations_path != dataset_path else False,
                'image_count': len(list(images_path.glob('*.jpg')) + list(images_path.glob('*.JPG')) + 
                                 list(images_path.glob('*.png')) + list(images_path.glob('*.PNG'))) if images_path.exists() else 0
            }
        
        return validation_results
    
    def get_recommended_splits(self) -> Dict[str, str]:
        """
        Get recommended dataset splits for domain adaptation.
        
        Returns:
            Dictionary with recommended source/target/validation splits
        """
        # Analyze available datasets
        validation_results = self.validate_dataset_paths()
        
        # Find the dataset with most images for source
        best_source = None
        max_images = 0
        
        for dataset_key, results in validation_results.items():
            if results['annotations_exist'] and results['image_count'] > max_images:
                max_images = results['image_count']
                best_source = dataset_key
        
        recommendations = {
            'source_domain': best_source or 'main_dataset.part_1',
            'target_domain': 'testing.simulated',  # Use simulated for target
            'validation_domain': 'main_dataset.part_2' if validation_results.get('main_dataset.part_2', {}).get('annotations_exist') else None
        }
        
        return recommendations
    
    def setup_domain_adaptation_config(self, source: str = None, target: str = None, 
                                     validation: str = None) -> Dict[str, str]:
        """
        Setup configuration for domain adaptation.
        
        Args:
            source: Source domain dataset key
            target: Target domain dataset key  
            validation: Validation domain dataset key
            
        Returns:
            Final configuration dictionary
        """
        # Use provided values or get recommendations
        if source is None or target is None:
            recommendations = self.get_recommended_splits()
            source = source or recommendations['source_domain']
            target = target or recommendations['target_domain']
            validation = validation or recommendations['validation_domain']
        
        config = {
            'source_domain': source,
            'target_domain': target,
            'validation_domain': validation
        }
        
        # Validate the configuration
        validation_results = self.validate_dataset_paths()
        for role, dataset_key in config.items():
            if dataset_key and dataset_key not in validation_results:
                logger.warning(f"Dataset '{dataset_key}' for {role} not found")
            elif dataset_key and not validation_results[dataset_key]['annotations_exist']:
                logger.warning(f"Dataset '{dataset_key}' for {role} has no annotations")
        
        logger.info(f"Domain adaptation configuration:")
        logger.info(f"  Source: {config['source_domain']}")
        logger.info(f"  Target: {config['target_domain']}")
        logger.info(f"  Validation: {config['validation_domain']}")
        
        return config
    
    def get_dataset_summary(self) -> Dict[str, any]:
        """Get comprehensive dataset summary."""
        available_datasets = self.list_available_datasets()
        validation_results = self.validate_dataset_paths()
        
        summary = {
            'total_datasets': len(available_datasets),
            'valid_datasets': sum(1 for v in validation_results.values() if v['path_exists']),
            'datasets_with_annotations': sum(1 for v in validation_results.values() if v['annotations_exist']),
            'total_images': sum(v['image_count'] for v in validation_results.values()),
            'dataset_details': {}
        }
        
        for dataset_key in available_datasets:
            validation = validation_results[dataset_key]
            summary['dataset_details'][dataset_key] = {
                'description': available_datasets[dataset_key]['description'],
                'status': 'valid' if validation['path_exists'] else 'missing',
                'has_annotations': validation['annotations_exist'],
                'image_count': validation['image_count']
            }
        
        return summary

def main():
    """Test dataset configuration."""
    config = DatasetConfig()
    
    # List available datasets
    print("Available Datasets:")
    available = config.list_available_datasets()
    for key, info in available.items():
        print(f"  {key}: {info['description']}")
    
    # Validate paths
    print("\nValidation Results:")
    validation = config.validate_dataset_paths()
    for key, results in validation.items():
        status = "✓" if results['path_exists'] else "✗"
        images = f"{results['image_count']} images" if results['image_count'] > 0 else "no images"
        annotations = "annotations ✓" if results['annotations_exist'] else "annotations ✗"
        print(f"  {status} {key}: {images}, {annotations}")
    
    # Get recommendations
    print("\nRecommended Configuration:")
    recommendations = config.get_recommended_splits()
    for role, dataset in recommendations.items():
        print(f"  {role}: {dataset}")
    
    # Get summary
    print("\nDataset Summary:")
    summary = config.get_dataset_summary()
    print(f"  Total datasets: {summary['total_datasets']}")
    print(f"  Valid datasets: {summary['valid_datasets']}")
    print(f"  Datasets with annotations: {summary['datasets_with_annotations']}")
    print(f"  Total images: {summary['total_images']}")

if __name__ == "__main__":
    main()
