"""
Data Visualization Utilities

This module provides utilities for visualizing datasets, annotations, and preprocessing results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from PIL import Image
import torch
import seaborn as sns

logger = logging.getLogger(__name__)

class DataVisualizer:
    """Utilities for visualizing dataset and preprocessing results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def visualize_image_with_boxes(self, image_path: Path, boxes: List[List[float]], 
                                 category_ids: List[int] = None, category_names: Dict[int, str] = None,
                                 title: str = None, save_path: Path = None) -> plt.Figure:
        """
        Visualize an image with bounding boxes.
        
        Args:
            image_path: Path to image file
            boxes: List of bounding boxes in COCO format [x, y, w, h]
            category_ids: List of category IDs for each box
            category_names: Mapping of category ID to name
            title: Plot title
            save_path: Path to save the visualization
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Load image
            if isinstance(image_path, (str, Path)):
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path
            
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.imshow(image)
            
            # Colors for different categories
            colors = plt.cm.Set3(np.linspace(0, 1, max(10, len(set(category_ids or [1])))))
            
            for i, box in enumerate(boxes):
                x, y, w, h = box
                
                # Get category info
                cat_id = category_ids[i] if category_ids else 1
                cat_name = category_names.get(cat_id, f"cat_{cat_id}") if category_names else f"object_{i}"
                color = colors[cat_id % len(colors)]
                
                # Draw bounding box
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                # Add label
                ax.text(x, y - 5, cat_name, fontsize=10, color=color, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            ax.set_title(title or f"Image with {len(boxes)} bounding boxes")
            ax.axis('off')
            
            if save_path:
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                logger.info(f"Saved visualization: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error visualizing image {image_path}: {e}")
            raise
    
    def visualize_dataset_samples(self, preprocessor, domain: str = 'source', 
                                num_samples: int = 6, save_path: Path = None) -> plt.Figure:
        """
        Visualize random samples from a dataset.
        
        Args:
            preprocessor: DataPreprocessor instance
            domain: 'source' or 'target'
            num_samples: Number of samples to visualize
            save_path: Path to save the visualization
            
        Returns:
            matplotlib Figure object
        """
        annotations = preprocessor.source_annotations if domain == 'source' else preprocessor.target_annotations
        
        if not annotations:
            logger.warning(f"No annotations found for {domain} domain")
            return None
        
        # Sample random annotations
        sample_indices = np.random.choice(len(annotations), min(num_samples, len(annotations)), replace=False)
        samples = [annotations[i] for i in sample_indices]
        
        # Calculate grid size
        cols = min(3, num_samples)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, ann in enumerate(samples):
            image_path = preprocessor._get_image_path(ann.file_name, domain)
            
            if image_path.exists():
                try:
                    # Load and display image
                    image = cv2.imread(str(image_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    axes[i].imshow(image)
                    
                    # Draw bounding boxes
                    colors = plt.cm.Set3(np.linspace(0, 1, max(10, len(set(ann.category_ids)))))
                    
                    for j, box in enumerate(ann.boxes):
                        x, y, w, h = box
                        cat_id = ann.category_ids[j] if j < len(ann.category_ids) else 1
                        color = colors[cat_id % len(colors)]
                        
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                        axes[i].add_patch(rect)
                        
                        # Add category label
                        cat_name = preprocessor.category_info.get(cat_id, f"cat_{cat_id}")
                        axes[i].text(x, y - 5, cat_name, fontsize=8, color=color,
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
                    
                    axes[i].set_title(f"{domain}: {ann.file_name}\n{len(ann.boxes)} objects")
                    axes[i].axis('off')
                    
                except Exception as e:
                    logger.warning(f"Error loading sample {i}: {e}")
                    axes[i].text(0.5, 0.5, f"Error loading\n{ann.file_name}", 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f"Image not found:\n{ann.file_name}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(samples), len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(f"{domain.capitalize()} Domain Samples", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved dataset samples: {save_path}")
        
        return fig
    
    def plot_dataset_statistics(self, stats: Dict[str, Any], save_path: Path = None) -> plt.Figure:
        """
        Plot dataset statistics.
        
        Args:
            stats: Dataset statistics from preprocessor
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Image count comparison
        domains = ['Source', 'Target']
        image_counts = [stats['source']['num_images'], stats['target']['num_images']]
        
        axes[0, 0].bar(domains, image_counts, color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Number of Images by Domain')
        axes[0, 0].set_ylabel('Number of Images')
        for i, count in enumerate(image_counts):
            axes[0, 0].text(i, count + max(image_counts) * 0.01, str(count), ha='center')
        
        # 2. Bounding box count comparison
        box_counts = [stats['source']['num_boxes'], stats['target']['num_boxes']]
        
        axes[0, 1].bar(domains, box_counts, color=['lightgreen', 'orange'])
        axes[0, 1].set_title('Number of Bounding Boxes by Domain')
        axes[0, 1].set_ylabel('Number of Boxes')
        for i, count in enumerate(box_counts):
            axes[0, 1].text(i, count + max(box_counts) * 0.01, str(count), ha='center')
        
        # 3. Category distribution (source)
        if stats['source']['categories']:
            categories = list(stats['source']['categories'].keys())
            counts = list(stats['source']['categories'].values())
            
            axes[1, 0].pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Source Domain Category Distribution')
        else:
            axes[1, 0].text(0.5, 0.5, 'No source categories', ha='center', va='center')
            axes[1, 0].set_title('Source Domain Category Distribution')
        
        # 4. Category distribution (target)  
        if stats['target']['categories']:
            categories = list(stats['target']['categories'].keys())
            counts = list(stats['target']['categories'].values())
            
            axes[1, 1].pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Target Domain Category Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'No target categories', ha='center', va='center')
            axes[1, 1].set_title('Target Domain Category Distribution')
        
        plt.suptitle('Dataset Statistics Overview', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved statistics plot: {save_path}")
        
        return fig
    
    def visualize_preprocessing_comparison(self, preprocessor, annotation, domain: str, 
                                        save_path: Path = None) -> plt.Figure:
        """
        Compare original vs preprocessed image.
        
        Args:
            preprocessor: DataPreprocessor instance
            annotation: ImageAnnotation object
            domain: 'source' or 'target'
            save_path: Path to save the visualization
            
        Returns:
            matplotlib Figure object
        """
        image_path = preprocessor._get_image_path(annotation.file_name, domain)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None
        
        try:
            # Load original image
            original_image = cv2.imread(str(image_path))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Get preprocessed image
            transform = preprocessor.augment_transform if domain == 'source' else preprocessor.target_transform
            processed = preprocessor.preprocess_image(image_path, transform, annotation.boxes, annotation.category_ids)
            
            # Convert tensor back to displayable format
            processed_image = processed['image']
            if isinstance(processed_image, torch.Tensor):
                # Denormalize
                mean = torch.tensor(preprocessor.imagenet_mean).view(3, 1, 1)
                std = torch.tensor(preprocessor.imagenet_std).view(3, 1, 1)
                processed_image = processed_image * std + mean
                processed_image = torch.clamp(processed_image, 0, 1)
                processed_image = (processed_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Original image
            axes[0].imshow(original_image)
            axes[0].set_title(f'Original Image\nSize: {original_image.shape[1]}x{original_image.shape[0]}')
            axes[0].axis('off')
            
            # Draw original bounding boxes
            colors = plt.cm.Set3(np.linspace(0, 1, max(10, len(set(annotation.category_ids)))))
            for i, box in enumerate(annotation.boxes):
                x, y, w, h = box
                cat_id = annotation.category_ids[i] if i < len(annotation.category_ids) else 1
                color = colors[cat_id % len(colors)]
                
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                axes[0].add_patch(rect)
            
            # Processed image
            axes[1].imshow(processed_image)
            axes[1].set_title(f'Preprocessed Image\nSize: {processed_image.shape[1]}x{processed_image.shape[0]}')
            axes[1].axis('off')
            
            # Draw transformed bounding boxes
            for i, box in enumerate(processed['boxes']):
                if len(box) == 4:
                    x, y, w, h = box
                    cat_id = processed['category_ids'][i] if i < len(processed['category_ids']) else 1
                    color = colors[cat_id % len(colors)]
                    
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                    axes[1].add_patch(rect)
            
            fig.suptitle(f'{domain.capitalize()} Domain: Preprocessing Comparison\n{annotation.file_name}', 
                        fontsize=14)
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                logger.info(f"Saved preprocessing comparison: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating preprocessing comparison: {e}")
            raise
    
    def plot_augmentation_examples(self, preprocessor, annotation, domain: str = 'source', 
                                 num_examples: int = 6, save_path: Path = None) -> plt.Figure:
        """
        Show examples of data augmentation.
        
        Args:
            preprocessor: DataPreprocessor instance
            annotation: ImageAnnotation object
            domain: Domain to use ('source' for augmentations)
            num_examples: Number of augmentation examples
            save_path: Path to save the visualization
            
        Returns:
            matplotlib Figure object
        """
        image_path = preprocessor._get_image_path(annotation.file_name, domain)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None
        
        # Calculate grid
        cols = min(3, num_examples)
        rows = (num_examples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        try:
            for i in range(num_examples):
                # Apply augmentation
                processed = preprocessor.preprocess_image(
                    image_path, preprocessor.augment_transform, 
                    annotation.boxes, annotation.category_ids
                )
                
                # Convert tensor to displayable format
                aug_image = processed['image']
                if isinstance(aug_image, torch.Tensor):
                    # Denormalize
                    mean = torch.tensor(preprocessor.imagenet_mean).view(3, 1, 1)
                    std = torch.tensor(preprocessor.imagenet_std).view(3, 1, 1)
                    aug_image = aug_image * std + mean
                    aug_image = torch.clamp(aug_image, 0, 1)
                    aug_image = (aug_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
                axes[i].imshow(aug_image)
                axes[i].set_title(f'Augmentation {i+1}')
                axes[i].axis('off')
                
                # Draw transformed boxes
                colors = plt.cm.Set3(np.linspace(0, 1, max(10, len(set(annotation.category_ids)))))
                for j, box in enumerate(processed['boxes']):
                    if len(box) == 4:
                        x, y, w, h = box
                        cat_id = processed['category_ids'][j] if j < len(processed['category_ids']) else 1
                        color = colors[cat_id % len(colors)]
                        
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                        axes[i].add_patch(rect)
            
            # Hide unused subplots
            for i in range(num_examples, len(axes)):
                axes[i].axis('off')
            
            fig.suptitle(f'Data Augmentation Examples\n{annotation.file_name}', fontsize=14)
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                logger.info(f"Saved augmentation examples: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating augmentation examples: {e}")
            raise

def main():
    """Test visualization functionality."""
    from data_preprocessing import DataPreprocessor
    
    # Initialize components
    preprocessor = DataPreprocessor(target_size=512, apply_augmentations=True)
    visualizer = DataVisualizer()
    
    # Load data
    source_ann, target_ann = preprocessor.load_datasets(annotation_format='coco')
    
    if source_ann:
        # Visualize dataset samples
        fig = visualizer.visualize_dataset_samples(preprocessor, 'source', num_samples=4)
        if fig:
            plt.show()
    
    # Plot statistics
    stats = preprocessor.get_dataset_statistics()
    fig = visualizer.plot_dataset_statistics(stats)
    plt.show()

if __name__ == "__main__":
    main()
