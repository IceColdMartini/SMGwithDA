"""
SAM (Segment Anything Model) Setup and Utilities

This module provides utilities for:
1. Loading and configuring SAM models
2. SAM model management and caching
3. Basic SAM inference utilities
4. Model configuration for domain adaptation
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)

class SAMSetup:
    """Handles SAM model setup and basic operations."""
    
    def __init__(self, model_type: str = 'vit_b', device: str = 'auto'):
        """
        Initialize SAM setup.
        
        Args:
            model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_type = model_type
        self.device = self._setup_device(device)
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "models"
        
        # Model configurations
        self.model_configs = {
            'vit_b': {
                'checkpoint': 'sam_vit_b_01ec64.pth',
                'model_class': 'sam_model_registry["vit_b"]',
                'encoder_embed_dim': 768,
                'encoder_depth': 12,
                'encoder_num_heads': 12,
                'encoder_global_attn_indexes': [2, 5, 8, 11]
            },
            'vit_l': {
                'checkpoint': 'sam_vit_l_0b3195.pth',
                'model_class': 'sam_model_registry["vit_l"]',
                'encoder_embed_dim': 1024,
                'encoder_depth': 24,
                'encoder_num_heads': 16,
                'encoder_global_attn_indexes': [5, 11, 17, 23]
            },
            'vit_h': {
                'checkpoint': 'sam_vit_h_4b8939.pth',
                'model_class': 'sam_model_registry["vit_h"]',
                'encoder_embed_dim': 1280,
                'encoder_depth': 32,
                'encoder_num_heads': 16,
                'encoder_global_attn_indexes': [7, 15, 23, 31]
            }
        }
        
        self.sam_model = None
        self.sam_predictor = None
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup and return the appropriate device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                logger.info("CUDA not available, using CPU")
        
        return torch.device(device)
    
    def load_sam_model(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Load SAM model from checkpoint.
        
        Args:
            checkpoint_path: Custom checkpoint path (optional)
            
        Returns:
            bool: Success status
        """
        try:
            # Import SAM components
            from segment_anything import sam_model_registry, SamPredictor
            
            # Determine checkpoint path
            if checkpoint_path is None:
                checkpoint_name = self.model_configs[self.model_type]['checkpoint']
                checkpoint_path = self.models_dir / checkpoint_name
            
            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path):
                logger.error(f"SAM checkpoint not found: {checkpoint_path}")
                logger.info("Please run environment setup to download the checkpoint")
                return False
            
            # Load model
            logger.info(f"Loading SAM {self.model_type} model...")
            self.sam_model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            self.sam_model.to(device=self.device)
            
            # Initialize predictor
            self.sam_predictor = SamPredictor(self.sam_model)
            
            logger.info(f"✓ SAM model loaded successfully on {self.device}")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import SAM: {e}")
            logger.info("Please install segment-anything: pip install segment-anything")
            return False
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current SAM model."""
        if self.sam_model is None:
            return {"status": "Model not loaded"}
        
        config = self.model_configs[self.model_type]
        
        info = {
            "model_type": self.model_type,
            "device": str(self.device),
            "encoder_embed_dim": config['encoder_embed_dim'],
            "encoder_depth": config['encoder_depth'],
            "encoder_num_heads": config['encoder_num_heads'],
            "status": "Loaded and ready"
        }
        
        if torch.cuda.is_available() and self.device.type == 'cuda':
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3
            info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3
        
        return info
    
    def extract_encoder_features(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract features using SAM encoder.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            
        Returns:
            torch.Tensor: Encoded features
        """
        if self.sam_predictor is None:
            raise RuntimeError("SAM model not loaded. Call load_sam_model() first.")
        
        # Set image for predictor (handles preprocessing internally)
        self.sam_predictor.set_image(image)
        
        # Get encoder features
        # Note: SAM predictor stores features after set_image()
        features = self.sam_predictor.features
        
        return features
    
    def predict_masks_from_boxes(self, 
                                image: np.ndarray, 
                                boxes: np.ndarray,
                                return_logits: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate masks from bounding boxes using SAM.
        
        Args:
            image: Input image (H, W, 3)
            boxes: Bounding boxes in format (N, 4) as [x1, y1, x2, y2]
            return_logits: Whether to return mask logits
            
        Returns:
            Tuple of (masks, scores, logits)
        """
        if self.sam_predictor is None:
            raise RuntimeError("SAM model not loaded. Call load_sam_model() first.")
        
        # Set image
        self.sam_predictor.set_image(image)
        
        all_masks = []
        all_scores = []
        all_logits = []
        
        for box in boxes:
            # Convert box to the format expected by SAM
            input_box = np.array(box)
            
            # Predict mask
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],  # Add batch dimension
                multimask_output=False,  # Single mask per box
                return_logits=return_logits
            )
            
            all_masks.append(masks[0])  # Take the first (and only) mask
            all_scores.append(scores[0])
            if return_logits:
                all_logits.append(logits[0])
        
        # Stack results
        masks = np.stack(all_masks, axis=0) if all_masks else np.array([])
        scores = np.array(all_scores) if all_scores else np.array([])
        logits = np.stack(all_logits, axis=0) if all_logits and return_logits else None
        
        return masks, scores, logits
    
    def get_encoder_for_adaptation(self) -> torch.nn.Module:
        """
        Get SAM encoder for domain adaptation.
        
        Returns:
            torch.nn.Module: SAM image encoder
        """
        if self.sam_model is None:
            raise RuntimeError("SAM model not loaded. Call load_sam_model() first.")
        
        # Return the image encoder
        encoder = self.sam_model.image_encoder
        
        # Freeze encoder parameters for domain adaptation
        for param in encoder.parameters():
            param.requires_grad = False
        
        logger.info("SAM encoder prepared for domain adaptation (frozen)")
        return encoder
    
    def get_decoder_for_finetuning(self) -> torch.nn.Module:
        """
        Get SAM decoder components for fine-tuning.
        
        Returns:
            torch.nn.Module: SAM mask decoder
        """
        if self.sam_model is None:
            raise RuntimeError("SAM model not loaded. Call load_sam_model() first.")
        
        # Return components needed for mask generation
        decoder_components = {
            'mask_decoder': self.sam_model.mask_decoder,
            'prompt_encoder': self.sam_model.prompt_encoder
        }
        
        # Enable gradients for fine-tuning
        for component in decoder_components.values():
            for param in component.parameters():
                param.requires_grad = True
        
        logger.info("SAM decoder prepared for fine-tuning")
        return decoder_components
    
    def preprocess_image(self, image_path: str, target_size: int = 1024) -> np.ndarray:
        """
        Preprocess image for SAM input.
        
        Args:
            image_path: Path to image file
            target_size: Target size for SAM (default 1024)
            
        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            from PIL import Image
            import cv2
            
            # Load image
            if isinstance(image_path, (str, Path)):
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path
            
            # SAM expects images in RGB format
            # The predictor will handle resizing internally
            return image
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise
    
    def benchmark_inference(self, image: np.ndarray, boxes: np.ndarray, num_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark SAM inference speed.
        
        Args:
            image: Test image
            boxes: Test bounding boxes
            num_runs: Number of benchmark runs
            
        Returns:
            Dict with timing statistics
        """
        if self.sam_predictor is None:
            raise RuntimeError("SAM model not loaded")
        
        import time
        
        times = []
        
        # Warmup
        self.predict_masks_from_boxes(image, boxes[:1])
        
        # Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            self.predict_masks_from_boxes(image, boxes)
            end_time = time.time()
            times.append(end_time - start_time)
        
        times = np.array(times)
        
        return {
            'mean_time': float(times.mean()),
            'std_time': float(times.std()),
            'min_time': float(times.min()),
            'max_time': float(times.max()),
            'fps': float(len(boxes) / times.mean())
        }

def create_sam_setup(model_type: str = 'vit_b', device: str = 'auto') -> SAMSetup:
    """
    Factory function to create SAM setup instance.
    
    Args:
        model_type: SAM model type
        device: Compute device
        
    Returns:
        SAMSetup: Configured SAM setup instance
    """
    sam_setup = SAMSetup(model_type=model_type, device=device)
    
    # Load model
    if sam_setup.load_sam_model():
        logger.info("SAM setup completed successfully")
    else:
        logger.warning("SAM setup incomplete - model loading failed")
    
    return sam_setup

def main():
    """Test SAM setup functionality."""
    sam_setup = create_sam_setup()
    
    # Print model info
    info = sam_setup.get_model_info()
    logger.info(f"SAM Model Info: {info}")

if __name__ == "__main__":
    main()
