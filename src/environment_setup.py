"""
Environment Setup Module for SAM-based Segmentation with Domain Adaptation

This module handles:
1. Environment validation (CUDA, GPU availability)
2. Dependency installation verification
3. SAM model setup and checkpoint management
4. Directory structure validation
"""

import os
import sys
import torch
import subprocess
import platform
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """Handles environment setup and validation for the segmentation pipeline."""
    
    def __init__(self, project_root: str = None):
        """
        Initialize environment setup.
        
        Args:
            project_root: Path to project root directory
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.models_dir = self.project_root / "models"
        self.dataset_dir = self.project_root / "dataset"
        
    def check_system_info(self):
        """Display system information."""
        logger.info("=== System Information ===")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Python executable: {sys.executable}")
        
    def check_cuda_availability(self):
        """
        Check CUDA availability and GPU information.
        
        Returns:
            dict: CUDA information including availability, version, and device info
        """
        logger.info("=== CUDA/GPU Information ===")
        
        cuda_info = {
            'available': torch.cuda.is_available(),
            'version': None,
            'device_count': 0,
            'devices': [],
            'current_device': None
        }
        
        if torch.cuda.is_available():
            cuda_info['version'] = torch.version.cuda
            cuda_info['device_count'] = torch.cuda.device_count()
            cuda_info['current_device'] = torch.cuda.current_device()
            
            logger.info(f"CUDA Available: ✓")
            logger.info(f"CUDA Version: {cuda_info['version']}")
            logger.info(f"Number of GPUs: {cuda_info['device_count']}")
            
            for i in range(cuda_info['device_count']):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    'name': device_props.name,
                    'memory': f"{device_props.total_memory / 1024**3:.1f} GB",
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                }
                cuda_info['devices'].append(device_info)
                
                logger.info(f"GPU {i}: {device_info['name']}")
                logger.info(f"  Memory: {device_info['memory']}")
                logger.info(f"  Compute Capability: {device_info['compute_capability']}")
        else:
            logger.warning("CUDA not available. The pipeline will run on CPU (much slower).")
        
        return cuda_info
    
    def check_pytorch_setup(self):
        """Verify PyTorch installation and functionality."""
        logger.info("=== PyTorch Setup ===")
        
        try:
            logger.info(f"PyTorch version: {torch.__version__}")
            
            # Check torchvision version properly
            try:
                import torchvision
                logger.info(f"Torchvision version: {torchvision.__version__}")
            except Exception as tv_error:
                logger.warning(f"Torchvision version check failed: {tv_error}")
            
            # Test basic tensor operations
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            test_tensor = torch.randn(10, 10, device=device)
            result = torch.matmul(test_tensor, test_tensor.T)
            
            logger.info(f"Basic tensor operations: ✓ (device: {device})")
            return True
            
        except Exception as e:
            logger.error(f"PyTorch setup issue: {e}")
            return False
    
    def verify_dependencies(self):
        """Verify that all required packages are installed."""
        logger.info("=== Dependency Verification ===")
        
        required_packages = [
            'torch', 'torchvision', 'segment_anything', 'transformers',
            'opencv-python', 'PIL', 'numpy', 'scipy', 'scikit-learn',
            'matplotlib', 'tqdm', 'pandas'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'opencv-python':
                    import cv2
                elif package == 'PIL':
                    from PIL import Image
                elif package == 'segment_anything':
                    import segment_anything
                else:
                    __import__(package)
                logger.info(f"✓ {package}")
            except ImportError:
                logger.warning(f"✗ {package} - Not installed")
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")
            logger.info("Run: pip install -r requirements.txt")
            return False
        
        logger.info("All required dependencies are available!")
        return True
    
    def setup_directories(self):
        """Create and validate directory structure."""
        logger.info("=== Directory Setup ===")
        
        directories = [
            self.project_root / "src",
            self.project_root / "models",
            self.project_root / "dataset",
            self.project_root / "dataset" / "source",
            self.project_root / "dataset" / "target",
            self.project_root / "outputs",
            self.project_root / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True, parents=True)
            logger.info(f"✓ {directory}")
        
        # Create dummy files in dataset directories
        self._create_dummy_dataset_structure()
        
        return True
    
    def _create_dummy_dataset_structure(self):
        """Create dummy dataset structure with README files."""
        # Source dataset structure
        source_readme = self.dataset_dir / "source" / "README.md"
        source_readme.write_text("""# Source Dataset Directory

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
""")
        
        # Target dataset structure
        target_readme = self.dataset_dir / "target" / "README.md"
        target_readme.write_text("""# Target Dataset Directory

This directory should contain:
- `images/` - Target domain images (unlabeled)
- `annotations/` - Bounding box annotations for target images

Example structure:
```
target/
├── images/
│   ├── target_001.jpg
│   ├── target_002.jpg
│   └── ...
└── annotations/
    └── annotations.json  # COCO format with bounding boxes
```
""")
        
        logger.info("Created dummy dataset structure with README files")
    
    def get_sam_model_info(self):
        """Get information about available SAM models."""
        logger.info("=== SAM Model Information ===")
        
        sam_models = {
            'vit_h': {
                'checkpoint': 'sam_vit_h_4b8939.pth',
                'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
                'size': '~2.4GB',
                'description': 'Huge model - best accuracy'
            },
            'vit_l': {
                'checkpoint': 'sam_vit_l_0b3195.pth', 
                'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
                'size': '~1.2GB',
                'description': 'Large model - good balance'
            },
            'vit_b': {
                'checkpoint': 'sam_vit_b_01ec64.pth',
                'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', 
                'size': '~350MB',
                'description': 'Base model - fastest'
            }
        }
        
        logger.info("Available SAM models:")
        for model_name, info in sam_models.items():
            logger.info(f"  {model_name}: {info['description']} ({info['size']})")
            checkpoint_path = self.models_dir / info['checkpoint']
            if checkpoint_path.exists():
                logger.info(f"    ✓ Checkpoint available locally")
            else:
                logger.info(f"    ✗ Checkpoint not downloaded")
        
        return sam_models
    
    def download_sam_checkpoint(self, model_type='vit_b'):
        """
        Download SAM checkpoint if not available.
        
        Args:
            model_type: Type of SAM model ('vit_b', 'vit_l', 'vit_h')
        """
        sam_models = self.get_sam_model_info()
        
        if model_type not in sam_models:
            logger.error(f"Invalid model type: {model_type}")
            return False
        
        model_info = sam_models[model_type]
        checkpoint_path = self.models_dir / model_info['checkpoint']
        
        if checkpoint_path.exists():
            logger.info(f"SAM checkpoint already exists: {checkpoint_path}")
            return True
        
        logger.info(f"Downloading SAM {model_type} checkpoint...")
        logger.info(f"This may take a while ({model_info['size']})...")
        
        try:
            import urllib.request
            urllib.request.urlretrieve(model_info['url'], checkpoint_path)
            logger.info(f"✓ Downloaded SAM checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download SAM checkpoint: {e}")
            return False
    
    def run_complete_setup(self, download_sam=True, sam_model='vit_b'):
        """
        Run complete environment setup.
        
        Args:
            download_sam: Whether to download SAM checkpoint
            sam_model: Which SAM model to download
        """
        logger.info("Starting complete environment setup...")
        
        # System info
        self.check_system_info()
        
        # CUDA check
        cuda_info = self.check_cuda_availability()
        
        # PyTorch check
        pytorch_ok = self.check_pytorch_setup()
        
        # Dependencies
        deps_ok = self.verify_dependencies()
        
        # Directories
        dirs_ok = self.setup_directories()
        
        # SAM model info
        self.get_sam_model_info()
        
        # Download SAM checkpoint
        sam_ok = True
        if download_sam:
            sam_ok = self.download_sam_checkpoint(sam_model)
        
        # Summary
        logger.info("=== Setup Summary ===")
        logger.info(f"CUDA Available: {'✓' if cuda_info['available'] else '✗'}")
        logger.info(f"PyTorch Working: {'✓' if pytorch_ok else '✗'}")
        logger.info(f"Dependencies: {'✓' if deps_ok else '✗'}")
        logger.info(f"Directories: {'✓' if dirs_ok else '✗'}")
        logger.info(f"SAM Model: {'✓' if sam_ok else '✗'}")
        
        all_ok = pytorch_ok and deps_ok and dirs_ok and sam_ok
        
        if all_ok:
            logger.info("🎉 Environment setup completed successfully!")
            logger.info("Ready to proceed with the segmentation pipeline.")
        else:
            logger.warning("⚠️ Some setup steps failed. Please resolve issues before proceeding.")
        
        return all_ok

def main():
    """Main function for standalone execution."""
    setup = EnvironmentSetup()
    success = setup.run_complete_setup(download_sam=True, sam_model='vit_b')
    return success

if __name__ == "__main__":
    main()
