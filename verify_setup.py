#!/usr/bin/env python3
"""
Quick setup verification script for SMGwithDA project.
Run this script to verify that the environment is properly configured.
"""

import sys
import os
from pathlib import Path

def main():
    """Run basic setup verification."""
    print("🔍 SMGwithDA Setup Verification")
    print("=" * 40)
    
    # Check project structure
    project_root = Path(__file__).parent
    required_dirs = [
        "src", "dataset", "models", "dataset/source", "dataset/target"
    ]
    
    print("\n📁 Directory Structure:")
    all_dirs_ok = True
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}")
        else:
            print(f"✗ {dir_name} - Missing")
            all_dirs_ok = False
    
    # Check key files
    print("\n📄 Key Files:")
    key_files = [
        "requirements.txt",
        "main_pipeline.ipynb", 
        "src/environment_setup.py",
        "src/sam_setup.py"
    ]
    
    all_files_ok = True
    for file_name in key_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name} - Missing")
            all_files_ok = False
    
    # Try importing key modules (if dependencies installed)
    print("\n🐍 Python Dependencies:")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA {torch.version.cuda} - GPU available")
        else:
            print("⚠ CUDA not available - will use CPU")
        
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Run: pip install -r requirements.txt")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not installed")
    
    try:
        from segment_anything import sam_model_registry
        print("✓ Segment Anything installed")
    except ImportError:
        print("✗ Segment Anything not installed")
    
    # Overall status
    print("\n" + "=" * 40)
    if all_dirs_ok and all_files_ok:
        print("🎉 Setup verification PASSED!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Open main_pipeline.ipynb in Jupyter")
        print("3. Run the environment setup cells")
        print("4. Add your dataset to dataset/ directory")
    else:
        print("⚠️ Setup verification FAILED!")
        print("\nPlease ensure all required files and directories exist.")
    
    return all_dirs_ok and all_files_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
