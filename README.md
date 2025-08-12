# SAM-based Segmentation with Domain Adaptation (SMGwithDA)

A comprehensive pipeline for generating segmentation masks from bounding boxes using the Segment Anything Model (SAM) with unsupervised domain adaptation. This solution is designed to be generalized across different datasets while optimizing for cluttered forest environment scenarios.

## 🎯 Project Overview

This project implements a sophisticated segmentation pipeline that:
- Uses Meta's SAM (Segment Anything Model) as the foundation
- Applies unsupervised domain adaptation for cross-domain generalization
- Generates high-quality segmentation masks from bounding box annotations
- Optimizes for cluttered forest environments while maintaining generalization capability
- Supports both CUDA GPU acceleration and CPU fallback

## 🏗️ Architecture

The pipeline follows a 9-step process:

1. **Environment Setup** - CUDA verification, dependency management, SAM model loading
2. **Data Ingestion** - Multi-domain dataset loading and preprocessing
3. **Zero-Shot Segmentation** - Initial mask generation using SAM
4. **Feature Extraction** - Deep feature extraction for domain adaptation
5. **Domain Alignment** - Adversarial training for domain confusion
6. **Self-Training** - Iterative pseudo-labeling on target domain
7. **Post-Processing** - CRF and morphological refinement
8. **Validation** - Performance evaluation and early stopping
9. **Inference Pipeline** - Production-ready deployment module

## 📁 Project Structure

```
```
SMGwithDA/
├── src/                          # Source code modules
│   ├── environment_setup.py      # Environment and dependency management (Step 1)
│   ├── sam_setup.py              # SAM model setup and utilities (Step 1)
│   ├── dataset_config.py         # Dataset structure configuration (Step 2)
│   ├── data_preprocessing.py     # Data loading and preprocessing (Step 2)
│   ├── data_visualization.py     # Dataset visualization tools (Step 2)
│   ├── zero_shot_segmentation.py # Initial SAM mask generation (Step 3)
│   ├── feature_extraction.py     # Feature extraction for DA (Step 4)
│   ├── domain_adaptation.py      # Domain alignment module (Step 5)
│   ├── self_training.py          # Iterative self-training (Step 6)
│   ├── post_processing.py        # CRF and morphological ops (Step 7)
│   ├── validation.py             # Evaluation metrics (Step 8)
│   └── inference_pipeline.py     # Final inference module (Step 9)
├── dataset/                      # Dataset directory
│   ├── Dataset/                  # Main forestry dataset
│   │   ├── part_1/              # Dataset part 1 with tree annotations
│   │   │   ├── 2k_dataset/      # High-resolution forest images (4608×3456)
│   │   │   ├── coco_annotations.json # COCO format tree annotations
│   │   │   ├── cropped_boxes/   # Pre-cropped bounding box regions
│   │   │   ├── depth_images/    # Depth information and maps
│   │   │   └── yolo_labels/     # YOLO format labels
│   │   └── part_2/              # Dataset part 2 with additional annotations
│   │       ├── raw_images/      # Additional forest images
│   │       ├── coco_annotations.json # Extended tree annotations
│   │       ├── depth_images/    # Depth data for part 2
│   │       └── yolo_labels/     # YOLO format labels
│   ├── Testing/                  # Test datasets
│   │   └── simulated_images/    # Simulated test data for validation
│   │       ├── images/          # Test images
│   │       ├── coco_annotations.json # Test annotations
│   │       └── yolo_labels/     # Test labels
│   ├── source/                   # Legacy source domain (placeholder)
│   └── target/                   # Legacy target domain (placeholder)
```
├── models/                       # Model checkpoints
│   ├── sam_vit_b_01ec64.pth     # SAM base model checkpoint
│   └── adapted_models/          # Domain-adapted model saves
├── outputs/                      # Generated outputs
│   ├── masks/                   # Generated segmentation masks
│   ├── visualizations/          # Result visualizations
│   └── metrics/                 # Evaluation results
├── logs/                        # Training and inference logs
├── main_pipeline.ipynb          # Main Jupyter notebook workflow
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Navigate to project directory
cd SMGwithDA

# Create virtual environment (recommended)
python -m venv smgda_env
source smgda_env/bin/activate  # On Windows: smgda_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

### 2. Test Current Implementation

```bash
# Test Step 1: Environment Setup
python src/environment_setup.py

# Test Step 2: Data Preprocessing
python test_step2.py

# View dataset configuration
python src/dataset_config.py
```

### 3. Launch the Main Pipeline

```bash
# Start Jupyter notebook
jupyter notebook main_pipeline.ipynb
```

### 4. Run Step-by-Step

The main notebook (`main_pipeline.ipynb`) guides you through each step:
1. **Step 1**: Execute environment setup cells (✅ Complete)
2. **Step 2**: Run data ingestion and preprocessing cells (✅ Complete)
3. **Step 3**: Zero-shot mask generation (Next)
4. Follow remaining steps sequentially

## 💾 Dataset Format

### Current Forestry Dataset Structure
Your dataset is automatically detected with this structure:
```
dataset/
├── Dataset/
│   ├── part_1/2k_dataset/           # 3 forest images (4608×3456)
│   ├── part_2/raw_images/           # 8 forest images with annotations
│   └── coco_annotations.json        # Tree bounding box annotations
├── Testing/simulated_images/        # 4 test images
└── Legacy source/target dirs        # Placeholder directories
```

### Legacy Dataset Structure (for reference)
```
dataset/source/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── annotations/
│   └── annotations.json         # COCO format or custom JSON
└── masks/                       # Optional ground truth masks
    ├── image_001_mask.png
    └── ...
```

### Target Dataset Structure
```
dataset/target/
├── images/
│   ├── target_001.jpg
│   └── ...
└── annotations/
    └── annotations.json         # Bounding boxes only
```

### Annotation Format
Supports COCO format or custom JSON:
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image_001.jpg",
      "width": 1024,
      "height": 768
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "bbox": [x, y, width, height],
      "category_id": 1
    }
  ]
}
```

## 🛠️ Implementation Steps

### ✅ Step 1: Environment Setup (COMPLETED)

**Objective**: Set up the development environment with SAM model and dependencies.

**What was implemented:**
- **Environment Validation**: CUDA/GPU detection, dependency verification
- **SAM Model Management**: Automatic download and loading of SAM checkpoints
- **Project Structure**: Created modular directory structure for scalability
- **Error Handling**: Comprehensive validation and troubleshooting utilities

**Key files created:**
- `src/environment_setup.py` - Environment validation and setup utilities
- `src/sam_setup.py` - SAM model loading and configuration
- `requirements.txt` - Complete dependency management
- `verify_setup.py` - Setup verification script

**Results achieved:**
- ✅ CUDA/GPU support with CPU fallback
- ✅ SAM VIT-B model (350MB) downloaded and configured
- ✅ Project structure ready for modular development
- ✅ Comprehensive error handling and user guidance

---

### ✅ Step 2: Data Ingestion and Preprocessing (COMPLETED)

**Objective**: Load and preprocess the forestry dataset with domain adaptation support.

**What was implemented:**
- **Dataset Configuration**: Smart analysis of complex dataset structure
- **Multi-Domain Support**: Automatic source/target domain configuration
- **COCO Integration**: Native support for COCO format tree annotations
- **Preprocessing Pipeline**: Image resizing, normalization, and augmentation
- **Data Validation**: Comprehensive dataset integrity checking
- **Visualization Tools**: Dataset exploration and debugging utilities

**Key files created:**
- `src/dataset_config.py` - Intelligent dataset structure management
- `src/data_preprocessing.py` - Complete data preprocessing pipeline
- `src/data_visualization.py` - Dataset visualization and exploration tools
- `test_step2.py` - Step 2 verification and testing

**Dataset analysis results:**
```
🌲 Forestry Dataset Detected:
├── main_dataset.part_1: 3 high-resolution forest images (4608×3456)
├── main_dataset.part_2: 8 images with tree annotations
├── testing.simulated: 4 simulated test images
└── Total: 15 images with COCO tree annotations

🎯 Optimal Domain Configuration:
├── Source Domain: main_dataset.part_2 (8 annotated forest images)
├── Target Domain: testing.simulated (4 test images)
└── Primary Category: Trees (forest environment optimization)
```

**Technical achievements:**
- ✅ Automatic dataset discovery and configuration
- ✅ COCO format tree annotations loaded and validated
- ✅ Image preprocessing: 4608×3456 → 512×512 with aspect ratio preservation
- ✅ ImageNet normalization for SAM compatibility
- ✅ Data augmentation pipeline for source domain (flip, color jitter, noise, blur)
- ✅ PyTorch DataLoaders with variable bounding box support
- ✅ Domain adaptation ready with source/target/validation splits

**Results achieved:**
- ✅ 15 forest images processed and ready for training
- ✅ Tree bounding box annotations validated and loaded
- ✅ Preprocessing pipeline optimized for forestry domain
- ✅ Data integrity: 100% valid images with proper annotations
- ✅ Memory-efficient batch processing configured

---

### 🔄 Next Steps:

**Step 3: Zero-Shot Mask Generation** (In Progress)
- Load SAM model for initial mask generation
- Process tree bounding boxes to create segmentation masks
- Establish baseline performance before domain adaptation
- Generate features for domain adaptation pipeline

**Step 4: Feature Extraction** (Planned)
- Extract deep features using SAM encoder
- Prepare features for domain adaptation
- Implement feature dimensionality reduction

**Steps 5-9**: Domain adaptation, self-training, post-processing, validation, and final inference pipeline.

---

## ⚙️ Configuration

### SAM Model Options
- `vit_b`: Base model (~350MB) - Fastest, good for development
- `vit_l`: Large model (~1.2GB) - Better accuracy
- `vit_h`: Huge model (~2.4GB) - Best accuracy

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only (slow)
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Optimal**: 32GB RAM, NVIDIA RTX 3080/4080+ with 12GB+ VRAM

### CUDA Support
The pipeline automatically detects and uses CUDA if available:
- Supports CUDA 11.x and 12.x
- Falls back to CPU if CUDA unavailable
- Multi-GPU support (experimental)

## 🔧 Advanced Usage

### Custom Dataset Integration
```python
from src.data_preprocessing import DataPreprocessor

# Initialize with custom dataset
preprocessor = DataPreprocessor(
    source_path="path/to/source",
    target_path="path/to/target",
    annotation_format="coco"  # or "custom"
)
```

### Model Configuration
```python
from src.sam_setup import create_sam_setup

# Use different SAM model
sam_setup = create_sam_setup(
    model_type='vit_l',  # vit_b, vit_l, vit_h
    device='cuda:0'      # specific GPU
)
```

### Domain Adaptation Parameters
```python
# In domain_adaptation.py
adaptation_config = {
    'learning_rate': 1e-4,
    'adversarial_weight': 0.1,
    'batch_size': 16,
    'num_epochs': 50
}
```

## 📊 Evaluation Metrics

The pipeline tracks multiple metrics:
- **IoU (Intersection over Union)**: Standard segmentation metric
- **mIoU (mean IoU)**: Average IoU across all objects/classes
- **Dice Coefficient**: Alternative overlap metric
- **Boundary F1**: Boundary accuracy metric
- **Domain Confusion**: Domain adaptation effectiveness

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or use smaller SAM model
   export CUDA_VISIBLE_DEVICES=0
   ```

2. **SAM Checkpoint Download Fails**
   ```bash
   # Manual download
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P models/
   ```

3. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install --upgrade segment-anything transformers torch torchvision
   ```

### Performance Optimization

1. **Use Mixed Precision Training**
   ```python
   # Enable in domain adaptation
   use_amp = True
   ```

2. **Optimize Data Loading**
   ```python
   # Increase num_workers
   dataloader = DataLoader(..., num_workers=8)
   ```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI** for the Segment Anything Model
- **PyTorch** team for the deep learning framework
- **Domain adaptation** research community
- **Forest environment** dataset contributors

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review closed issues on GitHub
3. Open a new issue with detailed description
4. Include system info and error logs

---

**Developed with ❤️ for Computer Vision by Kazi Fahim Tahmid** 🚀

