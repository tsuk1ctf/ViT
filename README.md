# Low-Light Image Enhancement Based on Vision Transformer

## Project Overview

A low-light image enhancement framework built on a frozen Vision Transformer (ViT) backbone, paired with a lightweight mask decoder for dark-region focus and an enhancement network for feature transformation.

### Core Features

- **ViT Feature Extraction**: Frozen pre-trained ViT-Base model for global feature extraction
- **Mask Decoder**: Automatically locates dark regions in images and generates attention masks
- **Enhancement Network**: Mask-based convolutional network that improves brightness while preserving details
- **Composite Loss Function**: L1 loss + VGG16 perceptual loss + dark region MSE loss, with optimal weights α=0.20, β=0.20, γ=0.60 (Dark Region Dominant)

## Model Architecture

```
Input Image (224×224)
    ↓
ViT-Base Feature Extraction [Frozen]
    ↓
Mask Decoder → Attention Mask
    ↓
RGB + Mask → Enhancement Network
    ↓
Enhanced Image Output
```

## Dataset

This study uses the **[LoLI-Street dataset](https://arxiv.org/abs/2410.09831)** for training and evaluation.

## Environment Dependencies

```bash
# Core dependencies (training & inference)
pip install torch torchvision transformers opencv-python albumentations tensorboard

# Evaluation dependencies (eval_fr.py / eval_nr.py)
pip install pyiqa lpips scikit-image natsort termcolor Pillow
```

**Main libraries**:

- PyTorch >= 1.9
- Transformers (Hugging Face)
- OpenCV
- Albumentations (data augmentation)
- TensorBoard (training visualization)
- pyiqa (NIQE / BRISQUE / PIQE)
- lpips (perceptual similarity)
- scikit-image (SSIM)

## Quick Start

### 1. Data Preparation

Organize the dataset in the following structure:

```
data/LoLI-Street/
├── train/
│   ├── low/    # Low-light images
│   └── high/   # Normal-light images
└── val/
    ├── low/
    └── high/
```

### 2. Configuration Parameters

Edit `config.py` to adjust training parameters:

```python
class Config:
    data_root = "./data/LoLI-Street"  # Dataset path
    input_size = 224                  # Input size
    batch_size = 8                    # Batch size
    epochs = 100                      # Training epochs
    lr = 1e-4                        # Learning rate
```

### 3. Model Training

```bash
python train.py

# View training logs
tensorboard --logdir=./runs
```

### 4. Inference Enhancement

```bash
python infer2.py
```

Modify path parameters in `infer2.py`:

```python
input_folder = "./test"      # Input folder
output_folder = "./output"   # Output folder
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

### 5. Quality Evaluation

Two evaluation scripts are provided, both unify image size to 512×512 by default for fair comparison.

**Full-Reference metrics** (require ground-truth images): PSNR / SSIM / LPIPS

```bash
python eval_fr.py --pred ./output --gt ./test_gt --use_lpips --output ./results_fr.csv
```

**No-Reference metrics** (no ground-truth needed): NIQE / BRISQUE / PIQE

```bash
python eval_nr.py --img ./output --output ./results_nr.csv
```

Add `--size 0` to keep original image size if needed.

## Project Structure

```
ViT-LLIE/
├── config.py                      # Configuration file
├── model.py                       # DarkEnhancer model definition
├── dataset.py                     # CustomLowLightDataset data loader
├── utils.py                       # LossCalculator composite loss function
├── train.py                       # Training script
├── infer2.py                      # Inference script
├── eval_fr.py                     # Full-Reference evaluation (PSNR / SSIM / LPIPS)
├── eval_nr.py                     # No-Reference evaluation (NIQE / BRISQUE / PIQE)
├── vit-base-patch16-224-in21k/   # ViT pre-trained model directory
├── data/                          # Dataset directory
│   └── LoLI-Street/
│       ├── train/
│       └── val/
├── checkpoints/                   # Model checkpoints
├── runs/                          # TensorBoard logs
├── test/                          # Test image input
└── output/                        # Enhanced result output
```

**Note**: The pre-trained model `vit-base-patch16-224-in21k` needs to be downloaded from [Hugging Face](https://huggingface.co/google/vit-base-patch16-224-in21k) and placed in the project root directory, or it will be automatically downloaded on first run.

## License

This project is open source under the MIT License. See [LICENSE](LICENSE) file for details.

## Evaluation Metrics

- **NIQE** (Natural Image Quality Evaluator): Lower is better, measures deviation from natural image statistics
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better, measures signal quality vs noise
- **SSIM** (Structural Similarity Index): Higher is better (closer to 1), measures structural similarity
- **LPIPS** (Learned Perceptual Image Patch Similarity): Lower is better, measures perceptual similarity
- **BRISQUE**: Lower is better, blind/referenceless image spatial quality evaluator
- **PIQE**: Lower is better, perception-based image quality evaluator
