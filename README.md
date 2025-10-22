# Low-Light Image Enhancement Based on Vision Transformer

> A deep learning project for low-light image enhancement using Vision Transformer (ViT) with global attention mechanism, combined with mask decoder and enhancement network.

## Project Overview

This project implements a low-light image enhancement technique based on Vision Transformer (ViT). By leveraging ViT's global attention mechanism combined with a mask decoder and enhancement network, we achieve intelligent image enhancement for low-light conditions.

### Core Features

- **ViT Feature Extraction**: Frozen pre-trained ViT-Base model for global feature extraction
- **Mask Decoder**: Automatically locates dark regions in images and generates attention masks
- **Enhancement Network**: Mask-based convolutional network that improves brightness while preserving details
- **Composite Loss Function**: L1 loss + VGG16 perceptual loss + dark region enhancement loss

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

This study uses the **[LoLI-Street dataset](https://arxiv.org/abs/2410.09831)** for training and evaluation:

- **Training set**: 30,000 image pairs (low-light and normal-light)
- **Validation set**: 3,000 image pairs
- **Test set**: 10,000 image pairs

For computational efficiency and experimental practicality, this research:

- Uses the complete training set for model training
- Randomly selects 200 image pairs from the validation set for model validation
- Further selects 200 image pairs as the final test set

## Environment Dependencies

```bash
# Core dependencies
pip install torch torchvision transformers opencv-python albumentations tensorboard
```

**Main libraries**:

- PyTorch >= 1.9
- Transformers (Hugging Face)
- OpenCV
- Albumentations (data augmentation)
- TensorBoard (training visualization)

## Quick Start

### 1. Data Preparation

Organize the dataset in the following structure:

```
data/LOL_dataset/
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
    data_root = "./data/LOL_dataset"  # Dataset path
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

## Project Structure

```
.
├── config.py                      # Configuration file
├── model.py                       # DarkEnhancer model definition
├── dataset.py                     # CustomLowLightDataset data loader
├── utils.py                       # LossCalculator composite loss function
├── train.py                       # Training script
├── infer2.py                      # Inference script
├── vit-base-patch16-224-in21k/   # ViT pre-trained model directory
├── data/                          # Dataset directory
│   └── LOL_dataset/
│       ├── train/
│       └── val/
├── checkpoints/                   # Model checkpoints
├── runs/                          # TensorBoard logs
├── test/                          # Test image input
└── output/                        # Enhanced result output
```

**Note**: The pre-trained model `vit-base-patch16-224-in21k` needs to be downloaded from [Hugging Face](https://huggingface.co/google/vit-base-patch16-224-in21k) and placed in the project root directory, or it will be automatically downloaded on first run.

## Experimental Results

| Method         | NIQE ↓   | PSNR ↑    | SSIM ↑   | LPIPS ↓  |
| -------------- | -------- | --------- | -------- | -------- |
| **ViT (Ours)** | **3.36** | **26.64** | **0.82** | **0.17** |

## License

This project is open source under the MIT License. See [LICENSE](LICENSE) file for details.

## Evaluation Metrics

- **NIQE** (Natural Image Quality Evaluator): Lower is better, measures deviation from natural image statistics
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better, measures signal quality vs noise
- **SSIM** (Structural Similarity Index): Higher is better (closer to 1), measures structural similarity
- **LPIPS** (Learned Perceptual Image Patch Similarity): Lower is better, measures perceptual similarity
- **BRISQUE**: Lower is better, blind/referenceless image spatial quality evaluator
- **PIQE**: Lower is better, perception-based image quality evaluator
