# GPU Training Setup Guide

This guide explains how to set up CUDA-enabled PyTorch for GPU training with YOLOv8 on your system.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA drivers installed
- Python 3.8 or higher

## Quick Setup (Automated)

Run the automated setup script:

```bash
./setup_gpu_venv.bat
```

This will:
- Create a clean virtual environment
- Install PyTorch with CUDA 12.9 support
- Install all required ML packages
- Verify the GPU setup

## Manual Setup

### 1. Create Virtual Environment

```bash
# Create a new virtual environment
python -m venv venv_gpu

# Activate it (Windows)
venv_gpu\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install PyTorch with CUDA Support

**Important**: Install PyTorch with CUDA FIRST before other packages:

```bash
pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 --index-url https://download.pytorch.org/whl/cu129
```

### 3. Install Other Requirements

```bash
pip install ultralytics opencv-python Pillow numpy pandas matplotlib tqdm PyYAML
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"
```

Expected output:
```
PyTorch version: 2.8.0+cu129
CUDA available: True
CUDA device count: 1
```

## Usage

### Activate Environment

Every time you want to train, activate the environment first:

```bash
venv_gpu\Scripts\activate
```

### Run Training

```bash
# Quick test (1 epoch)
python train_yolo.py --model n --epochs 1 --device cuda

# Full training (100 epochs)
python train_yolo.py --model n --epochs 100 --device cuda

# Different model sizes
python train_yolo.py --model s --epochs 50 --device cuda  # Small model
python train_yolo.py --model m --epochs 50 --device cuda  # Medium model
```

### Monitor GPU Usage

During training, you can monitor GPU usage with:

```bash
nvidia-smi
```

## Troubleshooting

### "CUDA not available" Error

If you get `torch.cuda.is_available(): False`:

1. **Check PyTorch installation**:
   ```bash
   pip show torch
   ```
   Should show version with `+cu129`, not `+cpu`

2. **Reinstall PyTorch with CUDA**:
   ```bash
   pip uninstall torch torchvision -y
   pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 --index-url https://download.pytorch.org/whl/cu129
   ```

3. **Verify CUDA drivers**: Run `nvidia-smi` to check if CUDA drivers are working

### RTX 5080 Compatibility Warning

If you see warnings about RTX 5080 compute capability, the training will still work but may not be fully optimized. The warning is informational only.

### Version Conflicts

If you encounter package conflicts:

1. Create a fresh virtual environment
2. Install PyTorch with CUDA FIRST
3. Then install other packages

## Performance Tips

- **Batch Size**: Start with 16, increase if you have enough GPU memory
- **Mixed Precision**: Enabled by default (AMP) for better performance
- **GPU Memory**: Monitor with `nvidia-smi` during training
- **Model Size**: Use `n` (nano) for fastest training, `s`/`m` for better accuracy

## Hardware Info

- **GPU**: NVIDIA GeForce RTX 5080 (16GB VRAM)
- **CUDA Version**: 12.9
- **PyTorch Version**: 2.8.0+cu129

## Support

If you encounter issues:

1. Check that your CUDA drivers are up to date
2. Verify PyTorch installation shows `+cu129` version
3. Test with a simple CUDA operation:
   ```python
   import torch
   x = torch.randn(5, 3)
   if torch.cuda.is_available():
       x = x.cuda()
       print("CUDA tensor created successfully!")
   ```

## Deactivate Environment

When finished training:

```bash
deactivate
```