# Drone Object Detection Training

**Objective**: Train YOLOv8 to detect objects from drone cameras using a blend of augmented and real-world data.

**Status**: Self-contained repository with all scripts and data for running complete training workflow.

## Problem Statement

Your drone can detect printed images when held close to the camera, but struggles when images are placed in recessed buildings. This is a **domain gap problem** - pretrained models don't fully capture the real-world conditions of drone-view detection (perspective, lighting, shadows, distance).

## Solution: Blending Augmented and Real Data

This project uses **fine-tuning** with a hybrid dataset:
- Your 6 classes (bird, car, cat, dog, motorcycle, truck) exist in COCO
- Start with `yolov8n.pt` pretrained weights
- Train on **both augmented COCO images** and **real drone photos**

### Why Blend Data?

**Augmented Data Benefits:**
- Large quantity (hundreds of images per class)
- Diverse object appearances and backgrounds
- Provides strong foundation for object recognition

**Real Drone Data Benefits:**
- Captures deployment conditions (drone perspective, recessed buildings)
- Real-world lighting, shadows, and environmental factors
- Addresses domain-specific edge cases

**Combined Effect:**
Training with both types of data allows the model to leverage general object recognition from COCO while adapting to the specific conditions of drone deployment. Just 30-50 real drone photos per class can significantly improve performance by teaching the model to handle real-world variations that augmentation cannot fully replicate.

## Training Workflow

### Step 1: Generate Augmented Dataset

Create baseline augmented dataset from the 6 original images:

```bash
python3 augment_dataset.py
```

This creates:
- `train/images/` - 240 augmented training images (40 per class)
- `train/labels/` - Corresponding YOLO labels
- `val/images/` - 240 augmented validation images (40 per class)
- `val/labels/` - Corresponding YOLO labels

### Step 2: Collect Real Drone Photos

See `DATA_COLLECTION_GUIDE.md` for detailed protocol.

**Guidelines:**
- 30-50 photos per class minimum
- Vary height, lighting, positioning
- Capture deployment conditions (recessed buildings, various distances)
- Store in `source_data/raw_drone_photos/<class>/`

### Step 3: Label Your Drone Photos

```bash
python3 label_images.py source_data/raw_drone_photos/<class> --class <class>
# Labels are saved automatically in the same directory
# Then copy to source_data/real_drone_photos/<class>/
```

### Step 4: Train with Combined Dataset

Train on augmented + real data together:

```bash
python3 train_with_real_data.py --epochs 50 --device mps
```

**Parameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--device`: Device to use (`mps` for Mac, `cuda` for NVIDIA GPU, `cpu` for CPU)
- `--batch`: Batch size (default: 8, increase to 16-32 for GPU)

**Output:**
- Model: `results/with_real_data/weights/best.pt`
- Training curves: `results/with_real_data/results.png`
- Validation metrics: Shown during training

### Step 5: Validate Your Model

Test your trained model on real drone photos to verify performance:

```bash
# Test on cat photos
yolo predict model=results/with_real_data/weights/best.pt \
    source=source_data/real_drone_photos/cat/images \
    conf=0.25 save=True project=results/predictions name=cats

# Test on dog photos
yolo predict model=results/with_real_data/weights/best.pt \
    source=source_data/real_drone_photos/dog/images \
    conf=0.25 save=True project=results/predictions name=dogs
```

**Review outputs:**
- Check `results/predictions/cats/` and `results/predictions/dogs/`
- Look for missed detections or misclassifications
- Verify confidence scores are reasonable (>0.5 for good detections)
- Identify failure cases for additional data collection

### Step 6: Iterate Based on Failure Cases

1. **Identify failure cases** - Which objects are missed or misclassified?
2. **Collect targeted data** - Gather more photos similar to failure cases
3. **Re-label and train** - Add new photos and retrain
4. **Validate again** - Check if failures are resolved

## Model Architecture: YOLOv8 vs YOLOv11

**Recommendation: Stick with YOLOv8**

Why:
- ✅ Proven ONNX export for Raspberry Pi
- ✅ Stable, well-documented (better for teaching)
- ✅ Performance difference minimal for this use case
- ✅ Focus should be on **data quality**, not architecture

YOLOv11 offers marginal improvements, but your bottleneck is data, not model architecture.

## Deployment Pipeline

After training:

```python
# Convert to ONNX for Raspberry Pi
from ultralytics import YOLO

model = YOLO('results/with_real_data/weights/best.pt')
model.export(format='onnx', imgsz=640)
```

Use in ROS2 node:
```python
import onnxruntime as ort
# Load and run inference on Raspberry Pi
```

## Files

- `README.md` - This file
- `DATA_COLLECTION_GUIDE.md` - Photo collection protocol
- `train_with_real_data.py` - Training script with combined dataset
- `augment_dataset.py` - Generate augmented COCO images
- `label_images.py` - Interactive labeling tool
- `source_data/` - Data storage directory
- `results/` - Training results and models

## Understanding Training Metrics

During training, you'll see these key metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| mAP50 | Mean Average Precision at 50% IoU | >0.75 is good |
| mAP50-95 | mAP averaged across IoU thresholds | >0.60 is good |
| Precision | How many detections are correct | >0.80 |
| Recall | How many objects are detected | >0.75 |

**What good performance looks like:**
- mAP50: 75-90%
- mAP50-95: 60-75%
- Consistent detection in validation images
- High confidence scores (>0.5) on correct detections

## Troubleshooting

**Q: No real photos found**
- Check directory structure: `source_data/real_drone_photos/<class>/`
- Ensure images are `.jpg` format
- Verify both `images/` and `labels/` subdirectories exist

**Q: Performance is poor**
- Need more real photos (aim for 50+ per class)
- Check real photo quality (focus, visibility)
- Verify labels are accurate
- Ensure photos match deployment scenario (recessed buildings!)
- Try more training epochs (100+)

**Q: Training is slow**
- Reduce batch size: `--batch 2`
- Use smaller model: `yolov8n` (nano)
- Reduce epochs: `--epochs 30`
- Use GPU if available (much faster!)

**Q: Model works on some objects but not others**
- Classes with fewer real photos will perform worse
- Collect more photos for underperforming classes
- Ensure photo variety (different lighting, angles, distances)

## Windows + NVIDIA GPU Setup

### Prerequisites

- Windows PC with NVIDIA GPU (CUDA support)
- CUDA drivers installed (run `nvidia-smi` to verify)
- Python 3.8 or higher
- Git installed

### Setup Instructions

```bash
# 1. Clone repository
git clone https://github.com/DroneBlocks/dexi_yolo_training.git
cd dexi_yolo_training

# 2. Create virtual environment
python -m venv venv_gpu
venv_gpu\Scripts\activate

# 3. Install PyTorch with CUDA FIRST
pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 --index-url https://download.pytorch.org/whl/cu129

# 4. Install other requirements
pip install -r requirements-gpu.txt

# 5. Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Training on GPU

```bash
# Generate dataset
python augment_dataset.py

# Train with GPU acceleration
python train_with_real_data.py --epochs 50 --device cuda --batch 16
```

**GPU Training Tips:**
- Monitor usage: `nvidia-smi` in another terminal
- Increase batch size to 16-32 for faster training
- GPU training is 5-10x faster than CPU
- 50 epochs typically takes 10-20 minutes on RTX 3060+

## Current Project Status

1. ✅ Augmented dataset generated (480 images total)
2. ✅ Real drone photos collected (259 images: 107 cats, 152 dogs)
3. ✅ Images labeled and organized in `source_data/real_drone_photos/`
4. ✅ Model trained with combined data: `results/with_real_data/weights/best.pt`
   - **mAP50:** 76.2%
   - **mAP50-95:** 75.2%
   - **Cat precision:** 85%
   - **Dog precision:** 87%
5. ⏳ Validate on additional real drone scenarios
6. ⏳ Deploy to Raspberry Pi via ONNX
7. ⏳ Iterate based on failure cases

## Adding More Classes

To add real drone photos for other classes (bird, car, motorcycle, truck):

```bash
# 1. Collect photos for the class
# Store in source_data/raw_drone_photos/<class_name>/

# 2. Label images
python3 label_images.py source_data/raw_drone_photos/<class_name> --class <class_name>

# 3. Copy to training directory
cp source_data/raw_drone_photos/<class_name>/images/* source_data/real_drone_photos/<class_name>/images/
cp source_data/raw_drone_photos/<class_name>/labels/* source_data/real_drone_photos/<class_name>/labels/

# 4. Re-train with updated data
python3 train_with_real_data.py --epochs 50 --device mps
```

---

**Remember**: The goal is to teach students the **systematic process** of domain adaptation through strategic data collection and iterative improvement.
