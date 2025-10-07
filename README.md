# Drone Domain Adaptation Experiments

**Objective**: Demonstrate the value of real-world data collection for fine-tuning YOLOv8 to detect objects from drone cameras.

**Status**: Self-contained repository with all scripts and data for running the complete experiment.

## Problem Statement

Your drone can detect printed images when held up to the camera, but struggles when images are placed in recessed buildings. This is a **domain gap problem** - augmented COCO images don't capture the real-world conditions of drone-view detection.

## Solution: Fine-tuning vs Transfer Learning

This is a **fine-tuning** problem, not transfer learning:
- Your 6 classes (bird, car, cat, dog, motorcycle, truck) exist in COCO
- You're adapting pretrained weights to a new **domain** (drone-view, recessed buildings)
- Strategy: Start with `yolov8n.pt` and continue training on your specific data

## Experiment Design

### Experiment 1: Baseline (Augmented Data Only)
- Train on augmented COCO images
- Establishes baseline performance
- Shows limits of synthetic augmentation

### Experiment 2: Fine-tuned (Augmented + Real Drone Data)
- Train on augmented + real drone photos (30-50 per class)
- Demonstrates value of domain-specific data
- Expected to significantly outperform baseline

### Expected Outcome
Real drone data should improve mAP by 20-40%, proving that **matching training distribution to deployment conditions** is critical.

## Workflow

This repository provides a complete **domain adaptation experiment** for drone-based object detection.

### Complete Workflow (From Scratch)

1. **Generate augmented baseline dataset**
   ```bash
   python3 augment_dataset.py
   ```

2. **Train baseline** (augmented data only)
   ```bash
   python3 train_baseline_augmented.py --epochs 50 --device mps
   ```

3. **Collect real drone photos** (see `DATA_COLLECTION_GUIDE.md`)
   - 30-50 photos per class
   - Vary height, lighting, positioning
   - Store in `source_data/raw_drone_photos/<class>/`

4. **Label your drone photos**
   ```bash
   python3 label_images.py source_data/raw_drone_photos/<class> --class <class>
   # Labels are saved automatically in the same directory
   # Then copy to source_data/real_drone_photos/<class>/
   ```

5. **Train with real data**
   ```bash
   python3 train_with_real_data.py --epochs 50 --device mps
   ```

6. **Compare results**
   ```bash
   python3 compare_experiments.py
   ```

### Current Workflow (Resume from where you are)

Since you've already trained with real data, just run:

1. **Generate augmented dataset** (if not done yet)
   ```bash
   python3 augment_dataset.py
   ```

2. **Train baseline**
   ```bash
   python3 train_baseline_augmented.py --epochs 50 --device mps
   ```

3. **Compare metrics**
   ```bash
   python3 compare_experiments.py
   ```

4. **Visual comparison on real drone photos**

   Test both models on your real drone photos to see which performs better:

   ```bash
   # Test baseline (augmented only) on real cat photos
   yolo predict model=results/baseline_augmented/weights/best.pt \
       source=source_data/real_drone_photos/cat/images \
       conf=0.25 save=True project=results/predictions name=baseline_cats

   # Test fine-tuned (augmented + real) on real cat photos
   yolo predict model=results/with_real_data/weights/best.pt \
       source=source_data/real_drone_photos/cat/images \
       conf=0.25 save=True project=results/predictions name=finetuned_cats

   # Test on dog photos
   yolo predict model=results/baseline_augmented/weights/best.pt \
       source=source_data/real_drone_photos/dog/images \
       conf=0.25 save=True project=results/predictions name=baseline_dogs

   yolo predict model=results/with_real_data/weights/best.pt \
       source=source_data/real_drone_photos/dog/images \
       conf=0.25 save=True project=results/predictions name=finetuned_dogs
   ```

   **Compare outputs:**
   - `results/predictions/baseline_cats/` vs `results/predictions/finetuned_cats/`
   - `results/predictions/baseline_dogs/` vs `results/predictions/finetuned_dogs/`

   **What to look for:**
   - Does the baseline model miss detections or misclassify objects?
   - Does the fine-tuned model have better confidence scores?
   - Which model handles drone perspective, lighting, and shadows better?
   - Count false positives and false negatives for each model

   This visual comparison will show the **real-world impact** of domain-specific data collection!

## Teaching Framework

### For Student Teams

**Phase 1: Understanding** (Week 1)
1. Predict: Which approach will work better?
2. Discuss: Why does domain gap matter?
3. Hypothesis: How much improvement will real data provide?

**Phase 2: Data Collection** (Week 2)
1. Protocol: Height variations, lighting conditions, positioning
2. Collect: 10-20 photos per class per team
3. Quality check: Focus, visibility, metadata

**Phase 3: Training & Evaluation** (Week 3)
1. Run baseline experiment
2. Run fine-tuned experiment
3. Compare results
4. Analyze: What worked? What didn't?

**Phase 4: Iteration** (Week 4)
1. Identify failure cases
2. Collect targeted data
3. Re-train and measure improvement
4. Document learnings

### Key Learning Objectives

1. **Data Distribution Matching**
   - Training data must match deployment conditions
   - Synthetic augmentation has limits
   - Real data captures nuances (shadows, perspective, lighting)

2. **Fine-tuning Strategy**
   - Leverage pretrained knowledge (COCO weights)
   - Use lower learning rate (0.001)
   - Combine synthetic + real data

3. **Iterative Development**
   - Measure baseline first
   - Collect targeted data
   - Re-train and evaluate
   - Repeat until satisfactory

4. **Data Efficiency**
   - Quality > Quantity
   - 30-50 well-chosen photos can significantly improve performance
   - Focus on domain-specific edge cases

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
- `train_baseline_augmented.py` - Baseline experiment script
- `train_with_real_data.py` - Fine-tuning experiment script
- `compare_experiments.py` - Results comparison script
- `source_data/` - Data storage directory
- `results/` - Training results and models

## Expected Results

| Experiment | mAP50 | mAP50-95 | Notes |
|------------|-------|----------|-------|
| Baseline (augmented) | ~0.60-0.70 | ~0.40-0.50 | Synthetic data only |
| Fine-tuned (+ real) | ~0.80-0.90 | ~0.60-0.70 | With domain data |
| **Improvement** | **+20-30%** | **+30-50%** | Real data impact |

## Troubleshooting

**Q: Baseline won't run**
- Ensure `augment_dataset.py` has been run first
- Check that `train/images` and `val/images` exist

**Q: No real photos found**
- Check directory structure: `source_data/real_drone_photos/<class>/`
- Ensure images are `.jpg` format

**Q: Performance didn't improve**
- Need more real photos (aim for 50+ per class)
- Check real photo quality (focus, visibility)
- Verify labels are accurate
- Ensure photos match deployment scenario (recessed buildings!)

**Q: Training is slow**
- Reduce batch size: `--batch 2`
- Use smaller model: `--model n`
- Reduce epochs: `--epochs 30`

## Quick Start

### Step 1: Generate Augmented Dataset

First, generate the baseline augmented dataset from the 6 original images:

```bash
python3 augment_dataset.py
```

This creates:
- `train/images/` - 240 augmented training images (40 per class)
- `train/labels/` - Corresponding YOLO labels
- `val/images/` - 240 augmented validation images (40 per class)
- `val/labels/` - Corresponding YOLO labels

### Step 2: Run Baseline Experiment

Train on augmented data only to establish baseline:

```bash
python3 train_baseline_augmented.py --epochs 50 --device mps
```

**Output:**
- Model: `results/baseline_augmented/weights/best.pt`
- Metrics: `results/baseline_augmented/metrics.json`

### Step 3: Compare Results

Compare baseline (augmented only) vs fine-tuned (augmented + real drone data):

```bash
python3 compare_experiments.py
```

This will show the performance improvement from adding real drone photos!

## Current Status

1. ✅ Real drone photos collected (259 images: 107 cats, 152 dogs)
2. ✅ Images labeled and organized in `source_data/real_drone_photos/`
3. ✅ Fine-tuned model trained: `results/with_real_data/weights/best.pt`
   - **mAP50:** 76.2%
   - **mAP50-95:** 75.2%
   - **Cat accuracy:** 85%
   - **Dog accuracy:** 87%
4. ✅ Baseline model trained: `results/baseline_augmented/weights/best.pt`
   - **mAP50:** 99.5% (on augmented validation set)
5. ⏳ Visual comparison on real drone photos (see workflow above)
6. ⏳ Deploy best model to Raspberry Pi
7. ⏳ Iterate based on failure cases

## Optional: Label Additional Drone Photos

If you want to add more real drone photos for other classes:

```bash
# Label images for any class
python3 label_images.py source_data/raw_drone_photos/<class_name> --class <class_name>

# Copy to real_drone_photos
cp source_data/raw_drone_photos/<class_name>/images/* source_data/real_drone_photos/<class_name>/images/
cp source_data/raw_drone_photos/<class_name>/labels/* source_data/real_drone_photos/<class_name>/labels/

# Re-train with updated data
python3 train_with_real_data.py --epochs 50 --device mps
```

---

**Remember**: The goal is not just to improve the model, but to teach students the **systematic process** of domain adaptation through data collection.
