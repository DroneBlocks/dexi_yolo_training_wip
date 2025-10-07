#!/usr/bin/env python3
"""
Fine-tuning Experiment: Train on augmented + real drone data
This demonstrates the value of domain-specific data collection
"""

from ultralytics import YOLO
import argparse
from pathlib import Path
import sys
import shutil
import yaml

def create_combined_dataset():
    """
    Create a combined dataset with augmented + real drone photos
    Returns path to new dataset config file
    """
    # Check if real drone photos exist
    real_photos_dir = Path('source_data/real_drone_photos')
    if not real_photos_dir.exists():
        print(f"❌ Real drone photos directory not found: {real_photos_dir}")
        print("\nPlease collect real drone photos first!")
        print("See DATA_COLLECTION_GUIDE.md for instructions")
        return None

    # Count real photos per class (standard YOLO structure: class/images/)
    classes = ['bird', 'car', 'cat', 'dog', 'motorcycle', 'truck']
    real_photo_counts = {}

    print("\n📸 Real drone photos found:")
    for cls in classes:
        cls_dir = real_photos_dir / cls / 'images'
        if cls_dir.exists():
            count = len(list(cls_dir.glob('*.jpg'))) + len(list(cls_dir.glob('*.JPG')))
            real_photo_counts[cls] = count
            print(f"   {cls}: {count} photos")
        else:
            real_photo_counts[cls] = 0
            print(f"   {cls}: 0 photos (missing)")

    total_real = sum(real_photo_counts.values())
    if total_real == 0:
        print("\n❌ No real drone photos found!")
        print("Please collect photos according to DATA_COLLECTION_GUIDE.md")
        return None

    print(f"\n   Total real photos: {total_real}")

    # Create combined dataset directory
    combined_dir = Path('combined')
    combined_train_img = combined_dir / 'train' / 'images'
    combined_train_lbl = combined_dir / 'train' / 'labels'
    combined_val_img = combined_dir / 'val' / 'images'
    combined_val_lbl = combined_dir / 'val' / 'labels'

    for dir in [combined_train_img, combined_train_lbl, combined_val_img, combined_val_lbl]:
        dir.mkdir(parents=True, exist_ok=True)

    # Copy augmented dataset (from baseline)
    print("\n📦 Creating combined dataset...")
    print("   Copying augmented images...")

    aug_train_img = Path('../train/images')
    aug_train_lbl = Path('../train/labels')
    aug_val_img = Path('../val/images')
    aug_val_lbl = Path('../val/labels')

    # Copy training images/labels
    if aug_train_img.exists():
        for img in aug_train_img.glob('*.jpg'):
            shutil.copy(img, combined_train_img / img.name)
        for lbl in aug_train_lbl.glob('*.txt'):
            shutil.copy(lbl, combined_train_lbl / lbl.name)

    # Copy validation images/labels
    if aug_val_img.exists():
        for img in aug_val_img.glob('*.jpg'):
            shutil.copy(img, combined_val_img / img.name)
        for lbl in aug_val_lbl.glob('*.txt'):
            shutil.copy(lbl, combined_val_lbl / lbl.name)

    print("   Adding real drone photos to training set...")

    # Class to ID mapping (must match dataset.yaml)
    class_to_id = {
        'car': 0,
        'motorcycle': 1,
        'truck': 2,
        'bird': 3,
        'cat': 4,
        'dog': 5,
    }

    # Copy real photos with labels (standard YOLO structure)
    real_added = 0
    for cls in classes:
        cls_dir = real_photos_dir / cls
        if not cls_dir.exists():
            continue

        class_id = class_to_id[cls]

        # Standard YOLO structure: class/images/ and class/labels/
        cls_images = cls_dir / 'images'
        cls_labels = cls_dir / 'labels'

        if not cls_images.exists():
            print(f"   Warning: {cls}/images/ not found, skipping")
            continue

        for img in list(cls_images.glob('*.jpg')) + list(cls_images.glob('*.JPG')):
            # Copy image
            new_name = f"{cls}_real_{img.stem}.jpg"
            shutil.copy(img, combined_train_img / new_name)

            # Copy corresponding label if it exists
            label_file = cls_labels / f"{img.stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, combined_train_lbl / f"{cls}_real_{img.stem}.txt")
                real_added += 1
            else:
                print(f"   Warning: No label for {img.name}, skipping")

    print(f"   Added {real_added} real drone photos to training set")

    # Create dataset config
    config = {
        'path': str(combined_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 6,
        'names': {
            0: 'car',
            1: 'motorcycle',
            2: 'truck',
            3: 'bird',
            4: 'cat',
            5: 'dog',
        }
    }

    config_file = combined_dir / 'dataset.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Print dataset statistics
    train_count = len(list(combined_train_img.glob('*.jpg')))
    val_count = len(list(combined_val_img.glob('*.jpg')))

    print(f"\n✅ Combined dataset created!")
    print(f"   Training images: {train_count} (augmented + {real_added} real)")
    print(f"   Validation images: {val_count}")
    print(f"   Config: {config_file}")

    return config_file

def train_with_real_data(model_size='n', epochs=50, batch_size=4, device='mps'):
    """
    Train model with augmented + real drone data

    Args:
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to use ('cpu', 'cuda', 'mps')
    """

    # Create combined dataset
    config_file = create_combined_dataset()
    if config_file is None:
        return None, None

    print("\n" + "="*60)
    print("FINE-TUNING EXPERIMENT: Augmented + Real Drone Data")
    print("="*60)
    print(f"🔧 Training Configuration:")
    print(f"   Model: YOLOv8{model_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    print(f"   Fine-tuning: YES (starting from COCO pretrained weights)")
    print("="*60 + "\n")

    # Load pretrained COCO model (fine-tuning approach)
    model = YOLO(f'yolov8{model_size}.pt')

    try:
        # Train the model
        results = model.train(
            data=str(config_file),
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            device=device,
            project='results',
            name='with_real_data',
            save_period=10,
            patience=20,

            # Light augmentation (data is diverse already)
            # Increased color augmentation to handle building color variations
            hsv_h=0.05,      # Moderate hue variation (building colors)
            hsv_s=0.4,       # Moderate saturation (different surfaces)
            hsv_v=0.4,       # Moderate brightness (lighting/color changes)
            degrees=5,       # Minimal rotation (drone is stable)
            translate=0.05,  # Minimal translation
            scale=0.1,       # Minimal scale
            shear=0.0,       # No shear
            perspective=0.0, # No perspective
            flipud=0.0,      # No vertical flip
            fliplr=0.0,      # No horizontal flip (drone view is consistent)
            mosaic=0.3,      # Light mosaic
            mixup=0.0,       # No mixup
            copy_paste=0.0,  # No copy-paste

            # Fine-tuning optimization (lower learning rate)
            optimizer='AdamW',
            lr0=0.001,       # Lower learning rate for fine-tuning
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,

            verbose=True,
        )

        model_path = 'results/with_real_data/weights/best.pt'
        print(f"\n{'='*60}")
        print(f"✅ FINE-TUNING COMPLETED!")
        print(f"{'='*60}")
        print(f"📁 Model saved: {model_path}")

        # Validation
        print(f"\n🧪 Running validation...")
        trained_model = YOLO(model_path)
        val_results = trained_model.val(data=str(config_file))

        print(f"\n📊 Fine-tuned Performance:")
        print(f"   mAP50: {val_results.box.map50:.4f}")
        print(f"   mAP50-95: {val_results.box.map:.4f}")

        # Save metrics for comparison
        train_count = len(list((Path(config_file).parent / 'train' / 'images').glob('*.jpg')))
        val_count = len(list((Path(config_file).parent / 'val' / 'images').glob('*.jpg')))

        metrics = {
            'experiment': 'with_real_data',
            'map50': float(val_results.box.map50),
            'map50_95': float(val_results.box.map),
            'train_images': train_count,
            'val_images': val_count,
        }

        import json
        metrics_file = Path('results/with_real_data/metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n💾 Metrics saved: {metrics_file}")
        print("\n" + "="*60)
        print("NEXT STEP: Compare results")
        print("="*60)
        print("Run: python3 compare_experiments.py")
        print("="*60 + "\n")

        return model, results

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    parser = argparse.ArgumentParser(
        description='Train with augmented + real drone data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 train_with_real_data.py
  python3 train_with_real_data.py --epochs 100 --device cuda

Prerequisites:
  1. Collect real drone photos (see DATA_COLLECTION_GUIDE.md)
  2. Place photos in source_data/real_drone_photos/<class>/
        """
    )
    parser.add_argument('--model', '-m', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (default: n)')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch', '-b', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--device', '-d', type=str, default='mps',
                       help='Device to use: cpu, cuda, mps (default: mps)')

    args = parser.parse_args()

    model, results = train_with_real_data(
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device
    )

    if model is None:
        sys.exit(1)

if __name__ == "__main__":
    main()
