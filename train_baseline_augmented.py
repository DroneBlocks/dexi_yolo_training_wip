#!/usr/bin/env python3
"""
Baseline Experiment: Train on augmented COCO images only
This establishes baseline performance before adding real drone data
"""

from ultralytics import YOLO
import argparse
from pathlib import Path
import sys

def train_baseline(model_size='n', epochs=50, batch_size=4, device='mps'):
    """
    Train baseline model on augmented data only

    Args:
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to use ('cpu', 'cuda', 'mps')
    """

    # Check if augmented dataset exists
    config_file = 'dataset.yaml'
    if not Path(config_file).exists():
        print(f"❌ Dataset config not found: {config_file}")
        print("\nPlease run augment_dataset.py first:")
        print("  python3 augment_dataset.py")
        return None, None

    # Check if train/val directories exist
    train_dir = Path('train/images')
    val_dir = Path('val/images')

    if not train_dir.exists() or not val_dir.exists():
        print(f"❌ Training/validation directories not found")
        print(f"   Train: {train_dir}")
        print(f"   Val: {val_dir}")
        print("\nPlease run augment_dataset.py first")
        return None, None

    train_count = len(list(train_dir.glob('*.jpg')))
    val_count = len(list(val_dir.glob('*.jpg')))

    print("\n" + "="*60)
    print("BASELINE EXPERIMENT: Augmented Data Only")
    print("="*60)
    print(f"📊 Dataset Statistics:")
    print(f"   Training images: {train_count}")
    print(f"   Validation images: {val_count}")
    print(f"\n🔧 Training Configuration:")
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
            data=config_file,
            epochs=epochs,
            imgsz=320,
            batch=batch_size,
            device=device,
            project='results',
            name='baseline_augmented',
            save_period=10,
            patience=20,

            # Light augmentation since dataset is already augmented
            hsv_h=0.01,      # Minimal hue variation
            hsv_s=0.3,       # Light saturation
            hsv_v=0.3,       # Light brightness
            degrees=10,      # Light rotation
            translate=0.1,   # Light translation
            scale=0.2,       # Light scale
            shear=0.0,       # No shear
            perspective=0.0, # No perspective
            flipud=0.0,      # No vertical flip
            fliplr=0.5,      # Horizontal flip OK
            mosaic=0.5,      # Moderate mosaic
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

        model_path = 'results/baseline_augmented/weights/best.pt'
        print(f"\n{'='*60}")
        print(f"✅ BASELINE TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"📁 Model saved: {model_path}")

        # Validation
        print(f"\n🧪 Running validation...")
        trained_model = YOLO(model_path)
        val_results = trained_model.val(data=config_file)

        print(f"\n📊 Baseline Performance:")
        print(f"   mAP50: {val_results.box.map50:.4f}")
        print(f"   mAP50-95: {val_results.box.map:.4f}")

        # Save metrics for comparison
        metrics = {
            'experiment': 'baseline_augmented',
            'map50': float(val_results.box.map50),
            'map50_95': float(val_results.box.map),
            'train_images': train_count,
            'val_images': val_count,
        }

        import json
        metrics_file = Path('results/baseline_augmented/metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n💾 Metrics saved: {metrics_file}")
        print("\n" + "="*60)
        print("NEXT STEP: Collect real drone photos")
        print("="*60)
        print("1. Read: DATA_COLLECTION_GUIDE.md")
        print("2. Collect 30-50 drone photos per class")
        print("3. Run: train_with_real_data.py")
        print("="*60 + "\n")

        return model, results

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    parser = argparse.ArgumentParser(
        description='Train baseline model on augmented data only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 train_baseline_augmented.py
  python3 train_baseline_augmented.py --epochs 100 --device cuda
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

    model, results = train_baseline(
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device
    )

    if model is None:
        sys.exit(1)

if __name__ == "__main__":
    main()
