#!/usr/bin/env python3
"""
Compare baseline (augmented only) vs fine-tuned (augmented + real data)
Demonstrates the value of domain-specific data collection
"""

import json
from pathlib import Path
from ultralytics import YOLO
import argparse

def load_metrics(experiment_name):
    """Load metrics from experiment results"""
    metrics_file = Path(f'results/{experiment_name}/metrics.json')

    if not metrics_file.exists():
        return None

    with open(metrics_file, 'r') as f:
        return json.load(f)

def compare_experiments():
    """Compare baseline vs fine-tuned experiments"""

    print("\n" + "="*70)
    print("EXPERIMENT COMPARISON: Augmented vs Augmented + Real Drone Data")
    print("="*70)

    # Load metrics
    baseline = load_metrics('baseline_augmented')
    real_data = load_metrics('with_real_data')

    if baseline is None and real_data is None:
        print("\n❌ No experiment results found!")
        print("\nRun experiments first:")
        print("  1. python3 train_baseline_augmented.py")
        print("  2. python3 train_with_real_data.py")
        return

    # Display results
    print("\n📊 RESULTS SUMMARY")
    print("-" * 70)

    if baseline:
        print("\n🔵 Baseline (Augmented Data Only):")
        print(f"   Training images: {baseline.get('train_images', 'N/A')}")
        print(f"   Validation images: {baseline.get('val_images', 'N/A')}")
        print(f"   mAP50: {baseline.get('map50', 0):.4f}")
        print(f"   mAP50-95: {baseline.get('map50_95', 0):.4f}")
        print(f"   Model: results/baseline_augmented/weights/best.pt")
    else:
        print("\n🔵 Baseline: NOT RUN")
        print("   Run: python3 train_baseline_augmented.py")

    if real_data:
        print("\n🟢 Fine-tuned (Augmented + Real Drone Data):")
        print(f"   Training images: {real_data.get('train_images', 'N/A')}")
        print(f"   Validation images: {real_data.get('val_images', 'N/A')}")
        print(f"   mAP50: {real_data.get('map50', 0):.4f}")
        print(f"   mAP50-95: {real_data.get('map50_95', 0):.4f}")
        print(f"   Model: results/with_real_data/weights/best.pt")
    else:
        print("\n🟢 Fine-tuned: NOT RUN")
        print("   Run: python3 train_with_real_data.py")

    # Calculate improvement
    if baseline and real_data:
        map50_improvement = real_data['map50'] - baseline['map50']
        map50_95_improvement = real_data['map50_95'] - baseline['map50_95']
        map50_pct_change = (map50_improvement / baseline['map50'] * 100) if baseline['map50'] > 0 else 0
        map50_95_pct_change = (map50_95_improvement / baseline['map50_95'] * 100) if baseline['map50_95'] > 0 else 0

        print("\n" + "="*70)
        print("📈 IMPROVEMENT WITH REAL DRONE DATA")
        print("="*70)
        print(f"   mAP50 improvement: {map50_improvement:+.4f} ({map50_pct_change:+.1f}%)")
        print(f"   mAP50-95 improvement: {map50_95_improvement:+.4f} ({map50_95_pct_change:+.1f}%)")

        if map50_improvement > 0:
            print("\n✅ Real drone data IMPROVED performance!")
            print("   → Domain-specific data collection is valuable")
        elif map50_improvement < -0.01:
            print("\n⚠️  Real drone data DECREASED performance")
            print("   → Possible causes:")
            print("     - Insufficient real data quantity")
            print("     - Real data quality issues")
            print("     - Labels may be inaccurate")
        else:
            print("\n➡️  Performance roughly equivalent")
            print("   → May need more real data to see improvement")

        # Teaching insights
        print("\n" + "="*70)
        print("🎓 TEACHING INSIGHTS FOR STUDENTS")
        print("="*70)
        print("\n1. Data Distribution Matching:")
        print("   → Training data should match deployment conditions")
        print("   → Synthetic augmentation has limits")
        print("   → Real-world data captures nuances (lighting, shadows, perspective)")

        print("\n2. Fine-tuning Strategy:")
        print("   → Started with COCO pretrained weights (transfer learning)")
        print("   → Used lower learning rate (0.001) for fine-tuning")
        print("   → Combined synthetic + real data for best results")

        print("\n3. Iterative Improvement:")
        print("   → Measure baseline performance first")
        print("   → Collect targeted data for failure cases")
        print("   → Re-train and measure improvement")
        print("   → Repeat until satisfactory performance")

        real_added = real_data['train_images'] - baseline['train_images']
        print(f"\n4. Data Efficiency:")
        print(f"   → Added only ~{real_added} real photos")
        print(f"   → Achieved {map50_improvement:+.4f} mAP50 improvement")
        print("   → Quality > Quantity for domain-specific data")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Test on actual drone footage:")
    print("   from ultralytics import YOLO")
    print("   model = YOLO('results/with_real_data/weights/best.pt')")
    print("   results = model('drone_test_video.mp4')")

    print("\n2. If performance is insufficient:")
    print("   → Collect more real drone photos from failure cases")
    print("   → Focus on recessed building scenarios")
    print("   → Ensure diverse lighting/height conditions")
    print("   → Re-run train_with_real_data.py")

    print("\n3. For student teams:")
    print("   → Have them predict which approach works better (before training)")
    print("   → Discuss why real data matters")
    print("   → Assign: collect 10 photos per class, measure improvement")
    print("   → Teach systematic data collection methodology")

    print("\n" + "="*70 + "\n")

def test_on_images(model_path, test_images_path):
    """Test a model on specific images"""
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return

    if not Path(test_images_path).exists():
        print(f"❌ Test images not found: {test_images_path}")
        return

    print(f"\n🧪 Testing model: {model_path}")
    print(f"   On images: {test_images_path}")

    model = YOLO(model_path)
    results = model(test_images_path, conf=0.25, save=True)

    # Count detections
    detection_count = 0
    for result in results:
        if result.boxes is not None:
            detection_count += len(result.boxes)

    print(f"✅ Test completed!")
    print(f"   Total detections: {detection_count}")
    print(f"   Results saved to: runs/detect/predict*/")

def main():
    parser = argparse.ArgumentParser(
        description='Compare experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 compare_experiments.py

  # Test specific model on images
  python3 compare_experiments.py --test results/with_real_data/weights/best.pt --images ../val/images
        """
    )
    parser.add_argument('--test', type=str,
                       help='Test a specific model on images')
    parser.add_argument('--images', type=str,
                       help='Path to test images (required with --test)')

    args = parser.parse_args()

    if args.test:
        if not args.images:
            print("❌ --images required with --test")
            return
        test_on_images(args.test, args.images)
    else:
        compare_experiments()

if __name__ == "__main__":
    main()
