#!/usr/bin/env python3
"""
Interactive labeling tool for YOLO format labels
Click and drag to create bounding boxes, save labels

Usage:
    python3 label_images.py source_data/raw_drone_photos/cat
    python3 label_images.py source_data/raw_drone_photos/dog --class dog
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

class InteractiveLabelTool:
    def __init__(self, images_dir, default_class='dog'):
        self.class_dir = Path(images_dir)

        if not self.class_dir.exists():
            raise FileNotFoundError(f"Directory not found: {images_dir}")

        # Standard YOLO structure: class/images/ and class/labels/
        self.images_dir = self.class_dir / 'images'
        self.labels_dir = self.class_dir / 'labels'

        # Create directories if they don't exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Find all image files in images/ subdirectory
        self.image_files = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG']:
            self.image_files.extend(sorted(self.images_dir.glob(ext)))

        if not self.image_files:
            raise FileNotFoundError(f"No images found in: {self.images_dir}\n" +
                                  f"Place images in {self.images_dir}/")

        self.current_idx = 0

        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        self.labels = []

        # Class settings
        self.class_names = ['car', 'motorcycle', 'truck', 'bird', 'cat', 'dog']

        # Set default class based on directory name or argument
        dir_name = self.images_dir.name.lower()
        self.current_class = 5  # Default to dog

        for idx, name in enumerate(self.class_names):
            if name in dir_name or name == default_class.lower():
                self.current_class = idx
                break

        print(f"\n{'='*70}")
        print(f"Interactive YOLO Labeler")
        print(f"{'='*70}")
        print(f"Directory: {self.images_dir}")
        print(f"Images found: {len(self.image_files)}")
        print(f"Labels will be saved to: {self.images_dir} (same directory)")
        print(f"\n{'='*70}")
        print("CONTROLS:")
        print(f"{'='*70}")
        print("  🖱️  Click and drag      - Draw bounding box")
        print("  n                     - Next image (auto-saves)")
        print("  p                     - Previous image (auto-saves)")
        print("  c                     - Clear all labels for current image")
        print("  u                     - Undo last box")
        print("  s                     - Save labels manually")
        print("  q                     - Quit (auto-saves)")
        print("  0-5                   - Change class:")
        print("                          0=car, 1=motorcycle, 2=truck")
        print("                          3=bird, 4=cat, 5=dog")
        print(f"{'='*70}")
        print(f"Current class: {self.current_class} ({self.class_names[self.current_class]})")
        print(f"{'='*70}\n")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_rect = (self.start_point[0], self.start_point[1], x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                self.drawing = False
                end_point = (x, y)

                # Add the bounding box to labels
                self.add_label(self.start_point, end_point)
                self.current_rect = None

    def add_label(self, start_point, end_point):
        """Convert pixel coordinates to YOLO format and add label"""
        x1, y1 = start_point
        x2, y2 = end_point

        # Ensure x1,y1 is top-left and x2,y2 is bottom-right
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Skip tiny boxes
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            print("⚠️  Box too small - skipped")
            return

        # Convert to YOLO format (normalized coordinates)
        img_h, img_w = self.current_image.shape[:2]

        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        # Add label (class_id, x_center, y_center, width, height)
        self.labels.append((self.current_class, x_center, y_center, width, height))
        print(f"✅ Added {self.class_names[self.current_class]} box (total: {len(self.labels)})")

    def load_labels(self, image_path):
        """Load existing YOLO format labels from labels/ directory"""
        label_path = self.labels_dir / (image_path.stem + '.txt')
        labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append((class_id, x_center, y_center, width, height))
            print(f"📂 Loaded {len(labels)} existing labels")

        return labels

    def save_labels(self, image_path):
        """Save labels to YOLO format file in labels/ directory"""
        label_path = self.labels_dir / (image_path.stem + '.txt')

        with open(label_path, 'w') as f:
            for class_id, x_center, y_center, width, height in self.labels:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        print(f"💾 Saved {len(self.labels)} labels to {label_path.name}")

    def draw_labels(self, image):
        """Draw existing labels and current rectangle"""
        img = image.copy()
        h, w = img.shape[:2]

        # Draw existing labels
        for class_id, x_center, y_center, width, height in self.labels:
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)

            # Different colors for different classes
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[class_id % len(colors)]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
            cv2.putText(img, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw current rectangle being drawn
        if self.current_rect:
            x1, y1, x2, y2 = self.current_rect
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return img

    def show_current_image(self):
        """Display current image with labels"""
        if self.current_idx >= len(self.image_files):
            return False

        image_path = self.image_files[self.current_idx]
        self.current_image = cv2.imread(str(image_path))

        if self.current_image is None:
            print(f"❌ Could not load image: {image_path}")
            return True

        # Load existing labels for this image
        self.labels = self.load_labels(image_path)

        # Create window and set mouse callback
        cv2.namedWindow('Interactive Labeler', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Interactive Labeler', self.mouse_callback)

        while True:
            # Draw image with labels
            display_img = self.draw_labels(self.current_image)

            # Add info text overlay
            info_text = f"Image {self.current_idx + 1}/{len(self.image_files)}: {image_path.name}"
            cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            class_text = f"Class: {self.current_class} ({self.class_names[self.current_class]})"
            cv2.putText(display_img, class_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            labels_text = f"Boxes: {len(self.labels)}"
            cv2.putText(display_img, labels_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Interactive Labeler', display_img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.save_labels(image_path)
                print("\n👋 Quitting...")
                return False
            elif key == ord('n'):
                self.save_labels(image_path)
                self.current_idx = min(self.current_idx + 1, len(self.image_files) - 1)
                print(f"\n➡️  Next image ({self.current_idx + 1}/{len(self.image_files)})")
                break
            elif key == ord('p'):
                self.save_labels(image_path)
                self.current_idx = max(self.current_idx - 1, 0)
                print(f"\n⬅️  Previous image ({self.current_idx + 1}/{len(self.image_files)})")
                break
            elif key == ord('c'):
                self.labels.clear()
                print("🗑️  Cleared all labels for current image")
            elif key == ord('u'):
                if self.labels:
                    removed = self.labels.pop()
                    print(f"↩️  Removed last label: {self.class_names[removed[0]]}")
                else:
                    print("⚠️  No labels to undo")
            elif key == ord('s'):
                self.save_labels(image_path)
            elif ord('0') <= key <= ord('5'):
                self.current_class = key - ord('0')
                print(f"🔄 Changed to class {self.current_class} ({self.class_names[self.current_class]})")

        return True

    def run(self):
        """Main loop"""
        while self.current_idx < len(self.image_files):
            if not self.show_current_image():
                break

        cv2.destroyAllWindows()

        print(f"\n{'='*70}")
        print("✅ Labeling session complete!")
        print(f"{'='*70}")
        print(f"Images: {self.images_dir}")
        print(f"Labels: {self.labels_dir}")
        print(f"\nNext steps:")
        print(f"1. Copy labeled data to real_drone_photos (images + labels):")
        print(f"   cp -r {self.class_dir}/images {self.class_dir}/labels source_data/real_drone_photos/{self.class_dir.name}/")
        print(f"2. Run training:")
        print(f"   python3 train_with_real_data.py")
        print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Interactive YOLO labeling tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 label_images.py source_data/raw_drone_photos/cat
  python3 label_images.py source_data/raw_drone_photos/dog --class dog
  python3 label_images.py source_data/raw_drone_photos/bird --class bird

Controls:
  Click and drag to draw bounding box
  'n' - Next image    'p' - Previous image
  'c' - Clear labels  'u' - Undo last box
  's' - Save          'q' - Quit
  '0-5' - Change class (0=car, 1=motorcycle, 2=truck, 3=bird, 4=cat, 5=dog)
        """
    )
    parser.add_argument('directory', type=str,
                       help='Directory containing images to label')
    parser.add_argument('--class', '-c', dest='default_class', type=str, default='dog',
                       choices=['car', 'motorcycle', 'truck', 'bird', 'cat', 'dog'],
                       help='Default class for labeling (default: dog)')

    args = parser.parse_args()

    try:
        labeler = InteractiveLabelTool(args.directory, args.default_class)
        labeler.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nUsage: python3 label_images.py <directory>")
        print(f"Example: python3 label_images.py source_data/raw_drone_photos/cat\n")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
