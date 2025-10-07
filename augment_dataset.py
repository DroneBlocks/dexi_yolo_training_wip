#!/usr/bin/env python3
"""
Data augmentation script for YOLO training dataset.
Generates rotated, scaled, and transformed variants of original images.
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
import argparse

class YOLODatasetAugmenter:
    def __init__(self, original_images_dir="source_data/original_images", output_dir="train", val_split=0.2):
        self.original_images_dir = Path(original_images_dir)
        self.output_dir = Path(output_dir)
        self.val_split = val_split
        
        # Create train directories
        self.train_images_dir = self.output_dir / "images"
        self.train_labels_dir = self.output_dir / "labels"
        
        # Create validation directories
        self.val_images_dir = Path("val") / "images"
        self.val_labels_dir = Path("val") / "labels"
        
        # Create all output directories
        self.train_images_dir.mkdir(parents=True, exist_ok=True)
        self.train_labels_dir.mkdir(parents=True, exist_ok=True)
        self.val_images_dir.mkdir(parents=True, exist_ok=True)
        self.val_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Class mapping: Sequential IDs (0-5) but ordered for better COCO transfer learning
        self.class_to_id = {
            'car': 0,        # Sequential ID 0 (maps to COCO ID 2)
            'motorcycle': 1, # Sequential ID 1 (maps to COCO ID 3)  
            'truck': 2,      # Sequential ID 2 (maps to COCO ID 7)
            'bird': 3,       # Sequential ID 3 (maps to COCO ID 14)
            'cat': 4,        # Sequential ID 4 (maps to COCO ID 15)
            'dog': 5,        # Sequential ID 5 (maps to COCO ID 16)
        }
        self.class_names = list(self.class_to_id.keys())
        
    def detect_class_from_filename(self, filename):
        """Detect class from filename"""
        filename_lower = filename.lower()
        for class_name in self.class_names:
            if class_name in filename_lower:
                return self.class_to_id[class_name]
        raise ValueError(f"Could not detect class from filename: {filename}")
    
    def rotate_image_and_bbox(self, image, angle):
        """Rotate image and return full bounding box for rotated object"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new canvas size to fit rotated image
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int((h * sin_a) + (w * cos_a))
        new_h = int((h * cos_a) + (w * sin_a))
        
        # Adjust rotation matrix for new canvas size
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Rotate image
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=(0, 0, 0))
        
        # For single object detection, bbox covers entire image with some padding
        padding = 0.05  # 5% padding
        x_center = 0.5
        y_center = 0.5
        width = 1.0 - 2 * padding
        height = 1.0 - 2 * padding
        
        return rotated, (x_center, y_center, width, height)
    
    def scale_image(self, image, scale_factor):
        """Scale image up or down"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        if scale_factor > 1.0:
            # Scale up then crop to original size
            scaled = cv2.resize(image, (new_w, new_h))
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            cropped = scaled[start_y:start_y+h, start_x:start_x+w]
            return cropped
        else:
            # Scale down then pad to original size
            scaled = cv2.resize(image, (new_w, new_h))
            # Create black canvas of original size
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = scaled
            return canvas
    
    def adjust_brightness_contrast(self, image, brightness=0, contrast=1.0):
        """Adjust image brightness and contrast"""
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted
    
    def add_noise(self, image, noise_factor=25):
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, noise_factor, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        return noisy
    
    def blur_image(self, image, blur_strength=3):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    
    def generate_augmentations(self, image_path, augmentations_per_image=100):
        """Generate augmented versions of a single image"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Detect class from filename
        class_id = self.detect_class_from_filename(image_path.name)
        # Find class name from class_to_id mapping
        class_name = None
        for name, id in self.class_to_id.items():
            if id == class_id:
                class_name = name
                break
        
        # Calculate train/val split
        val_count = int(augmentations_per_image * self.val_split)
        train_count = augmentations_per_image - val_count
        
        print(f"Processing {class_name} - generating {augmentations_per_image} variations")
        print(f"  Train: {train_count}, Validation: {val_count}")
        
        # Generate augmentations
        for i in range(augmentations_per_image):
            # Start with original image
            aug_image = image.copy()
            
            # Random rotation (0-360 degrees)
            angle = np.random.uniform(0, 360)
            aug_image, bbox = self.rotate_image_and_bbox(aug_image, angle)
            
            # Random scale (0.25x to 1.3x)
            scale = np.random.uniform(0.25, 1.3)
            aug_image = self.scale_image(aug_image, scale)
            
            # Random brightness (-30 to +30)
            brightness = np.random.randint(-30, 31)
            # Random contrast (0.7 to 1.3)
            contrast = np.random.uniform(0.7, 1.3)
            aug_image = self.adjust_brightness_contrast(aug_image, brightness, contrast)
            
            # Random noise (20% chance)
            if np.random.random() < 0.2:
                aug_image = self.add_noise(aug_image)
            
            # Random blur (15% chance)
            if np.random.random() < 0.15:
                blur_strength = np.random.choice([3, 5, 7])
                aug_image = self.blur_image(aug_image, blur_strength)
            
            # Determine if this image goes to train or validation
            if i < train_count:
                images_dir = self.train_images_dir
                labels_dir = self.train_labels_dir
            else:
                images_dir = self.val_images_dir
                labels_dir = self.val_labels_dir
            
            # Save augmented image
            img_filename = f"{class_name}_{i+1:03d}.jpg"
            img_path = images_dir / img_filename
            cv2.imwrite(str(img_path), aug_image)
            
            # Save corresponding label file
            label_filename = f"{class_name}_{i+1:03d}.txt"
            label_path = labels_dir / label_filename
            
            # YOLO format: class_id x_center y_center width height (normalized 0-1)
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    def augment_all_images(self, augmentations_per_image=100):
        """Process all images in the base directory"""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(self.original_images_dir.glob(ext))
            image_files.extend(self.original_images_dir.glob(ext.upper()))
        
        if not image_files:
            print(f"No image files found in {self.original_images_dir}")
            return
        
        print(f"Found {len(image_files)} original images")
        
        for image_path in image_files:
            self.generate_augmentations(image_path, augmentations_per_image)
        
        print(f"\nAugmentation complete!")
        print(f"Training images: {len(list(self.train_images_dir.glob('*.jpg')))}")
        print(f"Training labels: {len(list(self.train_labels_dir.glob('*.txt')))}")
        print(f"Validation images: {len(list(self.val_images_dir.glob('*.jpg')))}")
        print(f"Validation labels: {len(list(self.val_labels_dir.glob('*.txt')))}")
        print(f"Total images: {len(list(self.train_images_dir.glob('*.jpg'))) + len(list(self.val_images_dir.glob('*.jpg')))}")

def main():
    parser = argparse.ArgumentParser(description='Augment YOLO training dataset')
    parser.add_argument('--input', '-i', type=str, default="source_data/original_images",
                       help='Directory containing original images (default: source_data/original_images)')
    parser.add_argument('--output', '-o', type=str, default='train',
                       help='Output directory for augmented dataset')
    parser.add_argument('--count', '-c', type=int, default=100,
                       help='Number of augmentations per original image')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Fraction of data to use for validation (default: 0.2)')

    args = parser.parse_args()

    input_dir = args.input if args.input else "source_data/original_images"
    augmenter = YOLODatasetAugmenter(input_dir, args.output, args.val_split)
    augmenter.augment_all_images(args.count)

if __name__ == "__main__":
    main()