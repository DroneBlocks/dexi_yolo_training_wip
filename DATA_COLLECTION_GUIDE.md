# Drone Data Collection Guide

## Objective
Collect real-world drone camera images of the 6 COCO classes (bird, car, cat, dog, motorcycle, truck) to fine-tune YOLOv8 for drone-based detection in recessed buildings.

## Classes to Photograph
- bird
- car
- cat
- dog
- motorcycle
- truck

## Collection Protocol

### Setup
1. Print each of the 6 original images from `train/original_images/`
2. Place images in the recessed building setup (actual deployment scenario)
3. Ensure drone camera is pointing straight down (matches production)

### Photo Requirements Per Class

**Target: 40-60 photos per class**

#### Drone Approach Angles (CRITICAL for recessed buildings)
Rotate drone around building to capture how shadows/occlusion change:
- **North approach** (0°)
- **East approach** (90°)
- **South approach** (180°)
- **West approach** (270°)

WHY: Building shadows and edge occlusion change with approach angle - this is domain-specific data that augmentation cannot replicate!

#### Height Variations (per approach angle)
- 0.5 meters above court
- 1.0 meters above court
- 2.0 meters above court
- 3.0 meters above court
- 4.0 meters above court

#### Optimal Collection Matrix
For efficiency, collect at 2-3 heights per approach angle:
- 4 angles × 3 heights = 12 photos per class (minimum)
- 4 angles × 5 heights = 20 photos per class (recommended)

#### Additional Variations
- Lighting conditions (time of day, indoor/outdoor)
- Position in frame (center, off-center)
- Partial occlusion (edge of building)
- Distance variations within each height

### Naming Convention
```
<class>_drone_<height>m_<angle>deg_<condition>_<number>.jpg
```

Examples:
- `bird_drone_1m_0deg_shadow_001.jpg` (North approach, 1m height)
- `car_drone_3m_90deg_indoor_001.jpg` (East approach, 3m height)
- `dog_drone_2m_180deg_natural_001.jpg` (South approach, 2m height)
- `cat_drone_4m_270deg_shadow_001.jpg` (West approach, 4m height)

### Storage
Save all collected images to:
```
source_data/real_drone_photos/
  ├── bird/
  ├── car/
  ├── cat/
  ├── dog/
  ├── motorcycle/
  └── truck/
```

## Quality Checklist
- [ ] Image is in focus
- [ ] Target object visible (even if small)
- [ ] Camera pointed straight down
- [ ] Metadata recorded (height, lighting, angle)
- [ ] Matches actual deployment scenario

## Building Color Consideration

⚠️ **IMPORTANT**: If your training building color differs from deployment building color:

**Risk**: Model may overfit to background color/contrast instead of object features

**Solutions**:
1. **Test First**: Collect 5-10 photos on deployment building, test baseline model
2. **If performance drops >20%**:
   - Collect equal photos from BOTH building colors (20 each per class)
   - OR increase color augmentation in training (already enabled)
   - OR train primarily on deployment building color

**Best Practice**: When possible, collect majority of photos on the **actual deployment building**

## Next Steps
After collection:
1. Label images using interactive_label_tool.py
2. Run baseline experiment (augmented data only)
3. Run fine-tuning experiment (augmented + real data)
4. Compare results
