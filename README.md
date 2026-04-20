# Vehicle Detection for Autonomous Driving Systems

A fine-tuned **YOLOv8** object detection model for detecting vehicles and road users in autonomous driving scenarios, achieving a **mAP@0.5 of 0.938** on the KITTI 2D Object Detection dataset.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Transfer Learning Model](#transfer-learning-model)
- [Training](#training)
- [Results](#results)
- [Conclusion](#conclusion)
- [Recommendations](#recommendations)
- [Scripts](#scripts)
- [External Links](#external-links)

---

## Introduction

This project leverages transfer learning from a pre-trained YOLOv8 backbone, fine-tuned on the KITTI 2D Object Detection dataset. The resulting model is highly accurate in detecting vehicles and related road-user classes, making it suitable for real-world autonomous driving applications.

---

## Dataset

The **KITTI 2D Object Detection** dataset was used. As a benchmark in autonomous driving research, it provides a robust and well-annotated foundation for fine-tuning and evaluating object detection models.

### Class Categories

| Class | Instances |
|---|---|
| Car | 22,949 |
| Pedestrian | 3,649 |
| Van | 2,383 |
| Cyclist | 1,313 |
| Misc | 802 |
| Truck | 898 |
| Tram | 427 |
| Person_sitting | 182 |

> **Note:** The significant class imbalance in this dataset has a notable impact on the model's performance across different object categories, as discussed in the Results section.

### Data Split

The dataset contains **7,481 labeled images**. No separate validation set is provided, and the 7,518 test images do not include labels (evaluation requires submission to the KITTI server). The data was therefore split using **stratified sampling at the image level**:

| Split | Images |
|---|---|
| Train | ~5,984 (80%) |
| Validation | ~1,497 (20%) |
| **Total** | **7,481** |

Stratified sampling was applied at the **image level** — each image is assigned to a split based on its dominant class. This prevents data leakage and ensures proportional class representation in both train and validation sets.

> The stratified split script is available in the repository with inline comments for clarification.

---

## Transfer Learning Model

### YOLO Overview

YOLO is a family of pre-trained object detection models, most commonly trained on the COCO dataset. Versions include YOLOv5, YOLOv8, and YOLOv11, each available in multiple sizes: `n` (nano), `s` (small), `m` (medium), `l` (large), and `x` (extra-large), which differ in performance and resource requirements.

### Model Selected

**YOLOv8m** was selected as it offers the best balance of performance and computational cost for this project. Multiple models were benchmarked using `compare_results.py` before this decision was finalized.

---

## Training

### Train Command

```bash
yolo detect train \
  model=yolov8m.pt \
  data="/path/to/kitti.yaml" \
  epochs=100 \
  imgsz=640 \
  fraction=1 \
  amp=False \
  workers=2 \
  batch=16 \
  device=0 \
  project=/path/to/runs/detect \
  name=continue_to_epoch100 \
  resume=True
```

### Hyperparameter Explanation

| Parameter | Description |
|---|---|
| `model` | Pre-trained or custom model to use |
| `epochs` | Number of full passes over the training dataset |
| `imgsz` | Input image size; larger sizes improve performance but require more resources |
| `fraction` | Fraction of dataset to use (`1` = full dataset) |
| `amp` | Automatic Mixed Precision; set to `False` due to GPU compatibility issues |
| `workers` | Number of CPU threads for data loading |
| `batch` | Number of images processed simultaneously |
| `device` | GPU index to use for training |
| `resume` | Resumes training from the last saved checkpoint |

### Overfitting & Underfitting

YOLO mitigates overfitting by automatically saving two weight files after training:

- **`best.pt`** — Weights from the epoch with optimal evaluation metrics (mAP).
- **`last.pt`** — Weights from the final epoch, regardless of performance.

The `best.pt` model is used for inference and evaluation.

---

## Results

### Overall Performance

| Metric | Value |
|---|---|
| **mAP@0.5** | **0.938** |
| Best Confidence Threshold (F1) | 0.451 |
| Peak F1-Score | 0.91 |
| Peak Recall | 0.96 |

### Per-Class Average Precision (AP@0.5)

| Class | AP@0.5 |
|---|---|
| Truck | 0.988 |
| Car | 0.980 |
| Van | 0.978 |
| Tram | 0.972 |
| Misc | 0.960 |
| Cyclist | 0.929 |
| Pedestrian | 0.875 |
| Person_sitting | 0.825 |

### Metric Curves

- **Precision-Confidence:** Precision remains above 0.90 across most confidence levels, reaching a maximum of 1.00 at high confidence — indicating very few false positives when the model is certain.
- **Recall-Confidence:** Recall peaks at **0.96** near a confidence threshold of 0, as expected — lower thresholds detect more objects at the cost of more false positives.
- **F1-Confidence:** The F1-score peaks at **0.91** at a confidence threshold of **0.451**, making this the recommended default threshold for deployment.

### Confusion Matrix Highlights

- Strong diagonal dominance: **0.97** for Car, **0.99** for Truck.
- Primary confusion: Pedestrians vs. background (a known challenge in object detection).
- Minor confusion between structurally similar classes: Car/Van and Cyclist/Pedestrian.
- `Person_sitting` shows the lowest recall, likely due to its visual similarity to `Pedestrian` and low training sample count.

---

## Conclusion

The fine-tuned model achieves state-of-the-art performance on the KITTI dataset with a **mAP@0.5 of 93.8%**. It performs exceptionally well on large vehicle classes (Car, Truck, Tram) and reasonably well on vulnerable road users. The recommended operational confidence threshold is **0.45** to maximize the F1-score.

---

## Recommendations

- The KITTI dataset is relatively small, especially for underrepresented classes like pedestrians.
- The dataset lacks diversity in lighting conditions (nighttime) and weather scenarios.
- Training on additional or larger datasets (e.g., nuScenes, Waymo Open Dataset) would improve generalization, particularly for pedestrian detection.

---

## Repository Structure

```
├── Metrics/          # Visual representations of evaluation metrics
├── Model/            # Final trained model weights
├── Scripts/          # Utility scripts used throughout the project
├── Testing Videos/   # Real-time testing video results
└── Metrics.csv       # Primary training metrics log
```

---

## Scripts

| Script | Description |
|---|---|
| `test_ROCm_and_pytorch.py` | Verifies that PyTorch and ROCm are correctly installed |
| `stratified_split.py` | Performs the stratified train/val split on the KITTI dataset |
| `compare_results.py` | Compares CSV metrics across model runs to select the best configuration |
| `add_missing_labels.py` | Fills in empty label files for test images where YOLO detected no objects |

---

## External Links

- [KITTI 2D Object Detection Dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)
