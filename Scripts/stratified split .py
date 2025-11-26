#!/usr/bin/env python3
"""
Full pipeline:
- parse KITTI label_2/*.txt to a DataFrame (file, class, x1,y1,x2,y2)
- compute per-image dominant class for stratification
- perform image-level stratified train/val split
- create YOLO folder structure, copy images, write YOLO .txt labels (normalized)
"""

import os
import glob
import shutil
from collections import Counter, defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

# ---------- CONFIG ----------
kitti_label_dir  = "/home/omar-mohamed/ML_Project/KITTI/data_object_label_2/training/label_2"
kitti_image_dir  = "/home/omar-mohamed/ML_Project/KITTI/data_object_image_2/training/image_2"
output_dir       = "/home/omar-mohamed/ML_Project/kitti_yolo_fixed"   # output root
train_ratio      = 0.8
random_state     = 42

# class mapping used for YOLO indices
class_map = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
    "Truck": 3,
    "Van": 4,
    "Person_sitting": 5,
    "Tram": 6,
    "Misc": 7
}
# ----------------------------

os.makedirs(output_dir, exist_ok=True)
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

# --- 1) parse KITTI label files into DataFrame
records = []
label_files = sorted(glob.glob(os.path.join(kitti_label_dir, "*.txt")))
if not label_files:
    raise SystemExit(f"No KITTI label files found in {kitti_label_dir}")

for lf in label_files:
    base = os.path.basename(lf)
    img_name = base.replace(".txt", ".png")  # KITTI images are .png
    with open(lf, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            cls = parts[0]
            if cls == "DontCare":
                continue
            # x1,y1,x2,y2 are parts[4:8]
            x1, y1, x2, y2 = map(float, parts[4:8])
            records.append({
                "file": img_name,
                "class": cls,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

df = pd.DataFrame(records)
if df.empty:
    raise SystemExit("No objects parsed from KITTI labels (check paths).")

print("Parsed annotations:", df.shape)
print(df["class"].value_counts())

# --- 2) make image-level DataFrame for stratification
# choose image 'label' as the most frequent class in the image (mode)
img2classes = df.groupby("file")["class"].apply(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
img_df = img2classes.reset_index().rename(columns={0: "dominant_class", "class": "dominant_class"})
img_df.columns = ["file", "dominant_class"]
print("Unique images parsed:", len(img_df))

# --- 3) stratified split at image-level
train_imgs, val_imgs = train_test_split(
    img_df["file"].tolist(),
    train_size=train_ratio,
    stratify=img_df["dominant_class"],
    random_state=random_state
)

train_set = set(train_imgs)
val_set   = set(val_imgs)

print("Train images:", len(train_set))
print("Val images:", len(val_set))
print("Overlap check (should be 0):", len(train_set & val_set))

# --- 4) for each split, write YOLO labels and copy images
def convert_and_copy(split_set, split_name):
    # collect rows for only images in split_set
    df_split = df[df["file"].isin(split_set)]
    # group by file
    grouped = df_split.groupby("file")

    missing_images = []
    processed_images = 0

    for fname, group in grouped:
        img_src = os.path.join(kitti_image_dir, fname)
        if not os.path.exists(img_src):
            missing_images.append(fname)
            continue

        # read width,height dynamically (safer than hard-coding)
        with Image.open(img_src) as im:
            img_w, img_h = im.size

        # copy image
        dst_img = os.path.join(output_dir, f"images/{split_name}", fname)
        shutil.copy2(img_src, dst_img)

        # write label file
        label_dst = os.path.join(output_dir, f"labels/{split_name}", fname.replace(".png", ".txt"))
        with open(label_dst, "w") as w:
            for _, row in group.iterrows():
                cls_name = row["class"]
                if cls_name not in class_map:
                    # skip or map unknown classes if you prefer
                    continue
                cls_id = class_map[cls_name]
                x1, y1, x2, y2 = row[["x1","y1","x2","y2"]].astype(float)
                # normalize
                x_c = ((x1 + x2) / 2.0) / img_w
                y_c = ((y1 + y2) / 2.0) / img_h
                w_box = (x2 - x1) / img_w
                h_box = (y2 - y1) / img_h
                # clamp to [0,1]
                x_c = max(0.0, min(1.0, x_c))
                y_c = max(0.0, min(1.0, y_c))
                w_box = max(0.0, min(1.0, w_box))
                h_box = max(0.0, min(1.0, h_box))
                w.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_box:.6f} {h_box:.6f}\n")

        processed_images += 1

    return processed_images, missing_images

train_count, train_missing = convert_and_copy(train_set, "train")
val_count, val_missing     = convert_and_copy(val_set, "val")

print(f"Processed train images: {train_count}; missing: {len(train_missing)}")
print(f"Processed val   images: {val_count}; missing: {len(val_missing)}")
print(f"Total unique images expected: {len(img_df)}, processed total: {train_count + val_count}")
