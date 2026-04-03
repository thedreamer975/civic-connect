"""
generate_synthetic_dataset.py
Generates a synthetic YOLOv8 dataset for pothole, drainage, streetlight, and garbage detection.
Images are created by overlaying transparent PNG/JPG objects on random backgrounds, with augmentation.
"""
import cv2
import numpy as np
import os
import random
from glob import glob
from sklearn.model_selection import train_test_split

# ---------------- Paths ----------------
# Multi-extension support for backgrounds
backgrounds = glob("backgrounds/*.jpg") + glob("backgrounds/*.jpeg") + glob("backgrounds/*.png")
if len(backgrounds) == 0:
    raise ValueError("❌ No background images found in 'backgrounds/' (jpg/jpeg/png supported)")

# Object folders with multiple extensions
objects = {
    0: glob("potholes/*.jpg") + glob("potholes/*.jpeg") + glob("potholes/*.png"),
    1: glob("drainage/*.jpg") + glob("drainage/*.jpeg") + glob("drainage/*.png"),
    2: glob("streetlight/*.jpg") + glob("streetlight/*.jpeg") + glob("streetlight/*.png"),
    3: glob("garbage/*.jpg") + glob("garbage/*.jpeg") + glob("garbage/*.png")
}

# Warn if any class folder is empty
for cid, imgs in objects.items():
    if len(imgs) == 0:
        print(f"⚠️ Warning: Class {cid} has no images. Skipping during generation.")
    else:
        print(f"✅ Class {cid} has {len(imgs)} images")

output_base = "dataset"
splits = ["train", "val", "test"]
for s in splits:
    os.makedirs(f"{output_base}/images/{s}", exist_ok=True)
    os.makedirs(f"{output_base}/labels/{s}", exist_ok=True)

num_images = 1000
split_ratios = [0.7, 0.2, 0.1]  # train, val, test

# ---------------- Augmentation Functions ----------------
def augment_image(img):
    # Random flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    # Random brightness/contrast
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # Random blur
    if random.random() < 0.2:
        ksize = random.choice([3,5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img

# ---------------- Generate Images ----------------
all_images = []
all_labels = []

for i in range(num_images):
    bg_path = random.choice(backgrounds)
    bg = cv2.imread(bg_path)
    if bg is None:
        print(f"⚠️ Skipping invalid background: {bg_path}")
        continue

    h, w, _ = bg.shape
    labels = []

    num_objects = random.randint(1, 5)

    for _ in range(num_objects):
        class_id = random.choice(list(objects.keys()))
        if len(objects[class_id]) == 0:
            continue  # skip empty classes

        obj_path = random.choice(objects[class_id])
        obj = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)

        if obj is None:
            print(f"⚠️ Skipping invalid object: {obj_path}")
            continue

        # Resize object randomly
        scale = random.uniform(0.1, 0.3)
        obj_h, obj_w = int(obj.shape[0]*scale), int(obj.shape[1]*scale)
        if obj_h <= 0 or obj_w <= 0 or obj_h > h or obj_w > w:
            continue
        obj = cv2.resize(obj, (obj_w, obj_h))

        # Random placement
        x_offset = random.randint(0, w - obj_w)
        y_offset = random.randint(0, h - obj_h)

        # Overlay object
        if obj.shape[2] == 4:  # has alpha channel
            alpha = obj[:, :, 3] / 255.0
            for c in range(3):
                bg[y_offset:y_offset+obj_h, x_offset:x_offset+obj_w, c] = \
                    alpha * obj[:, :, c] + (1 - alpha) * bg[y_offset:y_offset+obj_h, x_offset:x_offset+obj_w, c]
        else:  # no alpha channel
            bg[y_offset:y_offset+obj_h, x_offset:x_offset+obj_w] = obj

        # YOLOv8 normalized bbox
        x_center = (x_offset + obj_w/2) / w
        y_center = (y_offset + obj_h/2) / h
        width = obj_w / w
        height = obj_h / h
        labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

    # Augment image
    bg = augment_image(bg)

    # Save image and label temporarily
    all_images.append(bg)
    all_labels.append(labels)

# ---------------- Split Dataset ----------------
train_idx, test_idx = train_test_split(range(len(all_images)), test_size=split_ratios[1]+split_ratios[2], random_state=42)
val_idx, test_idx = train_test_split(test_idx, test_size=split_ratios[2]/(split_ratios[1]+split_ratios[2]), random_state=42)

idx_split_map = {}
for idx in train_idx: idx_split_map[idx] = "train"
for idx in val_idx: idx_split_map[idx] = "val"
for idx in test_idx: idx_split_map[idx] = "test"

# ---------------- Save Images and Labels ----------------
for i, img in enumerate(all_images):
    split = idx_split_map[i]
    img_name = f"{output_base}/images/{split}/img_{i}.jpg"
    label_name = f"{output_base}/labels/{split}/img_{i}.txt"

    cv2.imwrite(img_name, img)
    with open(label_name, "w") as f:
        f.write("\n".join(all_labels[i]))

print("✅ Synthetic dataset generated successfully!")

# Example dataset.yaml for YOLOv8:
# train: dataset/images/train
# val: dataset/images/val
# test: dataset/images/test
# nc: 4
# names: ["pothole", "drainage", "streetlight", "garbage"]
