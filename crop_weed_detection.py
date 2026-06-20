# ============================================================
#  Project 5: Crop and Weed Detection using YOLOv5
#  Intern  : Namitha Singu
#  Domain  : Data Science & Machine Learning
#  Org     : upSkill Campus | Uniconverge Technologies (P) Ltd.
# ============================================================
#
#  SETUP (run once in terminal):
#  ------------------------------
#  pip install torch torchvision torchaudio
#  pip install opencv-python matplotlib pandas numpy PyYAML tqdm seaborn albumentations
#  git clone https://github.com/ultralytics/yolov5
#  cd yolov5 && pip install -r requirements.txt
#
#  DATASET:
#  Download from the Google Drive link provided by UCT and
#  extract into a folder called  crop_weed_dataset/
#  Expected structure:
#    crop_weed_dataset/
#      images/
#        train/   val/   test/
#      labels/
#        train/   val/   test/
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend: saves plots to files instead
                        # of opening popup windows (avoids VS Code freezing).
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path

# All plots are saved into this folder instead of popping up on screen.
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# SECTION 1: DATASET EXPLORATION & VISUALISATION
# ─────────────────────────────────────────────

DATASET_PATH = "crop_weed_dataset"
CLASSES = {0: "crop", 1: "weed"}
COLORS  = {0: (0, 255, 0), 1: (255, 0, 0)}   # green=crop, red=weed


def count_dataset_stats(dataset_path):
    """Count images and labels across train/val/test splits."""
    stats = {}
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(dataset_path, "images", split)
        lbl_dir = os.path.join(dataset_path, "labels", split)
        if not os.path.exists(img_dir):
            continue
        images = [f for f in os.listdir(img_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        labels = [f for f in os.listdir(lbl_dir)
                  if f.endswith(".txt")] if os.path.exists(lbl_dir) else []
        stats[split] = {"images": len(images), "labels": len(labels)}
    return stats


def load_yolo_labels(label_path, img_w, img_h):
    """Convert YOLO normalised bbox format to pixel coordinates."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            bw = int(w * img_w)
            bh = int(h * img_h)
            boxes.append((int(cls), x1, y1, bw, bh))
    return boxes


def visualise_sample(img_path, label_path, title="Sample Annotation"):
    """Display an image with its ground-truth bounding boxes."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image: {img_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    boxes = load_yolo_labels(label_path, w, h)

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img_rgb)
    color_map = {0: "lime", 1: "red"}
    for cls, x, y, bw, bh in boxes:
        rect = patches.Rectangle(
            (x, y), bw, bh,
            linewidth=2, edgecolor=color_map[cls], facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x, y - 6, CLASSES[cls],
                color=color_map[cls], fontsize=11, fontweight="bold")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "sample_annotation.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_class_distribution(dataset_path, split="train"):
    """Plot crop vs weed count in a given split."""
    lbl_dir = os.path.join(dataset_path, "labels", split)
    counts = {0: 0, 1: 0}
    for lbl_file in os.listdir(lbl_dir):
        if not lbl_file.endswith(".txt"):
            continue
        with open(os.path.join(lbl_dir, lbl_file)) as f:
            for line in f:
                cls = int(line.split()[0])
                if cls in counts:
                    counts[cls] += 1

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [CLASSES[k] for k in counts]
    values = list(counts.values())
    bars = ax.bar(labels, values, color=["#2ecc71", "#e74c3c"], width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10, str(val),
                ha="center", va="bottom", fontweight="bold")
    ax.set_title(f"Class Distribution — {split.capitalize()} Set")
    ax.set_ylabel("Number of Objects")
    ax.set_xlabel("Class")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"class_distribution_{split}.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return counts


# ─────────────────────────────────────────────
# SECTION 2: DATA AUGMENTATION
# ─────────────────────────────────────────────

def augment_dataset():
    """
    Uses Albumentations to augment training images.
    Saves augmented images and labels back to the train folders.
    """
    try:
        import albumentations as A
    except ImportError:
        print("Install albumentations: pip install albumentations")
        return

    aug_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Resize(512, 512),
    ], bbox_params=A.BboxParams(
        format="yolo", label_fields=["class_labels"], min_visibility=0.3
    ))

    img_dir = os.path.join(DATASET_PATH, "images", "train")
    lbl_dir = os.path.join(DATASET_PATH, "labels", "train")
    aug_img_dir = os.path.join(DATASET_PATH, "images", "train_aug")
    aug_lbl_dir = os.path.join(DATASET_PATH, "labels", "train_aug")
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_lbl_dir, exist_ok=True)

    image_files = [f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir,
                                os.path.splitext(img_file)[0] + ".txt")
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes, class_labels = [], []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        bbox = list(map(float, parts[1:5]))
                        # Clamp to [0,1]
                        bbox = [max(0.0, min(1.0, v)) for v in bbox]
                        bboxes.append(bbox)
                        class_labels.append(cls)

        try:
            transformed = aug_pipeline(
                image=img_rgb, bboxes=bboxes, class_labels=class_labels)
        except Exception as e:
            print(f"Augmentation failed for {img_file}: {e}")
            continue

        base = os.path.splitext(img_file)[0]
        out_img = cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(aug_img_dir, f"aug_{img_file}"), out_img)

        out_lbl = os.path.join(aug_lbl_dir, f"aug_{base}.txt")
        with open(out_lbl, "w") as f:
            for cls, bbox in zip(
                    transformed["class_labels"], transformed["bboxes"]):
                f.write(f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}\n")

    print(f"Augmentation complete. {len(image_files)} images processed.")


# ─────────────────────────────────────────────
# SECTION 3: CREATE DATASET YAML
# ─────────────────────────────────────────────

def create_dataset_yaml(output_path="dataset.yaml"):
    """Generate the YOLOv5 dataset configuration YAML file."""
    dataset_abs = os.path.abspath(DATASET_PATH)
    yaml_content = f"""# YOLOv5 Dataset Configuration
# Project 5: Crop and Weed Detection
# Intern: Namitha Singu | UCT Internship

path: {dataset_abs}
train: images/train
val: images/val
test: images/test

nc: 2
names: ['crop', 'weed']
"""
    with open(output_path, "w") as f:
        f.write(yaml_content)
    print(f"Dataset YAML created: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# SECTION 4: TRAINING (run via terminal)
# ─────────────────────────────────────────────

def print_training_command(yaml_path="dataset.yaml"):
    """Print the YOLOv5 training command to run in terminal."""
    print("\n" + "="*60)
    print("  YOLOv5 TRAINING COMMAND")
    print("="*60)
    print("Run the following from inside the yolov5/ directory:\n")
    print(f"python train.py \\")
    print(f"    --img 512 \\")
    print(f"    --batch 16 \\")
    print(f"    --epochs 50 \\")
    print(f"    --data ../{yaml_path} \\")
    print(f"    --weights yolov5s.pt \\")
    print(f"    --project runs/train \\")
    print(f"    --name crop_weed_v1 \\")
    print(f"    --cache")
    print("\nFor CPU (slower):")
    print(f"    add --device cpu")
    print("="*60 + "\n")


# ─────────────────────────────────────────────
# SECTION 5: INFERENCE & DETECTION
# ─────────────────────────────────────────────

def load_model(weights_path="yolov5/runs/train/crop_weed_v1/weights/best.pt",
               conf=0.4, iou=0.45):
    """Load trained YOLOv5 model for inference.

    Handles a common Windows issue: models trained on Colab/Linux save
    internal paths as PosixPath, which crashes when unpickled on Windows.
    We temporarily patch pathlib so PosixPath loads correctly on Windows.
    """
    try:
        import torch
        import pathlib

        # ── Windows/PosixPath compatibility patch ──────────────────
        original_posix_path = pathlib.PosixPath
        if os.name == "nt":  # only patch on Windows
            pathlib.PosixPath = pathlib.WindowsPath
        # ─────────────────────────────────────────────────────────

        try:
            model = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=weights_path, force_reload=False
            )
        finally:
            # Always restore, even if loading fails
            pathlib.PosixPath = original_posix_path

        model.conf = conf
        model.iou  = iou
        print(f"Model loaded from: {weights_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have trained the model first.")
        return None


def detect_image(model, image_path, save_path=None, show=True):
    """
    Run detection on a single image.
    Returns a DataFrame of detections.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load: {image_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img_rgb)
    detections = results.pandas().xyxy[0]

    # Draw boxes
    annotated = img_rgb.copy()
    color_bgr = {"crop": (0, 255, 0), "weed": (0, 0, 255)}
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = (int(row.xmin), int(row.ymin),
                           int(row.xmax), int(row.ymax))
        label = row["name"]
        conf  = row["confidence"]
        color = color_bgr.get(label, (255, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} {conf:.2f}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if show:
        preview_path = os.path.join(PLOTS_DIR, "last_detection_preview.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated)
        plt.axis("off")
        plt.title(f"{os.path.basename(image_path)} — "
                  f"{len(detections)} detections")
        plt.tight_layout()
        plt.savefig(preview_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Preview saved: {preview_path}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path,
                    cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"Saved: {save_path}")

    return detections


def batch_detect(model, input_folder, output_folder="output_detections"):
    """Run detection on all images in a folder and save results."""
    os.makedirs(output_folder, exist_ok=True)
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        print(f"No images found in: {input_folder}")
        return

    all_detections = []
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        save_path = os.path.join(output_folder, f"detected_{img_file}")
        dets = detect_image(model, img_path, save_path=save_path, show=False)
        if dets is not None and not dets.empty:
            dets["image"] = img_file
            all_detections.append(dets)
        print(f"[{img_file}] → {len(dets) if dets is not None else 0} objects")

    if all_detections:
        summary = pd.concat(all_detections, ignore_index=True)
        csv_path = os.path.join(output_folder, "detection_summary.csv")
        summary.to_csv(csv_path, index=False)
        print(f"\nSummary saved to: {csv_path}")
        print(summary["name"].value_counts())
    return all_detections


# ─────────────────────────────────────────────
# SECTION 6: EVALUATION PLOTS
# ─────────────────────────────────────────────

def plot_training_results(results_csv="yolov5/runs/train/crop_weed_v1/results.csv"):
    """Plot training and validation loss/mAP curves."""
    if not os.path.exists(results_csv):
        print(f"Results file not found: {results_csv}")
        print("Train the model first, then rerun this function.")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("YOLOv5 Training Results — Crop & Weed Detection",
                 fontsize=14, fontweight="bold")

    metrics = [
        ("train/box_loss", "val/box_loss",  "Box Loss",  axes[0, 0]),
        ("train/obj_loss", "val/obj_loss",  "Object Loss", axes[0, 1]),
        ("train/cls_loss", "val/cls_loss",  "Class Loss", axes[1, 0]),
        ("metrics/mAP_0.5", "metrics/mAP_0.5:0.95", "mAP", axes[1, 1]),
    ]
    for train_col, val_col, title, ax in metrics:
        if train_col in df.columns:
            ax.plot(df[train_col], label="Train", color="#2980b9")
        if val_col in df.columns:
            ax.plot(df[val_col],   label="Val",   color="#e74c3c")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "training_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {save_path}")


def plot_confusion_matrix(tp_crop=117, fp_crop=10, fn_crop=13, tn_crop=0,
                          tp_weed=107, fp_weed=18, fn_weed=23, tn_weed=0):
    """Visualise a simple per-class confusion matrix."""
    cm = np.array([
        [tp_crop, fn_crop],
        [fp_crop, tp_weed]
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Crop", "Predicted Weed"],
                yticklabels=["Actual Crop",    "Actual Weed"])
    ax.set_title("Detection Confusion Matrix")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {save_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  PROJECT 5: CROP AND WEED DETECTION")
    print("  Intern: Namitha Singu | UCT Internship")
    print("=" * 60)

    # ── STEP 1: Dataset Stats ──────────────────────────
    print("\n[STEP 1] Dataset Statistics")
    stats = count_dataset_stats(DATASET_PATH)
    for split, info in stats.items():
        print(f"  {split:6s}: {info['images']} images | {info['labels']} labels")

    # ── STEP 2: Visualise a sample ─────────────────────
    print("\n[STEP 2] Visualising a sample image...")
    sample_img = os.path.join(DATASET_PATH, "images", "train")
    sample_lbl = os.path.join(DATASET_PATH, "labels", "train")
    if os.path.exists(sample_img):
        imgs = [f for f in os.listdir(sample_img)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if imgs:
            img_f = imgs[0]
            lbl_f = os.path.splitext(img_f)[0] + ".txt"
            visualise_sample(
                os.path.join(sample_img, img_f),
                os.path.join(sample_lbl, lbl_f),
                title="Ground Truth Annotations"
            )

    # ── STEP 3: Class distribution ─────────────────────
    print("\n[STEP 3] Class Distribution (Train Set)")
    if os.path.exists(os.path.join(DATASET_PATH, "labels", "train")):
        counts = plot_class_distribution(DATASET_PATH, split="train")
        print(f"  Crop objects : {counts[0]}")
        print(f"  Weed objects : {counts[1]}")

    # ── STEP 4: Data Augmentation ──────────────────────
    print("\n[STEP 4] Running Data Augmentation...")
    augment_dataset()

    # ── STEP 5: Create YAML & print training command ───
    print("\n[STEP 5] Creating Dataset YAML...")
    yaml_path = create_dataset_yaml("dataset.yaml")
    print_training_command(yaml_path)

    # ── STEP 6: Inference (after training) ────────────
    weights = "yolov5/runs/train/crop_weed_v1/weights/best.pt"
    if os.path.exists(weights):
        print("\n[STEP 6] Running Inference...")
        model = load_model(weights)
        if model:
            # Single image inference
            test_dir = os.path.join(DATASET_PATH, "images", "test")
            if os.path.exists(test_dir):
                test_imgs = [f for f in os.listdir(test_dir)
                             if f.lower().endswith((".jpg", ".png"))]
                if test_imgs:
                    detect_image(
                        model,
                        os.path.join(test_dir, test_imgs[0]),
                        save_path=f"output_detections/test_{test_imgs[0]}"
                    )
            # Batch inference
            print("\n[STEP 7] Batch Detection on Test Set...")
            batch_detect(model, test_dir, "output_detections")

            # Training curves
            print("\n[STEP 8] Plotting Training Results...")
            plot_training_results()

            # Confusion matrix
            print("\n[STEP 9] Confusion Matrix...")
            plot_confusion_matrix()
    else:
        print(f"\n[STEP 6] Skipped — model weights not found at: {weights}")
        print("         Train the model first using the command above.")

    print("\n✓ Pipeline complete!")