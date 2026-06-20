import os
import shutil
import random

# ── CONFIGURE THESE TWO PATHS IF NEEDED ──────────────────────
SOURCE_DIR = r"C:\Users\Namitha\Downloads\archive (1)\agri_data\data"
DEST_ROOT  = r"C:\weed detection\crop_weed_project\crop_weed_dataset"
# ───────────────────────────────────────────────────────────

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_image_label_pairs(source_dir):
    all_files = os.listdir(source_dir)
    images = [f for f in all_files if f.lower().endswith(IMAGE_EXTENSIONS)]
    pairs = []
    for img in images:
        base = os.path.splitext(img)[0]
        label = base + ".txt"
        if label in all_files:
            pairs.append((img, label))
        else:
            print(f"  Warning: no matching label for {img}, skipping.")
    return pairs


def make_dest_folders(dest_root):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(dest_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dest_root, "labels", split), exist_ok=True)


def split_pairs(pairs, ratios, seed=42):
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    train_set = shuffled[:n_train]
    val_set = shuffled[n_train:n_train + n_val]
    test_set = shuffled[n_train + n_val:]
    return {"train": train_set, "val": val_set, "test": test_set}


def copy_split(split_name, pairs, source_dir, dest_root):
    img_dest = os.path.join(dest_root, "images", split_name)
    lbl_dest = os.path.join(dest_root, "labels", split_name)
    for img_file, lbl_file in pairs:
        shutil.copy2(os.path.join(source_dir, img_file), os.path.join(img_dest, img_file))
        shutil.copy2(os.path.join(source_dir, lbl_file), os.path.join(lbl_dest, lbl_file))
    print(f"  {split_name:6s}: {len(pairs)} image-label pairs copied.")


def main():
    print("=" * 60)
    print("  DATASET SPLITTER — Crop and Weed Detection")
    print("=" * 60)

    if not os.path.exists(SOURCE_DIR):
        print(f"\nERROR: Source folder not found:\n  {SOURCE_DIR}")
        print("Please check the path and update SOURCE_DIR at the top of this script.")
        return

    print(f"\nSource: {SOURCE_DIR}")
    print(f"Dest:   {DEST_ROOT}\n")

    print("[1/4] Scanning for image-label pairs...")
    pairs = get_image_label_pairs(SOURCE_DIR)
    print(f"  Found {len(pairs)} valid image-label pairs.")

    if len(pairs) == 0:
        print("\nERROR: No image-label pairs found. Check SOURCE_DIR.")
        return

    print("\n[2/4] Creating destination folders...")
    make_dest_folders(DEST_ROOT)
    print("  Done.")

    print("\n[3/4] Splitting into train/val/test (80/10/10)...")
    splits = split_pairs(pairs, SPLIT_RATIOS, seed=RANDOM_SEED)
    for name, split_pairs_list in splits.items():
        print(f"  {name:6s}: {len(split_pairs_list)} pairs")

    print("\n[4/4] Copying files...")
    for split_name, split_pairs_list in splits.items():
        copy_split(split_name, split_pairs_list, SOURCE_DIR, DEST_ROOT)

    print("\n" + "=" * 60)
    print("  ✓ DATASET SPLIT COMPLETE!")
    print("=" * 60)
    print(f"\nYour dataset is now ready at:\n  {DEST_ROOT}")
    print("\nNext step: run crop_weed_detection.py to verify the counts,")
    print("then start training with the printed YOLOv5 command.")


if __name__ == "__main__":
    main()