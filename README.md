# 🌱 Crop and Weed Detection — YOLOv5 Object Detection

**Free Summer Internship in Data Science & Machine Learning**
upSkill Campus | Uniconverge Technologies (P) Ltd.

**Author:** Namitha Singu

---

## 📌 Overview

This project implements a deep learning–based object detection system to distinguish between **sesame crop** and **weeds** in field images, using **YOLOv5**. The goal is to support precision agriculture by enabling targeted pesticide spraying — reducing chemical usage while protecting crop health.

## 🎯 Problem Statement

Uncontrolled weed growth can reduce crop yield by 20–80%. Traditional weed control relies on blanket pesticide spraying across entire fields, which is costly, labour-intensive, and environmentally harmful. This project builds a computer vision model that can:

- Detect and localise crop and weed plants in field images using bounding boxes
- Classify each detection as **Crop** or **Weed**
- Serve as the vision component of a precision pesticide-spraying system

## 🗂️ Dataset

- **Source:** Crop and Weed Detection Data with Bounding Boxes (Kaggle)
- **Size:** 1,300 images (512×512), YOLO-format labels
- **Classes:** `crop` (sesame), `weed`
- **Split:** 1,040 train / 130 validation / 130 test (80/10/10)

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Model | YOLOv5s (Ultralytics) |
| Framework | PyTorch |
| Augmentation | Albumentations |
| Training | Google Colab (T4 GPU) |
| Inference | Local (VS Code, CPU) |
| Language | Python 3.11 |

## 📁 Repository Contents

| File | Description |
|---|---|
| `crop_weed_detection.py` | Main pipeline — dataset stats, visualisation, augmentation, training config, inference, evaluation plots |
| `split_dataset.py` | Splits raw image-label pairs into train/val/test folders (80/10/10) |
| `dataset.yaml` | YOLOv5 dataset configuration (classes, paths) |
| `.gitignore` | Excludes large/generated folders (dataset images, venv, model weights) |

> **Note:** The dataset, virtual environment, YOLOv5 source, and trained weights are excluded from this repository due to size — see *Setup* below to reproduce locally.

## ⚙️ Setup & Usage

```bash
# 1. Clone this repo
git clone https://github.com/Namitha938/upskillcampus.git
cd upskillcampus

# 2. Create a virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch torchvision torchaudio
pip install opencv-python matplotlib pandas "numpy<2"
pip install PyYAML tqdm seaborn albumentations

# 3. Clone YOLOv5
git clone https://github.com/ultralytics/yolov5
cd yolov5 && pip install -r requirements.txt && cd ..

# 4. Add the dataset
# Download from Kaggle: "Crop and Weed Detection Data with Bounding Boxes"
# Then split it into train/val/test:
python split_dataset.py

# 5. Run the pipeline
python crop_weed_detection.py
```

### Training (on Google Colab GPU)

```bash
python train.py \
    --img 512 --batch 16 --epochs 50 \
    --data dataset.yaml \
    --weights yolov5s.pt \
    --project runs/train --name crop_weed_v1 --cache
```

## 📊 Results

| Class | Objects Detected (Test Set) |
|---|---|
| Crop | 137 |
| Weed | 77 |
| **Total** | **214** |

The model was evaluated on 130 held-out test images, successfully detecting and classifying crop and weed instances across varying densities (0–18 objects per image).

## 🧠 Key Learnings

- End-to-end object detection pipeline: dataset prep → augmentation → training → evaluation → inference
- Cloud GPU training (Google Colab) for compute-intensive deep learning when local GPU is unavailable
- Resolved real-world cross-platform deployment issues — `PosixPath`/`WindowsPath` conflicts and NumPy/PyTorch version incompatibility when moving a model from Linux (Colab) to Windows (local inference)

## 📄 License

This project was developed for educational purposes as part of the UCT/upSkill Campus internship programme.

---

*Part of the Free Summer Internship in Data Science & Machine Learning — upSkill Campus & Uniconverge Technologies (P) Ltd.*
