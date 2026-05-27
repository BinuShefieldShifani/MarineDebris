# 🌊 Marine Debris Detection — Underwater Plastic Segmentation

> Semantic segmentation of marine debris (plastic pollution) in underwater images using DeepLabV3+ with ResNet-50, trained on the TrashCan dataset.

---

## 📌 Overview

Plastic pollution in oceans is one of the most critical environmental challenges today. This project tackles it at the perception level — building a deep learning model that can **automatically detect and segment marine debris** in underwater imagery.

The model distinguishes between background and debris pixels, producing a segmentation mask that highlights plastic waste in underwater scenes.

---

## 🎯 Results

| Metric | Score |
|---|---|
| Mean Dice Score | **0.7165** |
| Mean IoU Score | **0.6111** |
| Debris Class Dice | 0.4902 |
| Background Class Dice | 0.9427 |
| Final Val Loss | 0.0512 |

> Trained for 10 epochs on a Tesla T4 GPU (Google Colab). Class imbalance handled via weighted Dice Loss (1:5 background:debris).

---

## 📊 Visual Results

### Class Distribution
 <img width="466" height="368" alt="Screenshot 2026-05-27 at 9 49 42 AM" src="https://github.com/user-attachments/assets/fec1beb3-8c00-4928-9d15-b9da963504cb" />

*Fig. 1 — Pixel count histogram of Class 0 (background) and Class 1 (debris). The significant imbalance motivated the use of weighted Dice Loss.*

### Predictions vs Ground Truth
<img width="428" height="445" alt="Screenshot 2026-05-27 at 9 50 02 AM" src="https://github.com/user-attachments/assets/0398cadd-2c6b-4ebe-ac46-5b9129aa7a2c" />

*Fig. 2 — Sample images, ground truth masks, and predicted masks from the validation set.*

### System Architecture
<img width="439" height="373" alt="Screenshot 2026-05-27 at 9 50 17 AM" src="https://github.com/user-attachments/assets/0fe76c46-31ad-46fa-b1c8-4d23bbe1b8d5" />

*Fig. 3 — Block diagram of the semantic segmentation system architecture.*

### Training Curves
<img width="430" height="298" alt="Screenshot 2026-05-27 at 9 50 27 AM" src="https://github.com/user-attachments/assets/252583b4-a0d3-4bf7-b348-76a4c7baeed0" />

*Fig. 4 — Training and validation loss over 10 epochs.*

---

## 🗂️ Dataset

**TrashCan** — an underwater image dataset for marine debris detection.

- **Training images:** 6,065
- **Validation images:** 1,147
- **Annotation format:** COCO-style JSON (instance segmentation)
- **Classes:** Background (0), Debris (1)

📥 Download the dataset from the [official TrashCan repository](https://conservancy.umn.edu/handle/11299/214865) and place it at:

```
trashcan/
└── dataset/
    └── instance_version/
        ├── train/
        ├── val/
        ├── instances_train_trashcan.json
        └── instances_val_trashcan.json
```

---

## 🏗️ Model Architecture

- **Backbone:** ResNet-50 (pretrained on ImageNet)
- **Segmentation head:** DeepLabV3+ with custom 2-class classifier
- **Auxiliary classifier:** Modified for 2-class output with dropout (0.1)
- **Input resolution:** 128 × 128 px
- **Output:** Per-pixel binary segmentation mask

---

## ⚙️ Training Details

| Component | Detail |
|---|---|
| Loss Function | Custom Dice Loss with class weights [1.0, 5.0] |
| Optimizer | Adam (lr = 1e-4) |
| LR Scheduler | StepLR (step=4, gamma=0.1) |
| Batch Size | 8 |
| Epochs | 10 |
| Hardware | Tesla T4 (Google Colab) |

**Augmentation pipeline (Albumentations):**
- Horizontal & vertical flips
- Random 90° rotation
- Brightness/contrast jitter
- Hue-saturation variation
- Random scaling
- ImageNet normalization

**Post-processing:**
- Morphological dilation + erosion to clean up noisy predictions

---

## 📁 Repository Structure

```
MarineDebris/
├── marine_debris_segmentation.ipynb   # Main training & evaluation notebook
├── requirements.txt                   # Dependencies
├── figures/
│   ├── pixel_histogram.png            # Fig. 1 — Class distribution
│   ├── sample_predictions.png         # Fig. 2 — Predictions vs ground truth
│   ├── block_diagram.png              # Fig. 3 — System architecture
│   └── loss_curve.png                 # Fig. 4 — Training curves
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/BinuShefieldShifani/MarineDebris.git
cd MarineDebris
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Follow the dataset instructions above to place the TrashCan dataset in the correct directory.

### 4. Run the notebook
Open `marine_debris_segmentation.ipynb` in Jupyter or Google Colab and run all cells.

> **Note:** A GPU is strongly recommended. The notebook was developed on Google Colab with a Tesla T4.

---

## 📦 Requirements

```
torch
torchvision
albumentations
opencv-python
numpy
matplotlib
tifffile
```

---

## 🔬 Key Design Decisions

- **Weighted Dice Loss** — debris covers only ~10% of pixels, so class weighting (5× for debris) prevents the model from predicting all background.
- **Morphological post-processing** — dilation followed by erosion removes small spurious detections and fills gaps in predicted debris regions.
- **Transfer learning** — starting from ImageNet-pretrained ResNet-50 significantly accelerates convergence on the relatively small TrashCan dataset.

---

## 🌍 Real-World Relevance

This work contributes to the broader problem of **automated ocean monitoring** — enabling ROVs and underwater drones to identify pollution without manual review. The segmentation approach can be extended to multi-class debris detection (e.g. plastic bags, bottles, fishing nets) with a richer annotation set.

---

## 👤 Author

**Binu Shefield Shifani**
Software Engineer (5 years, Cognizant Technology Solutions)
MS AI & Automation · University West, Trollhättan, Sweden
[GitHub](https://github.com/BinuShefieldShifani)
