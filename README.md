# SVM-Based LULC Change Detection using Multitemporal Satellite Images

This repository implements a **Support Vector Machine (SVM)**-based framework for **Land Use and Land Cover (LULC) change detection** using the **LEVIR-CD dataset**.  
The model classifies each pixel as *change* or *no-change* by analyzing two time-separated satellite images (Before & After) using handcrafted feature extraction.

---

## Project Overview

Change detection in satellite imagery helps monitor urban growth, deforestation, land degradation, and environmental change.  
Deep learning models often dominate this task, but they require large datasets and high computation.  
This project demonstrates that a **classical SVM**, when combined with **rich handcrafted feature extraction**, can still deliver competitive accuracy on modern datasets.

---

## Key Features

- **Pixel-wise classification** using SVM  
- **11-dimensional handcrafted feature vector** per pixel:
  - RGB values from both images (6)
  - Absolute difference |B - A| (3)
  - Gradient (Sobel) difference (1)
  - Texture (local mean) difference (1)
- **Class balancing** using random sampling and upsampling of minority class
- **Morphological post-processing** to refine prediction maps
- **RBF-kernel SVM** trained using `scikit-learn`
- Compatible with **LEVIR-CD** or similar bi-temporal datasets

---

## Dataset

This project uses the [LEVIR-CD Dataset](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd) available on Kaggle.

-Each image pair consists of:

  A/ → Before (t₁) image

  B/ → After (t₂) image

  label/ → Ground truth binary mask (change = 1, no change = 0)
