# PaRK-Detect
Patch-wise road keypoint detection from satellite imagery using deep learning.

**PaRK-Detect** is a deep learning-based pipeline for extracting **road network maps** from high-resolution **satellite imagery** using **patch-wise keypoint detection and linking**. It is built upon the work from [PaRK-Detect: Towards Efficient Multi-Task Satellite Imagery Road Extraction via Patch-Wise Keypoints Detection (BMVC 2022)](https://arxiv.org/abs/2302.13263).


## What is PaRK-Detect?

PaRK-Detect formulates road extraction as a **keypoint detection and linking problem** instead of traditional pixel-wise segmentation. This formulation reduces redundancy, improves connectivity, and is highly efficient for large-scale road graph construction.

It performs:
- **Keypoint classification**: Classifies each 16×16 patch in a 1024×1024 image as containing a road keypoint or not.
- **Keypoint localization**: Determines the exact position of a keypoint within the patch.
- **Keypoint linking**: Predicts links between neighboring keypoints to construct a road network graph.


## How It Works

1. **Input Preprocessing**:
   - Satellite masks are preprocessed into **scribble labels** via skeletonization and selective simplification.
   - 1024×1024 binary road masks are divided into 64×64 grid patches of 16×16 each.

2. **Keypoint Extraction**:
   - In each patch:
     - A keypoint is classified based on the patch's structure (intersection, endpoint, or line).
     - The keypoint's coordinates are computed from local road geometry.

3. **Link Prediction**:
   - Each keypoint is evaluated for connectivity to its 8 neighbors (N, NE, E, SE, S, SW, W, NW).
   - Valid links are stored in an `anchor_link` matrix.

4. **Model Training**:
   - The model is trained on 1024×1024 image-label pairs.
   - Loss functions optimize classification, regression (keypoint coordinates), and link prediction.

5. **Postprocessing**:
   - Outputs are reassembled into road network graphs.
   - Optionally converted to shapefiles for GIS applications.


## Inputs

- Satellite road masks (1024×1024 `.jpg`)
- Naming Convention xxx_sat.jpg, xxx_mask.png, xxx_mask.mat
- Each image is preprocessed into:
  - A **scribble mask** (`./scribble/`)
  - Keypoint presence (`if_key_points`)
  - Keypoint coordinates (`all_key_points_position`)
  - Link predictions (`anchor_link`)


## Installation

git clone https://github.com/lokrim/PaRK-Detect.git
cd PaRK-Detect
pip install -r requirements.txt


## Citation

This repository is based on the official implementation of:
PaRK-Detect: Towards Efficient Multi-Task Satellite Imagery Road Extraction via Patch-Wise Keypoints Detection
Shenwei Xie (BUPT PRIS), BMVC 2022
