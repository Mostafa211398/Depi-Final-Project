# 🛰️ Satellite Land-Cover Classification with YOLOv8

An end-to-end deep learning project for **classifying land-cover types in satellite imagery** using a fine-tuned **YOLOv8n** model and an interactive **Streamlit** application. The project includes dataset exploration, preprocessing, model training, evaluation, and an interface for analyzing images, videos, live webcam frames, and satellite map screenshots.

## Overview

This project uses the **EuroSAT RGB dataset**, based on Sentinel-2 satellite imagery, to recognize 10 land-cover categories. A pretrained YOLOv8n model is fine-tuned on EuroSAT images, and the saved model weights (`yolo_model.pt`) are loaded by a Streamlit application for inference.

**Important distinction:** EuroSAT provides *image-level land-cover labels*, not individual object bounding boxes. In the training notebook, each image is assigned a bounding box covering the **entire image** to adapt the dataset to YOLO's detection format. Consequently, this model is best understood as a **land-cover classification system implemented with a YOLO detection model**, rather than a system trained to pinpoint individual buildings, roads, or other objects.

## Highlights

- **10 land-cover classes** learned from EuroSAT RGB satellite images.
- **YOLOv8n fine-tuning** with a reproducible training workflow documented in a notebook.
- **Interactive Streamlit dashboard** with a satellite-themed interface.
- **Image inference:** upload an image and view annotated predictions, confidence scores, and detection details.
- **Video inference:** analyze uploaded video frames and download the processed output.
- **Webcam inference:** process frames from a camera connected to the machine running the app.
- **Satellite map workflow:** navigate an embedded map, capture a screenshot, and upload it for analysis.
- **Adjustable confidence threshold** (default: `0.25`).

## Land-Cover Categories

| | Class | | Class |
|---|---|---|---|
| 1 | AnnualCrop | 6 | Pasture |
| 2 | Forest | 7 | PermanentCrop |
| 3 | HerbaceousVegetation | 8 | Residential |
| 4 | Highway | 9 | River |
| 5 | Industrial | 10 | SeaLake |

## Results

The classification report saved in the training notebook reports the following performance on **810 test images**:

| Metric | Reported result |
|---|---:|
| Classification accuracy | **97%** |
| Macro-average precision | **0.97** |
| Macro-average recall | **0.97** |
| Macro-average F1-score | **0.97** |
| Test images | **810** |

These are **results recorded in the provided notebook**, not independently reproduced benchmarks. For the classification report, the notebook uses the **first predicted detection** as the image's predicted class. The reported classification accuracy should not be confused with YOLO bounding-box detection mAP.

## Repository Structure

```text
.
├── app.py                                   # Streamlit inference application
├── Land_Type_Classification_finalCode.ipynb # Dataset exploration, training, evaluation
├── yolo_model.pt                            # Trained YOLO weights
└── README.md                                # Project documentation
```

## Installation and Setup

### Prerequisites

- Python 3.10 or newer is recommended.
- An internet connection to install packages (and to use the embedded map).
- A GPU is optional for running the app; CUDA-capable hardware can accelerate inference.

### 1. Download the repository

Clone your GitHub repository or download its ZIP file, then open a terminal **inside the project folder**.

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install application dependencies

```bash
python -m pip install --upgrade pip
pip install streamlit ultralytics opencv-python numpy pillow
```

Ultralytics will also install its required machine-learning dependencies, including PyTorch, as appropriate for your environment. See [PyTorch's installation guide](https://pytorch.org/get-started/locally/) if you need a specific CUDA build.

### 4. Check the model file

Make sure `yolo_model.pt` is in the **same directory** as `app.py`. The app expects this exact filename and will stop if the file is missing.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit (usually `http://localhost:8501`).

## How to Use the App

Choose an input mode in the Streamlit sidebar and adjust the **Confidence Threshold** as needed.

| Mode | How it works | Supported uploads |
|---|---|---|
| **Image** | Upload a satellite image, click **Analyze Satellite Image**, and inspect the annotated image and confidence scores. | JPG, JPEG, PNG, BMP, WebP, TIF, TIFF |
| **Video** | Upload a video, click **Process Video Feed**, and download the processed MP4 output. | MP4, AVI, MOV, MKV |
| **Webcam** | Enable **Start Live Detection** to analyze frames captured by the machine running Streamlit. | Local camera |
| **Satellite Map** | Navigate the embedded map, take a screenshot, upload it, and click **Analyze Map Screenshot**. | JPG, JPEG, PNG, BMP, WebP |

**Tip:** For best results, use clear satellite or aerial imagery similar in appearance to the EuroSAT training data. Predictions on drastically different image sources may be less reliable.

### Optional: Run a prediction directly in Python

```python
from ultralytics import YOLO

model = YOLO("yolo_model.pt")
results = model.predict("path/to/satellite_image.jpg", conf=0.25)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        print(f"{model.names[class_id]}: {confidence:.2%}")
```

## Dataset and Training Pipeline

### Dataset

The notebook uses the **EuroSAT RGB** satellite image dataset and downloads a copy using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download("apollo2506/eurosat-dataset")
```

The notebook also links to the [EuroSAT RGB dataset download](https://madm.dfki.de/files/sentinel/EuroSAT.zip).

### Workflow

1. **Explore the dataset:** inspect class distributions, example satellite images, image dimensions, corrupted files, and RGB pixel-intensity distributions.
2. **Investigate preprocessing:** calculate RGB channel statistics and visualize transformations such as flipping, rotation, color jitter, and normalization using PyTorch.
3. **Prepare the YOLO dataset:** create training, validation, and testing directories; assign each image its land-cover category; generate a **full-image bounding box** in YOLO annotation format; and write `data.yaml`.
4. **Fine-tune YOLOv8n:** start from pretrained `yolov8n.pt` weights and train on the prepared dataset.
5. **Evaluate and export:** inspect validation metrics, generate a classification report and confusion matrix for the test images, then save the trained weights to `yolo_model.pt`.

The notebook's PyTorch augmentation section is an **exploratory preprocessing step**; the subsequent YOLO training uses its own dataset preparation and Ultralytics training pipeline.

### YOLO Training Configuration

| Parameter | Value used in notebook |
|---|---|
| Base model | `yolov8n.pt` |
| Task | YOLO detection with full-image land-cover labels |
| Maximum epochs | `50` |
| Early-stopping patience | `10` |
| Image size | `416` |
| Batch size | `16` |
| Optimizer | SGD |
| Initial learning rate | `0.001` |
| Warmup epochs | `3` |
| Training device | CUDA GPU |
| Intended YOLO dataset split | 80% train / 17% validation / 3% test |

The training notebook was written for **Google Colab** and uses paths under `/content/`. To rerun it elsewhere, update these paths and install the notebook's additional packages, including `kagglehub`, `torchvision`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `tqdm`, and `joblib`.

## Technical Stack

**Model and machine learning:** Python, Ultralytics YOLOv8, PyTorch, NumPy  
**Application:** Streamlit, OpenCV, Pillow  
**Training and analysis:** Google Colab, KaggleHub, Pandas, Matplotlib, Seaborn, scikit-learn

## Notes and Limitations

- **Classification rather than precise localization:** because the training annotations cover entire images, bounding-box coordinates should not be interpreted as verified outlines of individual land-cover features.
- **Dataset domain:** EuroSAT uses RGB satellite imagery. Results may not transfer equally well to other resolutions, multispectral imagery, map interfaces, or ground-level video/webcam feeds.
- **Map analysis requires a screenshot:** the app does **not** automatically capture imagery from the map or perform live geospatial inference.
- **Webcam uses OpenCV's local camera capture:** the camera must be accessible to the machine running the Streamlit process. Remote deployments will not automatically access a visitor's browser webcam.
- **Model weights:** `yolo_model.pt` is included, so retraining is not required to try the app.
- **Evaluation:** the notebook's reported metrics are specific to its own testing procedure; independent external validation would be needed to establish broader real-world performance.

## Future Improvements

- Train a dedicated **image classification** model or use a YOLO classification head for an architecture better aligned with EuroSAT's image-level labels.
- Add a documented, reproducible inference benchmark and independent test set.
- Improve satellite map integration and add in-app image capture or map-tile analysis where permitted.
- Add batch image inference and exportable prediction summaries.
- Expand evaluation to additional satellite datasets and spatial resolutions.

## Acknowledgments

- [EuroSAT](https://madm.dfki.de/files/sentinel/EuroSAT.zip) for the land-use/land-cover dataset.
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 and its training/inference tools.
- [Streamlit](https://streamlit.io/) for the interactive web interface.

---

*This repository is a machine-learning project and demonstration application for satellite land-cover analysis.*
