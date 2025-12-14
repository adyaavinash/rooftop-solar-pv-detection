# Rooftop Solar PV Detection

## Project Description
This project is a machine learning system that detects rooftop solar photovoltaic (PV) installations from satellite imagery using YOLO object detection models. YOLO itself trains a single model, but the project improves performance by applying post-training ensemble techniques, especially **Weighted Box Fusion (WBF)**.

## Project Overview
The aim of this project is to detect rooftop solar PV panels in high-resolution satellite imagery using:
- YOLO models (Ultralytics) for object detection  
- Ensemble learning (WBF) across multiple trained YOLO versions  

This approach increases robustness under varying conditions such as shadows, rooftop colors (blue and reflective asbestos or tarpauline), resolution shifts, and environmental noise.

## Features
- Detects presence/absence of solar PV panels using YOLO  
- Ensemble predictions using Weighted Box Fusion (WBF)  
- Works on high-resolution aerial/satellite imagery  
- Batch inference support (XLSX and CSV files)  
- Saves detection results to JSON as well as CSV files  

## Datasets
The datasets used for the project are as follows:
- Alfred Weber Institute of Economics. (2025). Custom Workflow Object Detection Dataset  
- ProjectSolarPanel. (2025). LSGI547 Project Dataset  
- Piscinas y Tenistable. (2023). Solar Panels Dataset  
- A dataset was generated for hard-negatives using Google API key and trained using Colab  

Typical image sizes: **640 × 640** patches

## Methodology
1. **Train YOLO Models**  
   Each YOLO model was trained independently (YOLOv8s)

2. **Generate Predictions**  
   Run inference using all trained models to get bounding boxes and confidence scores

3. **Apply Ensemble Method**  
   Combine predictions using Weighted Box Fusion (WBF)

4. **Evaluate**  
   Compute mAP, precision, recall, and visualize results

## Why Ensemble?
Single YOLO models may miss detections due to:
- Shadows  
- Low contrast  
- Angled rooftops  
- Panel variations  

Ensembling helps improve:
- Bounding box accuracy  
- Detection confidence  
- Overall mAP  
- Produces fewer false positives  

## Environment Setup
```bash                                             
python -m venv venv                                   #1. Create a virtual environment
.\venv\Scripts\activate                               #2. Activate virtual environment
pip install -r environment/requirements.txt           #3. Install dependencies

```md
## Training the Dataset
The dataset was trained on Google Colab:

```python
model = YOLO("yolov8s.pt")
model.train(
   data="<path>/data.yaml",
   epochs=50,
   imgsz=640,
   batch=16,
   device="cuda"
)

## Ensemble Algorithm
The main prediction logic lies inside pipeline_code/ensemble_api.py.

```md
Usage:
```python
from pipeline_code.ensemble_api import ensemble_predict

result = ensemble_predict("test_images/sample.png")
print(result)

```md
Example Output:
```json
{
   "has_solar": true,
   "confidence": 0.82,
   "pv_area_sqm_est": 12.55,
   "reason": "High-confidence solar detection",
   "detections": [[50, 120, 200, 260]]
}

## Lat/Lon Satellite Prediction
```md
Add your Google API key to `.env`:
```env
GOOGLE_API_KEY=YOUR_KEY

Usage:
```python
python pipeline_code/ensemble_latlon.py 13.152261 77.569047 prediction_files/test4.json

## Batch CSV/XLSX Prediction

Required Columns:
  sample_id, latitude, longitude

Run Batch:
```python
python -m pipeline_code.batch_predict_csv --csv test/samples.csv --out results/
python -m pipeline_code.batch_predict_csv --csv test/Sample1.xlsx --out results/

```md
Outputs:
```text
results/
├── 101.json
├── 102.json
└── results_summary.csv

```md
JSON Output Structure:
```json
  {
    "sample_id": "12345",
    "lat": 12.97,
    "lon": 77.59,
    "has_solar": true,
    "confidence": 0.81,
    "pv_area_sqm_est": 14.23,
    "buffer_radius_sqft": 1200,
    "qc_status": "VERIFIABLE",
    "bbox_or_mask": [[50,112,200,260]],
    "image_metadata": {
    "source": "Google Static Maps",
    "capture_date": "N/A"
  }
}

```md
## Project Strucute
```text
rooftop-solar-pv-detection-final/
├── artefacts/ # Raw negative images (no solar)
├── datasets/ # Main dataset from Roboflow + pre-processed negatives
│ ├── test/
│ ├── train/
│ ├── valid/
│ └── data.yml
├── environment/
│ ├── environment.yml
│ ├── python_version.txt
│ └── requirements.txt
├── model_card/
├── pipeline_code/
│ ├── ensemble_api.py # Solar detection + ensemble logic
│ ├── batch_predict_csv.py # Batch prediction (CSV/XLSX)
│ ├── ensemble_latlon.py
│ ├── ensemble_predict.py
│ ├── evaluate_model.py
│ ├── fetch_sarellite.py
│ ├── utils.py
│ └── init.py
├── trained_model/
│ ├── best_original.pt
│ ├── best_negative.pt
├── training_logs/
├── results/
├── prediction_files/
├── venv/
├── .env
└── README.md

## License

This project is licensed under the MIT License — see the LICENSE file for details.








