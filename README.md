# Real-Time Tube Defect Detection using YOLOv8 and Streamlit

## Project Overview

This project presents a real-time tube defect detection system using the YOLOv8 object detection framework. The system is capable of detecting defective and non-defective tubes from images, videos, and live webcam streams. A user-friendly Streamlit interface is used to visualize detections in real time.

The objective is to automate quality inspection in industrial environments, reducing manual inspection time and improving detection accuracy.

---

## Features

- Real-time tube defect detection
- YOLOv8-based object detection
- Streamlit web interface
- Webcam live detection
- Video file processing
- Image upload and prediction
- Confidence score display
- GPU acceleration support
- Easy deployment using Streamlit

---

## Classes

| Class ID | Class Name |
|-----------|------------|
| 0 | all_okay |
| 1 | defected |

---

## Dataset Structure

```text
tube/
│
├── train/
│   ├── images/
│   └── labels/
|
└── data.yaml
```

### data.yaml

```yaml
names:
  0: all_okay
  1: defected

path: /content/tube

train: train/images
val: valid/images
```

---

## Technologies Used

- Python
- YOLOv8
- OpenCV
- Streamlit
- PyTorch
- NumPy
- Google Colab
- VS Code

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/tube-defect-detection.git
cd tube-defect-detection
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

Create a file named `requirements.txt`

```text
streamlit
ultralytics
opencv-python
numpy
Pillow
torch
torchvision
```

Install using:

```bash
pip install -r requirements.txt
```

---

## Training the Model

### Import Libraries

```python
from ultralytics import YOLO
```

### Load YOLOv8 Model

```python
model = YOLO("yolov8n.pt")
```

### Train Model

```python
results = model.train(
    data="/content/tube/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=2,
    project="/content/drive/MyDrive/tube_YOLOv8_Results",
    name="tube_defects",
    exist_ok=True
)
```

### Best Weights

After training, the best model is saved as:

```text
runs/detect/tube_defects/weights/best.pt
```

---

## Testing on Images

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="test.jpg",
    conf=0.25
)

results[0].show()
```

---

## Testing on Videos

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="video.mp4",
    save=True,
    conf=0.25
)
```

---

## Streamlit Deployment

### Run Application

```bash
streamlit run app.py
```

The application opens at:

```text
http://localhost:8501
```

---

## Project Structure

```text
tube-defect-detection/
│
├── app.py
├── best.pt
├── requirements.txt
├── README.md
│
├── dataset/ https://drive.google.com/drive/folders/1C9gBm1ly0m_DBwwGF8Ctj9Rhd6RfxTF_?usp=sharing

---

## Working Methodology

### Step 1: Dataset Preparation

- Collect tube images.
- Annotate defective regions.
- Organize dataset into train and validation sets.

### Step 2: Model Training

- Train YOLOv8n on the prepared dataset.
- Monitor loss and mAP metrics.
- Save best-performing weights.

### Step 3: Real-Time Detection

- Load trained model.
- Capture frames from webcam/video.
- Perform inference frame-by-frame.
- Draw bounding boxes around detected defects.
- Display confidence scores.

### Step 4: Streamlit Visualization

- Upload image or video.
- View detection results.
- Perform live camera inspection.

---

## Results

The trained model successfully detects defective tubes in real-time environments and can be used for industrial quality inspection systems.

### Performance Highlights

- Fast inference speed
- Real-time processing capability
- Accurate defect detection
- Easy deployment using Streamlit

---

## Future Scope

- Defect severity classification
- Defect segmentation using YOLOv8-Seg
- Conveyor belt integration
- Object tracking using DeepSORT
- Edge deployment on Jetson Nano
- Cloud-based monitoring dashboard

---

## Applications

- Manufacturing Quality Control
- Industrial Inspection Systems
- Smart Factory Automation
- Conveyor Belt Monitoring
- Production Line Defect Detection

---
## Results
https://drive.google.com/drive/folders/1ft0KyTv5e4UkHv_p6b8ZSW_cUbk3xXef?usp=sharing

## Author

### Noor Tandon

B.Tech Computer Engineering 

Thapar Institute of Engineering and Technology

---

## Acknowledgements

- Ultralytics YOLOv8
- Streamlit
- OpenCV
- PyTorch
- Google Colab

---

## License

This project is developed for internship purposes.

© 2026 Noor Tandon Computer Engineering
