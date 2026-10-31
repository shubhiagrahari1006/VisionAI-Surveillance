# 🎯 VisionAI — Intelligent Object Detection & Traffic Analytics

> **Semester V Cornerstone Project**  
> **Team Name:** Team TrickedTriad  
> **Core Technologies:** YOLO11n · ByteTrack · OpenCV · Streamlit Multi-Page · Plotly · Pandas

---

## 📌 Executive Summary

**VisionAI** is a deep-learning-driven intelligent surveillance and campus traffic monitoring suite. It provides real-time object detection, multi-object tracking across video frames, bidirectional virtual line-crossing vehicle counting, and an interactive real-time telemetry analytics dashboard.

Built using **Ultralytics YOLO11n** and **ByteTrack**, the system accurately counts vehicles (cars, motorcycles, buses, trucks) while ensuring zero duplicate counts per trajectory, logging every event to a persistent telemetry data store.

---

## 🚀 Key Features

* **Unified Multi-Page Portal:** Seamlessly navigate between the **Detection Hub** and the **Analytics Dashboard** in a single application with full Light, Dark, and System theme support.
* **Image Object Detection:** Fast inference on static imagery with bounding box annotations, class-level breakdowns, and confidence score tables.
* **Bidirectional Gate Counting:** Distinguishes between **Inbound (Entry)** and **Outbound (Exit)** traffic with trajectory detection and live net on-campus occupancy.
* **Real-Time Speed Estimation:** Frame-to-frame centroid displacement tracking that calculates velocity in $km/h$ with Exponential Moving Average (EMA) jitter filtering.
* **Overspeeding Violation Alerts:** Configurable campus speed limits with visual red glowing bounding boxes on video feeds and dedicated telemetry logging.
* **Live Webcam Surveillance:** Real-time camera feed processing with persistent tracking IDs, on-screen HUD telemetry, and batch telemetry logging.
* **Modern Telemetry Dashboard:** Image 3-inspired design with floating metric cards, top accent glow borders, directional traffic splits, hourly trend charts, and CSV telemetry exports.
* **Telemetry Data Store:** Automatically records detection and crossing events (`timestamp`, `source`, `object_class`, `confidence`, `tracking_id`, `event`) into `data/detection_history.csv`.

---

## 🔄 System Architecture Pipeline

```
  ┌─────────────────────────────────────────────────────────────┐
  │                 Input Ingestion Layer                       │
  │     (Static Images / Recorded Video / Live Webcam Feed)     │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │             Deep Learning Inference Engine                  │
  │            Ultralytics YOLO11n (640x640 px)                 │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │            Multi-Object Tracking (ByteTrack)                │
  │      Kalman Filtering & Data Association (Track IDs)        │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │           Virtual Line-Crossing Counter Logic               │
  │   Coordinate Delta Checking & Single-Count ID Memory        │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │             Data Persistence & Telemetry Log                │
  │              data/detection_history.csv                     │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │        Interactive Streamlit & Plotly Dashboard             │
  │       (KPI Metrics · Time Trends · Filterable Tables)       │
  └─────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Directory Structure

```plaintext
SEM V_ CORNERSTONE PROJECT/
├── app/
│   ├── main.py                 # Multi-Page Unified Entrypoint (st.navigation)
│   ├── detection_view.py       # Detection Hub (Image, Video, Webcam)
│   ├── dashboard.py            # Analytics Dashboard & Visualizations
│   ├── ui_theme.py             # Shared Modern Glassmorphic CSS Theme
│   ├── detection.py            # YOLO11 Inference Engine
│   ├── tracking.py             # ByteTrack Video & Frame Tracking Pipeline
│   ├── counting.py             # Line-Crossing & Class Counter Classes
│   ├── storage.py              # Telemetry CSV Append & Read Helpers
│   ├── test_counting.py        # Unit Test: Class Counter Logic
│   ├── test_detection.py       # Unit Test: Image Inference
│   ├── test_line_counting.py   # Unit Test: Virtual Line Coordinate Thresholding
│   ├── test_storage.py         # Unit Test: Telemetry CSV Write Operations
│   ├── test_tracking.py        # Unit Test: ByteTrack Video Pipeline
│   └── test_video.py           # Unit Test: OpenCV Video Processing
├── data/
│   ├── detection_history.csv   # Main Telemetry Event Database
│   └── test_detection_history.csv
├── input/
│   ├── images/                 # Uploaded & Sample Test Images
│   └── videos/                 # Uploaded & Sample Test Videos
├── output/
│   ├── image_detection/        # Annotated Output Images
│   ├── streamlit_video/        # Processed & Tracked Videos
│   └── tracking_test/
├── models/                     # Dedicated Model Checkpoint Folder
├── yolo11n.pt                  # YOLO11 Nano Weights (5.6 MB)
├── requirements.txt            # Python Dependencies
└── README.md                   # Project Documentation
```

---

## 🛠️ Technology Stack & Dependencies

| Category | Technology / Library | Role |
| :--- | :--- | :--- |
| **Language** | Python 3.12 | Core programming language |
| **Object Detection** | Ultralytics YOLO11n | Real-time object recognition and classification |
| **Object Tracking** | ByteTrack (`bytetrack.yaml`) | Multi-object tracking with persistent IDs |
| **Computer Vision** | OpenCV (`opencv-python`) | Image manipulation, video streaming & writing |
| **User Interface** | Streamlit 1.66 | Reactive, multi-page web application framework |
| **Data Analytics** | Plotly & Pandas | Interactive charts, aggregations & data cleaning |
| **Image Processing** | Pillow (PIL) | Format conversions and display buffers |

---

## ⚡ Setup & Installation

### 1. Clone or Open the Repository
```bash
cd "SEM V_ CORNERSTONE PROJECT"
```

### 2. Activate Virtual Environment
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Unified Application
```bash
streamlit run app/main.py
```
> The application will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Running Validation Tests

You can verify each core module independently using the included unit tests:

```powershell
# Test Object Class Counting Logic
python app/test_counting.py

# Test Line-Crossing Math & Single-Count Mechanism
python app/test_line_counting.py

# Test Image Detection Engine
python app/test_detection.py

# Test CSV Telemetry Storage
python app/test_storage.py

# Test ByteTrack on Sample Video
python app/test_video.py
```

---

## 📊 Monitored Target Classes

The system monitors standard COCO classes, with high-priority specialized tracking for:
- 🚗 **Car**
- 🏍️ **Motorcycle**
- 🚌 **Bus**
- 🚚 **Truck**
- 🚶 **Person (Pedestrian Surveillance)**

---

## 🔮 Future Roadmap (Academic Extensions)

* [x] **Bidirectional Inbound/Outbound Logic:** Differentiating campus entrance vs exit lanes.
* [x] **Estimated Speed Detection:** Pixel displacement over time to flag overspeeding vehicles.
* [x] **Automated PDF/Excel Reporting:** One-click executive multi-page audit report generation with embedded vector charts.
* [ ] **Automatic Number Plate Recognition (ANPR):** OCR-based plate reading using EasyOCR/Tesseract.
* [ ] **SQLite/PostgreSQL Migration:** Transitioning from CSV to a relational database for large-scale production telemetry.

---

## 👨‍💻 Team & Academic Acknowledgements

* **Academic Curriculum:** Semester V Cornerstone Project
* **Team:** Team TrickedTriad
* **Institution:** Computer Science & Engineering
"# VisionAI-Surveillance" 
