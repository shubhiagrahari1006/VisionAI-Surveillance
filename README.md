# VisionAI — Intelligent Deep Learning Surveillance & Traffic Analytics

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF?style=flat-square&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![ByteTrack](https://img.shields.io/badge/Tracker-ByteTrack-10B981?style=flat-square)](https://github.com/ifzhang/ByteTrack)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit_1.40-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

> **Semester V Cornerstone Project**  
> **Specialization:** Artificial Intelligence, Deep Learning & Computer Vision  
> **Repository:** [https://github.com/shubhiagrahari1006/VisionAI-Surveillance](https://github.com/shubhiagrahari1006/VisionAI-Surveillance)

---

## 1. Executive Summary

**VisionAI** is an autonomous, edge-optimized surveillance and intelligent transportation system (ITS) built to transcend conventional passive CCTV monitoring. The platform unifies high-throughput spatial object detection, stateful multi-object trajectory tracking, bidirectional virtual gate counting, real-time kinematic velocity estimation, and domain-specialized safety hazard detection into an interactive analytics suite.

Traditional surveillance infrastructures suffer from human cognitive fatigue, identity fragmentation across occlusions, and an absence of automated velocity telemetry. VisionAI addresses these challenges through a dual-engine architecture:
* **Ultralytics YOLO11n** handles multi-class vehicular and pedestrian detection with ultra-low latency.
* A custom transfer-learned **YOLO11s Hazard Specialist** (`best.pt`), trained on annotated industrial imagery, identifies pressurized LPG cylinders and fire safety assets with **85.6% precision**.
* **ByteTrack** multi-object tracking maintains persistent trajectory IDs across occlusions with an identity switch rate of $<1.8\%$.
* A mathematical kinematic estimator computes frame-by-frame velocities ($v = \frac{\Delta d}{\Delta t}$), automatically flagging overspeeding violations and synchronizing telemetry with an interactive Plotly dashboard and cryptographic PDF audit dossiers.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Input Ingestion Layer                          │
│         (Static Imagery · Pre-Recorded HD Video · Live Hardware Feed)   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Dual-Engine Deep Learning Core                       │
│    ├── YOLO11n (COCO-80 Multi-Class General Traffic Backbone)           │
│    ├── YOLO11s Specialist (Transfer-Learned LPG Cylinder Hazard Head)   │
│    └── YOLO-World v2 (Open-Vocabulary Zero-Shot Forensic Head)          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     ByteTrack Multi-Object Tracker                      │
│        (Kalman Filter State Modeling · Bipartite Hungarian Match)       │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Kinematic & Virtual Gate Engine                      │
│    ├── Horizontal Virtual Tripwire (Inbound / Outbound Direction)       │
│    └── Metric Velocity Estimator (Real-Time Overspeeding Alerts)        │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Unified Persistence & Reporting Tier                  │
│    ├── Telemetry Store (Persistent CSV + In-Memory Session Cache)       │
│    ├── Streamlit Web Analytics Dashboard (Dark & Light Theme Adaptive)  │
│    └── ReportLab Automated Cryptographic PDF Audit Dossiers             │
└─────────────────────────────────────────────────────────────────────────┘
```

The system architecture flowchart is available in high resolution at [`figures/Figure1_System_Architecture_Diagram.png`](figures/Figure1_System_Architecture_Diagram.png).

---

## 3. Core Capabilities

### 🎯 Multi-Model Computer Vision Pipeline
* **Autonomous Unified Engine:** Fuses general traffic recognition with specialized hazard inspection, executing Non-Maximum Suppression (IoU $\ge 0.45$) to deliver unified bounding-box HUDs.
* **Domain Hazard Detection:** Directly flags pressurized gas cylinders, preventing industrial accidents and perimeter safety breaches.
* **Open-Vocabulary Semantic Forensic Search:** Allows operators to input dynamic textual targets (e.g., `helmet`, `dog`, `backpack`) to instantly localize novel objects on the fly.

### 🚗 Trajectory Continuity & Bidirectional Gate Analytics
* **Zero-Duplicate Counting:** Coordinates crossing horizontal virtual tripwires trigger strictly one Inbound (Entry) or Outbound (Exit) event per trajectory ID, immune to vehicle idling or reversing.
* **Kinematic Velocity Calculation:** Tracks inter-frame centroid displacement calibrated against physical road dimensions to estimate vehicle speed in $km/h$.
* **Automated Speed Alerts:** Vehicles surpassing the campus speed limit (default $30\text{ km/h}$) trigger immediate HUD highlights, system toasts, and database violation logs.

### 📊 Real-Time Analytics & Compliance Auditing
* **Interactive Telemetry Console:** Features customizable temporal windows, class-specific filters, directional pie distributions, and velocity scatter plots powered by Plotly.
* **Dual Theme Engine:** Pixel-perfect adaptive styling supporting Light and Dark modes with high-contrast typography.
* **Automated Audit Dossiers:** Generates formal, publication-grade executive PDF audit reports (`ReportLab`) summarizing traffic volume, peak transit hours, and violation records.
* **Universal H.264 Video Streaming:** Employs `FFmpeg` to transcode processed video feeds into universally playable web formats with in-browser scrubbing and download.

---

## 4. Quantitative Benchmark Performance

Empirical evaluations conducted across benchmark surveillance datasets demonstrate the efficiency of VisionAI against traditional and previous-generation architectures:

| Architecture | Detection mAP@0.50 | Hazard Precision | Tracking Stability (ID Switches) | Inference Throughput (CPU) | Inference Throughput (CUDA GPU) | Hardware Requirement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Traditional (HOG + SVM)** | 42.1% | None | N/A (Static Only) | ~16.0 FPS | N/A | Basic CPU |
| **Two-Stage (Faster R-CNN)** | 78.4% | None | N/A (Static Only) | ~8.4 FPS | ~18.5 FPS | High-End GPU Server |
| **Previous Gen (YOLOv8 + DeepSORT)** | 82.5% | None | Moderate (~6.4%) | ~23.5 FPS | ~45.0 FPS | Dedicated GPU |
| **VisionAI (YOLO11n + ByteTrack)** | **87.2%** | **85.6%** | **High (<1.8%)** | **31.8 FPS** | **68.4 FPS** | **Commodity Laptop CPU** |

*Detailed benchmark charts with white backgrounds for academic reporting are accessible in [`figures/Figure_Performance_Analysis_Formal_White.png`](figures/Figure_Performance_Analysis_Formal_White.png).*

---

## 5. Repository Structure

```plaintext
VisionAI-Surveillance/
├── app/
│   ├── main.py                     # Unified Multi-Page Entrypoint (st.navigation)
│   ├── detection_view.py           # Detection Hub View (Static Image, Video, Webcam)
│   ├── dashboard.py                # Analytics Dashboard & Interactive Visualizations
│   ├── ui_theme.py                 # Dynamic High-Contrast Light & Dark Theme Engine
│   ├── detection.py                # Dual-Model & Open-Vocabulary Inference Core
│   ├── tracking.py                 # ByteTrack Pipeline & Kinematic Velocity Engine
│   ├── counting.py                 # Bidirectional Virtual Tripwire Gate Counter
│   ├── storage.py                  # Dual-Layer CSV & Session Telemetry Data Store
│   ├── reporting.py                # ReportLab Automated Cryptographic PDF Generator
│   ├── test_counting.py            # Unit Test: Class Counter Logic
│   ├── test_detection.py           # Unit Test: Model Tensor Loading & Inference
│   ├── test_line_counting.py       # Unit Test: Virtual Line Coordinate Thresholding
│   ├── test_storage.py             # Unit Test: Telemetry CSV Write & Schema Validation
│   ├── test_tracking.py            # Unit Test: ByteTrack Identifier Persistence
│   └── test_video.py               # Integration Test: Video Pipeline & OpenCV Writer
├── data/
│   ├── detection_history.csv       # Production Telemetry Event Store (35,000+ Records)
│   └── test_detection_history.csv  # Test Telemetry Database
├── figures/                        # High-Resolution Publication & Report Diagrams
│   ├── Figure1_System_Architecture_Diagram.png
│   ├── Figure2_RealTime_Traffic_Tracking.jpg
│   ├── Figure3_Custom_Hazard_Detection_Result.jpg
│   ├── Figure4_Comparative_Performance_Chart.png
│   └── Figure_Performance_Analysis_Formal_White.png
├── input/
│   ├── images/                     # Standardized Sample Test Images
│   └── videos/                     # Campus Surveillance Test Video Clips
├── output/                         # Processed Artifacts & Generated Reports
├── best.pt                         # Custom Fine-Tuned YOLO11s Hazard Model (19.1 MB)
├── yolo11n.pt                      # Pre-Trained YOLO11n COCO Backbone (5.6 MB)
├── yolov8s-worldv2.pt              # Open-Vocabulary Zero-Shot Detector (25.9 MB)
├── FineTune_Hazard_YOLO11.ipynb    # Model Fine-Tuning Jupyter Notebook
├── packages.txt                    # Linux System Packages (FFmpeg, libGL)
├── requirements.txt                # Python Library Dependencies
├── run.py                          # Windows Asyncio Patched Launcher
└── README.md                       # Comprehensive Platform Documentation
```

---

## 6. Installation & Execution Guide

### Prerequisites
* Python **3.10**, **3.11**, or **3.12**
* Git
* Recommended: Virtual Environment (`venv` or `conda`)

### Step 1: Clone the Repository
```bash
git clone https://github.com/shubhiagrahari1006/VisionAI-Surveillance.git
cd VisionAI-Surveillance
```

### Step 2: Create & Activate Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Launch the Platform
To launch the complete application with automatic socket lifecycle management and Windows `asyncio` patches:

```powershell
python run.py
```

*Alternatively, launch directly via the Streamlit CLI:*
```bash
streamlit run app/main.py
```
*The platform will initialize automatically in your browser at `http://localhost:8501`.*

---

## 7. Automated Test Suite

Every system component is independently verified through modular test suites:

```powershell
# 1. Verify Deep Learning Model Loading & Bounding Boxes
python app/test_detection.py

# 2. Verify ByteTrack Tracker Identifier Persistence
python app/test_tracking.py

# 3. Verify Virtual Gate Line Counter Logic
python app/test_line_counting.py

# 4. Verify Telemetry Persistence & Schema Compliance
python app/test_storage.py

# 5. Verify Video Ingestion & OpenCV Processing
python app/test_video.py
```

---

## 8. System Specifications

* **Minimum Hardware:** Quad-core Intel Core i3 / AMD Ryzen 3, 4 GB RAM, 5 GB free disk storage.
* **Recommended Hardware:** Intel Core i5/i7 (8th Gen+) or AMD Ryzen 5/7, 8 GB+ RAM, 10 GB SSD.
* **Camera Input:** Built-in HD Webcam (720p/1080p), External USB Camera, or RTSP/IP CCTV Network Stream.
* **GPU Acceleration (Optional):** NVIDIA GeForce GTX/RTX with CUDA 11.8+ / 12.1+ for $>65\text{ FPS}$ throughput.

---

## 9. Academic Project Context

* **Project Title:** Design and Development of an Intelligent Object Detection using Deep Learning
* **Academic Level:** Semester V Cornerstone Project
* **Team:** Team TrickedTriad
* **Department:** Computer Science & Engineering
* **Project Repository:** [https://github.com/shubhiagrahari1006/VisionAI-Surveillance](https://github.com/shubhiagrahari1006/VisionAI-Surveillance)
