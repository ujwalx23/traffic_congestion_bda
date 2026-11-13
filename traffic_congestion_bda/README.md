# Smart City Traffic Congestion Prediction Using Big Data
### BDA Mini Project — Semester VII (Academic Year 2025–26)
**Department of Computer Engineering, Shree L. R. Tiwari College of Engineering, Thane, Maharashtra**  
- **Students:** Shaikh Rimsha Ahbar Ahmed (Roll 09), Singh Ujwal Rakesh (Roll 21), Suvarna Kshitij Damodar (Roll 25), Thakur Prabhakur Ramesh (Roll 29)  
- **Project Guide:** Ms. Praniti Patil  

---

## 🚦 Project Overview
This project delivers an end-to-end Big Data Analytics (BDA) pipeline that ingests continuous 15-minute sensor streams across six city junctions, performs distributed-style time-series feature engineering, and **anticipates traffic congestion levels 30 minutes in advance** (Low, Medium, High). 

The system benchmarks multiple machine learning algorithms against a **Persistence Baseline** ("no-change" heuristic) and addresses severe class imbalance (**82.7% Low, 10.2% Medium, 7.1% High**) using **Macro F1 evaluation**.

---

## 📁 Repository Structure
```text
traffic_congestion_bda/
│
├── train.py                   # Complete data simulation, feature engineering, and model training script
├── figs.py                    # Generates publication figures (Figures 4.1 to 4.8) into figures/
├── app.py                     # Interactive FastAPI Web Dashboard & Live 30-Min Predictor
├── spark_pipeline.py          # Apache Spark (PySpark) distributed scaling pipeline script
├── requirements.txt           # Python dependencies
│
├── run_dashboard.bat          # 1-Click launcher for the interactive Web Dashboard (port 8000)
├── run_pipeline.bat           # 1-Click launcher to re-train models and regenerate figures
│
├── traffic_large.csv          # 51,810 synthetic 15-minute sensor records over 90 days
├── traffic_sample.csv         # 50-row extract illustrating dataset schema (matches Table 4.2)
├── results.json               # Full model evaluation metrics, confusion matrix & rankings
├── predictions_test.csv       # Test set ground truth vs predictions across all models
│
├── figures/                   # Generated high-resolution academic charts (PNG, 300 DPI)
│   ├── fig1_class_dist.png            # Figure 4.1: Congestion class distribution
│   ├── fig2_hourly_profile.png        # Figure 4.2: Hourly bimodal congestion curves
│   ├── fig3_junction_hotspots.png      # Figure 4.3: Junction hot-spot ranking
│   ├── fig4_context_effects.png       # Figure 4.4: Context factors (rain, events, weekends)
│   ├── fig5_model_comparison.png      # Figure 4.5: Model comparison (Acc, Prec, Rec, F1)
│   ├── fig6_confusion_matrix.png      # Figure 4.6: Confusion matrix of Gradient Boosting
│   ├── fig7_feature_importance.png    # Figure 4.7: Random Forest top 10 features
│   └── fig8_prediction_timeline.png   # Figure 4.8: Actual vs predicted timeline for IT Park
│
├── VIVA_PREPARATION_GUIDE.md  # Comprehensive 25+ question & answer cheatsheet for External Viva
└── README.md                  # Project documentation & instructions
```

---

## ⚡ Quick Start Instructions

### 1. Launch the Interactive Web Dashboard (Recommended for Demo)
Simply double-click:
```bat
run_dashboard.bat
```
Or run from terminal:
```bash
.\.venv\Scripts\python.exe app.py
```
Open your browser at **`http://127.0.0.1:8000`**.

### 2. Run the Full ML Pipeline
To re-run simulation, feature engineering, model training, and chart generation:
```bat
run_pipeline.bat
```
Or manually:
```bash
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe figs.py
```

### 3. Inspect Apache Spark Distributed Code
```bash
.\.venv\Scripts\python.exe spark_pipeline.py
```

---

## 📊 Summary of Results (As Reported in Project Report)

| Model Architecture | Accuracy (%) | Macro Precision | Macro Recall | Macro F1 Score | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 87.3% | 0.684 | 0.684 | 0.684 | — |
| **Logistic Regression** | 89.2% | 0.731 | 0.691 | 0.708 | 0.5s |
| **Decision Tree** | 90.6% | 0.760 | 0.758 | 0.759 | 0.2s |
| **Random Forest** | 91.5% | 0.795 | 0.766 | 0.780 | 4.2s |
| **Gradient Boosting (Best)** | **92.0%** | **0.808** | **0.787** | **0.797** | 42.4s |

- **Best Model Gain:** +11.3 percentage points Macro F1 improvement over Persistence Baseline.
- **High-Congestion Class Detection:** **82.4% Precision**, **76.2% Recall**, **0.792 F1**.
- **Top Predictive Features:** Average Speed (23.6%), Vehicle Count (14.6%), Speed Lag 1 (13.4%), 1-Hour Rolling Speed (11.0%), Hour of Day (10.3%).

---

## 🎓 Viva Voce Quick Reference
For answers to tough external questions like *"Why BDA for 51k rows?"*, *"Why time-based split over K-Fold?"*, and *"How Spark scales this?"*, see [VIVA_PREPARATION_GUIDE.md](file:///C:/Users/singh/.gemini/antigravity/scratch/traffic_congestion_bda/VIVA_PREPARATION_GUIDE.md).

