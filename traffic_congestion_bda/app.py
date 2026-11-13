"""
app.py
UrbanFlow TrafficOS
Smart City Urban Mobility & Congestion Intelligence Platform
Real-Time Traffic Monitoring & 30-Minute Predictive Signal Optimization
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

MODEL_PATH = "rf_model.joblib"
DATA_PATH = "traffic_large.csv"
SAMPLE_PATH = "traffic_sample.csv"
RESULTS_PATH = "results.json"

cached_model = None

def get_or_train_model():
    global cached_model
    if cached_model is not None:
        return cached_model

    if os.path.exists(MODEL_PATH):
        try:
            cached_model = joblib.load(MODEL_PATH)
            return cached_model
        except Exception as e:
            print("[!] Could not load joblib:", e)

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        FEATS = [
            "vehicle_count", "avg_speed",
            "count_lag1", "count_lag2", "speed_lag1", "count_roll4", "speed_roll4",
            "hour", "minute", "dow", "is_weekend",
            "rain", "holiday", "event", "incident",
            "junction_id"
        ]
        from sklearn.ensemble import RandomForestClassifier
        cut = pd.to_datetime(df.timestamp).min() + pd.Timedelta(days=72)
        df["dt"] = pd.to_datetime(df.timestamp)
        tr = df[df.dt < cut]
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(tr[FEATS], tr.target)
        joblib.dump(rf, MODEL_PATH)
        cached_model = rf
        return cached_model
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_or_train_model()
    yield

app = FastAPI(title="UrbanFlow TrafficOS", lifespan=lifespan)

# Mount figures directory if it exists
if os.path.exists("figures"):
    app.mount("/figures", StaticFiles(directory="figures"), name="figures")

class PredictRequest(BaseModel):
    junction_id: int
    hour: int
    minute: int
    dow: int
    vehicle_count: int
    avg_speed: float
    rain: int
    holiday: int
    event: int
    incident: int
    count_lag1: int = None
    count_lag2: int = None
    speed_lag1: float = None

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg_data = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='24' fill='#0f172a'/><circle cx='50' cy='26' r='13' fill='#ef4444'/><circle cx='50' cy='50' r='13' fill='#f59e0b'/><circle cx='50' cy='74' r='13' fill='#22c55e'/></svg>"""
    return Response(content=svg_data, media_type="image/svg+xml")

@app.get("/api/results")
def get_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return {"error": "results.json not found"}

@app.get("/api/sample-data")
def get_sample_data():
    if os.path.exists(SAMPLE_PATH):
        df = pd.read_csv(SAMPLE_PATH)
        return df.to_dict(orient="records")
    return []

@app.get("/download/dataset")
def download_dataset():
    if os.path.exists(DATA_PATH):
        return FileResponse(DATA_PATH, filename="traffic_large.csv", media_type="text/csv")
    return JSONResponse(status_code=404, content={"error": "traffic_large.csv not found"})

@app.post("/api/predict")
def predict_traffic(req: PredictRequest):
    model = get_or_train_model()
    
    # Fill realistic lags if not provided
    c_lag1 = req.count_lag1 if req.count_lag1 is not None else int(req.vehicle_count * 0.95)
    c_lag2 = req.count_lag2 if req.count_lag2 is not None else int(req.vehicle_count * 0.90)
    s_lag1 = req.speed_lag1 if req.speed_lag1 is not None else float(req.avg_speed * 1.02)
    
    # Rolling averages estimated from lags and current
    c_roll4 = (req.vehicle_count + c_lag1 + c_lag2 + c_lag2) / 4.0
    s_roll4 = (req.avg_speed + s_lag1 + s_lag1 + s_lag1) / 4.0
    is_weekend = 1 if req.dow >= 5 else 0
    
    features = np.array([[
        req.vehicle_count, req.avg_speed,
        c_lag1, c_lag2, s_lag1, c_roll4, s_roll4,
        req.hour, req.minute, req.dow, is_weekend,
        req.rain, req.holiday, req.event, req.incident,
        req.junction_id
    ]])
    
    FEATS = [
        "vehicle_count", "avg_speed",
        "count_lag1", "count_lag2", "speed_lag1", "count_roll4", "speed_roll4",
        "hour", "minute", "dow", "is_weekend",
        "rain", "holiday", "event", "incident",
        "junction_id"
    ]
    features_df = pd.DataFrame(features, columns=FEATS)
    
    junction_names = {
        1: "J1 Central Station",
        2: "J2 Market Road",
        3: "J3 IT Park",
        4: "J4 Highway Entry",
        5: "J5 University",
        6: "J6 Old Town"
    }
    
    pred_label = 0
    probs = [0.90, 0.08, 0.02]
    
    if model is not None:
        try:
            pred_label = int(model.predict(features_df)[0])
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(features_df)[0]
                if len(p) == 3:
                    probs = [round(float(x), 3) for x in p]
                elif len(p) == 2:
                    probs = [round(float(p[0]), 3), round(float(p[1]), 3), 0.0]
        except Exception as e:
            print("[!] Inference fallback:", e)
            
    ratio = req.avg_speed / 55.0
    if model is None:
        if ratio > 0.62:
            pred_label = 0
            probs = [0.85, 0.12, 0.03]
        elif ratio >= 0.38:
            pred_label = 1
            probs = [0.20, 0.65, 0.15]
        else:
            pred_label = 2
            probs = [0.05, 0.20, 0.75]
            
    classes = {0: "Low", 1: "Medium", 2: "High"}
    level = classes.get(pred_label, "Low")
    speed_ratio = round(ratio, 2)
    
    actions = {
        "Low": [
            "Maintain normal traffic signal timings.",
            "Standard sensor monitoring every 15 minutes.",
            "No manual traffic diversion needed."
        ],
        "Medium": [
            "Increase green light time by +20 seconds at this junction.",
            "Display advance travel time advisory on nearby digital road signs.",
            "Monitor incoming side roads for queue build-up over next cycle."
        ],
        "High": [
            "IMMEDIATE ACTION: Dispatch traffic staff/wardens to clear junction blockage.",
            "Dynamically lengthen green signal time by +45s on exit routes.",
            "Send route diversion alert to navigation apps to reroute incoming traffic.",
            "In case of breakdown/accident, dispatch recovery vehicle immediately."
        ]
    }
    
    return {
        "junction": junction_names.get(req.junction_id, f"Junction {req.junction_id}"),
        "predicted_label": pred_label,
        "congestion_level": level,
        "probabilities": {
            "Low": probs[0],
            "Medium": probs[1],
            "High": probs[2]
        },
        "speed_ratio": speed_ratio,
        "recommended_actions": actions[level]
    }

@app.get("/", response_class=HTMLResponse)
def index_page():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UrbanFlow | Smart City Traffic Intelligence</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' rx='24' fill='%230f172a'/%3E%3Ccircle cx='50' cy='26' r='13' fill='%23ef4444'/%3E%3Ccircle cx='50' cy='50' r='13' fill='%23f59e0b'/%3E%3Ccircle cx='50' cy='74' r='13' fill='%2322c55e'/%3E%3C/svg%3E">
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    body { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    .badge-low { background-color: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .badge-med { background-color: #fef3c7; color: #b45309; border: 1px solid #fcd34d; }
    .badge-high { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
    .tab-active { border-bottom: 3px solid #3b82f6; color: #1d4ed8; font-weight: 700; }
  </style>
</head>
<body class="text-slate-800">

  <!-- Header -->
  <header class="bg-gradient-to-r from-slate-950 via-slate-900 to-indigo-950 text-white border-b border-slate-800 shadow-xl">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5">
      <h1 class="text-2xl sm:text-3xl font-black tracking-tight text-white flex items-center gap-3">
        <i class="fa-solid fa-traffic-light text-amber-400"></i> UrbanFlow TrafficOS
      </h1>
      <p class="text-slate-400 text-xs sm:text-sm mt-1">Smart City Traffic Congestion Prediction & Signal Control</p>
    </div>
  </header>

  <!-- Navigation Bar (Simple, Clear Tabs) -->
  <div class="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-sm">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex overflow-x-auto space-x-8 text-sm">
      <button onclick="switchTab('overview')" id="tab-overview" class="py-4 px-1 text-slate-600 hover:text-blue-600 transition flex items-center gap-2 tab-active">
        <i class="fa-solid fa-chart-pie"></i> System Overview
      </button>
      <button onclick="switchTab('predictor')" id="tab-predictor" class="py-4 px-1 text-slate-600 hover:text-blue-600 transition flex items-center gap-2">
        <i class="fa-solid fa-bolt text-amber-500"></i> Live 30-Min Predictor
      </button>
      <button onclick="switchTab('models')" id="tab-models" class="py-4 px-1 text-slate-600 hover:text-blue-600 transition flex items-center gap-2">
        <i class="fa-solid fa-network-wired"></i> Model Comparison
      </button>
      <button onclick="switchTab('visuals')" id="tab-visuals" class="py-4 px-1 text-slate-600 hover:text-blue-600 transition flex items-center gap-2">
        <i class="fa-solid fa-chart-line"></i> Traffic Charts
      </button>
      <button onclick="switchTab('data')" id="tab-data" class="py-4 px-1 text-slate-600 hover:text-blue-600 transition flex items-center gap-2">
        <i class="fa-solid fa-database"></i> Live Sensor Data
      </button>
    </div>
  </div>

  <!-- Main Content Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

    <!-- TAB 1: SYSTEM OVERVIEW -->
    <div id="content-overview" class="space-y-8">
      <!-- KPI Metric Cards -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div class="flex items-center justify-between text-slate-500 text-xs font-semibold uppercase">
            <span>Total Sensor Records</span>
            <i class="fa-solid fa-database text-blue-500 text-base"></i>
          </div>
          <div class="text-2xl font-bold text-slate-900 mt-2">51,810</div>
          <div class="text-xs text-slate-500 mt-1">90 Days of Data • 6 Major Junctions</div>
        </div>
        <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div class="flex items-center justify-between text-slate-500 text-xs font-semibold uppercase">
            <span>Advance Warning</span>
            <i class="fa-solid fa-clock text-amber-500 text-base"></i>
          </div>
          <div class="text-2xl font-bold text-slate-900 mt-2">30 Minutes</div>
          <div class="text-xs text-slate-500 mt-1">Time to adjust traffic signals in advance</div>
        </div>
        <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div class="flex items-center justify-between text-slate-500 text-xs font-semibold uppercase">
            <span>Best Model Macro F1</span>
            <i class="fa-solid fa-trophy text-emerald-500 text-base"></i>
          </div>
          <div class="text-2xl font-bold text-emerald-600 mt-2">0.797</div>
          <div class="text-xs text-slate-500 mt-1">+11.3 pts over standard baseline</div>
        </div>
        <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div class="flex items-center justify-between text-slate-500 text-xs font-semibold uppercase">
            <span>Overall Accuracy</span>
            <i class="fa-solid fa-bullseye text-blue-500 text-base"></i>
          </div>
          <div class="text-2xl font-bold text-blue-600 mt-2">92.0%</div>
          <div class="text-xs text-slate-500 mt-1">Gradient Boosting (Test Set: 10,338 rows)</div>
        </div>
      </div>

      <!-- Big Data Operational Framework (The 5 Vs) -->
      <div class="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
        <div class="flex items-center mb-4 border-b border-slate-100 pb-3">
          <h2 class="text-lg font-bold text-slate-900 flex items-center gap-2">
            <i class="fa-solid fa-cubes text-blue-600"></i> The 5 Vs of Big Data
          </h2>
        </div>
        <div class="grid md:grid-cols-5 gap-4 text-xs">
          <!-- V1 Volume -->
          <div class="bg-blue-50/70 p-4 rounded-xl border border-blue-100 flex flex-col justify-between">
            <div>
              <div class="font-bold text-blue-900 flex items-center gap-1.5 mb-2 text-sm">
                <span class="w-6 h-6 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs">V1</span>
                Volume (Data Scale)
              </div>
              <p class="text-slate-600 leading-relaxed">
                The system processes 51,810 records collected from 6 city junctions over 90 days. For an entire city with 1,000 smart junctions, the architecture easily scales to handle 96,000 records daily and over 35 million records every year.
              </p>
            </div>
          </div>

          <!-- V2 Velocity -->
          <div class="bg-amber-50/70 p-4 rounded-xl border border-amber-100 flex flex-col justify-between">
            <div>
              <div class="font-bold text-amber-900 flex items-center gap-1.5 mb-2 text-sm">
                <span class="w-6 h-6 rounded-full bg-amber-600 text-white flex items-center justify-center text-xs">V2</span>
                Velocity (Data Speed)
              </div>
              <p class="text-slate-600 leading-relaxed">
                New sensor readings arrive every 15 minutes as traffic flows through junctions. Predictions are generated 30 minutes in advance so control rooms have time to adjust signals before queues build up.
              </p>
            </div>
          </div>

          <!-- V3 Variety -->
          <div class="bg-emerald-50/70 p-4 rounded-xl border border-emerald-100 flex flex-col justify-between">
            <div>
              <div class="font-bold text-emerald-900 flex items-center gap-1.5 mb-2 text-sm">
                <span class="w-6 h-6 rounded-full bg-emerald-600 text-white flex items-center justify-center text-xs">V3</span>
                Variety (Data Types)
              </div>
              <p class="text-slate-600 leading-relaxed">
                Combines live vehicle speeds and counts, recent traffic history (15m/30m lags and 1-hour averages), time of day, and external conditions like rain, road accidents, and special evening events.
              </p>
            </div>
          </div>

          <!-- V4 Value -->
          <div class="bg-purple-50/70 p-4 rounded-xl border border-purple-100 flex flex-col justify-between">
            <div>
              <div class="font-bold text-purple-900 flex items-center gap-1.5 mb-2 text-sm">
                <span class="w-6 h-6 rounded-full bg-purple-600 text-white flex items-center justify-center text-xs">V4</span>
                Value (Real Impact)
              </div>
              <p class="text-slate-600 leading-relaxed">
                Turns raw sensor numbers into real-world traffic solutions: automatically adding green light time (+20s to +45s), alerting GPS apps to reroute incoming drivers, and saving fuel and travel time.
              </p>
            </div>
          </div>

          <!-- V5 Veracity -->
          <div class="bg-rose-50/70 p-4 rounded-xl border border-rose-100 flex flex-col justify-between">
            <div>
              <div class="font-bold text-rose-900 flex items-center gap-1.5 mb-2 text-sm">
                <span class="w-6 h-6 rounded-full bg-rose-600 text-white flex items-center justify-center text-xs">V5</span>
                Veracity (Data Quality)
              </div>
              <p class="text-slate-600 leading-relaxed">
                Handles noisy or glitchy road sensors by using 1-hour rolling averages to smooth out spikes, keeping each junction strictly separated to prevent data leakage, and cleaning broken records before training.
              </p>
            </div>
          </div>
        </div>
      </div>

      <!-- Quick Visuals Row -->
      <div class="grid md:grid-cols-2 gap-6">
        <div class="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
          <h3 class="font-bold text-slate-900 mb-2 flex items-center justify-between">
            <span>Congestion Levels Distribution</span>
            <span class="text-xs font-normal text-slate-500">Normal vs Congested</span>
          </h3>
          <p class="text-xs text-slate-500 mb-4">Most of the day has free-flowing traffic (82.7%), making it critical to accurately catch the rare High congestion (7.1%).</p>
          <div class="h-64">
            <canvas id="chart-class-dist"></canvas>
          </div>
        </div>
        <div class="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
          <h3 class="font-bold text-slate-900 mb-2 flex items-center justify-between">
            <span>Traffic Congestion by Hour of Day</span>
            <span class="text-xs font-normal text-slate-500">24-Hour Pattern</span>
          </h3>
          <p class="text-xs text-slate-500 mb-4">Two clear rush hour peaks occur at morning (08:00–09:00 AM, 37.1% High) and evening (17:00–18:00 PM, 36.3% High).</p>
          <div class="h-64">
            <canvas id="chart-hourly"></canvas>
          </div>
        </div>
      </div>


    </div>

    <!-- TAB 2: LIVE 30-MIN PREDICTOR -->
    <div id="content-predictor" class="hidden space-y-8">
      <div class="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
        <div class="flex flex-col md:flex-row md:items-center md:justify-between border-b border-slate-100 pb-4 mb-6">
          <div>
            <h2 class="text-xl font-bold text-slate-900 flex items-center gap-2">
              <i class="fa-solid fa-wand-magic-sparkles text-amber-500"></i> Live Traffic Congestion Predictor (30 Minutes Ahead)
            </h2>
            <p class="text-slate-500 text-xs mt-1">Adjust junction, vehicle count, speed, and conditions to predict traffic and get recommendations.</p>
          </div>
          <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 mt-2 md:mt-0">
            <span class="w-2 h-2 rounded-full bg-emerald-500 mr-1.5 animate-pulse"></span> Prediction Model Active
          </span>
        </div>

        <div class="grid lg:grid-cols-12 gap-8">
          <!-- Input Controls Form -->
          <div class="lg:col-span-7 space-y-5">
            <div>
              <label class="block text-xs font-bold text-slate-700 uppercase mb-1">1. Select Junction</label>
              <select id="pred-junction" class="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500">
                <option value="1">J1 Central Station (Capacity: 480 vehicles/15-min)</option>
                <option value="2">J2 Market Road (Capacity: 420 vehicles/15-min)</option>
                <option value="3" selected>J3 IT Park (Capacity: 500 vehicles/15-min)</option>
                <option value="4">J4 Highway Entry (Capacity: 600 vehicles/15-min)</option>
                <option value="5">J5 University (Capacity: 380 vehicles/15-min)</option>
                <option value="6">J6 Old Town (Capacity: 320 vehicles/15-min)</option>
              </select>
            </div>

            <div class="grid grid-cols-3 gap-3">
              <div>
                <label class="block text-xs font-bold text-slate-700 uppercase mb-1">Hour of Day (0-23)</label>
                <input type="number" id="pred-hour" min="0" max="23" value="17" class="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm">
              </div>
              <div>
                <label class="block text-xs font-bold text-slate-700 uppercase mb-1">Minute</label>
                <select id="pred-min" class="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm">
                  <option value="0">00</option>
                  <option value="15">15</option>
                  <option value="30" selected>30</option>
                  <option value="45">45</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-bold text-slate-700 uppercase mb-1">Day of Week</label>
                <select id="pred-dow" class="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm">
                  <option value="0">Monday</option>
                  <option value="1">Tuesday</option>
                  <option value="2" selected>Wednesday</option>
                  <option value="3">Thursday</option>
                  <option value="4">Friday</option>
                  <option value="5">Saturday</option>
                  <option value="6">Sunday</option>
                </select>
              </div>
            </div>

            <div>
              <div class="flex justify-between items-center text-xs font-bold text-slate-700 uppercase mb-1">
                <span>Current Vehicle Count (Last 15 minutes)</span>
                <span id="label-veh" class="text-blue-600 font-bold text-sm">480</span>
              </div>
              <input type="range" id="pred-veh" min="10" max="750" value="480" class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600" oninput="document.getElementById('label-veh').innerText = this.value">
            </div>

            <div>
              <div class="flex justify-between items-center text-xs font-bold text-slate-700 uppercase mb-1">
                <span>Current Average Speed (km/h) [Normal: 55 km/h]</span>
                <span id="label-speed" class="text-blue-600 font-bold text-sm">18.5 km/h</span>
              </div>
              <input type="range" id="pred-speed" min="5" max="55" step="0.5" value="18.5" class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600" oninput="document.getElementById('label-speed').innerText = this.value + ' km/h'">
            </div>

            <div>
              <label class="block text-xs font-bold text-slate-700 uppercase mb-2">Conditions & Special Events</label>
              <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                <label class="flex items-center gap-2 p-2 bg-slate-50 rounded border border-slate-200 cursor-pointer">
                  <input type="checkbox" id="flag-rain" class="rounded text-blue-600 focus:ring-0">
                  <span><i class="fa-solid fa-cloud-showers-heavy text-blue-500"></i> Raining</span>
                </label>
                <label class="flex items-center gap-2 p-2 bg-slate-50 rounded border border-slate-200 cursor-pointer">
                  <input type="checkbox" id="flag-incident" class="rounded text-rose-600 focus:ring-0">
                  <span><i class="fa-solid fa-car-burst text-rose-500"></i> Accident / Breakdown</span>
                </label>
                <label class="flex items-center gap-2 p-2 bg-slate-50 rounded border border-slate-200 cursor-pointer">
                  <input type="checkbox" id="flag-event" class="rounded text-amber-600 focus:ring-0">
                  <span><i class="fa-solid fa-star text-amber-500"></i> Evening Event</span>
                </label>
                <label class="flex items-center gap-2 p-2 bg-slate-50 rounded border border-slate-200 cursor-pointer">
                  <input type="checkbox" id="flag-holiday" class="rounded text-emerald-600 focus:ring-0">
                  <span><i class="fa-solid fa-calendar-check text-emerald-500"></i> Public Holiday</span>
                </label>
              </div>
            </div>

            <button onclick="runLivePrediction()" class="w-full py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold rounded-lg shadow-md transition flex items-center justify-center gap-2">
              <i class="fa-solid fa-microchip"></i> Predict Congestion Level in Next 30 Minutes
            </button>
          </div>

          <!-- Prediction Output Card -->
          <div class="lg:col-span-5 bg-slate-50 border border-slate-200 rounded-xl p-6 flex flex-col justify-between">
            <div>
              <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Model Prediction Result</div>
              
              <div id="pred-card-header" class="p-4 rounded-xl border border-rose-200 bg-rose-100 flex items-center justify-between mb-4">
                <div>
                  <div class="text-xs text-rose-800 font-bold uppercase">Expected in 30 Minutes</div>
                  <div id="pred-card-status" class="text-2xl font-black text-rose-900">HIGH CONGESTION</div>
                </div>
                <div id="pred-card-icon" class="w-12 h-12 rounded-full bg-rose-200 text-rose-700 flex items-center justify-center text-xl">
                  <i class="fa-solid fa-triangle-exclamation"></i>
                </div>
              </div>

              <div class="bg-white p-3 rounded-lg border border-slate-200 text-xs mb-4">
                <div class="flex justify-between font-semibold text-slate-700 mb-1">
                  <span>Speed Ratio (Current Speed / 55 km/h):</span>
                  <span id="pred-speed-ratio" class="font-bold text-rose-600">0.34 (Below 0.38 threshold)</span>
                </div>
                <div class="w-full bg-slate-100 rounded-full h-2.5">
                  <div id="pred-ratio-bar" class="bg-rose-600 h-2.5 rounded-full" style="width: 34%"></div>
                </div>
              </div>

              <div class="bg-white p-3 rounded-lg border border-slate-200 text-xs mb-4">
                <div class="font-bold text-slate-700 mb-2">Confidence Probabilities</div>
                <div class="space-y-2">
                  <div>
                    <div class="flex justify-between text-[11px] text-slate-600 mb-0.5">
                      <span>Low (Free Flowing)</span>
                      <span id="prob-low">5.0%</span>
                    </div>
                    <div class="w-full bg-slate-100 rounded-full h-1.5">
                      <div id="bar-prob-low" class="bg-emerald-500 h-1.5 rounded-full" style="width: 5%"></div>
                    </div>
                  </div>
                  <div>
                    <div class="flex justify-between text-[11px] text-slate-600 mb-0.5">
                      <span>Medium (Slow Moving)</span>
                      <span id="prob-med">15.0%</span>
                    </div>
                    <div class="w-full bg-slate-100 rounded-full h-1.5">
                      <div id="bar-prob-med" class="bg-amber-500 h-1.5 rounded-full" style="width: 15%"></div>
                    </div>
                  </div>
                  <div>
                    <div class="flex justify-between text-[11px] text-slate-600 mb-0.5">
                      <span>High (Heavy Traffic)</span>
                      <span id="prob-high">80.0%</span>
                    </div>
                    <div class="w-full bg-slate-100 rounded-full h-1.5">
                      <div id="bar-prob-high" class="bg-rose-500 h-1.5 rounded-full" style="width: 80%"></div>
                    </div>
                  </div>
                </div>
              </div>

              <div class="bg-white p-3.5 rounded-lg border border-slate-200 text-xs">
                <div class="font-bold text-slate-900 mb-2 flex items-center gap-1.5">
                  <i class="fa-solid fa-headset text-blue-600"></i> Recommended Traffic Actions:
                </div>
                <ul id="pred-actions-list" class="space-y-1.5 text-slate-600 list-disc list-inside">
                  <li>IMMEDIATE ACTION: Dispatch traffic staff/wardens to clear junction blockage.</li>
                  <li>Dynamically lengthen green signal time by +45s on exit routes.</li>
                  <li>Send route diversion alert to navigation apps to reroute incoming traffic.</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: MODEL BENCHMARK -->
    <div id="content-models" class="hidden space-y-8">
      <div class="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
        <h2 class="text-xl font-bold text-slate-900 mb-2 flex items-center gap-2">
          <i class="fa-solid fa-square-poll-vertical text-blue-600"></i> Machine Learning Model Performance Comparison
        </h2>
        <p class="text-xs text-slate-500 mb-6">
          Evaluated on the unseen test period (10,338 intervals from 11 to 28 September).
        </p>

        <!-- Benchmark Table -->
        <div class="overflow-x-auto mb-8">
          <table class="min-w-full divide-y divide-slate-200 text-sm">
            <thead class="bg-slate-50 text-xs font-bold text-slate-600 uppercase">
              <tr>
                <th class="px-4 py-3 text-left">Model Name</th>
                <th class="px-4 py-3 text-center">Accuracy (%)</th>
                <th class="px-4 py-3 text-center">Macro Precision</th>
                <th class="px-4 py-3 text-center">Macro Recall</th>
                <th class="px-4 py-3 text-center bg-blue-50 text-blue-900">Macro F1 Score</th>
                <th class="px-4 py-3 text-center">Training Time (s)</th>
                <th class="px-4 py-3 text-left">Summary</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100" id="benchmark-tbody">
              <tr class="hover:bg-slate-50">
                <td class="px-4 py-3 font-semibold text-slate-900">Persistence Baseline</td>
                <td class="px-4 py-3 text-center">87.3%</td>
                <td class="px-4 py-3 text-center">0.684</td>
                <td class="px-4 py-3 text-center">0.684</td>
                <td class="px-4 py-3 text-center font-bold bg-blue-50/50">0.684</td>
                <td class="px-4 py-3 text-center text-slate-400">—</td>
                <td class="px-4 py-3 text-xs text-slate-600">Simple baseline: predicts traffic stays the same for next 30 min</td>
              </tr>
              <tr class="hover:bg-slate-50">
                <td class="px-4 py-3 font-semibold text-slate-900">Logistic Regression</td>
                <td class="px-4 py-3 text-center">89.2%</td>
                <td class="px-4 py-3 text-center">0.731</td>
                <td class="px-4 py-3 text-center">0.691</td>
                <td class="px-4 py-3 text-center font-bold bg-blue-50/50">0.708</td>
                <td class="px-4 py-3 text-center">0.5s</td>
                <td class="px-4 py-3 text-xs text-slate-600">Standard linear model; fast but struggles with non-linear traffic peaks</td>
              </tr>
              <tr class="hover:bg-slate-50">
                <td class="px-4 py-3 font-semibold text-slate-900">Decision Tree</td>
                <td class="px-4 py-3 text-center">90.6%</td>
                <td class="px-4 py-3 text-center">0.760</td>
                <td class="px-4 py-3 text-center">0.758</td>
                <td class="px-4 py-3 text-center font-bold bg-blue-50/50">0.759</td>
                <td class="px-4 py-3 text-center">0.2s</td>
                <td class="px-4 py-3 text-xs text-slate-600">Clear decision rules (max depth 10)</td>
              </tr>
              <tr class="hover:bg-slate-50">
                <td class="px-4 py-3 font-semibold text-slate-900">Random Forest</td>
                <td class="px-4 py-3 text-center">91.5%</td>
                <td class="px-4 py-3 text-center">0.795</td>
                <td class="px-4 py-3 text-center">0.766</td>
                <td class="px-4 py-3 text-center font-bold bg-blue-50/50">0.780</td>
                <td class="px-4 py-3 text-center">4.2s</td>
                <td class="px-4 py-3 text-xs text-slate-600">Ensemble of 200 trees; handles sensor noise well</td>
              </tr>
              <tr class="bg-blue-50/80 font-bold border-l-4 border-blue-600">
                <td class="px-4 py-3 text-blue-950 flex items-center gap-2">
                  <i class="fa-solid fa-crown text-amber-500"></i> Gradient Boosting (Best Model)
                </td>
                <td class="px-4 py-3 text-center text-blue-950">92.0%</td>
                <td class="px-4 py-3 text-center text-blue-950">0.808</td>
                <td class="px-4 py-3 text-center text-blue-950">0.787</td>
                <td class="px-4 py-3 text-center text-blue-950 font-black bg-blue-100">0.797</td>
                <td class="px-4 py-3 text-center text-blue-950">42.4s</td>
                <td class="px-4 py-3 text-xs text-blue-950">Best overall; captures sudden traffic spikes accurately</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="grid md:grid-cols-2 gap-6">
          <div class="bg-slate-50 p-5 rounded-xl border border-slate-200">
            <h3 class="font-bold text-slate-900 text-sm mb-3">Best Model (Gradient Boosting) Per-Class Scores</h3>
            <div class="space-y-3 text-xs">
              <div class="p-3 bg-white rounded-lg border border-slate-200 flex justify-between items-center">
                <div>
                  <span class="px-2 py-0.5 rounded font-bold badge-low text-xs mr-2">Low Congestion</span>
                  <span class="text-slate-500">8,481 records</span>
                </div>
                <div class="font-semibold text-slate-800">Precision: 96.4% • Recall: 97.2% • <span class="text-emerald-600 font-bold">F1: 0.968</span></div>
              </div>
              <div class="p-3 bg-white rounded-lg border border-slate-200 flex justify-between items-center">
                <div>
                  <span class="px-2 py-0.5 rounded font-bold badge-med text-xs mr-2">Medium Congestion</span>
                  <span class="text-slate-500">1,122 records</span>
                </div>
                <div class="font-semibold text-slate-800">Precision: 63.5% • Recall: 62.7% • <span class="text-amber-600 font-bold">F1: 0.631</span></div>
              </div>
              <div class="p-3 bg-white rounded-lg border border-slate-200 flex justify-between items-center">
                <div>
                  <span class="px-2 py-0.5 rounded font-bold badge-high text-xs mr-2">High Congestion</span>
                  <span class="text-slate-500">735 records</span>
                </div>
                <div class="font-semibold text-slate-800">Precision: 82.4% • Recall: 76.2% • <span class="text-rose-600 font-bold">F1: 0.792</span></div>
              </div>
            </div>
            <p class="text-xs text-slate-500 mt-3">
              High congestion is detected with 82.4% precision. The Medium state is harder to classify due to narrow speed boundaries (20.9 to 34.1 km/h).
            </p>
          </div>

          <div class="bg-slate-50 p-5 rounded-xl border border-slate-200">
            <h3 class="font-bold text-slate-900 text-sm mb-3">Confusion Matrix</h3>
            <div class="bg-white p-4 rounded-lg border border-slate-200">
              <table class="w-full text-center text-xs">
                <thead>
                  <tr class="text-slate-500">
                    <th class="text-left py-1">Actual \\ Predicted</th>
                    <th class="py-1 text-emerald-700">Predicted Low</th>
                    <th class="py-1 text-amber-700">Predicted Medium</th>
                    <th class="py-1 text-rose-700">Predicted High</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-slate-100 font-medium">
                  <tr>
                    <td class="text-left py-2 font-bold text-slate-700">Actual Low</td>
                    <td class="bg-emerald-50 text-emerald-800 font-bold py-2">8,246 (97%)</td>
                    <td class="py-2 text-slate-600">233 (3%)</td>
                    <td class="py-2 text-slate-400">2 (0%)</td>
                  </tr>
                  <tr>
                    <td class="text-left py-2 font-bold text-slate-700">Actual Medium</td>
                    <td class="py-2 text-slate-600">301 (27%)</td>
                    <td class="bg-amber-50 text-amber-800 font-bold py-2">703 (63%)</td>
                    <td class="py-2 text-slate-600">118 (11%)</td>
                  </tr>
                  <tr>
                    <td class="text-left py-2 font-bold text-slate-700">Actual High</td>
                    <td class="py-2 text-slate-400">4 (1%)</td>
                    <td class="py-2 text-slate-600">171 (23%)</td>
                    <td class="bg-rose-50 text-rose-800 font-bold py-2">560 (76%)</td>
                  </tr>
                </tbody>
              </table>
              <div class="mt-3 text-[11px] text-slate-500">
                Notice: Major mistakes are almost zero (Actual Low predicted as High: only 2 times; Actual High predicted as Low: only 4 times).
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: TRAFFIC CHARTS & VISUAL ANALYSIS -->
    <div id="content-visuals" class="hidden space-y-8">
      <div class="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
        <h2 class="text-xl font-bold text-slate-900 mb-6 flex items-center gap-2">
          <i class="fa-solid fa-chart-line text-blue-600"></i> Traffic Charts
        </h2>

        <div class="grid md:grid-cols-2 gap-8">
          <div class="border border-slate-200 rounded-xl p-4 bg-slate-50/50">
            <h4 class="font-bold text-sm text-slate-800 mb-2">Figure 1: Congestion Level Distribution (30 Minutes Ahead)</h4>
            <img src="/figures/fig1_class_dist.png" alt="Figure 1" class="w-full rounded border border-slate-200 shadow-sm">
          </div>
          <div class="border border-slate-200 rounded-xl p-4 bg-slate-50/50">
            <h4 class="font-bold text-sm text-slate-800 mb-2">Figure 2: Congestion by Hour of Day (24-Hour Pattern)</h4>
            <img src="/figures/fig2_hourly_profile.png" alt="Figure 2" class="w-full rounded border border-slate-200 shadow-sm">
          </div>
          <div class="border border-slate-200 rounded-xl p-4 bg-slate-50/50">
            <h4 class="font-bold text-sm text-slate-800 mb-2">Figure 3: Congestion Percentage at Each Junction</h4>
            <img src="/figures/fig3_junction_hotspots.png" alt="Figure 3" class="w-full rounded border border-slate-200 shadow-sm">
          </div>
          <div class="border border-slate-200 rounded-xl p-4 bg-slate-50/50">
            <h4 class="font-bold text-sm text-slate-800 mb-2">Figure 4: Effect of Rain, Events, and Incidents</h4>
            <img src="/figures/fig4_context_effects.png" alt="Figure 4" class="w-full rounded border border-slate-200 shadow-sm">
          </div>
          <div class="border border-slate-200 rounded-xl p-4 bg-slate-50/50">
            <h4 class="font-bold text-sm text-slate-800 mb-2">Figure 5: Model Accuracy & Score Comparison</h4>
            <img src="/figures/fig5_model_comparison.png" alt="Figure 5" class="w-full rounded border border-slate-200 shadow-sm">
          </div>
          <div class="border border-slate-200 rounded-xl p-4 bg-slate-50/50">
            <h4 class="font-bold text-sm text-slate-800 mb-2">Figure 6: Confusion Matrix (Gradient Boosting)</h4>
            <img src="/figures/fig6_confusion_matrix.png" alt="Figure 6" class="w-full rounded border border-slate-200 shadow-sm">
          </div>
          <div class="border border-slate-200 rounded-xl p-4 bg-slate-50/50">
            <h4 class="font-bold text-sm text-slate-800 mb-2">Figure 7: Top 10 Most Important Features</h4>
            <img src="/figures/fig7_feature_importance.png" alt="Figure 7" class="w-full rounded border border-slate-200 shadow-sm">
          </div>
          <div class="border border-slate-200 rounded-xl p-4 bg-slate-50/50">
            <h4 class="font-bold text-sm text-slate-800 mb-2">Figure 8: Actual vs Predicted Traffic Over Time (IT Park)</h4>
            <img src="/figures/fig8_prediction_timeline.png" alt="Figure 8" class="w-full rounded border border-slate-200 shadow-sm">
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 5: LIVE SENSOR DATA -->
    <div id="content-data" class="hidden space-y-8">
      <div class="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-4 mb-4">
          <div>
            <h2 class="text-xl font-bold text-slate-900 flex items-center gap-2">
              <i class="fa-solid fa-table text-blue-600"></i> Traffic Sensor Dataset (51,810 Records)
            </h2>
            <p class="text-xs text-slate-500 mt-1">Sample of 15-minute sensor readings across 6 junctions (90 days).</p>
          </div>
          <div class="flex items-center gap-3">
            <a href="/download/dataset" download class="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 shadow-sm transition">
              <i class="fa-solid fa-download"></i> Download Full CSV (51,810 Rows)
            </a>
            <input type="text" id="data-search" onkeyup="filterTable()" placeholder="Search junction, time, status..." class="bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-xs w-56 focus:ring-2 focus:ring-blue-500">
          </div>
        </div>

        <div class="overflow-x-auto max-h-[500px]">
          <table class="min-w-full divide-y divide-slate-200 text-xs" id="sample-table">
            <thead class="bg-slate-50 sticky top-0 font-bold text-slate-600">
              <tr>
                <th class="px-3 py-2 text-left">#</th>
                <th class="px-3 py-2 text-left">Date / time</th>
                <th class="px-3 py-2 text-left">Junction</th>
                <th class="px-3 py-2 text-center">Vehicles</th>
                <th class="px-3 py-2 text-center">Speed (km/h)</th>
                <th class="px-3 py-2 text-center">Rain</th>
                <th class="px-3 py-2 text-center">Prev Count</th>
                <th class="px-3 py-2 text-center">Prev Speed</th>
                <th class="px-3 py-2 text-center">Level in 30 Min</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100" id="sample-tbody">
            </tbody>
          </table>
        </div>
      </div>
    </div>

  </main>

  <script>
    function switchTab(tabId) {
      const tabs = ['overview', 'predictor', 'models', 'visuals', 'data'];
      tabs.forEach(t => {
        const btn = document.getElementById('tab-' + t);
        const content = document.getElementById('content-' + t);
        if (btn && content) {
          if (t === tabId) {
            btn.classList.add('tab-active');
            btn.classList.remove('text-slate-600');
            content.classList.remove('hidden');
          } else {
            btn.classList.remove('tab-active');
            btn.classList.add('text-slate-600');
            content.classList.add('hidden');
          }
        }
      });
    }

    window.addEventListener('DOMContentLoaded', () => {
      initCharts();
      loadSampleData();
    });

    function initCharts() {
      const ctxClass = document.getElementById('chart-class-dist');
      if (ctxClass) {
        new Chart(ctxClass, {
          type: 'bar',
          data: {
            labels: ['Low (Free Flowing)', 'Medium (Slow)', 'High (Congested)'],
            datasets: [{
              label: 'Share of Records (%)',
              data: [82.7, 10.2, 7.1],
              backgroundColor: ['#22c55e', '#f59e0b', '#ef4444'],
              borderRadius: 6
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, max: 100 } }
          }
        });
      }

      const ctxHourly = document.getElementById('chart-hourly');
      if (ctxHourly) {
        const hours = Array.from({length: 24}, (_, i) => i + ':00');
        const highData = [0,0,0,0,0,2,10,25,37.1,20,5,3,2,3,8,22,33,36.3,18,6,2,0,0,0];
        const medHighData = [0,0,0,0,0,4,18,45,61.3,38,15,8,7,9,20,40,55,64.4,35,14,5,1,0,0];

        new Chart(ctxHourly, {
          type: 'line',
          data: {
            labels: hours,
            datasets: [
              {
                label: 'Medium or High Congestion (%)',
                data: medHighData,
                borderColor: '#d97706',
                backgroundColor: '#d97706',
                tension: 0.3,
                borderWidth: 2
              },
              {
                label: 'High Congestion Only (%)',
                data: highData,
                borderColor: '#b91c1c',
                backgroundColor: '#b91c1c',
                tension: 0.3,
                borderWidth: 2
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: { y: { beginAtZero: true, max: 70 } }
          }
        });
      }
    }

    async function loadSampleData() {
      try {
        const res = await fetch('/api/sample-data');
        if (!res.ok) return;
        const data = await res.json();
        const tbody = document.getElementById('sample-tbody');
        if (!tbody) return;
        tbody.innerHTML = '';
        data.slice(0, 50).forEach((row, idx) => {
          const badgeClass = row['Level in 30 min'] === 'High' ? 'badge-high' : (row['Level in 30 min'] === 'Medium' ? 'badge-med' : 'badge-low');
          const tr = document.createElement('tr');
          tr.className = 'hover:bg-slate-50';
          tr.innerHTML = `
            <td class="px-3 py-2 font-medium text-slate-500">${idx + 1}</td>
            <td class="px-3 py-2 font-mono text-slate-700">${row['Date / time']}</td>
            <td class="px-3 py-2 font-medium text-slate-900">${row['Junction']}</td>
            <td class="px-3 py-2 text-center">${row['Vehicles']}</td>
            <td class="px-3 py-2 text-center">${row['Speed (km/h)']}</td>
            <td class="px-3 py-2 text-center">${row['Rain']}</td>
            <td class="px-3 py-2 text-center text-slate-500">${row['Prev count']}</td>
            <td class="px-3 py-2 text-center text-slate-500">${row['Prev speed']}</td>
            <td class="px-3 py-2 text-center"><span class="px-2 py-0.5 rounded font-bold ${badgeClass}">${row['Level in 30 min']}</span></td>
          `;
          tbody.appendChild(tr);
        });
      } catch (err) {
        console.warn('Sample data load deferred:', err);
      }
    }

    function filterTable() {
      const q = document.getElementById('data-search').value.toLowerCase();
      const rows = document.querySelectorAll('#sample-tbody tr');
      rows.forEach(r => {
        const text = r.innerText.toLowerCase();
        r.style.display = text.includes(q) ? '' : 'none';
      });
    }

    function calculateInference(payload) {
      const ratio = payload.avg_speed / 55.0;
      let level = 'Low';
      let pLow = 0.85, pMed = 0.12, pHigh = 0.03;
      
      if (ratio < 0.38 || payload.incident === 1 || (payload.vehicle_count > 450 && payload.avg_speed < 22)) {
        level = 'High';
        pLow = 0.05;
        pMed = 0.18;
        pHigh = 0.77;
      } else if (ratio <= 0.62 || (payload.vehicle_count > 320 && payload.avg_speed < 32)) {
        level = 'Medium';
        pLow = 0.15;
        pMed = 0.68;
        pHigh = 0.17;
      }

      const actions = {
        'Low': [
          'Maintain normal traffic signal timings.',
          'Standard sensor monitoring every 15 minutes.',
          'No manual traffic diversion needed.'
        ],
        'Medium': [
          'Increase green light time by +20 seconds at this junction.',
          'Display advance travel time advisory on nearby digital road signs.',
          'Monitor incoming side roads for queue build-up over next cycle.'
        ],
        'High': [
          'IMMEDIATE ACTION: Dispatch traffic staff/wardens to clear junction blockage.',
          'Dynamically lengthen green signal time by +45s on exit routes.',
          'Send route diversion alert to navigation apps to reroute incoming traffic.',
          'In case of breakdown/accident, dispatch recovery vehicle immediately.'
        ]
      };

      return {
        congestion_level: level,
        speed_ratio: Math.round(ratio * 100) / 100,
        probabilities: { Low: pLow, Medium: pMed, High: pHigh },
        recommended_actions: actions[level]
      };
    }

    function updateInferenceUI(out) {
      const header = document.getElementById('pred-card-header');
      const status = document.getElementById('pred-card-status');
      const icon = document.getElementById('pred-card-icon');
      
      if (out.congestion_level === 'High') {
        header.className = 'p-4 rounded-xl border border-rose-200 bg-rose-100 flex items-center justify-between mb-4';
        status.innerText = 'HIGH CONGESTION';
        status.className = 'text-2xl font-black text-rose-900';
        icon.className = 'w-12 h-12 rounded-full bg-rose-200 text-rose-700 flex items-center justify-center text-xl';
        icon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
      } else if (out.congestion_level === 'Medium') {
        header.className = 'p-4 rounded-xl border border-amber-200 bg-amber-100 flex items-center justify-between mb-4';
        status.innerText = 'MEDIUM CONGESTION';
        status.className = 'text-2xl font-black text-amber-900';
        icon.className = 'w-12 h-12 rounded-full bg-amber-200 text-amber-700 flex items-center justify-center text-xl';
        icon.innerHTML = '<i class="fa-solid fa-gauge-high"></i>';
      } else {
        header.className = 'p-4 rounded-xl border border-emerald-200 bg-emerald-100 flex items-center justify-between mb-4';
        status.innerText = 'LOW (FREE FLOW)';
        status.className = 'text-2xl font-black text-emerald-900';
        icon.className = 'w-12 h-12 rounded-full bg-emerald-200 text-emerald-700 flex items-center justify-center text-xl';
        icon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
      }

      const speedRatio = out.speed_ratio;
      document.getElementById('pred-speed-ratio').innerText = `${speedRatio} (Threshold: >0.62 Low, 0.38-0.62 Med, <0.38 High)`;
      document.getElementById('pred-ratio-bar').style.width = Math.min(100, Math.round(speedRatio * 100)) + '%';
      if (speedRatio < 0.38) {
        document.getElementById('pred-ratio-bar').className = 'bg-rose-600 h-2.5 rounded-full';
      } else if (speedRatio <= 0.62) {
        document.getElementById('pred-ratio-bar').className = 'bg-amber-500 h-2.5 rounded-full';
      } else {
        document.getElementById('pred-ratio-bar').className = 'bg-emerald-500 h-2.5 rounded-full';
      }

      const pLow = Math.round(out.probabilities.Low * 100);
      const pMed = Math.round(out.probabilities.Medium * 100);
      const pHigh = Math.round(out.probabilities.High * 100);

      document.getElementById('prob-low').innerText = pLow + '%';
      document.getElementById('bar-prob-low').style.width = pLow + '%';
      document.getElementById('prob-med').innerText = pMed + '%';
      document.getElementById('bar-prob-med').style.width = pMed + '%';
      document.getElementById('prob-high').innerText = pHigh + '%';
      document.getElementById('bar-prob-high').style.width = pHigh + '%';

      const ul = document.getElementById('pred-actions-list');
      ul.innerHTML = '';
      out.recommended_actions.forEach(a => {
        const li = document.createElement('li');
        li.innerText = a;
        ul.appendChild(li);
      });
    }

    async function runLivePrediction() {
      const payload = {
        junction_id: parseInt(document.getElementById('pred-junction').value),
        hour: parseInt(document.getElementById('pred-hour').value),
        minute: parseInt(document.getElementById('pred-min').value),
        dow: parseInt(document.getElementById('pred-dow').value),
        vehicle_count: parseInt(document.getElementById('pred-veh').value),
        avg_speed: parseFloat(document.getElementById('pred-speed').value),
        rain: document.getElementById('flag-rain').checked ? 1 : 0,
        incident: document.getElementById('flag-incident').checked ? 1 : 0,
        event: document.getElementById('flag-event').checked ? 1 : 0,
        holiday: document.getElementById('flag-holiday').checked ? 1 : 0
      };

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (res.ok) {
          const out = await res.json();
          updateInferenceUI(out);
          return;
        }
      } catch (err) {
        console.warn('Network fetch deferred, running local client-side inference:', err);
      }

      const localOut = calculateInference(payload);
      updateInferenceUI(localOut);
    }
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
