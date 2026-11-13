@echo off
title Smart City Traffic Congestion Prediction - Training Pipeline
echo ======================================================================
echo  Running Big Data Analytics End-to-End Pipeline
echo ======================================================================
cd /d "%~dp0"

echo [Step 1/2] Simulating sensor data and training 4 ML models...
".\.venv\Scripts\python.exe" train.py

echo.
echo [Step 2/2] Generating report publication figures (4.1 to 4.8)...
".\.venv\Scripts\python.exe" figs.py

echo.
echo ======================================================================
echo  Pipeline Execution Complete! Outputs generated in project directory.
echo ======================================================================
pause

