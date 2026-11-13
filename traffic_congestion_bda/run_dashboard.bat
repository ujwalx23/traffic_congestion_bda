@echo off
title Smart City Traffic Congestion Prediction - Web Dashboard
echo ======================================================================
echo  Launching Smart City Traffic Congestion Dashboard...
echo  Shree L. R. Tiwari College of Engineering - BDA Mini Project
echo ======================================================================
cd /d "%~dp0"

echo Starting server on http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
".\.venv\Scripts\python.exe" app.py
pause

