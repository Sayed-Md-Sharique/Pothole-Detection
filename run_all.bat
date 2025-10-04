@echo off
echo ================================
echo    Pothole Detection Pipeline
echo ================================
echo.

echo Step 1: Preparing dataset...
python prepare_dataset.py
if %errorlevel% neq 0 (
    echo Error in dataset preparation!
    pause
    exit /b 1
)
echo.

echo Step 2: Training YOLOv8 model...
python train_yolov8.py --size m
if %errorlevel% neq 0 (
    echo Error in training!
    pause
    exit /b 1
)
echo.

echo Step 3: Validating model...
python valid.py
if %errorlevel% neq 0 (
    echo Error in validation!
    pause
    exit /b 1
)
echo.

echo Step 4: Starting live detection...
python live_detection.py
if %errorlevel% neq 0 (
    echo Error in live detection!
    pause
    exit /b 1
)
echo.

echo All steps completed successfully!
pause