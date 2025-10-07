@echo off
echo Setting up GPU Virtual Environment for YOLO Training...
echo.

REM Remove existing venv if it exists
if exist venv_gpu_clean (
    echo Removing existing venv_gpu_clean...
    rmdir /s /q venv_gpu_clean
)

REM Create new virtual environment
echo Creating new virtual environment...
python -m venv venv_gpu_clean

REM Activate the virtual environment
echo Activating virtual environment...
call venv_gpu_clean\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA 12.9
echo Installing PyTorch with CUDA 12.9 support...
pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 --index-url https://download.pytorch.org/whl/cu129

REM Install other requirements
echo Installing other ML packages...
pip install ultralytics opencv-python Pillow numpy pandas matplotlib tqdm PyYAML

REM Verify installation
echo.
echo Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

echo.
echo Setup complete! To activate this environment in the future, run:
echo     venv_gpu_clean\Scripts\activate.bat
echo.
echo Then you can run your training with:
echo     python train_yolo.py --model n --epochs 100 --device cuda