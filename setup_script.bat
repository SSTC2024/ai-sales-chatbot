@echo off
echo ===============================================
echo Natural Language AI Sales ChatBot Setup
echo Configuration-Enabled Version
echo Compatible with RTX 4070Ti Super / RTX 4090
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python version detected:
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not available
    echo Please reinstall Python with pip included
    pause
    exit /b 1
)

echo Creating project directory structure...
if not exist "config" mkdir config
if not exist "data" mkdir data
if not exist "data\excel" mkdir data\excel
if not exist "data\pdf" mkdir data\pdf
if not exist "data\word" mkdir data\word
if not exist "data\images" mkdir data\images
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "backups" mkdir backups
if not exist "docs" mkdir docs

echo ✓ Directory structure created
echo.

REM Check CUDA availability
echo Checking NVIDIA GPU and CUDA...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: NVIDIA GPU driver not detected
    echo The system will run in CPU mode (slower performance)
    echo For optimal performance, install NVIDIA drivers and CUDA toolkit
    set USE_GPU=0
) else (
    echo ✓ NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo.
    echo Checking CUDA installation...
    nvcc --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo WARNING: CUDA toolkit not found
        echo Please install CUDA 11.8+ or 12.1+ for optimal performance
        set USE_GPU=1
    ) else (
        echo ✓ CUDA toolkit detected
        nvcc --version | findstr "release"
        set USE_GPU=2
    )
)

echo.
echo Installing Python packages...
echo This may take 10-20 minutes depending on your internet connection...
echo.

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)

REM Install PyTorch based on GPU availability
if %USE_GPU%==0 (
    echo Installing PyTorch for CPU...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
) else (
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo ✓ PyTorch installed successfully
echo.

REM Install core AI/ML packages
echo Installing core AI/ML packages...
pip install transformers>=4.35.0
pip install sentence-transformers>=2.2.2
pip install accelerate>=0.24.0
pip install bitsandbytes>=0.41.0
pip install huggingface-hub>=0.17.0
pip install safetensors>=0.4.0

if %errorlevel% neq 0 (
    echo WARNING: Some AI packages may have failed to install
    echo This could affect model performance
)

echo Installing configuration management...
pip install PyYAML>=6.0.1

echo Installing data processing packages...
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install scikit-learn>=1.3.0
pip install PyPDF2>=3.0.1
pip install python-docx>=0.8.11
pip install openpyxl>=3.1.2

echo Installing image processing and OCR...
pip install Pillow>=10.0.0
pip install opencv-python>=4.8.0
pip install easyocr>=1.7.0

echo Installing web search capabilities...
pip install requests>=2.31.0
pip install beautifulsoup4>=4.12.0
pip install googlesearch-python>=1.2.3
pip install lxml>=4.9.3

echo Installing system monitoring...
pip install psutil>=5.9.0
pip install GPUtil>=1.4.0
pip install nvidia-ml-py3>=7.352.0

echo Installing utilities...
pip install tqdm>=4.66.0

echo.
echo Testing installation...
echo.

REM Test core imports
echo Testing PyTorch...
python -c "import torch; print('✓ PyTorch version:', torch.__version__)"
if %errorlevel% neq 0 (
    echo ✗ PyTorch import failed
    goto :error
)

echo Testing CUDA availability...
python -c "import torch; print('✓ CUDA available:' if torch.cuda.is_available() else '⚠ CUDA not available (CPU mode)')"

echo Testing Transformers...
python -c "import transformers; print('✓ Transformers version:', transformers.__version__)"
if %errorlevel% neq 0 (
    echo ✗ Transformers import failed
    goto :error
)

echo Testing YAML configuration support...
python -c "import yaml; print('✓ PyYAML installed - Configuration support enabled')"
if %errorlevel% neq 0 (
    echo ✗ PyYAML import failed - Configuration files won't work
    goto :error
)

echo Testing Sentence Transformers...
python -c "import sentence_transformers; print('✓ Sentence Transformers installed')"
if %errorlevel% neq 0 (
    echo ✗ Sentence Transformers import failed
    goto :error
)

echo Testing Google Search...
python -c "import googlesearch; print('✓ Google Search capability available')"
if %errorlevel% neq 0 (
    echo ✗ Google Search import failed - web fallback may not work
)

echo Testing OCR...
python -c "import easyocr; print('✓ OCR capability available')"
if %errorlevel% neq 0 (
    echo ✗ OCR import failed - image processing may not work
)

echo.
echo Creating startup scripts...

REM Create main startup script
echo @echo off > start_chatbot.bat
echo echo Starting Natural Language AI Sales ChatBot... >> start_chatbot.bat
echo echo Configuration-enabled version with YAML support >> start_chatbot.bat
echo echo. >> start_chatbot.bat
echo echo Loading AI models... This may take a few minutes on first run. >> start_chatbot.bat
echo python natural_language_chatbot.py >> start_chatbot.bat
echo echo. >> start_chatbot.bat
echo echo ChatBot stopped. Press any key to exit. >> start_chatbot.bat
echo pause >> start_chatbot.bat

REM Create requirements check script
echo @echo off > check_requirements.bat
echo echo Checking Python package requirements... >> check_requirements.bat
echo python -c "import sys; print('Python version:', sys.version)" >> check_requirements.bat
echo python -c "import torch; print('PyTorch:', torch.__version__, '- CUDA:',