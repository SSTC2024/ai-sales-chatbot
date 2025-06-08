@echo off
echo ===============================================
echo Natural Language AI Sales ChatBot Setup
echo Compatible with Latest sales_chatbot.py
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
echo This may take 10-15 minutes depending on your internet connection...
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

if %errorlevel% neq 0 (
    echo WARNING: Some AI packages may have failed to install
    echo This could affect model performance
)

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
pip install PyYAML>=6.0.1
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
echo echo. >> start_chatbot.bat
echo echo Loading AI models... This may take a few minutes on first run. >> start_chatbot.bat
echo python sales_chatbot.py >> start_chatbot.bat
echo echo. >> start_chatbot.bat
echo echo ChatBot stopped. Press any key to exit. >> start_chatbot.bat
echo pause >> start_chatbot.bat

REM Create requirements check script
echo @echo off > check_requirements.bat
echo echo Checking Python package requirements... >> check_requirements.bat
echo python -c "import sys; print('Python version:', sys.version)" >> check_requirements.bat
echo python -c "import torch; print('PyTorch:', torch.__version__, '- CUDA:', torch.cuda.is_available())" >> check_requirements.bat
echo python -c "import transformers; print('Transformers:', transformers.__version__)" >> check_requirements.bat
echo python -c "import sentence_transformers; print('Sentence Transformers: OK')" >> check_requirements.bat
echo pause >> check_requirements.bat

echo ✓ Startup scripts created
echo.

echo Creating configuration file...
(
echo # Natural Language AI Sales ChatBot Configuration
echo # Compatible with latest sales_chatbot.py
echo.
echo # AI Model Configuration
echo ai_models:
echo   primary_llm: "meta-llama/Llama-2-7b-chat-hf"  # Main text generation
echo   fallback_llm: "microsoft/DialoGPT-medium"     # Backup model
echo   embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
echo.
echo # GPU Configuration
echo gpu_config:
echo   enable_multi_gpu: true
echo   use_quantization: true  # 8-bit quantization to save VRAM
echo   mixed_precision: true
echo   max_memory_per_gpu: 0.8  # Use 80%% of available VRAM
echo.
echo # Search Configuration
echo search_config:
echo   local_similarity_threshold: 0.7
echo   enable_google_search: true
echo   max_google_results: 3
echo   search_timeout: 10  # seconds
echo.
echo # Response Generation
echo response_config:
echo   max_response_length: 512
echo   temperature: 0.7
echo   do_sample: true
echo   repetition_penalty: 1.1
echo.
echo # Database Configuration
echo database_config:
echo   auto_backup: true
echo   backup_interval: 24  # hours
echo   max_conversation_history: 1000
echo.
echo # Performance Monitoring
echo monitoring:
echo   enable_performance_tracking: true
echo   log_response_times: true
echo   gpu_monitoring: true
) > config\chatbot_config.yaml

echo ✓ Configuration file created
echo.

echo ===============================================
echo 🎉 SETUP COMPLETED SUCCESSFULLY!
echo ===============================================
echo.
echo Next steps:
echo 1. Place your sales_chatbot.py file in this directory
echo 2. Double-click 'start_chatbot.bat' to launch the application
echo 3. Or run: python sales_chatbot.py
echo.
echo First-time usage:
echo - The AI models will download automatically (2-5 GB)
echo - Add sample products using the Database tab
echo - Upload your product catalogs using the Training tab
echo.
echo Hardware detected:
if %USE_GPU%==2 (
    echo ✅ Optimal setup - NVIDIA GPU with CUDA toolkit
    echo ✅ Models will run on GPU for maximum performance
) else if %USE_GPU%==1 (
    echo ⚠️  GPU detected but CUDA toolkit missing
    echo ⚠️  Install CUDA 11.8+ for optimal performance
) else (
    echo ⚠️  CPU mode - Install NVIDIA drivers for better performance
)

echo.
echo GPU Memory Requirements:
echo - Minimum: 12GB VRAM ^(RTX 4070Ti or better^)
echo - Recommended: 16GB+ VRAM ^(RTX 4070Ti Super+^)
echo - Optimal: 24GB+ VRAM ^(RTX 4090+^)
echo.
echo Troubleshooting:
echo - Run 'check_requirements.bat' to verify installation
echo - Check 'logs\chatbot.log' for detailed error messages
echo - Ensure at least 32GB system RAM for large models
echo.
goto :end

:error
echo.
echo ===============================================
echo ❌ INSTALLATION FAILED
echo ===============================================
echo.
echo Common solutions:
echo 1. Ensure Python 3.9+ is installed with pip
echo 2. Run as Administrator if permission errors occur
echo 3. Check internet connection for package downloads
echo 4. Install Microsoft Visual C++ Redistributable
echo 5. For CUDA errors, install NVIDIA drivers and CUDA toolkit
echo.
echo For detailed help, check the installation logs above.
echo.

:end
pause