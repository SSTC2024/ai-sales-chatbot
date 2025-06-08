# 📖 Installation Guide - Natural Language AI Sales ChatBot

## System Requirements

### Hardware Requirements

#### Minimum Configuration
- **GPU**: NVIDIA RTX 4070Ti Super 16GB or RTX 4080 16GB
- **CPU**: Intel i7-10700K or AMD Ryzen 7 3700X (8+ cores recommended)
- **RAM**: 32GB DDR4/DDR5
- **Storage**: 50GB free space (SSD recommended for model caching)
- **OS**: Windows 10/11 (64-bit)

#### Recommended Configuration
- **Primary GPU**: NVIDIA RTX 4090 24GB (for LLM processing)
- **Secondary GPU**: NVIDIA RTX 4070Ti Super 16GB (for embeddings)
- **CPU**: Intel i9-12900K or AMD Ryzen 9 5900X
- **RAM**: 64GB DDR5
- **Storage**: 1TB NVMe SSD
- **OS**: Windows 11 (64-bit)

#### Future-Proof Configuration
- **Multi-GPU**: 2-4x RTX 5070Ti/5080 16GB (when available)
- **CPU**: Latest Intel i9 or AMD Ryzen 9
- **RAM**: 128GB DDR5
- **Storage**: 2TB NVMe SSD

### Software Requirements
- **Python**: 3.9, 3.10, or 3.11 (3.12+ not yet fully supported by all dependencies)
- **CUDA**: 11.8 or 12.1+ (for GPU acceleration)
- **Visual Studio Build Tools**: For compiling certain packages
- **Git**: For repository management (optional)

## Installation Methods

### Method 1: Automated Setup (Recommended for Windows)

#### Step 1: Download Repository
```bash
# Using Git (recommended)
git clone https://github.com/yourusername/ai-sales-chatbot.git
cd ai-sales-chatbot

# Or download ZIP and extract
```

#### Step 2: Run Automated Setup
```bash
# Run as Administrator for best results
setup.bat
```

The setup script will:
- ✅ Verify Python and pip installation
- ✅ Detect NVIDIA GPU and CUDA support
- ✅ Create project directory structure
- ✅ Install PyTorch with CUDA support
- ✅ Install all required dependencies
- ✅ Test package imports
- ✅ Create configuration files
- ✅ Generate startup scripts

### Method 2: Manual Installation

#### Step 1: Install Python
1. Download Python 3.9+ from [python.org](https://python.org)
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation:
```bash
python --version
pip --version
```

#### Step 2: Install CUDA Toolkit
1. Download CUDA 11.8 or 12.1 from [NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit)
2. Follow NVIDIA installation instructions
3. Verify installation:
```bash
nvidia-smi
nvcc --version
```

#### Step 3: Install PyTorch
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (not recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Step 4: Install Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# Or install manually:
pip install transformers>=4.35.0
pip install sentence-transformers>=2.2.2
pip install accelerate>=0.24.0
pip install bitsandbytes>=0.41.0
pip install PyYAML>=6.0.1
pip install pandas numpy scikit-learn
pip install PyPDF2 python-docx openpyxl
pip install easyocr opencv-python Pillow
pip install requests beautifulsoup4 googlesearch-python
pip install psutil GPUtil nvidia-ml-py3
```

#### Step 5: Create Configuration
Create `chatbot_config.yaml` in the project directory:
```yaml
version: "1.0"
environment: "production"

ai_models:
  primary_llm: "meta-llama/Llama-3.2-3B-Instruct"
  fallback_llm: "microsoft/DialoGPT-medium"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

gpu_config:
  use_quantization: true
  max_memory_per_gpu: 0.85

search_config:
  local_similarity_threshold: 0.7
  enable_google_search: true

performance:
  temperature: 0.7
  max_response_length: 512
```

### Method 3: Docker Installation (Advanced)

#### Prerequisites
- Docker Desktop with WSL2
- NVIDIA Container Toolkit

#### Build and Run
```bash
# Clone repository
git clone https://github.com/yourusername/ai-sales-chatbot.git
cd ai-sales-chatbot

# Build Docker image
docker build -t ai-sales-chatbot .

# Run with GPU support
docker run --gpus all -p 8000:8000 ai-sales-chatbot
```

## Verification and Testing

### Step 1: Import Testing
```bash
# Test core imports
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import sentence_transformers; print('Sentence Transformers: OK')"
python -c "import yaml; print('PyYAML: Configuration support enabled')"
```

### Step 2: GPU Testing
```bash
# Check GPU detection
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} - {props.total_memory/1024**3:.1f}GB')
"
```

### Step 3: Model Loading Test
```bash
# Test model loading (this will download models)
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
print('Model loading test: SUCCESS')
"
```

### Step 4: Application Launch
```bash
# Launch the application
python natural_language_chatbot.py
```

You should see:
```
✅ Configuration loaded from chatbot_config.yaml
Primary LLM: meta-llama/Llama-3.2-3B-Instruct
Found 1 GPU(s)
GPU 0: NVIDIA GeForce RTX 4070 Ti SUPER - 16.0GB
✅ Language model loaded successfully
✅ Embedding model loaded successfully
✅ All AI models loaded successfully!
```

## Configuration Setup

### Directory Structure
After installation, your project should look like:
```
ai-sales-chatbot/
├── natural_language_chatbot.py      # Main application
├── chatbot_config.yaml             # Configuration file
├── requirements.txt                # Dependencies
├── setup.bat                       # Setup script
├── start_chatbot.bat              # Launch script
├── README.md                       # Documentation
├── data/                           # Training data
│   ├── excel/                      # Product catalogs
│   ├── pdf/                        # Documentation
│   └── images/                     # Product images
├── logs/                           # Application logs
├── models/                         # AI model cache
├── backups/                        # Database backups
└── docs/                           # Documentation
    ├── QUICK_START.md
    ├── INSTALLATION.md
    └── CONFIGURATION.md
```

### Configuration File Location
The `chatbot_config.yaml` file must be in the **same directory** as `natural_language_chatbot.py`.

**Correct Structure:**
```
your-project-folder/
├── natural_language_chatbot.py
├── chatbot_config.yaml          ← Same directory
└── other files...
```

### Model Download and Caching
On first run, the system will download AI models:
- **Llama 3.2 3B**: ~6GB download, ~7GB VRAM usage (8-bit)
- **Sentence Transformers**: ~500MB download, ~1GB VRAM usage
- **Total storage**: ~20GB with cache and temporary files

Models are cached in:
- **Windows**: `C:\Users\{username}\.cache\huggingface\`
- **Custom location**: Set `HF_HOME` environment variable

## Troubleshooting

### Common Installation Issues

#### Issue 1: Python Not Found
```
'python' is not recognized as an internal or external command
```
**Solution:**
1. Reinstall Python with "Add to PATH" checked
2. Or manually add Python to PATH environment variable

#### Issue 2: CUDA Not Available
```
CUDA not available. Using CPU mode.
```
**Solutions:**
1. Install NVIDIA drivers: [geforce.com/drivers](https://geforce.com/drivers)
2. Install CUDA toolkit: [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)
3. Verify with `nvidia-smi` command

#### Issue 3: PyTorch Installation Fails
```
ERROR: Could not find a version that satisfies the requirement torch
```
**Solutions:**
1. Update pip: `python -m pip install --upgrade pip`
2. Use specific PyTorch index: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
3. Install Visual Studio Build Tools if on Windows

#### Issue 4: Memory Errors During Model Loading
```
OutOfMemoryError: CUDA out of memory
```
**Solutions:**
1. Enable quantization in config:
```yaml
gpu_config:
  use_quantization: true
  max_memory_per_gpu: 0.8
```
2. Close other GPU applications
3. Restart system to clear GPU memory

#### Issue 5: Configuration File Not Found
```
⚠️ chatbot_config.yaml not found, using default settings
```
**Solutions:**
1. Ensure `chatbot_config.yaml` is in the same directory as the Python script
2. Check file name spelling and extension
3. Run setup script to create default configuration

#### Issue 6: Import Errors
```
ModuleNotFoundError: No module named 'transformers'
```
**Solutions:**
1. Activate virtual environment if using one
2. Reinstall requirements: `pip install -r requirements.txt`
3. Check Python environment: `pip list | grep transformers`

### Performance Issues

#### Slow Model Loading
- **Cause**: Models downloading from internet
- **Solution**: Wait for initial download, subsequent loads will be faster

#### Slow Response Times
- **Cause**: Running on CPU instead of GPU
- **Solution**: Verify CUDA installation and GPU detection

#### High Memory Usage
- **Cause**: Large models loaded in full precision
- **Solution**: Enable quantization in configuration

### Getting Help

#### Log Files
Check these locations for detailed error information:
- `logs/chatbot.log` - Application logs
- Console output during installation
- Windows Event Viewer for system-level errors

#### System Information
Gather this information when reporting issues:
```bash
# Python environment
python --version
pip list

# GPU information
nvidia-smi

# System information
systeminfo  # Windows
```

#### Community Support
- **GitHub Issues**: [Repository Issues Page]
- **Documentation**: Check docs/ folder
- **Configuration Help**: Use Configuration tab in application

## Advanced Installation Options

### Virtual Environment Setup
```bash
# Create virtual environment
python -m venv chatbot_env

# Activate (Windows)
chatbot_env\Scripts\activate

# Activate (Linux/Mac)
source chatbot_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Multiple Python Versions
If you have multiple Python versions:
```bash
# Use specific Python version
python3.9 -m pip install -r requirements.txt
python