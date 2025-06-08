# Installation Guide - Natural Language AI Sales ChatBot

## 🎯 **Compatible with Latest sales_chatbot.py**

This guide is specifically updated for the latest `NaturalLanguageSalesChatBot` class with all new features.

## System Requirements

### **Minimum Hardware**
- **GPU**: RTX 4070Ti Super 16GB or equivalent
- **RAM**: 32GB DDR4/DDR5  
- **Storage**: 50GB free space
- **OS**: Windows 10/11 (64-bit)

### **Optimal Hardware (Recommended)**
- **Primary GPU**: RTX 4090 24GB (for LLM)
- **Secondary GPU**: RTX 4070Ti Super 16GB (for embeddings)
- **RAM**: 32GB+ DDR5
- **Storage**: NVMe SSD for model caching

### **Software Requirements**
- **Python**: 3.9, 3.10, or 3.11 (3.12 not fully supported yet)
- **CUDA**: 11.8+ or 12.1+ for optimal GPU performance
- **NVIDIA Drivers**: Latest stable version

## Quick Installation

### **Option 1: Automated Setup (Recommended)**
1. Download `setup.bat` from the project
2. Run as Administrator: Right-click → "Run as administrator"
3. Follow the prompts
4. Wait 10-15 minutes for package installation

### **Option 2: Manual Installation**

#### **Step 1: Install Python**
```bash
# Download Python 3.10 from python.org
# ⚠️ IMPORTANT: Check "Add Python to PATH" during installation
python --version  # Should show 3.10.x
pip --version     # Verify pip is available
```

#### **Step 2: Install CUDA Toolkit (for GPU acceleration)**
```bash
# Download from: https://developer.nvidia.com/cuda-toolkit
# Install CUDA 11.8 or 12.1
# Verify installation:
nvidia-smi
nvcc --version
```

#### **Step 3: Create Project Directory**
```bash
mkdir ai-sales-chatbot
cd ai-sales-chatbot

# Create subdirectories
mkdir config data logs models backups
mkdir data\excel data\pdf data\word data\images
```

#### **Step 4: Install Python Packages**
```bash
# Create virtual environment (recommended)
python -m venv chatbot_env
chatbot_env\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install AI/ML packages
pip install transformers>=4.35.0
pip install sentence-transformers>=2.2.2
pip install accelerate>=0.24.0
pip install bitsandbytes>=0.41.0

# Install document processing
pip install PyPDF2 python-docx openpyxl
pip install easyocr opencv-python Pillow

# Install web search capabilities
pip install googlesearch-python requests beautifulsoup4

# Install system monitoring
pip install psutil GPUtil nvidia-ml-py3

# Install utilities
pip install PyYAML tqdm numpy pandas scikit-learn
```

## Configuration

### **Default Configuration File**
The system will auto-create `config/chatbot_config.yaml`:

```yaml
# AI Model Configuration
ai_models:
  primary_llm: "meta-llama/Llama-2-7b-chat-hf"
  fallback_llm: "microsoft/DialoGPT-medium"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

# GPU Configuration  
gpu_config:
  enable_multi_gpu: true
  use_quantization: true
  mixed_precision: true
  max_memory_per_gpu: 0.8

# Search Configuration
search_config:
  local_similarity_threshold: 0.7
  enable_google_search: true
  max_google_results: 3

# Response Generation
response_config:
  max_response_length: 512
  temperature: 0.7
  repetition_penalty: 1.1
```

### **GPU Memory Allocation**
For RTX 4090 + RTX 4070Ti Super setup:

```yaml
gpu_allocation:
  cuda:0:  # RTX 4090 24GB
    - primary_llm: "Llama-2-7b-chat-hf"
    - text_generation_pipeline
    - conversation_context_management
    
  cuda:1:  # RTX 4070Ti Super 16GB  
    - embedding_model: "all-MiniLM-L6-v2"
    - product_search_index
    - intent_classification
    - ocr_processing
```

## First Run

### **1. Launch the Application**
```bash
# Option A: Use startup script
start_chatbot.bat

# Option B: Direct Python execution
python sales_chatbot.py
```

### **2. Initial Model Download**
- **First run takes 10-15 minutes** (downloads AI models)
- **Llama 2 7B**: ~13GB download
- **Sentence Transformers**: ~90MB download
- **EasyOCR models**: ~500MB download

### **3. Verify Installation**
Check the chat window for these messages:
```
✅ Language model loaded successfully
✅ Embedding model loaded successfully  
✅ OCR model loaded successfully
🤖 Natural Language AI Sales ChatBot Initialized!
```

## Testing the System

### **1. Add Sample Data**
1. Go to **"Database"** tab
2. Click **"Add Sample Products"**
3. Click **"View Products"** to verify
4. Click **"Search Test"** to test semantic search

### **2. Test Natural Language Generation**
Try these queries in the **"Chat"** tab:
```
"I'm looking for a gaming laptop under $2000"
"What business computers do you recommend?"
"Tell me about wireless gaming accessories" 
"Compare your laptop models for me"
```

### **3. Expected Response Flow**
```
User: "I need a gaming laptop for under $2000"

AI Process:
1. 🔍 Searches local database using embeddings
2. ✅ Finds "Gaming Laptop Pro X1" (similarity: 0.89)
3. 🤖 Generates natural language response using LLM
4. 📊 Source: local_database | Time: 2.3s

Response: "I'd recommend our Gaming Laptop Pro X1 at $1,899.99. 
It features an RTX 4070 graphics card and Intel i7-12700H 
processor with 32GB DDR5 RAM, making it excellent for gaming 
within your budget..."
```

## Training the System

### **1. Upload Product Data**
**Excel/CSV Format** (required columns):
- `name` - Product name
- `description` - Product description  
- `category` - Product category
- `price` - Product price
- `features` - Key features
- `specifications` - Technical specifications
- `availability` - Stock status

**Upload Process**:
1. Go to **"Training"** tab
2. Click **"Upload Excel Files"**
3. Select your product catalog
4. Monitor processing status

### **2. Document Processing**
- **PDF Files**: Product manuals, brochures
- **Word Files**: Product documentation  
- **Images**: Product photos (OCR extracts text)

## Performance Optimization

### **GPU Memory Management**
```python
# Automatic optimization based on available VRAM
RTX 4090 24GB:
  - Llama 2 7B (8-bit): ~7GB
  - Context buffer: ~5GB
  - Available: ~12GB

RTX 4070Ti Super 16GB:
  - Embeddings: ~0.1GB
  - Product database: ~2-4GB
  - OCR models: ~1GB  
  - Available: ~10GB
```

### **Model Selection by Hardware**
```python
# Automatic model selection
if gpu_memory >= 20:  # RTX 4090
    model = "meta-llama/Llama-2-7b-chat-hf"
    quantization = False
elif gpu_memory >= 12:  # RTX 4070Ti Super
    model = "meta-llama/Llama-2-7b-chat-hf" 
    quantization = True
else:  # Fallback
    model = "microsoft/DialoGPT-medium"
    quantization = True
```

## Troubleshooting

### **Common Installation Issues**

#### **1. PyTorch CUDA Issues**
```bash
# Test CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **2. Out of Memory Errors**
```python
# In config/chatbot_config.yaml, reduce:
gpu_config:
  max_memory_per_gpu: 0.6  # Reduce from 0.8 to 0.6
  use_quantization: true   # Enable if not already

response_config:
  max_response_length: 256  # Reduce from 512
```

#### **3. Model Download Failures**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python sales_chatbot.py
```

#### **4. Google Search Not Working**
```bash
# Install/reinstall googlesearch
pip uninstall googlesearch-python
pip install googlesearch-python==1.2.3
```

### **Performance Issues**

#### **Slow Response Times**
- Check GPU utilization: `nvidia-smi`
- Verify models loaded on GPU, not CPU
- Reduce conversation context length
- Enable quantization for large models

#### **Memory Leaks**
- Restart application periodically
- Monitor RAM usage in Task Manager
- Clear conversation history regularly

### **Log Analysis**
Check `logs/chatbot.log` for detailed error information:
```bash
# View recent errors
tail -50 logs/chatbot.log

# Search for specific errors
findstr "ERROR" logs/chatbot.log
```

## Advanced Configuration

### **Custom Model Integration**
```python
# To use different LLM models, modify load_language_model():
model_options = [
    "meta-llama/Llama-2-13b-chat-hf",     # Larger model
    "mistralai/Mistral-7B-Instruct-v0.1", # Alternative
    "microsoft/DialoGPT-large",           # Fallback
]
```

### **Multi-GPU Scaling**
For 4+ GPU setups:
```yaml
gpu_allocation:
  cuda:0: ["primary_llm"]
  cuda:1: ["embedding_model", "product_search"]  
  cuda:2: ["intent_classification", "ocr"]
  cuda:3: ["backup_models", "batch_processing"]
```

### **Production Deployment**
- Enable automatic model caching
- Set up database backups
- Configure performance monitoring
- Implement user authentication
- Add API endpoints for integration

## Support

### **Getting Help**
1. Check the **Quick Start Guide** for common questions
2. Review `logs/chatbot.log` for error details
3. Use `check_requirements.bat` to verify installation
4. Monitor GPU/RAM usage during operation

### **Known Limitations**
- Google search has rate limits (10-15 queries/minute)
- First model load takes 10-15 minutes
- Large datasets require significant VRAM
- OCR accuracy varies with image quality

This installation guide ensures compatibility with the latest `NaturalLanguageSalesChatBot` implementation with all advanced features enabled.