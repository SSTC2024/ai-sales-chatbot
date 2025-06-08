# 🚀 Quick Start Guide - Natural Language AI Sales ChatBot

## Overview

This AI ChatBot uses **natural language generation** with local LLM models to provide human-like responses. It searches your local product database first, then falls back to Google search if needed.

## Key Features

✅ **Natural Language Generation** - Uses local LLM (Llama 2, Mistral, etc.)  
✅ **Local Database Priority** - Searches your products first  
✅ **Google Search Fallback** - Finds info online when local data is insufficient  
✅ **Conversation Context** - Remembers previous conversation turns  
✅ **Multi-GPU Support** - RTX 4090 + RTX 4070Ti Super optimization  
✅ **Document Processing** - Excel, PDF, Word file support  

## Hardware Requirements

### Current Setup (Single GPU)
- **RTX 4070Ti Super 16GB** or better
- **32GB RAM** minimum
- **Windows 10/11**

### Optimal Setup (Multi-GPU)
- **RTX 4090 24GB** (Primary - for LLM)
- **RTX 4070Ti Super 16GB** (Secondary - for embeddings)
- **32GB+ RAM**

## Installation

### 1. Install Python 3.9+
```bash
# Download from python.org
python --version  # Should show 3.9+
```

### 2. Install CUDA Toolkit
```bash
# For NVIDIA GPUs - install CUDA 11.8 or 12.1
# Download from: https://developer.nvidia.com/cuda-toolkit
nvidia-smi  # Verify installation
```

### 3. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv chatbot_env
chatbot_env\Scripts\activate  # Windows
# source chatbot_env/bin/activate  # Linux/Mac

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install transformers>=4.35.0
pip install sentence-transformers>=2.2.2
pip install bitsandbytes>=0.41.0
pip install accelerate>=0.24.0
pip install pandas numpy scikit-learn
pip install PyPDF2 python-docx openpyxl
pip install easyocr opencv-python
pip install requests beautifulsoup4 googlesearch-python
pip install PyYAML psutil GPUtil nvidia-ml-py3
```

### 4. Run the ChatBot
```bash
python natural_language_chatbot.py
```

## First Time Setup

### 1. Add Sample Data
1. Click the **"Database"** tab
2. Click **"Add Sample Products"** to add test data
3. Click **"View Products"** to verify data was added

### 2. Test Search Functionality
1. Click **"Search Test"** to verify semantic search works
2. Check that products are found with similarity scores

### 3. Start Chatting
1. Go to the **"Chat"** tab
2. Try these example queries:

```
"I'm looking for a gaming laptop under $2000"
"What business computers do you recommend?"
"Tell me about wireless gaming accessories"
"Compare your laptop models for me"
"I need a computer for video editing"
```

## Expected Behavior

### Response Flow Priority
1. **Local Database Search** - Searches your products using AI embeddings
2. **Google Search** - If no local results, searches web for information
3. **Natural Language Generation** - LLM creates human-like response

### Example Interaction
```
User: "I need a gaming laptop for under $2000"

AI Process:
1. ✅ Searches local database → finds "Gaming Laptop Pro X1"
2. 🤖 Generates natural response using LLM
3. 📊 Source: local_database | Time: 2.3s

AI Response: "I'd recommend our Gaming Laptop Pro X1, which is perfect for 
your budget at $1,899.99. It features an RTX 4070 graphics card and Intel 
i7-12700H processor with 32GB of DDR5 RAM, making it excellent for gaming. 
The 15.6-inch 144Hz display ensures smooth gameplay, and it's currently in 
stock. Would you like to know more about its specific gaming performance 
or see some alternatives?"
```

## Training the System

### Adding Your Product Data

#### Excel/CSV Format
Create files with these columns:
- `name` - Product name
- `description` - Product description  
- `category` - Product category
- `price` - Product price
- `features` - Key features
- `specifications` - Technical specs
- `availability` - Stock status

#### Upload Process
1. Go to **"Training"** tab
2. Click **"Upload Excel Files"**
3. Select your product catalog files
4. Monitor processing in the status window

### Document Processing
- **PDF Files** - Product manuals, brochures
- **Word Files** - Product documentation
- **Images** - Product photos (OCR extracts text)

## Performance Optimization

### GPU Memory Usage
```yaml
RTX 4090 24GB (Primary):
  - Llama 2 7B (8-bit): ~7GB
  - Llama 2 13B (8-bit): ~13GB
  - Available for context: ~10-17GB

RTX 4070Ti Super 16GB (Secondary):
  - Sentence embeddings: ~0.1GB
  - Product database: ~2-4GB
  - OCR processing: ~1GB
  - Available for batching: ~10GB
```

### Model Selection
The system automatically chooses the best model for your GPU:

- **24GB+ VRAM**: Llama 2 7B (full precision)
- **16GB VRAM**: Llama 2 7B (8-bit quantized)
- **12GB VRAM**: Llama 2 7B (heavily quantized)
- **8GB VRAM**: DialoGPT-medium (fallback)

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce model size in code
use_quantization = True  # Enable 8-bit quantization
batch_size = 8          # Reduce batch size
```

#### 2. Slow Response Times
- Check GPU utilization with `nvidia-smi`
- Ensure models are loaded on GPU, not CPU
- Reduce conversation context length

#### 3. Poor Search Results
- Add more product data to database
- Check that embeddings are being generated
- Verify product descriptions are detailed

#### 4. Google Search Not Working
- Check internet connection
- Verify `googlesearch-python` is installed
- May hit rate limits with frequent searches

### Monitoring Performance

#### Check System Status
```python
# In the Analytics tab, you can monitor:
- Response times by data source
- Local vs web search usage
- GPU memory utilization
- Conversation patterns
```

#### Log Files
- Check `chatbot.log` for detailed error information
- Monitor GPU usage with `nvidia-smi`
- Use Task Manager to check RAM usage

## Advanced Configuration

### Custom Response Prompts
The system builds prompts like this:
```python
prompt = f"""You are a professional AI sales assistant.

Previous conversation:
Customer: {previous_input}
Assistant: {previous_response}

Relevant products from our catalog:
- Gaming Laptop Pro X1: High-performance gaming laptop...

Current customer question: {user_input}

Assistant response:"""
```

### Adding Custom Models
To use different LLM models, modify the model selection:
```python
# In load_language_model() function
model_options = [
    "meta-llama/Llama-2-7b-chat-hf",      # 7B model
    "meta-llama/Llama-2-13b-chat-hf",     # 13B model  
    "mistralai/Mistral-7B-Instruct-v0.1", # Mistral
    "microsoft/DialoGPT-large"             # Fallback
]
```

## Business Integration

### CRM Integration
The conversation data can be exported and integrated with:
- Salesforce
- HubSpot  
- Microsoft Dynamics
- Custom CRM systems

### Analytics Export
- Export conversation data to CSV
- Analyze customer inquiry patterns
- Track response accuracy
- Monitor system performance

## Next Steps

1. **Test thoroughly** with sample data
2. **Upload your real product catalogs**
3. **Train the system** with your documentation
4. **Monitor performance** and adjust settings
5. **Scale to multi-GPU** setup when ready

For technical support, check the log files and error messages. The system provides detailed logging to help diagnose issues.