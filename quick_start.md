# 🚀 Quick Start Guide - Natural Language AI Sales ChatBot

## Overview

This AI ChatBot uses **natural language generation** with local LLM models and **YAML configuration** to provide human-like responses. It searches your local product database first, then falls back to Google search if needed.

## Key Features

✅ **YAML Configuration** - Easy model and parameter management  
✅ **Natural Language Generation** - Uses local LLM (Llama 3.2, etc.)  
✅ **Local Database Priority** - Searches your products first  
✅ **Google Search Fallback** - Finds info online when local data is insufficient  
✅ **Conversation Context** - Remembers previous conversation turns  
✅ **Multi-GPU Support** - RTX 4070Ti Super + RTX 4090 optimization  
✅ **Real-time Config Reload** - Change settings without restart  

## Hardware Requirements

### Minimum Setup
- **RTX 4070Ti Super 16GB** or RTX 4080 16GB
- **32GB RAM** minimum
- **Windows 10/11** with CUDA 11.8+

### Optimal Setup
- **RTX 4090 24GB** (Primary - for LLM)
- **RTX 4070Ti Super 16GB** (Secondary - for embeddings)
- **32GB+ RAM**

## Installation

### 1. Automated Setup (Recommended)
```bash
# Download the repository
git clone https://github.com/yourusername/ai-sales-chatbot.git
cd ai-sales-chatbot

# Run automated setup (Windows)
setup.bat
```

### 2. Manual Installation
```bash
# Install Python 3.9+
python --version  # Should show 3.9+

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Verify YAML support
python -c "import yaml; print('Configuration support enabled')"
```

### 3. Launch Application
```bash
# Option 1: Use batch file (Windows)
start_chatbot.bat

# Option 2: Direct Python execution
python natural_language_chatbot.py
```

## First Time Setup

### 1. Verify Configuration Loading
When you start the application, you should see:
```
✅ Configuration loaded from chatbot_config.yaml
Primary LLM: meta-llama/Llama-3.2-3B-Instruct
Fallback LLM: microsoft/DialoGPT-medium
Embedding Model: sentence-transformers/all-MiniLM-L6-v2
Quantization: True
```

### 2. Add Sample Data
1. Go to the **"Database"** tab
2. Click **"Add Sample Products"** to add test data
3. Click **"View Products"** to verify data was added
4. Click **"Search Test"** to verify semantic search works

### 3. Test Configuration Management
1. Go to the **"Configuration"** tab
2. Click **"View Config"** to see current settings
3. Click **"Test Models"** to verify all models are working
4. Click **"Reload Config"** to test configuration reloading

### 4. Start Chatting
Go to the **"Chat"** tab and try these example queries:

```
"Hello!" (should trigger greeting template)
"I'm looking for a gaming laptop under $2000"
"What business computers do you recommend?"
"Tell me about wireless gaming accessories"
"Compare your laptop models for me"
"I need help with technical specifications"
```

## Configuration Management

### Main Configuration File (`chatbot_config.yaml`)

The system uses YAML configuration for easy customization:

```yaml
# AI Model Selection
ai_models:
  primary_llm: "meta-llama/Llama-3.2-3B-Instruct"
  fallback_llm: "microsoft/DialoGPT-medium"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

# GPU Settings
gpu_config:
  use_quantization: true    # 8-bit for memory efficiency
  max_memory_per_gpu: 0.85  # Use 85% of VRAM

# Response Parameters
performance:
  temperature: 0.7          # Response creativity (0.0-1.0)
  max_response_length: 512  # Maximum response tokens
  repetition_penalty: 1.1   # Reduce repetition

# Search Settings
search_config:
  local_similarity_threshold: 0.7  # Database search sensitivity
  enable_google_search: true       # Web search fallback
  max_google_results: 3            # Number of web results
```

### Real-time Configuration Changes

1. **Edit Configuration**: Modify `chatbot_config.yaml`
2. **Reload in App**: Configuration tab → "Reload Config"
3. **No Restart Required**: Changes apply immediately (except model changes)

### Response Templates

Configure automatic responses:
```yaml
response_templates:
  greeting: "Hello! I'm your AI sales assistant..."
  no_products_found: "I couldn't find matching products, but..."
  out_of_stock: "This item is currently unavailable..."
```

## Expected Behavior

### Response Flow Priority
1. **Configuration Check** - Loads settings from YAML
2. **Greeting Detection** - Checks for greeting patterns from config
3. **Local Database Search** - Searches products using configurable threshold
4. **Google Search** - If enabled and no local results found
5. **Natural Language Generation** - LLM creates response using config parameters

### Example Interaction with Configuration
```
User: "Hello!"

AI Process:
1. ✅ Detected greeting trigger from config
2. 🤖 Used greeting template from config
3. 📊 Source: template_response | Time: 0.1s

AI Response: "Hello! I'm your AI sales assistant. I can help you 
find the perfect products for your needs. What are you looking for today?"

---

User: "I need a gaming laptop under $2000"

AI Process:
1. ✅ Searches local database (threshold: 0.7 from config)
2. ✅ Found "Gaming Laptop Pro X1" with 89% similarity
3. 🤖 Generated response using Llama 3.2 (temp: 0.7 from config)
4. 📊 Source: local_database | Time: 2.3s

AI Response: "Based on your budget, I'd recommend our Gaming Laptop Pro X1 
at $1,899.99. It features an RTX 4070 graphics card and Intel i7-12700H 
processor with 32GB DDR5 RAM. The 15.6-inch 144Hz display ensures smooth 
gaming performance. It's currently in stock. Would you like to know more 
about its gaming capabilities or see some alternatives?"
```

## Training the System

### Adding Your Product Data

#### Excel/CSV Format
Create files with these columns (configurable in training tab):
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

### Automatic GPU Configuration
The system automatically detects and optimizes for your hardware:

```yaml
RTX 4070Ti Super 16GB:
  - Llama 3.2 3B (8-bit quantized): ~7GB VRAM
  - Sentence embeddings: ~1GB VRAM
  - Available for context: ~8GB VRAM

RTX 4090 24GB:
  - Llama 3.2 3B (full precision): ~6GB VRAM
  - Sentence embeddings: ~1GB VRAM
  - Available for larger models: ~17GB VRAM
```

### Configuration Tuning
Adjust these settings in `chatbot_config.yaml` for better performance:

```yaml
# For faster responses (less quality)
performance:
  temperature: 0.3          # More focused responses
  max_response_length: 256  # Shorter responses

# For better quality (slower)
performance:
  temperature: 0.8          # More creative responses
  max_response_length: 1024 # Longer responses

# For memory optimization
gpu_config:
  use_quantization: true    # Enable 8-bit
  max_memory_per_gpu: 0.8   # Reduce VRAM usage
```

## Troubleshooting

### Common Issues

#### 1. Configuration Not Loading
```
⚠️ chatbot_config.yaml not found, using default settings
```
**Solution**: Ensure `chatbot_config.yaml` is in the same directory as the Python script.

#### 2. Model Loading Errors
```
❌ Error loading model from config: ...
```
**Solution**: Check model names in config file, verify internet connection for model downloads.

#### 3. YAML Syntax Errors
```
❌ Error loading config: scanner.ScannerError...
```
**Solution**: Validate YAML syntax, check indentation (use spaces, not tabs).

#### 4. GPU Memory Issues
```
CUDA out of memory...
```
**Solution**: Enable quantization in config:
```yaml
gpu_config:
  use_quantization: true
  max_memory_per_gpu: 0.8
```

### Configuration Validation

Use the **Configuration tab** to:
- **View Config**: Check current settings
- **Test Models**: Verify all models load correctly
- **Reload Config**: Apply changes without restart

### Log Files
- Check `logs/chatbot.log` for detailed error information
- Monitor GPU usage with `nvidia-smi`
- Use Configuration tab for real-time diagnostics

## Advanced Configuration

### Custom Models
To use different models, edit the config:
```yaml
ai_models:
  primary_llm: "mistralai/Mistral-7B-Instruct-v0.1"
  embedding_model: "BAAI/bge-large-en-v1.5"
```

### Custom Response Flows
```yaml
conversation_flows:
  technical_support:
    triggers: ["help", "support", "problem"]
    response_template: "I'm here to help with technical support..."
    next_stage: "support_resolution"
```

### Performance Monitoring
The system tracks (configurable in analytics section):
- Response times by data source
- Configuration parameter effectiveness
- Model performance metrics
- GPU utilization and optimization

## Next Steps

1. **Test thoroughly** with the sample data and configuration
2. **Upload your real product catalogs** via Training tab
3. **Customize configuration** in `chatbot_config.yaml`
4. **Monitor performance** via Analytics and Configuration tabs
5. **Fine-tune parameters** based on your specific use case

For advanced configuration options, see the [Configuration Guide](CONFIGURATION.md).