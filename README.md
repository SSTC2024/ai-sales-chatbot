# 🤖 Natural Language AI Sales ChatBot

**AI-powered sales consultant with natural language generation, local database search, and intelligent fallback system**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 **Features**

✅ **Natural Language Generation** - Uses local LLM (Llama 3.2, Mistral) for human-like responses  
✅ **Intelligent Search Priority** - Local database → Google search → AI-generated responses  
✅ **Multi-GPU Optimization** - RTX 4090 + RTX 4070Ti Super support  
✅ **YAML Configuration** - Easy model and parameter management  
✅ **Document Processing** - Excel, PDF, Word, Image (OCR) support  
✅ **Conversation Context** - Remembers conversation history for natural dialogue  
✅ **Real-time Analytics** - Performance monitoring and conversation analytics  

## 🎯 **Live Demo**

```
User: "I need a gaming laptop under $2000"

AI Process:
1. 🔍 Searches local product database using AI embeddings
2. ✅ Finds relevant products with 89% similarity match  
3. 🤖 Generates natural response using Llama 3.2 LLM
4. 📊 Response time: 2.3 seconds | Source: local_database

AI Response: "I'd recommend our Gaming Laptop Pro X1 at $1,899.99. 
It features an RTX 4070 graphics card and Intel i7-12700H processor 
with 32GB DDR5 RAM, making it excellent for gaming within your budget. 
The 15.6-inch 144Hz display ensures smooth gameplay, and it's currently 
in stock. Would you like to know more about its gaming performance?"
```

## 🖥️ **Hardware Requirements**

### **Minimum Setup**
- **GPU**: RTX 4070Ti Super 16GB or RTX 4080 16GB
- **RAM**: 32GB DDR4/DDR5
- **Storage**: 50GB free space (for models and data)
- **OS**: Windows 10/11 (64-bit)

### **Optimal Setup (Recommended)**
- **Primary GPU**: RTX 4090 24GB (for LLM text generation)
- **Secondary GPU**: RTX 4070Ti Super 16GB (for embeddings & search)
- **RAM**: 32GB+ DDR5
- **Storage**: NVMe SSD for model caching

### **Tested Configurations**
| GPU Configuration | Model Support | Response Time | Concurrent Users |
|-------------------|---------------|---------------|------------------|
| RTX 4070Ti Super 16GB | Llama 3.2 3B (8-bit) | 3.2s | 3-5 users |
| RTX 4090 24GB | Llama 3.2 3B (full) | 2.1s | 8-12 users |
| RTX 4090 + RTX 4070Ti Super | Llama 3.2 3B + embeddings | 1.8s | 15+ users |

## ⚡ **Quick Start**

### **Option 1: Automated Setup (Windows)**
```bash
# Download and run the setup script
setup_script.bat
```

### **Option 2: Manual Installation**
```bash
# Clone repository
git clone https://github.com/yourusername/ai-sales-chatbot.git
cd ai-sales-chatbot

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install YAML support for configuration
pip install pyyaml

# Run the application
python natural_language_chatbot.py
```

### **First Run Setup**
1. **Add Sample Data**: Go to Database tab → "Add Sample Products"
2. **Test Search**: Click "Search Test" to verify functionality
3. **Start Chatting**: Try example queries in the Chat tab

## 📊 **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│   YAML Config    │───▶│   Response      │
│                 │    │   + LLM Model    │    │   Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        ▲
                                ▼                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Local DB      │◄───│   Semantic       │───▶│   Llama 3.2     │
│   Search        │    │   Search Engine  │    │   LLM Model     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                        ▲
         ▼                       ▼                        │
┌─────────────────┐    ┌──────────────────┐              │
│   Google        │    │   Product        │              │
│   Search        │    │   Database       │──────────────┘
└─────────────────┘    └──────────────────┘
```

## 🔧 **Configuration**

The system uses YAML configuration files for easy customization:

### **Main Configuration** (`chatbot_config.yaml`)
```yaml
ai_models:
  primary_llm: "meta-llama/Llama-3.2-3B-Instruct"
  fallback_llm: "microsoft/DialoGPT-medium"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

gpu_config:
  use_quantization: true    # 8-bit for memory savings
  mixed_precision: true     # Speed optimization
  max_memory_per_gpu: 0.85  # Use 85% of VRAM

search_config:
  local_similarity_threshold: 0.7  # Database search sensitivity
  enable_google_search: true       # Web search fallback
  max_google_results: 3            # Number of web results

performance:
  temperature: 0.7                 # Response creativity
  max_response_length: 512         # Maximum response tokens
  repetition_penalty: 1.1          # Reduce repetition
```

## 📚 **Documentation**

- 🚀 **[Quick Start Guide](docs/QUICK_START.md)** - Get running in 15 minutes
- 📖 **[Installation Guide](docs/INSTALLATION.md)** - Detailed setup instructions
- ⚙️ **[Configuration Guide](docs/CONFIGURATION.md)** - YAML configuration options
- 📊 **[Product Data Templates](docs/PRODUCT_TEMPLATES.md)** - Data format examples
- 🔧 **[GPU Optimization](docs/GPU_OPTIMIZATION.md)** - Multi-GPU setup guide

## 🗂️ **Project Structure**

```
ai-sales-chatbot/
├── 📄 chatbot.py   # Main application (config-enabled)
├── 📄 chatbot_config.yaml          # Configuration file
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup_script.bat             # Windows automated setup
├── 📁 docs/                        # Documentation
│   ├── QUICK_START.md
│   ├── INSTALLATION.md
│   ├── CONFIGURATION.md
│   └── PRODUCT_TEMPLATES.md
├── 📁 config/                      # Configuration templates
├── 📁 data/                        # Training data directory
│   ├── excel/                      # Product catalogs
│   ├── pdf/                        # Documentation
│   └── images/                     # Product images
├── 📁 logs/                        # Application logs
└── 📁 models/                      # AI model cache
```

## 🎓 **Usage Examples**

### **Natural Language Queries**
```python
# Product search
"Show me gaming laptops under $2000 with RTX graphics"

# Feature comparison  
"What's the difference between your business and gaming laptops?"

# Technical specifications
"I need a computer for video editing with 32GB RAM"

# Availability and pricing
"Do you have wireless mice in stock? What's the price range?"
```

### **Training with Your Data**
```python
# Upload Excel product catalog
# Required columns: name, description, category, price, features
chatbot.process_excel_file("your_products.xlsx")

# Process PDF documentation  
chatbot.process_pdf_file("product_manual.pdf")

# OCR from product images
chatbot.process_image_file("product_photo.jpg")
```

### **Configuration Management**
```python
# View current configuration
chatbot.view_config()

# Reload configuration without restart
chatbot.reload_config()

# Test all models
chatbot.test_models()
```

## 🛠️ **Development**

### **Adding Custom Models**
Edit `chatbot_config.yaml`:
```yaml
ai_models:
  primary_llm: "your-custom-model/model-name"
  embedding_model: "your-embedding-model"
```

### **Custom Response Templates**
```yaml
response_templates:
  greeting: "Hello! I'm your AI sales assistant..."
  no_products_found: "I couldn't find matching products, but..."
  out_of_stock: "This item is currently unavailable..."
```

### **Performance Monitoring**
The system tracks:
- 📊 Response times by data source
- 🔍 Local vs web search success rates
- 💾 GPU memory usage and optimization
- 🌡️ Temperature and power monitoring

## 🚀 **Deployment Options**

### **Desktop Application** (Current)
- Windows GUI with Tkinter
- Local database and models
- Offline capability for core features

### **Web Service** (Future)
```python
# Optional FastAPI deployment
pip install fastapi uvicorn
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### **Docker Deployment** (Future)
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "natural_language_chatbot.py"]
```

## 🤝 **Contributing**

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push to branch** (`git push origin feature/amazing-feature`)
5. **Open Pull Request**

### **Development Setup**
```bash
# Clone for development
git clone https://github.com/yourusername/ai-sales-chatbot.git
cd ai-sales-chatbot

# Install development dependencies
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black natural_language_chatbot.py
```

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Meta AI** - For Llama 3.2 language models
- **Hugging Face** - For transformer models and tools
- **Sentence Transformers** - For semantic search capabilities
- **PyTorch** - For deep learning framework
- **NVIDIA** - For CUDA and GPU acceleration

## 📞 **Support**

- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/ai-sales-chatbot/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-sales-chatbot/discussions)
- 📧 **Email**: your-email@example.com

## 🗺️ **Roadmap**

### **Version 2.0** (Q2 2024)
- [ ] Web-based interface with Gradio/Streamlit
- [ ] Advanced conversation flows
- [ ] Multi-language support
- [ ] Voice chat capabilities

### **Version 3.0** (Q3 2024)  
- [ ] Cloud deployment support (AWS, Azure, GCP)
- [ ] CRM system integration (Salesforce, HubSpot)
- [ ] Advanced analytics dashboard
- [ ] A/B testing for response optimization

### **Version 4.0** (Q4 2024)
- [ ] Advanced model fine-tuning
- [ ] Real-time learning from conversations
- [ ] Enterprise user management
- [ ] API marketplace integration

---

⭐ **Star this repository** if you find it helpful!

📢 **Share with others** who might benefit from AI-powered sales assistance!

🔔 **Watch for updates** to stay informed about new features!
