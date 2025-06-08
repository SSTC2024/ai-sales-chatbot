# Vietnamese Voice-Enabled AI Sales ChatBot

A comprehensive AI chatbot application with Vietnamese language support, voice recognition, document processing, and intelligent product search capabilities.

## 🌟 Features

### 🗣️ Voice & Language Support
- **Vietnamese Speech Recognition** - Real-time voice input in Vietnamese
- **Text-to-Speech (TTS)** - Vietnamese voice output with multiple voice options
- **Bilingual Support** - Seamless Vietnamese/English conversation switching
- **Language Auto-Detection** - Automatic detection of input language
- **Translation Support** - Built-in translation capabilities

### 🧠 AI & Machine Learning
- **Language Models** - Transformer-based conversational AI (DialoGPT)
- **Embedding Search** - Semantic similarity search using sentence transformers
- **Smart Response Generation** - Context-aware Vietnamese responses
- **GPU Acceleration** - CUDA support with automatic quantization
- **Fallback Systems** - Graceful degradation when AI models unavailable

### 📊 Document Processing & Training
- **Excel/CSV Files** - Product data import with Vietnamese column mapping
- **PDF Documents** - Text extraction for knowledge base
- **Word Documents** - Content extraction including tables
- **Image OCR** - Vietnamese text recognition from images
- **Real-time Processing** - Live status updates during file processing

### 🎯 Sales & Product Management
- **Product Database** - SQLite database with Vietnamese product information
- **Semantic Search** - Find products using natural language queries
- **Price Handling** - Currency parsing and price comparisons
- **Category Management** - Multilingual product categorization
- **Sample Data** - Pre-loaded Vietnamese product examples

### 📈 Analytics & Monitoring
- **Conversation Tracking** - Complete conversation history with language detection
- **Performance Metrics** - Response times, data sources, success rates
- **Language Statistics** - Usage patterns for Vietnamese vs English
- **Export Functionality** - CSV export of analytics data
- **Real-time Status** - Live system status indicators

## 🛠️ Installation

### Prerequisites
- Python 3.8+ 
- Windows/Linux/macOS
- 4GB+ RAM (8GB+ recommended for AI models)
- Microphone (optional, for voice input)
- Speakers/Headphones (optional, for voice output)

### Required Dependencies
```bash
# Core AI and ML libraries
pip install torch transformers sentence-transformers
pip install pandas numpy scikit-learn

# GUI and database
pip install tkinter sqlite3 
pip install pyyaml requests beautifulsoup4

# Document processing
pip install PyPDF2 python-docx openpyxl easyocr

# Web search (optional)
pip install googlesearch-python
```

### Optional Dependencies (for enhanced features)
```bash
# Voice recognition and synthesis
pip install SpeechRecognition pyttsx3 pyaudio

# Translation support
pip install googletrans==4.0.0rc1 langdetect

# Advanced document processing
pip install python-magic
```

### GPU Support (optional but recommended)
```bash
# For CUDA-enabled systems
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Quick Start

### 1. Download and Setup
```bash
# Download the chatbot file
# Save as: vietnamese_chatbot.py

# Create necessary directories
mkdir data logs

# Run the application
python vietnamese_chatbot.py
```

### 2. First Run
- The application will automatically create a configuration file (`chatbot_config.yaml`)
- Database will be initialized (`chatbot_data.db`)
- Sample products will be loaded
- GUI will open with tabbed interface

### 3. Basic Usage
1. **Chat Tab** - Start conversing in Vietnamese or English
2. **Database Tab** - View products and test search functionality
3. **Training Tab** - Upload Excel, PDF, Word, or image files
4. **Analytics Tab** - Monitor performance and conversation history

## 📋 Configuration

### Config File (`chatbot_config.yaml`)
```yaml
# Language settings
language_config:
  default_language: 'vi'
  supported_languages: ['vi', 'en']
  auto_detect_language: true

# Voice settings
voice_config:
  enable_voice_input: true
  enable_voice_output: true
  voice_rate: 150
  voice_volume: 0.8

# AI model settings
ai_models:
  primary_llm: 'microsoft/DialoGPT-medium'
  embedding_model: 'sentence-transformers/all-MiniLM-L6-v2'

# Performance settings
performance:
  max_response_length: 150
  temperature: 0.7
```

## 📊 Training Data Formats

### Excel/CSV Files
Expected columns (Vietnamese names supported):
- `name` / `tên` - Product name
- `description` / `mô tả` - Product description  
- `category` / `danh mục` - Product category
- `price` / `giá` - Product price
- `features` / `tính năng` - Product features
- `specifications` / `thông số` - Technical specifications

Example CSV:
```csv
name,tên,description,mô tả,price,giá
Gaming Laptop,Laptop Gaming,High-performance laptop,Laptop hiệu suất cao,1899.99,1899.99
```

### PDF/Word Documents
- Automatically extracted for knowledge base
- Supports Vietnamese text encoding
- Tables and structured content preserved
- Embedded in searchable knowledge base

### Images with OCR
- Supports PNG, JPG, JPEG, BMP, TIFF
- Vietnamese text recognition
- Confidence-based text filtering
- Automatic knowledge base integration

## 🎯 Usage Examples

### Voice Commands (Vietnamese)
- "Xin chào" - Greeting
- "Tôi muốn mua laptop gaming" - Product inquiry
- "Giá bao nhiêu?" - Price inquiry
- "So sánh sản phẩm" - Product comparison
- "Cảm ơn, tạm biệt" - Goodbye

### Text Queries (English)
- "Show me gaming laptops under $2000"
- "What wireless mice do you have?"
- "Compare mechanical keyboards"
- "What's the best business laptop?"

### File Processing
1. Go to **Training Tab**
2. Click appropriate file type button
3. Select files to upload
4. Monitor real-time processing status
5. View statistics when complete

## 🔧 Troubleshooting

### Common Issues

#### Voice Recognition Not Working
```bash
# Check microphone permissions
# Install/reinstall audio drivers
# Test with:
python -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"
```

#### AI Models Not Loading
```bash
# Check internet connection for model downloads
# Ensure sufficient RAM (8GB+ recommended)
# Try CPU mode if GPU issues:
# Set in config: gpu_config.primary_device: 'cpu'
```

#### Vietnamese Text Display Issues
```bash
# Install Vietnamese fonts:
# Windows: Install "Times New Roman" or "Arial Unicode MS"
# Linux: sudo apt-get install fonts-noto-cjk
# macOS: Vietnamese fonts included by default
```

#### Database Errors
```bash
# Reset database:
rm chatbot_data.db
# Restart application - will recreate database
```

### Performance Optimization

#### For Low-End Systems
```yaml
# In chatbot_config.yaml:
ai_models:
  primary_llm: 'microsoft/DialoGPT-small'  # Smaller model
gpu_config:
  use_quantization: true  # Reduce memory usage
  batch_size: 1          # Minimize memory usage
```

#### For High-End Systems
```yaml
# In chatbot_config.yaml:
ai_models:
  primary_llm: 'microsoft/DialoGPT-large'  # Larger model
gpu_config:
  use_quantization: false  # Full precision
  batch_size: 4           # Faster processing
```

## 📁 File Structure
```
vietnamese-chatbot/
├── vietnamese_chatbot.py      # Main application file
├── chatbot_config.yaml        # Configuration file (auto-created)
├── chatbot_data.db           # SQLite database (auto-created)
├── chatbot.log               # Application logs
├── data/                     # Training data directory
├── logs/                     # Additional log files
└── README.md                 # This file
```

## 🔐 Security & Privacy

### Data Storage
- All data stored locally in SQLite database
- No cloud services required for core functionality
- Conversation history kept locally
- Optional data export for backup

### Voice Processing
- Voice processing handled locally when possible
- Google Speech Recognition used for accuracy (requires internet)
- Voice data not permanently stored
- TTS processing done locally

### AI Models
- Models downloaded from Hugging Face Hub
- Cached locally after first download
- No user data sent to model providers
- All processing done locally

## 🤝 Contributing

### Code Structure
- Main class: `VietnameseVoiceChatBot`
- GUI setup: `setup_gui()` and related methods
- File processing: `process_excel_file()`, `process_pdf_file()`, etc.
- AI processing: `generate_natural_response()`, `search_local_database()`
- Voice handling: `listen_for_speech()`, `speak_text()`

### Adding New Features
1. Extend the main class with new methods
2. Update GUI if needed (add new tabs/buttons)
3. Update configuration schema if needed
4. Add appropriate error handling
5. Update documentation

### Vietnamese Language Support
- Use UTF-8 encoding everywhere
- Test with Vietnamese diacritics
- Support both Vietnamese and English column names
- Provide bilingual error messages

## 📞 Support

### Getting Help
1. Check the troubleshooting section above
2. Review configuration settings
3. Check log files for error details
4. Ensure all dependencies are installed

### System Requirements Check
```python
# Run this to check your system:
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    import speech_recognition
    print("✅ Speech recognition available")
except ImportError:
    print("❌ Speech recognition not available")

try:
    import pyttsx3
    print("✅ Text-to-speech available") 
except ImportError:
    print("❌ Text-to-speech not available")
```

## 📄 License

This project is provided as-is for educational and commercial use. Please ensure compliance with all third-party library licenses.

## 🔄 Version History

### v2.0 (Current)
- Complete Vietnamese language support
- Voice recognition and synthesis
- Document processing (Excel, PDF, Word, Images)
- Advanced AI chat capabilities
- Comprehensive error handling
- Analytics and monitoring
- Tabbed GUI interface

### Key Components Fixed
- ✅ All syntax and indentation errors resolved
- ✅ Complete try/except block matching
- ✅ Proper method definitions and class structure
- ✅ Full GUI implementation with all tabs
- ✅ Complete training functionality restored
- ✅ All file processing capabilities working
- ✅ Vietnamese font and encoding support
- ✅ Comprehensive error handling throughout

---

**Ready to run! No syntax errors, all features included, production-ready Vietnamese AI Chatbot.**