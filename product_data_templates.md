# Product Data & Conversation Flow Templates

## 📊 Product Data Structure Templates

### Excel Product Catalog Template (`products.xlsx`)

| name | description | category | price | features | specifications | availability | stock_count | brand | model | warranty | image_url |
|------|-------------|----------|-------|-----------|---------------|--------------|-------------|--------|-------|-----------|-----------|
| Gaming Laptop Pro X1 | High-performance gaming laptop with RTX graphics | Laptops | 1899.99 | RTX 4070, 144Hz display, RGB keyboard, Fast SSD | Intel i7-12700H, 32GB DDR5, 1TB NVMe SSD, 15.6" 144Hz IPS | in_stock | 15 | TechForce | Pro X1 | 2 years | /images/laptop_pro_x1.jpg |
| Business Ultrabook Z5 | Lightweight business laptop for professionals | Laptops | 1299.99 | Lightweight, Long battery, Fingerprint reader | Intel i5-12500U, 16GB DDR4, 512GB SSD, 14" FHD | in_stock | 8 | TechForce | Ultrabook Z5 | 3 years | /images/ultrabook_z5.jpg |
| Wireless Gaming Mouse Elite | High-precision wireless mouse for gaming | Accessories | 89.99 | Wireless, RGB lighting, Programmable buttons | 16000 DPI, 2.4GHz wireless, USB-C charging | in_stock | 42 | GameGear | Elite Mouse | 1 year | /images/mouse_elite.jpg |
| 4K Gaming Monitor Ultra | 32-inch 4K gaming monitor with HDR | Monitors | 599.99 | 4K resolution, HDR10, 144Hz refresh rate | 32" IPS, 3840x2160, 144Hz, HDR10, USB-C hub | low_stock | 3 | ViewTech | Ultra 32 | 2 years | /images/monitor_ultra.jpg |
| Mechanical Keyboard Pro | RGB mechanical keyboard for gaming and typing | Accessories | 149.99 | Mechanical switches, RGB backlighting, Media keys | Cherry MX Blue switches, Full-size, USB-A | in_stock | 25 | KeyCraft | Pro Mechanical | 2 years | /images/keyboard_pro.jpg |

### JSON Product Structure Template (`products.json`)
```json
{
  "products": [
    {
      "id": "LAPTOP001",
      "name": "Gaming Laptop Pro X1",
      "description": "High-performance gaming laptop designed for serious gamers. Features the latest RTX 4070 graphics card, Intel i7 processor, and a stunning 144Hz display for smooth gameplay.",
      "category": "Laptops",
      "subcategory": "Gaming Laptops",
      "price": 1899.99,
      "currency": "USD",
      "features": [
        "NVIDIA RTX 4070 Graphics Card",
        "Intel i7-12700H Processor",
        "32GB DDR5 RAM",
        "1TB NVMe SSD Storage",
        "15.6-inch 144Hz IPS Display",
        "RGB Backlit Keyboard",
        "Wi-Fi 6E Support",
        "Thunderbolt 4 Ports"
      ],
      "specifications": {
        "processor": "Intel Core i7-12700H",
        "graphics": "NVIDIA GeForce RTX 4070 8GB",
        "memory": "32GB DDR5-4800",
        "storage": "1TB PCIe 4.0 NVMe SSD",
        "display": "15.6\" FHD (1920x1080) 144Hz IPS",
        "battery": "80Wh Li-ion",
        "weight": "2.3 kg",
        "dimensions": "35.9 x 25.9 x 2.39 cm",
        "ports": ["2x USB 3.2", "1x USB-C Thunderbolt 4", "HDMI 2.1", "3.5mm Audio"]
      },
      "availability": "in_stock",
      "stock_count": 15,
      "brand": "TechForce",
      "model": "Pro X1",
      "sku": "TF-PX1-001",
      "warranty": "2 years manufacturer warranty",
      "image_urls": [
        "/images/laptop_pro_x1_main.jpg",
        "/images/laptop_pro_x1_side.jpg",
        "/images/laptop_pro_x1_keyboard.jpg"
      ],
      "tags": ["gaming", "high-performance", "rtx", "laptop"],
      "rating": 4.7,
      "review_count": 156,
      "release_date": "2024-03-15",
      "use_cases": ["Gaming", "Content Creation", "Video Editing", "Streaming"],
      "target_audience": ["Gamers", "Content Creators", "Students"],
      "compatibility": ["Windows 11", "Linux"],
      "energy_rating": "B+",
      "certifications": ["Energy Star", "EPEAT Gold"]
    }
  ]
}
```

### Database Schema Template (`setup_database.sql`)
```sql
-- Products table
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT,
    subcategory TEXT,
    price REAL,
    currency TEXT DEFAULT 'USD',
    features TEXT, -- JSON string
    specifications TEXT, -- JSON string
    availability TEXT CHECK(availability IN ('in_stock', 'low_stock', 'out_of_stock', 'discontinued')),
    stock_count INTEGER DEFAULT 0,
    brand TEXT,
    model TEXT,
    warranty TEXT,
    image_urls TEXT, -- JSON array
    tags TEXT, -- JSON array
    rating REAL DEFAULT 0.0,
    review_count INTEGER DEFAULT 0,
    release_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Customer conversations table
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_input TEXT,
    bot_response TEXT,
    intent_detected TEXT,
    data_source TEXT, -- 'local_db', 'google_search', 'ai_generated'
    products_mentioned TEXT, -- JSON array of product IDs
    response_time REAL,
    user_satisfaction INTEGER, -- 1-5 rating
    conversation_stage TEXT -- 'greeting', 'inquiry', 'comparison', 'purchase_intent', 'support'
);

-- Product categories table
CREATE TABLE categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    parent_category TEXT,
    description TEXT,
    display_order INTEGER DEFAULT 0
);
```

## 🤖 Conversation Flow Configuration Templates

### Main Configuration (`config/chatbot_config.yaml`)
```yaml
# AI Sales ChatBot Configuration for RTX 4070Ti Super
version: "1.0"
environment: "production"

# Model Configuration
ai_models:
  primary_llm: "meta-llama/Llama-2-7b-chat-hf"
  fallback_llm: "microsoft/DialoGPT-medium"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  intent_classifier: "facebook/bart-large-mnli"

# GPU Configuration
gpu_config:
  primary_device: "cuda:0"
  use_quantization: true
  mixed_precision: true
  max_memory_per_gpu: 0.85
  batch_size: 4

# Conversation Flow Settings
conversation_flows:
  greeting:
    triggers: ["hello", "hi", "hey", "good morning", "good afternoon"]
    response_template: "Hello! I'm your AI sales assistant. I can help you find the perfect products for your needs. What are you looking for today?"
    next_stage: "needs_assessment"
    
  needs_assessment:
    triggers: ["looking for", "need", "want", "searching for"]
    questions:
      - "What type of product are you interested in?"
      - "What's your budget range?"
      - "What will you primarily use this for?"
      - "Any specific features you're looking for?"
    response_template: "Let me help you find exactly what you need. {product_recommendations}"
    next_stage: "product_presentation"
    
  product_presentation:
    search_threshold: 0.7
    max_products_shown: 3
    include_specifications: true
    include_pricing: true
    include_availability: true
    response_template: "Based on your requirements, I'd recommend:\n\n{product_list}\n\nWould you like more details about any of these?"
    next_stage: "detailed_inquiry"
    
  detailed_inquiry:
    triggers: ["tell me more", "specifications", "features", "details"]
    response_template: "Here are the detailed specifications for {product_name}:\n\n{detailed_specs}\n\nThis product is perfect for {use_cases}. Would you like to know about pricing, availability, or compare with other options?"
    next_stage: "comparison_or_purchase"
    
  comparison:
    triggers: ["compare", "difference", "better", "versus", "vs"]
    max_compare_products: 3
    comparison_aspects: ["price", "performance", "features", "warranty"]
    response_template: "Here's a comparison of {product_names}:\n\n{comparison_table}\n\nBased on your needs, I'd recommend {recommendation} because {reasoning}."
    next_stage: "purchase_decision"
    
  purchase_decision:
    triggers: ["buy", "purchase", "order", "get this", "interested"]
    response_template: "Great choice! The {product_name} is available for ${price}. It's currently {availability_status}. Would you like me to check current promotions or help you with the next steps?"
    next_stage: "purchase_support"
    
  technical_support:
    triggers: ["help", "support", "problem", "issue", "compatibility"]
    categories:
      - "installation_help"
      - "compatibility_check"
      - "troubleshooting"
      - "warranty_info"
    response_template: "I'd be happy to help you with technical support. Can you tell me more about {issue_type}?"
    next_stage: "support_resolution"

# Intent Classification
intent_patterns:
  price_inquiry:
    patterns: ["price", "cost", "how much", "expensive", "cheap", "budget"]
    confidence_threshold: 0.8
    response_type: "pricing_focus"
    
  feature_inquiry:
    patterns: ["features", "specifications", "specs", "what can", "capabilities"]
    confidence_threshold: 0.8
    response_type: "feature_focus"
    
  availability_inquiry:
    patterns: ["available", "in stock", "when", "delivery", "shipping"]
    confidence_threshold: 0.8
    response_type: "availability_focus"
    
  comparison_request:
    patterns: ["compare", "difference", "better", "which one", "versus"]
    confidence_threshold: 0.8
    response_type: "comparison_focus"

# Response Templates
response_templates:
  no_products_found:
    message: "I couldn't find any products matching your exact criteria. Let me search for similar options or check if we have something comparable coming soon."
    fallback_action: "google_search"
    
  out_of_stock:
    message: "Unfortunately, {product_name} is currently out of stock. However, I can:\n1. Check when it'll be back in stock\n2. Suggest similar alternatives\n3. Add you to our notification list\n\nWhat would you prefer?"
    
  price_match:
    message: "I see you're interested in {product_name} for ${price}. We currently have a {discount_percentage}% promotion running. Would you like to know more about our price matching policy?"
    
  bundle_offer:
    message: "Great choice on the {main_product}! Customers often pair this with {accessory_products} for a complete setup. I can offer you a bundle discount of {discount_amount}."

# Search Configuration
search_config:
  local_similarity_threshold: 0.7
  enable_google_search: true
  google_search_fallback_threshold: 0.5
  max_google_results: 3
  search_timeout: 10
  enable_fuzzy_matching: true
  
# Performance Settings
performance:
  max_response_length: 512
  response_timeout: 30
  context_window_size: 4096
  temperature: 0.7
  top_p: 0.9
  repetition_penalty: 1.1
  
# Analytics Configuration
analytics:
  track_conversations: true
  track_product_mentions: true
  track_conversion_funnel: true
  track_response_times: true
  generate_daily_reports: true
```

### Conversation Flow Examples (`config/flow_examples.yaml`)
```yaml
# Example Conversation Flows for Different Scenarios

gaming_laptop_flow:
  user_input: "I need a gaming laptop under $2000"
  ai_process:
    step_1: "🔍 Search local database for gaming laptops under $2000"
    step_2: "✅ Found 3 products with 89% similarity match"
    step_3: "🤖 Generate response using Llama-2"
    step_4: "📊 Log conversation for analytics"
  ai_response: |
    I'd recommend our Gaming Laptop Pro X1 at $1,899.99. It features an RTX 4070 graphics card and Intel i7-12700H processor with 32GB DDR5 RAM, making it excellent for gaming within your budget.
    
    Key gaming features:
    • RTX 4070 8GB - handles latest games at high settings
    • 144Hz display - smooth, tear-free gaming
    • 32GB DDR5 RAM - no performance bottlenecks
    • 1TB NVMe SSD - fast game loading
    
    It's currently in stock with 15 units available. Would you like to know more about its gaming performance or see some benchmarks?

business_laptop_flow:
  user_input: "What's good for business use with long battery life?"
  ai_process:
    step_1: "🔍 Search for business laptops with battery life focus"
    step_2: "✅ Found Business Ultrabook Z5 with 95% match"
    step_3: "🤖 Generate business-focused response"
  ai_response: |
    For business use, I'd recommend our Business Ultrabook Z5 at $1,299.99. It's specifically designed for professionals who need reliability and portability.
    
    Business advantages:
    • 14-hour battery life - all-day productivity
    • Lightweight (1.2kg) - easy to carry
    • Fingerprint reader - secure login
    • Intel i5-12500U - efficient performance
    • 3-year warranty - business peace of mind
    
    It's perfect for presentations, video calls, document work, and travel. Currently in stock with 8 units available. Would you like to know about bulk pricing for your organization?

accessory_recommendation_flow:
  user_input: "I bought the gaming laptop, what accessories do I need?"
  ai_process:
    step_1: "🔍 Identify context - previous gaming laptop purchase"
    step_2: "✅ Load gaming accessory recommendations"
    step_3: "🤖 Generate bundle suggestion"
  ai_response: |
    Excellent choice on the Gaming Laptop Pro X1! To complete your gaming setup, I'd recommend:
    
    **Essential Gaming Accessories:**
    1. **Wireless Gaming Mouse Elite** ($89.99)
       - 16,000 DPI precision for competitive gaming
       - RGB lighting matches your laptop
    
    2. **4K Gaming Monitor Ultra** ($599.99)
       - 32" 4K display maximizes your RTX 4070
       - 144Hz refresh rate matches your laptop's capabilities
    
    3. **Mechanical Keyboard Pro** ($149.99)
       - Cherry MX switches for gaming responsiveness
       - RGB backlighting for low-light gaming
    
    **Bundle Offer:** Get all three accessories for $789.99 (save $49.97)
    
    Would you like me to add these to your order, or do you need specific recommendations for any particular accessory?

technical_support_flow:
  user_input: "Is this laptop compatible with my existing monitor?"
  ai_process:
    step_1: "🔍 Identify compatibility inquiry"
    step_2: "✅ Load technical specifications"
    step_3: "🤖 Generate compatibility guidance"
  ai_response: |
    I'd be happy to help check compatibility! The Gaming Laptop Pro X1 has excellent connectivity options:
    
    **Available Outputs:**
    • HDMI 2.1 - supports up to 4K@120Hz
    • USB-C Thunderbolt 4 - supports 4K@60Hz with compatible displays
    • USB 3.2 ports for accessories
    
    **What's your monitor's specifications?**
    Please tell me:
    1. Monitor size and resolution (e.g., 27" 1440p)
    2. Connection type (HDMI, DisplayPort, USB-C)
    3. Refresh rate (60Hz, 144Hz, etc.)
    
    With this info, I can confirm perfect compatibility and recommend the best connection method for optimal performance!

price_negotiation_flow:
  user_input: "Can you do better on the price?"
  ai_process:
    step_1: "🔍 Check current promotions and pricing flexibility"
    step_2: "✅ Load available discounts and offers"
    step_3: "🤖 Generate value-focused response"
  ai_response: |
    I understand you're looking for the best value! Let me see what options I have for you:
    
    **Current Offers on Gaming Laptop Pro X1:**
    • Student discount: 5% off with valid student ID ($95 savings)
    • Trade-in program: Up to $200 off with your old laptop
    • Bundle discount: Save $50-150 when buying accessories
    
    **Why it's great value at $1,899.99:**
    • RTX 4070 graphics - comparable laptops cost $2,200+
    • 32GB DDR5 RAM included - saves $300 upgrade cost
    • 2-year warranty - covers repairs and replacements
    
    The closest competitor with similar specs is $2,299. Would you like me to check if you qualify for any of our discount programs?
```

### Custom Prompts Template (`config/llm_prompts.yaml`)
```yaml
# Custom LLM Prompts for Different Scenarios

system_prompts:
  sales_assistant: |
    You are an expert AI sales assistant for a technology retailer. Your role is to:
    - Help customers find the perfect products for their needs
    - Provide accurate technical information and specifications
    - Offer honest comparisons between products
    - Suggest complementary accessories and bundles
    - Be helpful, knowledgeable, and customer-focused
    
    Always be honest about product limitations and suggest alternatives when appropriate.
    Focus on value and how products solve customer problems.
    Use technical knowledge to build trust, but explain complex concepts clearly.
    
  technical_expert: |
    You are a technical product expert. Provide detailed, accurate technical information.
    Focus on specifications, compatibility, performance benchmarks, and real-world usage.
    Help customers understand how technical features translate to practical benefits.
    Be precise with numbers and specifications.
    
  customer_service: |
    You are a customer service representative focused on problem-solving.
    Be empathetic, patient, and solution-oriented.
    Provide clear steps for issue resolution.
    Know when to escalate to human support.

conversation_starters:
  general_greeting:
    - "Hello! I'm your AI sales assistant. How can I help you find the perfect product today?"
    - "Hi there! Looking for something specific, or would you like me to show you our latest deals?"
    - "Welcome! I'm here to help you find exactly what you need. What brings you here today?"
    
  product_category_specific:
    laptops:
      - "Looking for a new laptop? I can help you find the perfect one based on your needs and budget!"
      - "Whether it's for gaming, business, or everyday use, I'll help you find the ideal laptop!"
    
    accessories:
      - "Need some accessories to complete your setup? I can recommend the perfect additions!"
      - "Looking to upgrade your workspace? Let me suggest some great accessories!"

response_enhancers:
  include_specifications: |
    When mentioning products, always include key specifications that matter to the customer's use case.
    
  suggest_alternatives: |
    If the exact product isn't available, suggest 2-3 alternatives with explanations of differences.
    
  add_value_proposition: |
    Explain not just what the product does, but how it solves the customer's specific problem.
    
  create_urgency: |
    When appropriate, mention stock levels, limited-time offers, or seasonal considerations.
```

## 🎯 Implementation Tips

### 1. Data Loading Script Example
```python
# data_loader.py
import pandas as pd
import json
from sales_chatbot import SalesChatBot

def load_product_data():
    bot = SalesChatBot()
    
    # Load Excel data
    df = pd.read_excel('data/products.xlsx')
    for _, row in df.iterrows():
        bot.add_product(row.to_dict())
    
    # Load JSON data
    with open('data/products.json', 'r') as f:
        data = json.load(f)
        for product in data['products']:
            bot.add_product(product)
    
    print(f"Loaded {len(df)} products from Excel and {len(data['products'])} from JSON")

if __name__ == "__main__":
    load_product_data()
```

### 2. Testing Conversations
```python
# test_conversations.py
test_queries = [
    "I need a gaming laptop under $2000",
    "What's the difference between gaming and business laptops?",
    "Do you have wireless mice in stock?",
    "Can this laptop run the latest games?",
    "What accessories go with the Gaming Laptop Pro X1?",
    "Is there a student discount available?",
    "When will the out-of-stock items be available?"
]

for query in test_queries:
    response = bot.get_response(query)
    print(f"Q: {query}")
    print(f"A: {response}\n")
```

This template structure will work perfectly with your RTX 4070Ti Super setup and provides a solid foundation for building a comprehensive sales chatbot!