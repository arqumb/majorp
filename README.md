# AutoScraper AI 🤖

An intelligent hybrid web scraping system inspired by AUTOSCRAPER that automatically generates XPath extraction rules through progressive refinement and cross-page validation.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🚀 Features

- **🔄 Hybrid Loading**: Automatically detects and handles both static and dynamic (JavaScript-rendered) websites
- **🎯 Progressive XPath Generation**: Uses iterative refinement with step-back logic to build robust extraction rules
- **✅ Cross-Page Validation**: Phase 2 synthesis validates rules across multiple pages for better reliability
- **🏗️ DOM-Based Structuring**: Groups extracted data into meaningful structured objects
- **💾 Persistent Rule Storage**: Saves and reuses successful extraction patterns
- **🛡️ Ethical Scraping**: No security bypass, CAPTCHA circumvention, or anti-bot evasion

## 📁 Architecture

```
autoscraper_ai/
├── autoscraper/          # Core AUTOSCRAPER algorithm
│   ├── phase1_progressive.py    # Progressive XPath generation
│   ├── phase2_synthesis.py      # Cross-page validation
│   └── ai_assistant.py          # AI-powered helpers
├── loader/               # Page loading (static + dynamic)
├── preprocessing/        # HTML cleaning and DOM analysis
├── executor/             # XPath execution engine
├── storage/              # Rule persistence
├── input/                # URL handling and validation
└── config/               # Configuration and logging
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/arqumb/majorp.git
cd majorp
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install Playwright browsers (for dynamic content):**
```bash
python -m playwright install
```

## 🎮 Usage

### Command Line Interface

```bash
# Basic usage
python main.py "https://example.com" "extract product names and prices"

# Save results to file
python main.py "https://example.com" "extract quotes and authors" --output results.json

# Force new rule generation (ignore cached rules)
python main.py "https://example.com" "extract data" --force-new
```

### 📝 Examples

```bash
# E-commerce product extraction
python main.py "https://urbannucleus.in/collections/new-balance-shoes" "extract New Balance product names and prices" --output products.json

# Quote extraction
python main.py "https://quotes.toscrape.com" "extract quote text and author name" --output quotes.json

# News article extraction
python main.py "https://news-site.com" "extract article titles and dates" --output news.json
```

## ⚙️ How It Works

### 🔄 Phase 1: Progressive XPath Generation
1. **HTML Analysis**: Analyzes DOM structure and extraction task requirements
2. **XPath Generation**: Creates initial XPath suggestions using pattern recognition
3. **Iterative Refinement**: Tests and refines using step-back logic for failed extractions
4. **Action Sequence**: Builds ordered sequence of successful extraction steps

### ✅ Phase 2: Cross-Page Validation
1. **Multi-Page Testing**: Tests Action Sequences across multiple similar pages
2. **Robustness Scoring**: Calculates success rates and reliability metrics
3. **Pattern Selection**: Chooses most reliable extraction patterns for reuse

### 🏗️ DOM Structuring
1. **Container Grouping**: Groups extracted elements by parent containers
2. **Field Inference**: Automatically infers semantic field names (title, price, description)
3. **Structured Output**: Creates meaningful objects instead of flat text lists

## 🔧 Configuration

### Environment Variables

```bash
# Scraping behavior
AUTOSCRAPER_MAX_ITERATIONS=5
AUTOSCRAPER_MIN_CONFIDENCE=0.7
AUTOSCRAPER_TIMEOUT_MS=5000

# Page loading
AUTOSCRAPER_STATIC_TIMEOUT=10
AUTOSCRAPER_DYNAMIC_TIMEOUT=30
AUTOSCRAPER_USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Storage
AUTOSCRAPER_DB_PATH=scraper_rules.db
```

### Configuration Files

The system uses centralized configuration in `autoscraper_ai/config/settings.py` with support for environment variable overrides.

## 📊 Output Format

```json
{
  "success": true,
  "data": [
    {
      "title": "New Balance 550 'Chicago'",
      "price": "Rs. 12,999.00"
    },
    {
      "title": "New Balance 9060 'Mushroom'",
      "price": "Rs. 15,999.00"
    }
  ],
  "metadata": {
    "extraction_task": "extract product names and prices",
    "domain": "urbannucleus.in",
    "sequence_id": "seq_bb5f8295aa93",
    "is_new_rule": true,
    "data_count": 38,
    "execution_time_seconds": 6.11,
    "success_rate": 1.0,
    "robustness_score": 0.71,
    "timestamp": "2026-02-03T00:15:14.537933"
  }
}
```

## 📋 Requirements

- **Python 3.8+**
- **Core Dependencies:**
  - `requests` - HTTP requests
  - `lxml` - XPath processing
  - `beautifulsoup4` - HTML parsing
  - `playwright` - Dynamic content rendering
  - `sqlite3` - Rule storage (built-in)

## 🚀 Recent Improvements

- ✅ Enhanced XPath generation for e-commerce sites
- ✅ Improved DOM structuring with better field inference
- ✅ Advanced product name and price pairing
- ✅ Better handling of dynamic content
- ✅ Comprehensive logging and configuration system
- ✅ URL validation and preprocessing
- ✅ Advanced DOM analysis capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚖️ Ethical Usage

This tool is designed for **legitimate web scraping purposes**. Please:

- ✅ Respect `robots.txt` files
- ✅ Don't overload servers with requests
- ✅ Follow website terms of service
- ✅ Use appropriate delays between requests
- ✅ Respect rate limits and server resources
- ❌ Don't use for malicious purposes
- ❌ Don't bypass security measures

## 🐛 Issues & Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/arqumb/majorp/issues) page
2. Create a new issue with detailed information
3. Include error messages and steps to reproduce

## 🌟 Acknowledgments

- Inspired by the original AUTOSCRAPER research
- Built with modern Python web scraping best practices
- Uses ethical scraping principles throughout