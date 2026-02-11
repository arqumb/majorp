# 🚀 AutoScraper AI - Complete Setup Guide

This guide will help you set up and run the AutoScraper AI project on any Windows/Mac/Linux machine.

---

## 📋 Prerequisites

Before starting, make sure you have:
- **Python 3.8 or higher** installed
- **Git** installed
- **Internet connection** (for downloading dependencies)
- **Command line/Terminal** access

---

## 🔧 Step-by-Step Installation

### Step 1: Check Python Installation

Open Command Prompt (Windows) or Terminal (Mac/Linux) and run:

```bash
python --version
```

**Expected output:** `Python 3.8.x` or higher

If Python is not installed:
- **Windows:** Download from [python.org](https://www.python.org/downloads/)
- **Mac:** `brew install python3` or download from python.org
- **Linux:** `sudo apt-get install python3 python3-pip`

---

### Step 2: Check Git Installation

```bash
git --version
```

**Expected output:** `git version 2.x.x`

If Git is not installed:
- **Windows:** Download from [git-scm.com](https://git-scm.com/download/win)
- **Mac:** `brew install git` or download from git-scm.com
- **Linux:** `sudo apt-get install git`

---

### Step 3: Clone the Repository

Open Command Prompt/Terminal and navigate to where you want to install the project:

```bash
# Navigate to your desired location (example)
cd Desktop

# Clone the repository
git clone https://github.com/arqumb/majorp.git

# Enter the project directory
cd majorp
```

**Expected output:** Project files downloaded successfully

---

### Step 4: Create Virtual Environment (Recommended)

Creating a virtual environment keeps dependencies isolated:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Expected output:** Your command prompt should now show `(venv)` at the beginning

---

### Step 5: Install Python Dependencies

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

**Expected output:** All packages installed successfully

**This will install:**
- requests (HTTP requests)
- lxml (XPath processing)
- beautifulsoup4 (HTML parsing)
- playwright (Dynamic content)

**Installation time:** 2-5 minutes depending on internet speed

---

### Step 6: Install Playwright Browsers

Playwright needs to download browser binaries:

```bash
python -m playwright install
```

**Expected output:** Chromium, Firefox, and WebKit browsers downloaded

**Installation time:** 3-10 minutes (downloads ~300MB)

**Note:** If you get an error, try:
```bash
playwright install
```

---

### Step 7: Verify Installation

Test if everything is working:

```bash
python main.py --help
```

**Expected output:**
```
Usage: python main.py <url> <task> [--force-new] [--output file.json]
Example: python main.py 'https://example.com' 'extract product titles'
```

---

## ✅ Test the Installation

### Test 1: Simple Quote Extraction

```bash
python main.py "https://quotes.toscrape.com" "extract quote text and author name" --output test_quotes.json
```

**Expected output:**
- Scraping progress messages
- "Results saved to test_quotes.json"
- File created with extracted quotes

**Time:** 5-10 seconds

---

### Test 2: E-commerce Product Extraction

```bash
python main.py "https://urbannucleus.in/collections/new-balance-shoes" "extract product names and prices" --output test_products.json
```

**Expected output:**
- Dynamic loading messages
- "Results saved to test_products.json"
- File created with product data

**Time:** 10-15 seconds

---

## 🎯 Usage Examples

### Basic Usage

```bash
python main.py "URL" "TASK_DESCRIPTION"
```

### Save to File

```bash
python main.py "URL" "TASK_DESCRIPTION" --output results.json
```

### Force New Rule (Ignore Cache)

```bash
python main.py "URL" "TASK_DESCRIPTION" --force-new
```

### Real Examples

```bash
# Extract news headlines
python main.py "https://news-site.com" "extract article titles and dates" --output news.json

# Extract product information
python main.py "https://shop.com/products" "extract product names, prices, and descriptions" --output products.json

# Extract quotes
python main.py "https://quotes.toscrape.com" "extract quotes and authors" --output quotes.json
```

---

## 🔍 Troubleshooting

### Problem 1: "python: command not found"

**Solution:**
- Try `python3` instead of `python`
- Reinstall Python and check "Add to PATH" during installation

---

### Problem 2: "pip: command not found"

**Solution:**
```bash
python -m pip install --upgrade pip
```

---

### Problem 3: "playwright install failed"

**Solution:**
```bash
# Try with admin/sudo privileges
# Windows (Run as Administrator)
python -m playwright install

# Mac/Linux
sudo python3 -m playwright install
```

---

### Problem 4: "ModuleNotFoundError: No module named 'xxx'"

**Solution:**
```bash
# Make sure virtual environment is activated
# Then reinstall requirements
pip install -r requirements.txt
```

---

### Problem 5: "Static loading failed: 404"

**Solution:**
- This is normal! The system automatically tries dynamic loading
- Wait for "Dynamic loading" message
- The scraper will still work

---

### Problem 6: Slow Performance

**Solution:**
- First run is slower (downloads browser binaries)
- Subsequent runs use cached rules and are faster
- Use `--force-new` only when needed

---

## 📁 Project Structure

After installation, your directory should look like:

```
majorp/
├── autoscraper_ai/          # Main package
│   ├── autoscraper/         # Core algorithm
│   ├── config/              # Configuration
│   ├── executor/            # XPath execution
│   ├── input/               # URL handling
│   ├── loader/              # Page loading
│   ├── preprocessing/       # HTML processing
│   └── storage/             # Rule storage
├── venv/                    # Virtual environment (if created)
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── README.md               # Documentation
├── SETUP.md                # This file
└── .gitignore              # Git ignore rules
```

---

## 🎓 Quick Start Tutorial

### 1. Activate Virtual Environment

**Windows:**
```bash
cd majorp
venv\Scripts\activate
```

**Mac/Linux:**
```bash
cd majorp
source venv/bin/activate
```

---

### 2. Run Your First Scrape

```bash
python main.py "https://quotes.toscrape.com" "extract quotes" --output my_first_scrape.json
```

---

### 3. View Results

Open `my_first_scrape.json` in any text editor to see the extracted data!

---

### 4. Try Different Websites

```bash
# Try different extraction tasks
python main.py "https://example.com" "extract titles and links"
python main.py "https://shop.com" "extract product names and prices"
python main.py "https://blog.com" "extract article titles and dates"
```

---

## ⚙️ Configuration (Optional)

### Environment Variables

Create a `.env` file in the project root:

```bash
# Scraping behavior
AUTOSCRAPER_MAX_ITERATIONS=5
AUTOSCRAPER_MIN_CONFIDENCE=0.7

# Page loading
AUTOSCRAPER_STATIC_TIMEOUT=10
AUTOSCRAPER_DYNAMIC_TIMEOUT=30

# Storage
AUTOSCRAPER_DB_PATH=scraper_rules.db
```

---

## 🔄 Updating the Project

To get the latest updates:

```bash
# Navigate to project directory
cd majorp

# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
```

---

## 🧹 Uninstallation

To completely remove the project:

```bash
# Deactivate virtual environment
deactivate

# Delete project folder
cd ..
rm -rf majorp  # Mac/Linux
rmdir /s majorp  # Windows
```

---

## 📞 Getting Help

If you encounter issues:

1. **Check this guide** - Most common issues are covered
2. **Check GitHub Issues** - https://github.com/arqumb/majorp/issues
3. **Create new issue** - Include error messages and steps to reproduce

---

## ✅ Installation Checklist

Use this checklist to verify your setup:

- [ ] Python 3.8+ installed and working
- [ ] Git installed and working
- [ ] Repository cloned successfully
- [ ] Virtual environment created and activated
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Playwright browsers installed (`python -m playwright install`)
- [ ] Test scrape completed successfully
- [ ] Output JSON file created and readable

---

## 🎉 Success!

If all tests pass, you're ready to use AutoScraper AI!

**Next Steps:**
- Try scraping different websites
- Experiment with different extraction tasks
- Check the README.md for advanced features
- Explore the code to understand how it works

---

## 📚 Additional Resources

- **Main Documentation:** README.md
- **GitHub Repository:** https://github.com/arqumb/majorp
- **Python Documentation:** https://docs.python.org/3/
- **Playwright Documentation:** https://playwright.dev/python/

---

**Happy Scraping! 🚀**