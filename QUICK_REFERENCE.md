# 🚀 AutoScraper AI - Quick Reference Guide

## 📥 Installation (One-Time Setup)

### Windows (Easiest Method)
```
1. Download project from GitHub
2. Double-click "QUICK_START.bat"
3. Wait 5-15 minutes
4. Done! ✅
```

### Mac/Linux (Easiest Method)
```bash
1. Download project from GitHub
2. Open Terminal in project folder
3. Run: chmod +x QUICK_START.sh
4. Run: ./QUICK_START.sh
5. Wait 5-15 minutes
6. Done! ✅
```

---

## 🎯 Daily Usage

### Step 1: Open Terminal/Command Prompt
```bash
cd majorp
```

### Step 2: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### Step 3: Run Scraper
```bash
python main.py "YOUR_URL" "YOUR_TASK" --output results.json
```

---

## 💡 Common Commands

### Extract Quotes
```bash
python main.py "https://quotes.toscrape.com" "extract quotes and authors" --output quotes.json
```

### Extract Products
```bash
python main.py "https://shop.com/products" "extract product names and prices" --output products.json
```

### Extract News
```bash
python main.py "https://news-site.com" "extract article titles and dates" --output news.json
```

### Force New Rule (Ignore Cache)
```bash
python main.py "URL" "TASK" --force-new --output results.json
```

---

## 🔧 Command Structure

```
python main.py "URL" "TASK_DESCRIPTION" [OPTIONS]

Required:
  URL               - Website to scrape (in quotes)
  TASK_DESCRIPTION  - What to extract (in quotes)

Optional:
  --output FILE     - Save results to file
  --force-new       - Generate new rule (ignore cache)
```

---

## 📊 Output Format

Results are saved as JSON:
```json
{
  "success": true,
  "data": [
    {
      "title": "Product Name",
      "price": "$99.99"
    }
  ],
  "metadata": {
    "data_count": 10,
    "execution_time_seconds": 5.2
  }
}
```

---

## ⚡ Quick Tips

1. **First run is slower** - Downloads browsers (~300MB)
2. **Use quotes** - Always put URL and task in "quotes"
3. **Be specific** - "extract product names and prices" works better than "extract data"
4. **Check output** - Open the JSON file to see results
5. **Reuse rules** - Second run on same site is faster (uses cached rules)

---

## 🐛 Quick Fixes

| Problem | Solution |
|---------|----------|
| `python: command not found` | Try `python3` instead |
| `pip: command not found` | Run `python -m pip install --upgrade pip` |
| `ModuleNotFoundError` | Activate virtual environment first |
| Slow performance | Normal for first run, faster after |
| `404 error` | Normal! System tries dynamic loading next |

---

## 📁 File Locations

```
majorp/
├── main.py              ← Run this file
├── requirements.txt     ← Dependencies list
├── SETUP.md            ← Detailed setup guide
├── QUICK_START.bat     ← Windows auto-installer
├── QUICK_START.sh      ← Mac/Linux auto-installer
└── results.json        ← Your scraped data (after running)
```

---

## 🎓 Example Session

```bash
# 1. Navigate to project
cd majorp

# 2. Activate environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 3. Run scraper
python main.py "https://quotes.toscrape.com" "extract quotes" --output my_quotes.json

# 4. Check results
# Open my_quotes.json in any text editor

# 5. Done! Deactivate when finished
deactivate
```

---

## 📞 Need Help?

1. **Read SETUP.md** - Detailed instructions
2. **Read INSTALL_INSTRUCTIONS.txt** - Quick troubleshooting
3. **Check GitHub Issues** - https://github.com/arqumb/majorp/issues
4. **Create Issue** - Include error message and steps

---

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Git installed
- [ ] Project cloned/downloaded
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Playwright browsers installed
- [ ] Test scrape successful

---

**That's it! You're ready to scrape! 🎉**