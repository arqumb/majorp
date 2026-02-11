@echo off
echo ========================================
echo AutoScraper AI - Quick Start Setup
echo ========================================
echo.

echo Step 1: Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo Python found!
echo.

echo Step 2: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)
echo Virtual environment created!
echo.

echo Step 3: Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated!
echo.

echo Step 4: Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo Dependencies installed!
echo.

echo Step 5: Installing Playwright browsers...
echo This may take 5-10 minutes...
python -m playwright install
if errorlevel 1 (
    echo ERROR: Failed to install Playwright browsers!
    pause
    exit /b 1
)
echo Playwright browsers installed!
echo.

echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To test the installation, run:
echo python main.py "https://quotes.toscrape.com" "extract quotes" --output test.json
echo.
echo To activate the virtual environment in future:
echo venv\Scripts\activate
echo.
pause
