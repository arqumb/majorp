#!/bin/bash

echo "========================================"
echo "AutoScraper AI - Quick Start Setup"
echo "========================================"
echo ""

echo "Step 1: Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed!"
    echo "Please install Python 3.8+ from https://www.python.org/downloads/"
    exit 1
fi
python3 --version
echo "Python found!"
echo ""

echo "Step 2: Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment!"
    exit 1
fi
echo "Virtual environment created!"
echo ""

echo "Step 3: Activating virtual environment..."
source venv/bin/activate
echo "Virtual environment activated!"
echo ""

echo "Step 4: Installing Python dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies!"
    exit 1
fi
echo "Dependencies installed!"
echo ""

echo "Step 5: Installing Playwright browsers..."
echo "This may take 5-10 minutes..."
python -m playwright install
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Playwright browsers!"
    exit 1
fi
echo "Playwright browsers installed!"
echo ""

echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To test the installation, run:"
echo "python main.py \"https://quotes.toscrape.com\" \"extract quotes\" --output test.json"
echo ""
echo "To activate the virtual environment in future:"
echo "source venv/bin/activate"
echo ""
