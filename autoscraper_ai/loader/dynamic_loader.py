# Dynamic page loading with Playwright WebDriver

"""
Dynamic Loader Module

This module provides Playwright-based dynamic page loading with improved error handling
for missing browser binaries.
"""

from playwright.sync_api import sync_playwright
from typing import Optional


class DynamicLoader:
    """
    Dynamic page loader using Playwright for JavaScript-rendered content.
    """
    
    def __init__(self, timeout: int = 30, user_agent: Optional[str] = None):
        """
        Initialize the DynamicLoader.
        
        Args:
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    
    def load_page(self, url: str) -> str:
        """
        Load page content using Playwright.
        
        Args:
            url: The URL to load
            
        Returns:
            Rendered HTML content after JavaScript execution
            
        Raises:
            Exception: If Playwright fails to load the page
        """
        try:
            with sync_playwright() as p:
                # Launch browser in headless mode
                browser = p.chromium.launch(headless=True)
                
                try:
                    # Create new page with custom user agent
                    page = browser.new_page(user_agent=self.user_agent)
                    
                    # Set timeout and navigate to URL
                    page.set_default_timeout(self.timeout * 1000)  # Convert to milliseconds
                    page.goto(url, wait_until='networkidle')
                    
                    # Wait for initial dynamic content to load
                    page.wait_for_timeout(2000)  # 2 second buffer
                    
                    # Perform small controlled scroll to trigger lazy-loaded content
                    page.evaluate("window.scrollBy(0, 500)")
                    
                    # Wait for any DOM updates triggered by scrolling
                    page.wait_for_timeout(1500)  # 1.5 second buffer for DOM updates
                    
                    # Scroll back to top for consistent extraction
                    page.evaluate("window.scrollTo(0, 0)")
                    
                    # Final wait for any remaining updates
                    page.wait_for_timeout(500)
                    
                    # Get the final HTML content
                    html_content = page.content()
                    
                    return html_content
                    
                finally:
                    browser.close()
        except Exception as e:
            error_msg = str(e).lower()
            if 'executable doesn\'t exist' in error_msg or 'browser executable not found' in error_msg:
                raise Exception("Playwright browser binaries not found. Please run: python -m playwright install")
            else:
                raise e