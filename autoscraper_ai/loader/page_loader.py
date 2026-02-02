"""
Page Loader Module for Hybrid Web Scraping

This module provides functionality to load web pages using both static and dynamic methods.
It automatically detects if a page requires JavaScript rendering and switches to Playwright
when necessary.
"""

import requests
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse


class PageLoader:
    """
    A hybrid page loader that handles both static and dynamic web pages.
    
    Automatically detects JavaScript-rendered content and falls back to
    browser automation when needed.
    """
    
    def __init__(self, timeout: int = 30, user_agent: Optional[str] = None):
        """
        Initialize the PageLoader.
        
        Args:
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
    
    def load_page(self, url: str) -> str:
        """
        Load a web page using the most appropriate method.
        
        First attempts static loading with requests. If the page appears to be
        JavaScript-rendered, falls back to Playwright for dynamic rendering.
        
        Args:
            url: The URL to load
            
        Returns:
            The final HTML content as a string
            
        Raises:
            ValueError: If URL is invalid
            requests.RequestException: If static loading fails
            Exception: If dynamic loading fails
        """
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")
        
        self.logger.info(f"Loading page: {url}")
        
        # Step 1: Try static loading first
        try:
            static_html = self._load_static(url)
            
            # Step 2: Check if content is JavaScript-rendered
            if self._is_dynamic_content(static_html):
                self.logger.info("Dynamic content detected, switching to Playwright")
                return self._load_dynamic(url)
            else:
                self.logger.info("Static content loaded successfully")
                return static_html
                
        except requests.RequestException as e:
            self.logger.warning(f"Static loading failed: {e}, trying dynamic loading")
            return self._load_dynamic(url)
    
    def _load_static(self, url: str) -> str:
        """
        Load page content using requests (static method).
        
        Args:
            url: The URL to load
            
        Returns:
            Raw HTML content
            
        Raises:
            requests.RequestException: If the request fails
        """
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        
        # Handle different encodings
        if response.encoding is None:
            response.encoding = 'utf-8'
        
        return response.text
    
    def _load_dynamic(self, url: str) -> str:
        """
        Load page content using Playwright (dynamic method).
        
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
                    
                    # Wait for potential dynamic content to load
                    page.wait_for_timeout(2000)  # 2 second buffer
                    
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
    
    def _is_dynamic_content(self, html: str) -> bool:
        """
        Detect if the HTML content appears to be JavaScript-rendered.
        
        Checks for common indicators of dynamic content:
        - Empty or minimal body content
        - Missing essential HTML tags
        - Presence of JavaScript frameworks
        - Very low content-to-markup ratio
        
        Args:
            html: The HTML content to analyze
            
        Returns:
            True if content appears to be JavaScript-rendered, False otherwise
        """
        if not html or len(html.strip()) < 100:
            return True
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Check 1: Empty or minimal body
        body = soup.find('body')
        if not body:
            return True
        
        body_text = body.get_text(strip=True)
        if len(body_text) < 50:  # Very little actual content
            return True
        
        # Check 2: High script-to-content ratio (indicates heavy JS usage)
        scripts = soup.find_all('script')
        script_content = sum(len(script.get_text()) for script in scripts)
        
        if script_content > len(body_text) * 3:  # Scripts are 3x larger than content
            return True
        
        # Check 3: Common JavaScript framework indicators
        js_indicators = [
            'ng-app',           # Angular
            'data-reactroot',   # React
            'v-app',           # Vue.js
            '__NEXT_DATA__',   # Next.js
            'nuxt',            # Nuxt.js
        ]
        
        html_lower = html.lower()
        for indicator in js_indicators:
            if indicator.lower() in html_lower:
                return True
        
        # Check 4: Minimal meaningful content tags
        content_tags = soup.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        meaningful_content = [tag for tag in content_tags if len(tag.get_text(strip=True)) > 10]
        
        if len(meaningful_content) < 3:  # Very few content elements
            return True
        
        # Check 5: Common "loading" or placeholder indicators
        loading_indicators = [
            'loading',
            'please wait',
            'javascript is required',
            'enable javascript',
            'js-disabled'
        ]
        
        for indicator in loading_indicators:
            if indicator in body_text.lower():
                return True
        
        return False
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Validate if the provided URL is properly formatted.
        
        Args:
            url: The URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    def close(self):
        """Close the requests session to free up resources."""
        self.session.close()


def load_page(url: str, timeout: int = 30, user_agent: Optional[str] = None) -> str:
    """
    Convenience function to load a single page.
    
    Args:
        url: The URL to load
        timeout: Request timeout in seconds
        user_agent: Custom user agent string
        
    Returns:
        The final HTML content as a string
    """
    loader = PageLoader(timeout=timeout, user_agent=user_agent)
    try:
        return loader.load_page(url)
    finally:
        loader.close()


# Example usage
if __name__ == "__main__":
    # Configure logging for demo
    logging.basicConfig(level=logging.INFO)
    
    # Test URLs
    test_urls = [
        "https://httpbin.org/html",  # Static content
        "https://example.com",       # Static content
    ]
    
    loader = PageLoader()
    
    try:
        for url in test_urls:
            print(f"\n--- Loading: {url} ---")
            html = loader.load_page(url)
            print(f"Content length: {len(html)} characters")
            print(f"First 200 characters: {html[:200]}...")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        loader.close()