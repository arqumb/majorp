"""
HTML Preprocessor Module

This module provides functionality to clean and preprocess raw HTML content
while preserving the DOM structure and essential attributes needed for
web scraping operations.
"""

from bs4 import BeautifulSoup, Comment
import re
from typing import Optional, List


class HTMLPreprocessor:
    """
    A class for preprocessing HTML content to remove unwanted elements
    while preserving the DOM structure and essential attributes.
    """
    
    def __init__(self, parser: str = 'lxml'):
        """
        Initialize the HTML preprocessor.
        
        Args:
            parser: The parser to use with BeautifulSoup ('lxml', 'html.parser', etc.)
        """
        self.parser = parser
    
    def clean_html(self, raw_html: str) -> str:
        """
        Clean raw HTML by removing scripts, styles, and other unwanted content
        while preserving DOM structure and class attributes.
        
        Args:
            raw_html: The raw HTML string to clean
            
        Returns:
            Cleaned HTML string with preserved DOM structure
        """
        if not raw_html or not raw_html.strip():
            return ""
        
        # Step 1: Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(raw_html, self.parser)
        
        # Step 2: Remove script tags and their content
        self._remove_script_tags(soup)
        
        # Step 3: Remove style tags and their content
        self._remove_style_tags(soup)
        
        # Step 4: Remove HTML comments
        self._remove_comments(soup)
        
        # Step 5: Remove other unwanted tags
        self._remove_unwanted_tags(soup)
        
        # Step 6: Clean up whitespace while preserving structure
        self._normalize_whitespace(soup)
        
        # Step 7: Convert back to string and return
        return str(soup)
    
    def _remove_script_tags(self, soup: BeautifulSoup) -> None:
        """
        Remove all <script> tags and their content from the soup.
        
        This includes:
        - Inline JavaScript code
        - External script references
        - Event handlers and other script-related content
        
        Args:
            soup: BeautifulSoup object to modify in-place
        """
        # Find and remove all script tags
        for script in soup.find_all('script'):
            script.decompose()  # Completely remove the tag and its content
    
    def _remove_style_tags(self, soup: BeautifulSoup) -> None:
        """
        Remove all <style> tags and their content from the soup.
        
        This removes:
        - Inline CSS styles
        - Internal stylesheets
        - CSS imports and other style-related content
        
        Note: This preserves 'class' and 'id' attributes which are essential
        for element identification and selection.
        
        Args:
            soup: BeautifulSoup object to modify in-place
        """
        # Find and remove all style tags
        for style in soup.find_all('style'):
            style.decompose()  # Completely remove the tag and its content
    
    def _remove_comments(self, soup: BeautifulSoup) -> None:
        """
        Remove HTML comments from the soup.
        
        HTML comments can contain:
        - Developer notes
        - Conditional comments (IE-specific)
        - Commented-out code
        
        Args:
            soup: BeautifulSoup object to modify in-place
        """
        # Find and remove all HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
    
    def _remove_unwanted_tags(self, soup: BeautifulSoup) -> None:
        """
        Remove other unwanted tags that don't contribute to content extraction.
        
        This includes:
        - <noscript> tags (fallback content for disabled JavaScript)
        - <meta> tags (metadata not needed for scraping)
        - <link> tags (external resources)
        - <base> tags (base URL definitions)
        
        Args:
            soup: BeautifulSoup object to modify in-place
        """
        unwanted_tags = ['noscript', 'meta', 'link', 'base']
        
        for tag_name in unwanted_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
    
    def _normalize_whitespace(self, soup: BeautifulSoup) -> None:
        """
        Normalize whitespace in text content while preserving DOM structure.
        
        This function:
        - Removes excessive whitespace from text nodes
        - Preserves single spaces between words
        - Maintains line breaks where semantically important
        - Keeps the DOM hierarchy intact
        
        Args:
            soup: BeautifulSoup object to modify in-place
        """
        # Find all text nodes and normalize whitespace
        for text_node in soup.find_all(string=True):
            if text_node.parent.name not in ['pre', 'code', 'textarea']:
                # Normalize whitespace but preserve single spaces
                normalized_text = re.sub(r'\s+', ' ', text_node.strip())
                if normalized_text:
                    text_node.replace_with(normalized_text)
                else:
                    # Remove empty text nodes
                    text_node.extract()
    
    def get_text_content(self, raw_html: str) -> str:
        """
        Extract clean text content from HTML without any markup.
        
        Args:
            raw_html: The raw HTML string to process
            
        Returns:
            Plain text content with normalized whitespace
        """
        # First clean the HTML
        cleaned_html = self.clean_html(raw_html)
        
        # Parse the cleaned HTML
        soup = BeautifulSoup(cleaned_html, self.parser)
        
        # Extract text content
        text_content = soup.get_text(separator=' ', strip=True)
        
        # Normalize whitespace in the final text
        return re.sub(r'\s+', ' ', text_content)
    
    def preserve_structure_info(self, raw_html: str) -> dict:
        """
        Clean HTML and return both cleaned HTML and structural information.
        
        Args:
            raw_html: The raw HTML string to process
            
        Returns:
            Dictionary containing:
            - 'cleaned_html': The cleaned HTML string
            - 'tag_count': Number of remaining tags
            - 'text_length': Length of text content
            - 'has_classes': Whether class attributes are present
        """
        cleaned_html = self.clean_html(raw_html)
        soup = BeautifulSoup(cleaned_html, self.parser)
        
        # Gather structural information
        all_tags = soup.find_all()
        tags_with_classes = soup.find_all(class_=True)
        text_content = soup.get_text(strip=True)
        
        return {
            'cleaned_html': cleaned_html,
            'tag_count': len(all_tags),
            'text_length': len(text_content),
            'has_classes': len(tags_with_classes) > 0,
            'class_count': len(tags_with_classes)
        }


def clean_html(raw_html: str, parser: str = 'lxml') -> str:
    """
    Convenience function to clean HTML content.
    
    Args:
        raw_html: The raw HTML string to clean
        parser: The parser to use with BeautifulSoup
        
    Returns:
        Cleaned HTML string
    """
    preprocessor = HTMLPreprocessor(parser=parser)
    return preprocessor.clean_html(raw_html)


def extract_text(raw_html: str, parser: str = 'lxml') -> str:
    """
    Convenience function to extract clean text from HTML.
    
    Args:
        raw_html: The raw HTML string to process
        parser: The parser to use with BeautifulSoup
        
    Returns:
        Plain text content
    """
    preprocessor = HTMLPreprocessor(parser=parser)
    return preprocessor.get_text_content(raw_html)


# Example usage and testing
if __name__ == "__main__":
    # Sample HTML with various elements to test
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
        <script>
            console.log("This script should be removed");
            var data = {test: "value"};
        </script>
        <style>
            body { background-color: #fff; }
            .container { margin: 20px; }
        </style>
        <meta charset="UTF-8">
        <link rel="stylesheet" href="styles.css">
    </head>
    <body>
        <!-- This is a comment that should be removed -->
        <div class="container">
            <h1 class="title">Main Heading</h1>
            <p class="content">This is some content with <strong>bold text</strong>.</p>
            <ul class="list">
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
        <noscript>
            <p>JavaScript is disabled in your browser.</p>
        </noscript>
        <script>
            // Another script to remove
            alert("Hello World");
        </script>
    </body>
    </html>
    """
    
    # Test the preprocessor
    preprocessor = HTMLPreprocessor()
    
    print("=== Original HTML ===")
    print(sample_html[:200] + "...")
    
    print("\n=== Cleaned HTML ===")
    cleaned = preprocessor.clean_html(sample_html)
    print(cleaned)
    
    print("\n=== Text Content Only ===")
    text_only = preprocessor.get_text_content(sample_html)
    print(text_only)
    
    print("\n=== Structure Information ===")
    info = preprocessor.preserve_structure_info(sample_html)
    for key, value in info.items():
        if key != 'cleaned_html':  # Don't print the full HTML again
            print(f"{key}: {value}")