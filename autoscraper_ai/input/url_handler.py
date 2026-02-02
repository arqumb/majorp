"""
URL Validation and Preprocessing

This module provides utilities for validating, normalizing, and preprocessing URLs
before they are used in the scraping process.
"""

import re
from urllib.parse import urlparse, urljoin, quote, unquote
from typing import Optional, Dict, Any, List
import logging


class URLHandler:
    """Handles URL validation, normalization, and preprocessing."""
    
    def __init__(self):
        """Initialize URL handler."""
        self.logger = logging.getLogger(__name__)
        
        # Common URL patterns
        self.url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        # Blocked domains/patterns for security
        self.blocked_patterns = [
            r'localhost',
            r'127\.0\.0\.1',
            r'0\.0\.0\.0',
            r'192\.168\.',
            r'10\.',
            r'172\.(1[6-9]|2[0-9]|3[01])\.'  # Private IP ranges
        ]
    
    def validate_url(self, url: str) -> Dict[str, Any]:
        """
        Validate a URL for scraping.
        
        Args:
            url: URL to validate
            
        Returns:
            Dictionary with validation results:
            - 'valid': Boolean indicating if URL is valid
            - 'normalized': Normalized URL (if valid)
            - 'issues': List of validation issues
            - 'metadata': Additional URL metadata
        """
        issues = []
        
        if not url or not isinstance(url, str):
            return {
                'valid': False,
                'normalized': None,
                'issues': ['URL is empty or not a string'],
                'metadata': {}
            }
        
        # Basic format validation
        if not self.url_pattern.match(url):
            issues.append('Invalid URL format')
        
        # Parse URL components
        try:
            parsed = urlparse(url)
        except Exception as e:
            return {
                'valid': False,
                'normalized': None,
                'issues': [f'URL parsing failed: {str(e)}'],
                'metadata': {}
            }
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, parsed.netloc, re.IGNORECASE):
                issues.append(f'Blocked domain pattern: {pattern}')
        
        # Check scheme
        if parsed.scheme not in ['http', 'https']:
            issues.append(f'Unsupported scheme: {parsed.scheme}')
        
        # Check for missing domain
        if not parsed.netloc:
            issues.append('Missing domain name')
        
        # Normalize URL
        normalized_url = self._normalize_url(url) if not issues else None
        
        # Extract metadata
        metadata = self._extract_url_metadata(parsed) if not issues else {}
        
        return {
            'valid': len(issues) == 0,
            'normalized': normalized_url,
            'issues': issues,
            'metadata': metadata
        }
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL by removing unnecessary components and standardizing format.
        
        Args:
            url: Raw URL
            
        Returns:
            Normalized URL
        """
        parsed = urlparse(url)
        
        # Normalize scheme to lowercase
        scheme = parsed.scheme.lower()
        
        # Normalize domain to lowercase
        netloc = parsed.netloc.lower()
        
        # Remove default ports
        if ':80' in netloc and scheme == 'http':
            netloc = netloc.replace(':80', '')
        elif ':443' in netloc and scheme == 'https':
            netloc = netloc.replace(':443', '')
        
        # Normalize path
        path = parsed.path or '/'
        if path != '/' and path.endswith('/'):
            path = path.rstrip('/')
        
        # URL encode path if needed
        path = quote(unquote(path), safe='/:@!$&\'()*+,;=')
        
        # Reconstruct URL
        normalized = f"{scheme}://{netloc}{path}"
        
        # Add query string if present
        if parsed.query:
            normalized += f"?{parsed.query}"
        
        # Note: We typically ignore fragments (#) for scraping
        
        return normalized
    
    def _extract_url_metadata(self, parsed_url) -> Dict[str, Any]:
        """
        Extract metadata from parsed URL.
        
        Args:
            parsed_url: ParseResult from urlparse
            
        Returns:
            Dictionary with URL metadata
        """
        metadata = {
            'scheme': parsed_url.scheme,
            'domain': parsed_url.netloc,
            'path': parsed_url.path,
            'has_query': bool(parsed_url.query),
            'has_fragment': bool(parsed_url.fragment),
            'is_secure': parsed_url.scheme == 'https',
            'port': parsed_url.port,
            'path_segments': [seg for seg in parsed_url.path.split('/') if seg]
        }
        
        # Detect common site types
        domain_lower = parsed_url.netloc.lower()
        if any(pattern in domain_lower for pattern in ['shop', 'store', 'buy', 'cart']):
            metadata['site_type'] = 'ecommerce'
        elif any(pattern in domain_lower for pattern in ['blog', 'news', 'article']):
            metadata['site_type'] = 'content'
        elif any(pattern in domain_lower for pattern in ['api', 'rest', 'graphql']):
            metadata['site_type'] = 'api'
        else:
            metadata['site_type'] = 'general'
        
        return metadata
    
    def build_absolute_url(self, base_url: str, relative_url: str) -> Optional[str]:
        """
        Build absolute URL from base URL and relative URL.
        
        Args:
            base_url: Base URL
            relative_url: Relative URL or path
            
        Returns:
            Absolute URL or None if invalid
        """
        try:
            # Validate base URL first
            base_validation = self.validate_url(base_url)
            if not base_validation['valid']:
                return None
            
            # Join URLs
            absolute_url = urljoin(base_validation['normalized'], relative_url)
            
            # Validate result
            result_validation = self.validate_url(absolute_url)
            return result_validation['normalized'] if result_validation['valid'] else None
            
        except Exception as e:
            self.logger.error(f"Failed to build absolute URL: {e}")
            return None
    
    def extract_domain(self, url: str) -> Optional[str]:
        """
        Extract domain from URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain name or None if invalid
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower() if parsed.netloc else None
        except:
            return None
    
    def is_same_domain(self, url1: str, url2: str) -> bool:
        """
        Check if two URLs belong to the same domain.
        
        Args:
            url1: First URL
            url2: Second URL
            
        Returns:
            True if same domain, False otherwise
        """
        domain1 = self.extract_domain(url1)
        domain2 = self.extract_domain(url2)
        
        return domain1 is not None and domain1 == domain2
    
    def get_robots_txt_url(self, url: str) -> Optional[str]:
        """
        Get robots.txt URL for a given site URL.
        
        Args:
            url: Site URL
            
        Returns:
            robots.txt URL or None if invalid
        """
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return None
            
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            return robots_url
        except:
            return None


# Global URL handler instance
url_handler = URLHandler()


# Convenience functions
def validate_url(url: str) -> Dict[str, Any]:
    """Validate a URL using the global handler."""
    return url_handler.validate_url(url)


def normalize_url(url: str) -> Optional[str]:
    """Normalize a URL using the global handler."""
    result = url_handler.validate_url(url)
    return result['normalized'] if result['valid'] else None


def extract_domain(url: str) -> Optional[str]:
    """Extract domain from URL using the global handler."""
    return url_handler.extract_domain(url)


def is_valid_url(url: str) -> bool:
    """Check if URL is valid using the global handler."""
    return url_handler.validate_url(url)['valid']


# Example usage and testing
if __name__ == "__main__":
    # Test URL validation
    test_urls = [
        "https://example.com",
        "http://shop.example.com/products?category=shoes",
        "https://localhost:8080/test",  # Should be blocked
        "invalid-url",
        "ftp://example.com",  # Unsupported scheme
        "https://example.com/path with spaces",
        "https://example.com:443/secure-path"
    ]
    
    handler = URLHandler()
    
    for url in test_urls:
        print(f"\n--- Testing: {url} ---")
        result = handler.validate_url(url)
        print(f"Valid: {result['valid']}")
        if result['valid']:
            print(f"Normalized: {result['normalized']}")
            print(f"Metadata: {result['metadata']}")
        else:
            print(f"Issues: {result['issues']}")
    
    # Test URL building
    print(f"\n--- URL Building Test ---")
    base = "https://example.com/products/"
    relative = "../about.html"
    absolute = handler.build_absolute_url(base, relative)
    print(f"Base: {base}")
    print(f"Relative: {relative}")
    print(f"Absolute: {absolute}")