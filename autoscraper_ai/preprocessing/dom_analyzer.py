"""
DOM Structure Analysis and Normalization

This module provides advanced DOM analysis capabilities to better understand
HTML structure and identify patterns for more effective scraping.
"""

from lxml import html, etree
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
import re
import logging


class DOMAnalyzer:
    """Advanced DOM structure analyzer for better scraping insights."""
    
    def __init__(self):
        """Initialize DOM analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_structure(self, html_content: str) -> Dict[str, Any]:
        """
        Perform comprehensive DOM structure analysis.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Dictionary with detailed structure analysis
        """
        try:
            tree = html.fromstring(html_content)
        except Exception as e:
            self.logger.error(f"Failed to parse HTML: {e}")
            return {'error': f'HTML parsing failed: {str(e)}'}
        
        analysis = {
            'basic_stats': self._get_basic_stats(tree),
            'element_distribution': self._analyze_element_distribution(tree),
            'class_patterns': self._analyze_class_patterns(tree),
            'content_patterns': self._analyze_content_patterns(tree),
            'repetitive_structures': self._find_repetitive_structures(tree)
        }
        
        return analysis
    
    def _get_basic_stats(self, tree: etree._Element) -> Dict[str, Any]:
        """Get basic DOM statistics."""
        all_elements = tree.xpath('//*')
        
        return {
            'total_elements': len(all_elements),
            'max_depth': self._calculate_max_depth(tree),
            'text_nodes': len(tree.xpath('//text()[normalize-space()]')),
            'elements_with_classes': len(tree.xpath('//*[@class]')),
            'elements_with_ids': len(tree.xpath('//*[@id]')),
            'links': len(tree.xpath('//a[@href]')),
            'images': len(tree.xpath('//img'))
        }
    
    def _calculate_max_depth(self, element: etree._Element, current_depth: int = 0) -> int:
        """Calculate maximum depth of DOM tree."""
        if len(element) == 0:
            return current_depth
        
        return max(self._calculate_max_depth(child, current_depth + 1) for child in element)
    
    def _analyze_element_distribution(self, tree: etree._Element) -> Dict[str, int]:
        """Analyze distribution of HTML elements."""
        all_elements = tree.xpath('//*')
        element_counts = Counter(elem.tag for elem in all_elements)
        
        return dict(element_counts.most_common(15))
    
    def _analyze_class_patterns(self, tree: etree._Element) -> Dict[str, Any]:
        """Analyze CSS class usage patterns."""
        elements_with_classes = tree.xpath('//*[@class]')
        
        all_classes = []
        for elem in elements_with_classes:
            class_attr = elem.get('class', '')
            classes = class_attr.split()
            all_classes.extend(classes)
        
        class_counts = Counter(all_classes)
        semantic_classes = self._find_semantic_classes(all_classes)
        
        return {
            'total_classes': len(set(all_classes)),
            'most_common_classes': dict(class_counts.most_common(10)),
            'semantic_classes': semantic_classes
        }
    
    def _find_semantic_classes(self, classes: List[str]) -> Dict[str, List[str]]:
        """Find semantically meaningful class names."""
        semantic_patterns = {
            'product': r'.*(?:product|item|card|listing).*',
            'price': r'.*(?:price|cost|amount|money).*',
            'title': r'.*(?:title|heading|name|label).*',
            'content': r'.*(?:content|text|body|description).*'
        }
        
        semantic_classes = defaultdict(list)
        
        for class_name in set(classes):
            for category, pattern in semantic_patterns.items():
                if re.match(pattern, class_name, re.IGNORECASE):
                    semantic_classes[category].append(class_name)
        
        return dict(semantic_classes)
    
    def _analyze_content_patterns(self, tree: etree._Element) -> Dict[str, Any]:
        """Analyze text content patterns."""
        text_nodes = tree.xpath('//text()[normalize-space()]')
        texts = [text.strip() for text in text_nodes if text.strip()]
        
        patterns = {
            'prices': self._find_price_patterns(texts),
            'total_text_nodes': len(texts)
        }
        
        return patterns
    
    def _find_price_patterns(self, texts: List[str]) -> List[str]:
        """Find price-like patterns in text."""
        price_pattern = re.compile(r'[\$€£¥₹Rs\.]\s*\d+[.,]?\d*', re.IGNORECASE)
        prices = []
        
        for text in texts:
            matches = price_pattern.findall(text)
            prices.extend(matches)
        
        return list(set(prices))[:10]
    
    def _find_repetitive_structures(self, tree: etree._Element) -> Dict[str, Any]:
        """Find repetitive structures that might contain data."""
        class_counts = Counter()
        for elem in tree.xpath('//*[@class]'):
            class_attr = elem.get('class')
            class_counts[class_attr] += 1
        
        repeated_classes = {cls: count for cls, count in class_counts.items() if count >= 3}
        
        return {
            'repeated_classes': dict(list(repeated_classes.items())[:5])
        }


# Global analyzer instance
dom_analyzer = DOMAnalyzer()


def analyze_dom(html_content: str) -> Dict[str, Any]:
    """Analyze DOM structure using the global analyzer."""
    return dom_analyzer.analyze_structure(html_content)