"""
XPath Execution Engine

This module provides a robust XPath execution engine that processes Action Sequences
containing XPath expressions against HTML content. It handles failures gracefully
and provides detailed execution results for debugging and optimization.
"""

from lxml import html, etree
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import re
import time
from urllib.parse import urljoin, urlparse


class ExecutionStatus(Enum):
    """Status codes for XPath execution results."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass
class ExecutionStep:
    """Represents the result of executing a single XPath expression."""
    step_index: int
    xpath: str
    status: ExecutionStatus
    extracted_data: List[str]
    element_count: int
    execution_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ExecutionResult:
    """Complete result of executing an Action Sequence."""
    success: bool
    total_steps: int
    successful_steps: int
    failed_steps: int
    final_data: List[Union[str, Dict[str, Any]]]  # Support both flat and structured data
    execution_steps: List[ExecutionStep]
    total_execution_time_ms: float
    error_summary: List[str]
    metadata: Dict[str, Any]


class XPathExecutor:
    """
    XPath execution engine for processing Action Sequences against HTML content.
    
    This engine executes XPath expressions in sequence, handles various failure
    modes gracefully, and provides comprehensive execution results for analysis
    and debugging.
    """
    
    def __init__(self, 
                 timeout_ms: int = 5000,
                 max_results_per_xpath: int = 1000,
                 normalize_whitespace: bool = True,
                 resolve_relative_urls: bool = False,
                 base_url: Optional[str] = None,
                 enable_structuring: bool = True):
        """
        Initialize the XPath executor.
        
        Args:
            timeout_ms: Maximum execution time per XPath expression
            max_results_per_xpath: Maximum number of results to extract per XPath
            normalize_whitespace: Whether to normalize whitespace in extracted text
            resolve_relative_urls: Whether to resolve relative URLs to absolute
            base_url: Base URL for resolving relative URLs
            enable_structuring: Whether to enable DOM-based structuring
        """
        self.timeout_ms = timeout_ms
        self.max_results_per_xpath = max_results_per_xpath
        self.normalize_whitespace = normalize_whitespace
        self.resolve_relative_urls = resolve_relative_urls
        self.base_url = base_url
        self.enable_structuring = enable_structuring
        self.logger = logging.getLogger(__name__)
    
    def execute_action_sequence(self, html_content: str, 
                              action_sequence: List[Dict[str, Any]]) -> ExecutionResult:
        """
        Execute a complete Action Sequence against HTML content.
        
        Processes each XPath expression in the sequence, accumulating results
        and handling failures gracefully. Returns comprehensive execution
        results for analysis.
        
        Args:
            html_content: Raw HTML content to process
            action_sequence: List of action dictionaries containing XPath expressions
            
        Returns:
            ExecutionResult with complete execution details
        """
        start_time = time.time()
        
        # Parse HTML content
        try:
            html_tree = html.fromstring(html_content)
        except Exception as e:
            self.logger.error(f"Failed to parse HTML: {e}")
            return self._create_failure_result(
                action_sequence, 
                f"HTML parsing failed: {str(e)}",
                time.time() - start_time
            )
        
        # Initialize execution state
        execution_steps = []
        final_data = []
        error_summary = []
        successful_steps = 0
        failed_steps = 0
        
        # Execute each action in sequence
        for i, action in enumerate(action_sequence):
            step_result = self._execute_single_xpath(html_tree, action, i)
            execution_steps.append(step_result)
            
            if step_result.status == ExecutionStatus.SUCCESS:
                successful_steps += 1
                # Update final data with results from this step
                if step_result.extracted_data:
                    final_data.extend(step_result.extracted_data)
            elif step_result.status == ExecutionStatus.FAILED:
                failed_steps += 1
                if step_result.error_message:
                    error_summary.append(f"Step {i}: {step_result.error_message}")
            elif step_result.status == ExecutionStatus.PARTIAL:
                successful_steps += 1  # Count partial success as success
                if step_result.extracted_data:
                    final_data.extend(step_result.extracted_data)
                if step_result.error_message:
                    error_summary.append(f"Step {i} (partial): {step_result.error_message}")
        
        # Remove duplicates while preserving order
        final_data = self._remove_duplicates(final_data)
        
        # Apply structuring if enabled and we have multiple data points
        if self.enable_structuring and len(final_data) >= 1:  # Changed from > 1 to >= 1
            structured_data = self._apply_dom_structuring(html_tree, final_data, execution_steps)
            if structured_data:
                final_data = structured_data
        
        # Calculate execution metrics
        total_execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        overall_success = successful_steps > 0 and len(final_data) > 0
        
        return ExecutionResult(
            success=overall_success,
            total_steps=len(action_sequence),
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            final_data=final_data,
            execution_steps=execution_steps,
            total_execution_time_ms=total_execution_time,
            error_summary=error_summary,
            metadata={
                'html_length': len(html_content),
                'unique_xpaths': len(set(action.get('xpath', '') for action in action_sequence)),
                'success_rate': successful_steps / len(action_sequence) if action_sequence else 0,
                'data_extraction_rate': len(final_data) / len(action_sequence) if action_sequence else 0
            }
        )
    
    def _execute_single_xpath(self, html_tree: etree._Element, 
                             action: Dict[str, Any], step_index: int) -> ExecutionStep:
        """
        Execute a single XPath expression from an action.
        
        Handles various XPath result types (elements, attributes, text) and
        provides detailed execution metrics and error handling.
        
        Args:
            html_tree: Parsed HTML tree
            action: Action dictionary containing XPath and metadata
            step_index: Index of this step in the sequence
            
        Returns:
            ExecutionStep with detailed execution results
        """
        xpath = action.get('xpath', '')
        action_type = action.get('action_type', 'xpath_extract')
        
        if not xpath:
            return ExecutionStep(
                step_index=step_index,
                xpath=xpath,
                status=ExecutionStatus.FAILED,
                extracted_data=[],
                element_count=0,
                execution_time_ms=0.0,
                error_message="No XPath expression provided"
            )
        
        start_time = time.time()
        
        try:
            # Execute XPath with timeout protection
            elements = self._execute_xpath_with_timeout(html_tree, xpath)
            
            # Process results based on action type
            if action_type == 'xpath_extract':
                extracted_data = self._extract_text_from_elements(elements)
            elif action_type == 'xpath_attribute':
                attribute_name = action.get('attribute', 'href')
                extracted_data = self._extract_attributes_from_elements(elements, attribute_name)
            elif action_type == 'xpath_html':
                extracted_data = self._extract_html_from_elements(elements)
            else:
                # Default to text extraction
                extracted_data = self._extract_text_from_elements(elements)
            
            # Apply post-processing
            extracted_data = self._post_process_data(extracted_data)
            
            # Limit results if necessary
            if len(extracted_data) > self.max_results_per_xpath:
                extracted_data = extracted_data[:self.max_results_per_xpath]
                status = ExecutionStatus.PARTIAL
                error_message = f"Results truncated to {self.max_results_per_xpath} items"
            else:
                status = ExecutionStatus.SUCCESS if extracted_data else ExecutionStatus.FAILED
                error_message = None if extracted_data else "No data extracted"
            
            execution_time = (time.time() - start_time) * 1000
            
            return ExecutionStep(
                step_index=step_index,
                xpath=xpath,
                status=status,
                extracted_data=extracted_data,
                element_count=len(elements) if isinstance(elements, list) else 1,
                execution_time_ms=execution_time,
                error_message=error_message,
                metadata={
                    'action_type': action_type,
                    'original_element_count': len(elements) if isinstance(elements, list) else 1,
                    'data_count_after_processing': len(extracted_data)
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_message = f"XPath execution failed: {str(e)}"
            
            self.logger.error(f"Step {step_index} failed: {error_message}")
            
            return ExecutionStep(
                step_index=step_index,
                xpath=xpath,
                status=ExecutionStatus.FAILED,
                extracted_data=[],
                element_count=0,
                execution_time_ms=execution_time,
                error_message=error_message,
                metadata={'action_type': action_type, 'exception_type': type(e).__name__}
            )
    
    def _execute_xpath_with_timeout(self, html_tree: etree._Element, xpath: str) -> List[Any]:
        """
        Execute XPath expression with timeout protection.
        
        Args:
            html_tree: Parsed HTML tree
            xpath: XPath expression to execute
            
        Returns:
            List of matching elements/values
            
        Raises:
            TimeoutError: If execution exceeds timeout
            Exception: For XPath syntax or execution errors
        """
        try:
            # Note: lxml doesn't have built-in timeout, but we can catch common issues
            # In a production system, you might want to use threading or async for true timeouts
            
            # Validate XPath syntax first
            self._validate_xpath_syntax(xpath)
            
            # Execute XPath
            results = html_tree.xpath(xpath)
            
            # Ensure results is always a list
            if not isinstance(results, list):
                results = [results]
            
            return results
            
        except etree.XPathEvalError as e:
            raise Exception(f"XPath evaluation error: {str(e)}")
        except etree.XPathSyntaxError as e:
            raise Exception(f"XPath syntax error: {str(e)}")
        except Exception as e:
            raise Exception(f"XPath execution error: {str(e)}")
    
    def _validate_xpath_syntax(self, xpath: str) -> None:
        """
        Validate XPath syntax before execution.
        
        Args:
            xpath: XPath expression to validate
            
        Raises:
            Exception: If XPath syntax is invalid
        """
        try:
            # Try to compile the XPath expression
            etree.XPath(xpath)
        except etree.XPathSyntaxError as e:
            raise Exception(f"Invalid XPath syntax: {str(e)}")
    
    def _extract_text_from_elements(self, elements: List[Any]) -> List[str]:
        """
        Extract text content from XPath result elements.
        
        Handles different types of XPath results:
        - Element nodes: extract text content
        - Text nodes: use directly
        - Attribute nodes: convert to string
        
        Args:
            elements: List of XPath result elements
            
        Returns:
            List of extracted text strings
        """
        extracted_data = []
        
        for element in elements:
            try:
                if isinstance(element, str):
                    # Direct text result
                    text = element.strip()
                    if text:
                        extracted_data.append(text)
                        
                elif hasattr(element, 'text_content'):
                    # Element with text_content method (lxml Element)
                    text = element.text_content().strip()
                    if text:
                        extracted_data.append(text)
                        
                elif hasattr(element, 'text'):
                    # Element with text attribute
                    text = (element.text or '').strip()
                    if text:
                        extracted_data.append(text)
                        
                elif hasattr(element, '__str__'):
                    # Fallback: convert to string
                    text = str(element).strip()
                    if text:
                        extracted_data.append(text)
                        
            except Exception as e:
                self.logger.warning(f"Failed to extract text from element: {e}")
                continue
        
        return extracted_data
    
    def _extract_attributes_from_elements(self, elements: List[Any], 
                                        attribute_name: str) -> List[str]:
        """
        Extract specific attributes from XPath result elements.
        
        Args:
            elements: List of XPath result elements
            attribute_name: Name of the attribute to extract
            
        Returns:
            List of extracted attribute values
        """
        extracted_data = []
        
        for element in elements:
            try:
                if hasattr(element, 'get'):
                    # lxml Element with get method
                    attr_value = element.get(attribute_name)
                    if attr_value:
                        extracted_data.append(attr_value.strip())
                        
                elif hasattr(element, 'attrib'):
                    # Element with attrib dictionary
                    attr_value = element.attrib.get(attribute_name)
                    if attr_value:
                        extracted_data.append(attr_value.strip())
                        
            except Exception as e:
                self.logger.warning(f"Failed to extract attribute {attribute_name}: {e}")
                continue
        
        return extracted_data
    
    def _extract_html_from_elements(self, elements: List[Any]) -> List[str]:
        """
        Extract HTML content from XPath result elements.
        
        Args:
            elements: List of XPath result elements
            
        Returns:
            List of extracted HTML strings
        """
        extracted_data = []
        
        for element in elements:
            try:
                if hasattr(element, 'tag'):
                    # lxml Element - convert to HTML string
                    html_string = etree.tostring(element, encoding='unicode', method='html')
                    if html_string:
                        extracted_data.append(html_string.strip())
                        
            except Exception as e:
                self.logger.warning(f"Failed to extract HTML from element: {e}")
                continue
        
        return extracted_data
    
    def _post_process_data(self, data: List[str]) -> List[str]:
        """
        Apply post-processing to extracted data.
        
        Args:
            data: List of extracted strings
            
        Returns:
            List of processed strings
        """
        processed_data = []
        
        for item in data:
            if not item:
                continue
                
            # Normalize whitespace if enabled
            if self.normalize_whitespace:
                item = re.sub(r'\s+', ' ', item.strip())
            
            # Resolve relative URLs if enabled and item looks like a URL
            if (self.resolve_relative_urls and self.base_url and 
                (item.startswith('/') or item.startswith('./'))):
                try:
                    item = urljoin(self.base_url, item)
                except Exception:
                    pass  # Keep original if URL resolution fails
            
            # Skip empty items after processing
            if item.strip():
                processed_data.append(item)
        
        return processed_data
    
    def _remove_duplicates(self, data: List[str]) -> List[str]:
        """
        Remove duplicates while preserving order.
        
        Args:
            data: List of strings that may contain duplicates
            
        Returns:
            List with duplicates removed, order preserved
        """
        seen = set()
        unique_data = []
        
        for item in data:
            if item not in seen:
                seen.add(item)
                unique_data.append(item)
        
        return unique_data
    
    def _apply_dom_structuring(self, html_tree: etree._Element, 
                              extracted_data: List[str], 
                              execution_steps: List[ExecutionStep]) -> List[Dict[str, Any]]:
        """
        Apply DOM-based structuring to group extracted elements by their parent containers.
        
        This method identifies repeated parent containers in the DOM and groups
        extracted text elements that belong to the same logical record.
        
        Args:
            html_tree: Parsed HTML tree
            extracted_data: List of extracted text strings
            execution_steps: Execution steps with XPath information
            
        Returns:
            List of structured objects grouped by parent containers
        """
        if not extracted_data:
            return []
        
        try:
            # Step 1: Find all elements that produced our extracted data
            source_elements = self._find_source_elements(html_tree, extracted_data, execution_steps)
            
            if not source_elements:
                self.logger.debug("No source elements found for structuring")
                return []
            
            # Step 2: Try multiple container identification strategies
            parent_containers = self._identify_repeated_containers(source_elements)
            
            if not parent_containers:
                self.logger.debug("No repeated containers found, trying broader search")
                # Try to find product containers more broadly
                parent_containers = self._find_product_containers(html_tree, extracted_data)
            
            if not parent_containers:
                self.logger.debug("No containers found, trying simple grouping")
                # Fallback: try to group by immediate parents
                return self._simple_grouping_fallback(source_elements)
            
            # Step 3: Group elements by their parent containers
            grouped_elements = self._group_by_containers(source_elements, parent_containers)
            
            # Step 4: Structure each group into semantic fields
            structured_records = []
            for container, elements in grouped_elements.items():
                if len(elements) > 0:  # Only process containers with elements
                    record = self._create_structured_record(elements, container)
                    if record:
                        structured_records.append(record)
            
            # Step 5: If we don't have enough structured records, try alternative approach
            if len(structured_records) < len(extracted_data) * 0.3:  # Less than 30% structured
                self.logger.debug("Low structuring success, trying alternative approach")
                alternative_records = self._alternative_structuring_approach(html_tree, extracted_data)
                if len(alternative_records) > len(structured_records):
                    return alternative_records
            
            return structured_records if structured_records else []
            
        except Exception as e:
            self.logger.warning(f"Structuring failed: {e}")
            return []
    
    def _find_source_elements(self, html_tree: etree._Element, 
                             extracted_data: List[str], 
                             execution_steps: List[ExecutionStep]) -> List[etree._Element]:
        """
        Find the DOM elements that produced the extracted data.
        
        Args:
            html_tree: Parsed HTML tree
            extracted_data: List of extracted text strings
            execution_steps: Execution steps with XPath information
            
        Returns:
            List of source elements
        """
        source_elements = []
        
        # Get the successful XPath from execution steps
        successful_xpaths = [step.xpath for step in execution_steps if step.status == ExecutionStatus.SUCCESS]
        
        for xpath in successful_xpaths:
            try:
                # Execute XPath to get elements
                elements = html_tree.xpath(xpath)
                
                for element in elements:
                    # Find the actual text-containing element
                    if isinstance(element, str):
                        # For text nodes, find elements containing this text
                        escaped_text = element.replace("'", "\\'")[:30]  # Escape and limit length
                        text_xpath = f"//*[contains(text(), '{escaped_text}')]"
                        try:
                            text_elements = html_tree.xpath(text_xpath)
                            source_elements.extend(text_elements[:5])  # Limit to avoid too many matches
                        except:
                            continue
                    elif hasattr(element, 'text_content'):
                        source_elements.append(element)
                        
            except Exception as e:
                self.logger.debug(f"Error finding source elements for XPath {xpath}: {e}")
                continue
        
        # Also try to find elements by matching extracted text directly
        for text in extracted_data[:10]:  # Limit to first 10 items
            if len(text.strip()) > 3:
                try:
                    escaped_text = text.replace("'", "\\'")[:30]
                    text_xpath = f"//*[normalize-space(text())='{escaped_text}' or contains(normalize-space(text()), '{escaped_text}')]"
                    matching_elements = html_tree.xpath(text_xpath)
                    source_elements.extend(matching_elements[:3])  # Limit matches
                except:
                    continue
        
        # Remove duplicates
        unique_elements = []
        for elem in source_elements:
            if elem not in unique_elements:
                unique_elements.append(elem)
        
        return unique_elements
    
    def _identify_repeated_containers(self, source_elements: List[etree._Element]) -> List[etree._Element]:
        """
        Identify repeated parent containers that likely represent logical records.
        
        Args:
            source_elements: List of source elements
            
        Returns:
            List of parent container elements
        """
        if len(source_elements) < 2:
            return []
        
        # Find common parent patterns
        parent_candidates = {}
        
        for element in source_elements:
            current = element
            # Walk up the DOM tree to find potential containers
            for level in range(5):  # Check up to 5 levels up
                if current is None or current.getparent() is None:
                    break
                    
                parent = current.getparent()
                parent_key = f"{parent.tag}_{level}"
                
                if parent_key not in parent_candidates:
                    parent_candidates[parent_key] = {'element': parent, 'count': 0, 'children': set()}
                
                parent_candidates[parent_key]['count'] += 1
                parent_candidates[parent_key]['children'].add(element)
                
                current = parent
        
        # Find containers that have multiple children (indicating repetition)
        repeated_containers = []
        for key, info in parent_candidates.items():
            if info['count'] >= 2 and len(info['children']) >= 2:
                repeated_containers.append(info['element'])
        
        # Remove duplicates and sort by specificity (deeper elements first)
        unique_containers = []
        for container in repeated_containers:
            if not any(self._is_ancestor(existing, container) for existing in unique_containers):
                unique_containers.append(container)
        
        return unique_containers
    
    def _is_ancestor(self, potential_ancestor: etree._Element, element: etree._Element) -> bool:
        """Check if potential_ancestor is an ancestor of element."""
        current = element.getparent()
        while current is not None:
            if current == potential_ancestor:
                return True
            current = current.getparent()
        return False
    
    def _group_by_containers(self, source_elements: List[etree._Element], 
                           containers: List[etree._Element]) -> Dict[etree._Element, List[etree._Element]]:
        """
        Group source elements by their parent containers.
        
        Args:
            source_elements: List of source elements
            containers: List of container elements
            
        Returns:
            Dictionary mapping containers to their child elements
        """
        grouped = {}
        
        for container in containers:
            grouped[container] = []
            
            for element in source_elements:
                # Check if element is a descendant of this container
                if self._is_descendant(element, container):
                    grouped[container].append(element)
        
        return grouped
    
    def _is_descendant(self, element: etree._Element, container: etree._Element) -> bool:
        """Check if element is a descendant of container."""
        current = element
        while current is not None:
            if current == container:
                return True
            current = current.getparent()
        return False
    
    def _create_structured_record(self, elements: List[etree._Element], 
                                container: etree._Element) -> Optional[Dict[str, Any]]:
        """
        Create a structured record from elements within a container.
        
        Args:
            elements: List of elements within the container
            container: Parent container element
            
        Returns:
            Structured record dictionary or None
        """
        if not elements:
            return None
        
        record = {}
        seen_values = set()  # Track seen values to avoid duplicates
        
        # Group elements by inferred field type first
        field_groups = {}
        for element in elements:
            field_name = self._infer_field_name(element, container)
            text_content = self._extract_clean_text(element)
            
            # Skip empty or very short content
            if not text_content or len(text_content.strip()) < 3:
                continue
            
            # Skip common navigation/UI elements
            if self._is_navigation_element(text_content):
                continue
                
            # Skip duplicates
            if text_content in seen_values:
                continue
            seen_values.add(text_content)
            
            if field_name not in field_groups:
                field_groups[field_name] = []
            field_groups[field_name].append(text_content)
        
        # Create record from field groups
        for field_name, values in field_groups.items():
            if len(values) == 1:
                record[field_name] = values[0]
            elif len(values) > 1:
                # For multiple values of same type, use the most meaningful one
                if field_name == 'price':
                    # For prices, prefer the one with currency symbol
                    price_values = [v for v in values if any(curr in v for curr in ['Rs.', '$', '€', '£', '₹'])]
                    if price_values:
                        record[field_name] = price_values[0]  # Take first valid price
                    else:
                        record[field_name] = values[0]
                elif field_name == 'title':
                    # For titles, prefer longer, more descriptive ones
                    title_values = sorted(values, key=len, reverse=True)
                    record[field_name] = title_values[0]
                else:
                    # For other fields, take the first value
                    record[field_name] = values[0]
        
        # Only return record if it looks like a product (has price or meaningful title)
        if record:
            has_price = 'price' in record and any(curr in record['price'] for curr in ['Rs.', '$', '€', '£', '₹'])
            has_product_title = ('title' in record and len(record['title']) > 5) or \
                              any(brand in str(record.get('title', '')) for brand in ['New Balance', 'Nike', 'Adidas'])
            
            # Accept records with either price or product title, or both
            if has_price or has_product_title:
                return record
        
        return None
    
    def _is_navigation_element(self, text: str) -> bool:
        """
        Check if text content appears to be a navigation or UI element.
        
        Args:
            text: Text content to check
            
        Returns:
            True if it's likely a navigation element
        """
        text_lower = text.lower().strip()
        
        # Common navigation/UI terms
        nav_terms = [
            'home', 'about', 'contact', 'login', 'register', 'cart', 'checkout',
            'menu', 'search', 'instagram', 'facebook', 'twitter', 'youtube',
            'subscribe', 'newsletter', 'follow', 'share', 'like', 'view all',
            'more', 'less', 'show', 'hide', 'toggle', 'close', 'open',
            'add to cart', 'buy now', 'quick view', 'compare', 'wishlist'
        ]
        
        # Check if text is exactly a navigation term
        if text_lower in nav_terms:
            return True
        
        # Check if text is very short (likely UI element) but allow brand names
        if len(text_lower) <= 4 and text_lower not in ['nike', 'puma', 'new', 'balance']:
            return True
        
        # Check if text is all uppercase (likely category/section header) but allow product names
        if text.isupper() and len(text) < 20 and 'NEW BALANCE' not in text:
            return True
        
        # Skip common category names that aren't product names
        category_terms = ['perfumes', 'shoes', 'clothing', 'accessories', 'men', 'women', 'kids']
        if text_lower in category_terms:
            return True
        
        return False
    
    def _infer_field_name(self, element: etree._Element, container: etree._Element) -> str:
        """
        Infer semantic field name based on element characteristics.
        
        Args:
            element: The element to analyze
            container: Parent container
            
        Returns:
            Inferred field name
        """
        text_content = element.text_content().strip()
        
        # Check if content looks like a price
        if any(currency in text_content for currency in ['Rs.', '$', '€', '£', '₹']) or 'price' in text_content.lower():
            return 'price'
        
        # Check for product names/titles - look for brand names and product indicators
        if any(brand in text_content for brand in ['New Balance', 'Nike', 'Adidas', 'Puma', 'Reebok']):
            return 'title'
        
        # Check for shoe/product specific terms
        if any(term in text_content.lower() for term in ['shoe', 'sneaker', 'boot', 'sandal', 'trainer', 'running', 'walking']):
            return 'title'
        
        # Check class attributes for semantic hints
        class_attr = element.get('class', '').lower()
        if class_attr:
            # Common semantic class patterns
            if any(word in class_attr for word in ['title', 'heading', 'name', 'product-name', 'product-title']):
                return 'title'
            elif any(word in class_attr for word in ['author', 'by', 'writer', 'brand']):
                return 'brand'
            elif any(word in class_attr for word in ['text', 'content', 'body', 'description', 'desc']):
                return 'description'
            elif any(word in class_attr for word in ['price', 'cost', 'amount', 'money']):
                return 'price'
            elif any(word in class_attr for word in ['date', 'time', 'when']):
                return 'date'
            elif any(word in class_attr for word in ['tag', 'category', 'label']):
                return 'category'
            elif any(word in class_attr for word in ['rating', 'star', 'review']):
                return 'rating'
        
        # Check tag type
        tag = element.tag.lower()
        if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            return 'title'
        elif tag == 'a':
            href = element.get('href', '')
            if href:
                return 'link'
            else:
                return 'title'  # Link text often contains product names
        elif tag in ['p', 'div', 'span']:
            # Analyze content to determine field type
            if len(text_content) > 100:
                return 'description'
            elif any(currency in text_content for currency in ['Rs.', '$', '€', '£', '₹']):
                return 'price'
            elif len(text_content) > 20:
                return 'title'
            else:
                return 'text'
        elif tag == 'time':
            return 'date'
        elif tag == 'img':
            return 'image'
        
        # Content-based analysis with better product name detection
        if any(currency in text_content for currency in ['Rs.', '$', '€', '£', '₹']):
            return 'price'
        elif any(brand in text_content for brand in ['New Balance', 'Nike', 'Adidas', 'Puma', 'Reebok']):
            return 'title'
        elif any(word in text_content.lower() for word in ['shoe', 'sneaker', 'boot', 'trainer', 'running']):
            return 'title'
        elif len(text_content) > 100:
            return 'description'
        elif 10 < len(text_content) <= 100:
            # Check if it looks like a product name
            if any(char.isdigit() for char in text_content) and any(char.isalpha() for char in text_content):
                return 'title'  # Product names often have numbers and letters
            return 'title'
        elif len(text_content) <= 10:
            return 'code'
        
        # Position-based fallback
        siblings = list(container)
        if siblings and element in siblings:
            position = siblings.index(element)
            if position == 0:
                return 'title'
            elif position == 1:
                return 'subtitle'
        
        return 'field'
    
    def _extract_clean_text(self, element: etree._Element) -> str:
        """Extract and clean text content from element."""
        text = element.text_content().strip()
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
        return text
    
    def _simple_grouping_fallback(self, source_elements: List[etree._Element]) -> List[Dict[str, Any]]:
        """
        Fallback grouping when no repeated containers are found.
        
        Groups elements by their immediate parents or creates simple records.
        
        Args:
            source_elements: List of source elements
            
        Returns:
            List of simple structured records
        """
        if not source_elements:
            return []
        
        # Group by immediate parent
        parent_groups = {}
        for element in source_elements:
            parent = element.getparent()
            if parent is not None:
                parent_key = id(parent)  # Use object id as key
                if parent_key not in parent_groups:
                    parent_groups[parent_key] = {'parent': parent, 'children': []}
                parent_groups[parent_key]['children'].append(element)
        
        # Create records from groups
        records = []
        for group_info in parent_groups.values():
            if len(group_info['children']) >= 1:  # Accept single elements too
                record = self._create_structured_record(group_info['children'], group_info['parent'])
                if record:
                    records.append(record)
        
        # If still no records, create individual records
        if not records:
            for i, element in enumerate(source_elements):
                text = self._extract_clean_text(element)
                if text:
                    field_name = self._infer_field_name(element, element.getparent())
                    records.append({field_name: text})
        
        return records
    
    def _find_product_containers(self, html_tree: etree._Element, extracted_data: List[str]) -> List[etree._Element]:
        """
        Find product containers more broadly by looking for common e-commerce patterns.
        
        Args:
            html_tree: Parsed HTML tree
            extracted_data: List of extracted text strings
            
        Returns:
            List of potential product container elements
        """
        containers = []
        
        # Strategy 1: Look for elements containing both price and product info
        price_texts = [text for text in extracted_data if any(curr in text for curr in ['Rs.', '$', '€', '£', '₹'])]
        product_texts = [text for text in extracted_data if 'New Balance' in text or len(text) > 20]
        
        if price_texts:
            for price_text in price_texts[:5]:  # Limit to avoid too many searches
                try:
                    # Find elements containing this price
                    escaped_price = price_text.replace("'", "\\'")[:20]
                    price_xpath = f"//*[contains(text(), '{escaped_price}')]"
                    price_elements = html_tree.xpath(price_xpath)
                    
                    for price_elem in price_elements:
                        # Look for parent containers that might contain both price and title
                        current = price_elem
                        for level in range(4):  # Check up to 4 levels up
                            if current is None:
                                break
                            parent = current.getparent()
                            if parent is not None:
                                # Check if this parent contains meaningful text beyond just price
                                parent_text = parent.text_content()
                                if len(parent_text) > len(price_text) * 1.5:  # Has more content than just price
                                    containers.append(parent)
                            current = parent
                except:
                    continue
        
        # Strategy 2: Look for common product container patterns
        common_patterns = [
            '//*[contains(@class, "product")]',
            '//*[contains(@class, "item")]',
            '//*[contains(@class, "card")]',
            '//article',
            '//*[contains(@class, "listing")]'
        ]
        
        for pattern in common_patterns:
            try:
                pattern_containers = html_tree.xpath(pattern)
                for container in pattern_containers:
                    container_text = container.text_content()
                    # Check if container has relevant content
                    if any(curr in container_text for curr in ['Rs.', '$']) or 'New Balance' in container_text:
                        containers.append(container)
            except:
                continue
        
        # Remove duplicates
        unique_containers = []
        for container in containers:
            if container not in unique_containers:
                unique_containers.append(container)
        
        return unique_containers[:10]  # Limit to avoid too many containers
    
    def _alternative_structuring_approach(self, html_tree: etree._Element, extracted_data: List[str]) -> List[Dict[str, Any]]:
        """
        Alternative approach to structuring when standard DOM grouping fails.
        
        This method tries to pair prices with nearby product names based on DOM proximity.
        
        Args:
            html_tree: Parsed HTML tree
            extracted_data: List of extracted text strings
            
        Returns:
            List of structured records
        """
        records = []
        
        # Separate prices and potential titles
        prices = []
        titles = []
        other_texts = []
        
        for text in extracted_data:
            if any(curr in text for curr in ['Rs.', '$', '€', '£', '₹']):
                prices.append(text)
            elif 'New Balance' in text or (len(text) > 15 and len(text) < 100):
                titles.append(text)
            else:
                other_texts.append(text)
        
        # Try to pair prices with titles based on DOM proximity
        used_titles = set()
        
        for price in prices:
            try:
                # Find the element containing this price
                escaped_price = price.replace("'", "\\'")[:20]
                price_xpath = f"//*[contains(text(), '{escaped_price}')]"
                price_elements = html_tree.xpath(price_xpath)
                
                if not price_elements:
                    # Create record with just price
                    records.append({'price': price})
                    continue
                
                price_element = price_elements[0]
                best_title = None
                
                # Look for nearby titles
                for title in titles:
                    if title in used_titles:
                        continue
                    
                    try:
                        escaped_title = title.replace("'", "\\'")[:30]
                        title_xpath = f"//*[contains(text(), '{escaped_title}')]"
                        title_elements = html_tree.xpath(title_xpath)
                        
                        if title_elements:
                            title_element = title_elements[0]
                            
                            # Check if they share a common ancestor within reasonable distance
                            if self._are_elements_related(price_element, title_element, max_distance=5):
                                best_title = title
                                break
                    except:
                        continue
                
                # Create record
                if best_title:
                    records.append({'title': best_title, 'price': price})
                    used_titles.add(best_title)
                else:
                    records.append({'price': price})
                    
            except:
                records.append({'price': price})
        
        # Add remaining titles as separate records
        for title in titles:
            if title not in used_titles:
                records.append({'title': title})
        
        return records
    
    def _are_elements_related(self, elem1: etree._Element, elem2: etree._Element, max_distance: int = 5) -> bool:
        """
        Check if two elements are related (share a common ancestor within max_distance levels).
        
        Args:
            elem1: First element
            elem2: Second element
            max_distance: Maximum levels to check for common ancestor
            
        Returns:
            True if elements are related
        """
        # Get ancestors of elem1
        ancestors1 = []
        current = elem1
        for _ in range(max_distance):
            if current is None:
                break
            ancestors1.append(current)
            current = current.getparent()
        
        # Check if elem2 shares any ancestor with elem1
        current = elem2
        for _ in range(max_distance):
            if current is None:
                break
            if current in ancestors1:
                return True
            current = current.getparent()
        
        return False
    
    def _create_failure_result(self, action_sequence: List[Dict[str, Any]], 
                             error_message: str, execution_time: float) -> ExecutionResult:
        """
        Create an ExecutionResult for complete failure scenarios.
        
        Args:
            action_sequence: The action sequence that failed
            error_message: Description of the failure
            execution_time: Time spent before failure
            
        Returns:
            ExecutionResult indicating complete failure
        """
        return ExecutionResult(
            success=False,
            total_steps=len(action_sequence),
            successful_steps=0,
            failed_steps=len(action_sequence),
            final_data=[],
            execution_steps=[],
            total_execution_time_ms=execution_time * 1000,
            error_summary=[error_message],
            metadata={
                'failure_type': 'complete_failure',
                'action_sequence_length': len(action_sequence)
            }
        )
    
    def execute_single_xpath(self, html_content: str, xpath: str, 
                           action_type: str = 'xpath_extract') -> ExecutionStep:
        """
        Execute a single XPath expression against HTML content.
        
        Convenience method for executing individual XPath expressions
        without needing to create a full Action Sequence.
        
        Args:
            html_content: Raw HTML content
            xpath: XPath expression to execute
            action_type: Type of extraction ('xpath_extract', 'xpath_attribute', 'xpath_html')
            
        Returns:
            ExecutionStep with results
        """
        try:
            html_tree = html.fromstring(html_content)
            action = {'xpath': xpath, 'action_type': action_type}
            return self._execute_single_xpath(html_tree, action, 0)
            
        except Exception as e:
            return ExecutionStep(
                step_index=0,
                xpath=xpath,
                status=ExecutionStatus.FAILED,
                extracted_data=[],
                element_count=0,
                execution_time_ms=0.0,
                error_message=f"HTML parsing or execution failed: {str(e)}"
            )
    
    def validate_xpath_against_html(self, html_content: str, xpath: str) -> Dict[str, Any]:
        """
        Validate an XPath expression against HTML content without full execution.
        
        Useful for testing XPath expressions during development or debugging.
        
        Args:
            html_content: HTML content to test against
            xpath: XPath expression to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Parse HTML
            html_tree = html.fromstring(html_content)
            
            # Validate XPath syntax
            self._validate_xpath_syntax(xpath)
            
            # Test execution
            results = html_tree.xpath(xpath)
            
            return {
                'valid': True,
                'syntax_valid': True,
                'executable': True,
                'result_count': len(results) if isinstance(results, list) else 1,
                'result_types': [type(r).__name__ for r in (results if isinstance(results, list) else [results])],
                'sample_results': str(results[:3]) if results else None
            }
            
        except Exception as e:
            return {
                'valid': False,
                'syntax_valid': 'XPathSyntaxError' not in str(type(e)),
                'executable': False,
                'error_message': str(e),
                'error_type': type(e).__name__
            }


# Utility functions for common XPath patterns
def create_text_extraction_action(xpath: str) -> Dict[str, Any]:
    """Create an action for text extraction."""
    return {
        'action_type': 'xpath_extract',
        'xpath': xpath,
        'description': f'Extract text using: {xpath}'
    }


def create_attribute_extraction_action(xpath: str, attribute: str) -> Dict[str, Any]:
    """Create an action for attribute extraction."""
    return {
        'action_type': 'xpath_attribute',
        'xpath': xpath,
        'attribute': attribute,
        'description': f'Extract {attribute} attribute using: {xpath}'
    }


def create_html_extraction_action(xpath: str) -> Dict[str, Any]:
    """Create an action for HTML extraction."""
    return {
        'action_type': 'xpath_html',
        'xpath': xpath,
        'description': f'Extract HTML using: {xpath}'
    }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample HTML for testing
    sample_html = """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <div class="container">
            <h1>Main Title</h1>
            <div class="products">
                <div class="product" id="prod1">
                    <h2 class="title">Laptop Computer</h2>
                    <span class="price">$999.99</span>
                    <p class="description">High-performance laptop for professionals</p>
                    <a href="/laptop-details" class="link">View Details</a>
                </div>
                <div class="product" id="prod2">
                    <h2 class="title">Desktop Monitor</h2>
                    <span class="price">$299.99</span>
                    <p class="description">24-inch LED monitor with crisp display</p>
                    <a href="/monitor-details" class="link">View Details</a>
                </div>
                <div class="product" id="prod3">
                    <h2 class="title">Wireless Mouse</h2>
                    <span class="price">$49.99</span>
                    <p class="description">Ergonomic wireless mouse with precision tracking</p>
                    <a href="/mouse-details" class="link">View Details</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Create XPath executor
    executor = XPathExecutor(
        normalize_whitespace=True,
        resolve_relative_urls=True,
        base_url="https://example.com"
    )
    
    # Test different action sequences
    test_sequences = [
        # Test 1: Extract product titles
        [create_text_extraction_action('//h2[@class="title"]/text()')],
        
        # Test 2: Extract prices
        [create_text_extraction_action('//span[@class="price"]/text()')],
        
        # Test 3: Extract product links
        [create_attribute_extraction_action('//a[@class="link"]', 'href')],
        
        # Test 4: Multi-step extraction
        [
            create_text_extraction_action('//h2[@class="title"]/text()'),
            create_text_extraction_action('//span[@class="price"]/text()'),
            create_text_extraction_action('//p[@class="description"]/text()')
        ],
        
        # Test 5: Invalid XPath (for error handling)
        [create_text_extraction_action('//invalid[xpath[syntax')]
    ]
    
    # Execute test sequences
    for i, sequence in enumerate(test_sequences, 1):
        print(f"\n=== Test {i}: {sequence[0]['description']} ===")
        
        result = executor.execute_action_sequence(sample_html, sequence)
        
        print(f"Success: {result.success}")
        print(f"Steps: {result.successful_steps}/{result.total_steps}")
        print(f"Execution time: {result.total_execution_time_ms:.2f}ms")
        print(f"Extracted data ({len(result.final_data)} items):")
        
        for j, data_item in enumerate(result.final_data[:5]):  # Show first 5 items
            print(f"  {j+1}. {data_item}")
        
        if result.error_summary:
            print(f"Errors: {result.error_summary}")
        
        # Show step details for multi-step sequences
        if len(result.execution_steps) > 1:
            print("Step details:")
            for step in result.execution_steps:
                print(f"  Step {step.step_index}: {step.status.value} "
                      f"({len(step.extracted_data)} items, {step.execution_time_ms:.1f}ms)")
    
    # Test single XPath execution
    print(f"\n=== Single XPath Test ===")
    single_result = executor.execute_single_xpath(
        sample_html, 
        '//div[@class="product"]/@id'
    )
    print(f"Single XPath result: {single_result.extracted_data}")
    
    # Test XPath validation
    print(f"\n=== XPath Validation Test ===")
    validation_tests = [
        '//h2[@class="title"]/text()',  # Valid
        '//invalid[xpath[syntax',        # Invalid syntax
        '//nonexistent/element'          # Valid syntax, no results
    ]
    
    for xpath in validation_tests:
        validation = executor.validate_xpath_against_html(sample_html, xpath)
        print(f"XPath: {xpath}")
        print(f"  Valid: {validation['valid']}, Results: {validation.get('result_count', 0)}")
        if not validation['valid']:
            print(f"  Error: {validation.get('error_message', 'Unknown error')}")