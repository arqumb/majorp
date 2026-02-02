"""
Phase 1 Progressive Module - AUTOSCRAPER Algorithm Implementation

This module implements Phase 1 of the AUTOSCRAPER algorithm, which progressively
builds XPath extraction rules through iterative refinement and step-back logic.

The algorithm works by:
1. Analyzing the extraction task and HTML structure
2. Generating initial XPath suggestions via LLM
3. Testing XPath expressions and applying step-back logic on failures
4. Building an ordered Action Sequence of successful XPath steps
"""

from lxml import html, etree
from typing import List, Dict, Any, Optional, Tuple
import re
import logging
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Types of actions in the extraction sequence."""
    XPATH_EXTRACT = "xpath_extract"
    STEP_BACK = "step_back"
    REFINE_XPATH = "refine_xpath"


@dataclass
class ExtractionAction:
    """Represents a single action in the extraction sequence."""
    action_type: ActionType
    xpath: str
    description: str
    success: bool
    extracted_data: Optional[List[str]] = None
    confidence_score: float = 0.0


class Phase1Progressive:
    """
    Phase 1 of AUTOSCRAPER algorithm - Progressive XPath generation and refinement.
    
    This class implements the core logic for automatically generating XPath expressions
    through iterative refinement, step-back logic, and progressive building of
    extraction sequences.
    """
    
    def __init__(self, max_iterations: int = 5, min_confidence: float = 0.7):
        """
        Initialize Phase 1 processor.
        
        Args:
            max_iterations: Maximum number of refinement iterations
            min_confidence: Minimum confidence score for accepting XPath
        """
        self.max_iterations = max_iterations
        self.min_confidence = min_confidence
        self.action_sequence: List[ExtractionAction] = []
        self.logger = logging.getLogger(__name__)
    
    def process_extraction_task(self, cleaned_html: str, extraction_task: str) -> Dict[str, Any]:
        """
        Main entry point for Phase 1 processing.
        
        Orchestrates the entire Phase 1 workflow:
        1. Analyze HTML structure and extraction requirements
        2. Generate initial XPath suggestions
        3. Execute progressive refinement with step-back logic
        4. Build final Action Sequence
        
        Args:
            cleaned_html: Preprocessed HTML content
            extraction_task: Natural language description of what to extract
            
        Returns:
            Dictionary containing:
            - 'success': Whether extraction was successful
            - 'action_sequence': Ordered list of extraction actions
            - 'final_xpath': Best performing XPath expression
            - 'extracted_data': Final extracted data
            - 'confidence_score': Overall confidence in the result
        """
        self.logger.info(f"Starting Phase 1 processing for task: {extraction_task}")
        
        # Step 1: Parse HTML and analyze structure
        try:
            html_tree = html.fromstring(cleaned_html)
        except Exception as e:
            self.logger.error(f"Failed to parse HTML: {e}")
            return self._create_failure_result("HTML parsing failed")
        
        # Step 2: Analyze HTML structure for context
        structure_analysis = self._analyze_html_structure(html_tree, extraction_task)
        
        # Step 3: Generate initial XPath suggestion via LLM
        initial_xpath = self._generate_initial_xpath(cleaned_html, extraction_task, structure_analysis)
        
        if not initial_xpath:
            return self._create_failure_result("Failed to generate initial XPath")
        
        # Step 4: Execute progressive refinement process
        final_result = self._execute_progressive_refinement(
            html_tree, initial_xpath, extraction_task, structure_analysis
        )
        
        return final_result
    
    def _analyze_html_structure(self, html_tree: etree._Element, task: str) -> Dict[str, Any]:
        """
        Analyze HTML structure to provide context for XPath generation.
        
        This analysis helps the LLM understand the DOM structure and
        identify potential extraction targets more effectively.
        
        Args:
            html_tree: Parsed HTML tree
            task: Extraction task description
            
        Returns:
            Dictionary with structural analysis results
        """
        analysis = {
            'total_elements': len(html_tree.xpath('//*')),
            'unique_tags': list(set([elem.tag for elem in html_tree.xpath('//*')])),
            'elements_with_classes': len(html_tree.xpath('//*[@class]')),
            'elements_with_ids': len(html_tree.xpath('//*[@id]')),
            'text_nodes_count': len(html_tree.xpath('//text()[normalize-space()]')),
            'max_depth': self._calculate_max_depth(html_tree),
            'common_patterns': self._identify_common_patterns(html_tree, task)
        }
        
        self.logger.info(f"HTML structure analysis: {analysis}")
        return analysis
    
    def _calculate_max_depth(self, element: etree._Element, current_depth: int = 0) -> int:
        """Calculate maximum depth of the HTML tree."""
        if len(element) == 0:
            return current_depth
        
        return max(self._calculate_max_depth(child, current_depth + 1) for child in element)
    
    def _identify_common_patterns(self, html_tree: etree._Element, task: str) -> List[str]:
        """
        Identify common HTML patterns that might be relevant to the extraction task.
        
        Args:
            html_tree: Parsed HTML tree
            task: Extraction task description
            
        Returns:
            List of identified patterns
        """
        patterns = []
        
        # Look for list patterns
        lists = html_tree.xpath('//ul | //ol | //div[count(child::*) > 2]')
        if lists:
            patterns.append("list_structure")
        
        # Look for table patterns
        tables = html_tree.xpath('//table | //div[contains(@class, "table")]')
        if tables:
            patterns.append("table_structure")
        
        # Look for card/item patterns
        cards = html_tree.xpath('//*[contains(@class, "card") or contains(@class, "item")]')
        if cards:
            patterns.append("card_structure")
        
        # Task-specific pattern detection
        task_lower = task.lower()
        if any(word in task_lower for word in ['price', 'cost', 'amount']):
            patterns.append("price_pattern")
        if any(word in task_lower for word in ['title', 'heading', 'name']):
            patterns.append("title_pattern")
        if any(word in task_lower for word in ['link', 'url', 'href']):
            patterns.append("link_pattern")
        
        return patterns
    
    def _generate_initial_xpath(self, html: str, task: str, analysis: Dict[str, Any]) -> Optional[str]:
        """
        Generate initial XPath suggestion using LLM.
        
        This function creates a structured prompt for the LLM that includes:
        - The extraction task
        - HTML structure analysis
        - Context about common patterns
        
        Args:
            html: Cleaned HTML content
            task: Extraction task description
            analysis: HTML structure analysis results
            
        Returns:
            Initial XPath expression or None if generation fails
        """
        # Create structured prompt for LLM
        prompt = self._create_xpath_generation_prompt(html, task, analysis)
        
        # Call LLM (placeholder function)
        llm_response = self._call_llm_for_xpath(prompt)
        
        if llm_response and 'xpath' in llm_response:
            xpath = llm_response['xpath']
            confidence = llm_response.get('confidence', 0.5)
            
            # Log the initial XPath generation
            action = ExtractionAction(
                action_type=ActionType.XPATH_EXTRACT,
                xpath=xpath,
                description=f"Initial XPath for: {task}",
                success=False,  # Will be updated after testing
                confidence_score=confidence
            )
            self.action_sequence.append(action)
            
            return xpath
        
        return None
    
    def _create_xpath_generation_prompt(self, html: str, task: str, analysis: Dict[str, Any]) -> str:
        """
        Create a structured prompt for XPath generation.
        
        Args:
            html: HTML content (truncated for prompt)
            task: Extraction task
            analysis: Structure analysis
            
        Returns:
            Formatted prompt string
        """
        # Truncate HTML for prompt (keep first 2000 characters)
        html_sample = html[:2000] + "..." if len(html) > 2000 else html
        
        prompt = f"""
        TASK: Generate an XPath expression to extract data from HTML.
        
        EXTRACTION REQUIREMENT: {task}
        
        HTML STRUCTURE ANALYSIS:
        - Total elements: {analysis['total_elements']}
        - Available tags: {', '.join(analysis['unique_tags'][:10])}
        - Elements with classes: {analysis['elements_with_classes']}
        - Elements with IDs: {analysis['elements_with_ids']}
        - Identified patterns: {', '.join(analysis['common_patterns'])}
        
        HTML SAMPLE:
        {html_sample}
        
        INSTRUCTIONS:
        1. Analyze the HTML structure and the extraction requirement
        2. Generate a precise XPath expression that targets the required data
        3. Prefer specific selectors (class, id) over generic ones
        4. Consider the identified patterns in your XPath design
        5. Return response in JSON format with 'xpath' and 'confidence' fields
        
        RESPONSE FORMAT:
        {{"xpath": "//your/xpath/expression", "confidence": 0.8, "reasoning": "explanation"}}
        """
        
        return prompt
    
    def _execute_progressive_refinement(self, html_tree: etree._Element, initial_xpath: str, 
                                      task: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the progressive refinement process with step-back logic.
        
        This is the core of Phase 1 AUTOSCRAPER algorithm:
        1. Test initial XPath
        2. If it fails, apply step-back logic (move to parent elements)
        3. Refine XPath based on results
        4. Iterate until success or max iterations reached
        
        Args:
            html_tree: Parsed HTML tree
            initial_xpath: Starting XPath expression
            task: Extraction task description
            analysis: HTML structure analysis
            
        Returns:
            Final processing result
        """
        current_xpath = initial_xpath
        iteration = 0
        best_result = None
        best_confidence = 0.0
        
        while iteration < self.max_iterations:
            self.logger.info(f"Refinement iteration {iteration + 1}: Testing XPath: {current_xpath}")
            
            # Step 1: Execute current XPath
            execution_result = self._execute_xpath(html_tree, current_xpath)
            
            # Step 2: Evaluate results
            evaluation = self._evaluate_extraction_result(execution_result, task)
            
            # Step 3: Update action sequence
            action = ExtractionAction(
                action_type=ActionType.XPATH_EXTRACT,
                xpath=current_xpath,
                description=f"Iteration {iteration + 1} XPath test",
                success=evaluation['success'],
                extracted_data=execution_result,
                confidence_score=evaluation['confidence']
            )
            self.action_sequence.append(action)
            
            # Step 4: Check if we have a good result
            if evaluation['success'] and evaluation['confidence'] >= self.min_confidence:
                self.logger.info(f"Successful extraction achieved with confidence: {evaluation['confidence']}")
                return self._create_success_result(current_xpath, execution_result, evaluation['confidence'])
            
            # Step 5: Track best result so far
            if evaluation['confidence'] > best_confidence:
                best_confidence = evaluation['confidence']
                best_result = {
                    'xpath': current_xpath,
                    'data': execution_result,
                    'confidence': evaluation['confidence']
                }
            
            # Step 6: Apply step-back logic or refinement only if confidence is very low
            if evaluation['confidence'] < 0.4 or (evaluation.get('is_single_element_weak') and evaluation['confidence'] < 0.7):
                next_xpath = self._apply_step_back_logic(html_tree, current_xpath, task, evaluation)
                
                if next_xpath == current_xpath:
                    # No improvement possible, try LLM refinement
                    next_xpath = self._refine_xpath_with_llm(current_xpath, task, evaluation, analysis)
                
                if next_xpath == current_xpath:
                    # Still no improvement, break the loop
                    self.logger.warning("No further refinement possible")
                    break
                
                current_xpath = next_xpath
            
            iteration += 1
        
        # Return best result found, even if not meeting minimum confidence
        if best_result and best_result['confidence'] >= 0.3:  # Lower threshold for acceptance
            self.logger.info(f"Returning best result with confidence: {best_confidence}")
            return self._create_success_result(
                best_result['xpath'], 
                best_result['data'], 
                best_result['confidence']
            )
        
        return self._create_failure_result("No viable XPath found after all iterations")
    
    def _execute_xpath(self, html_tree: etree._Element, xpath: str) -> List[str]:
        """
        Execute XPath expression on HTML tree and return extracted text.
        
        Args:
            html_tree: Parsed HTML tree
            xpath: XPath expression to execute
            
        Returns:
            List of extracted text values
        """
        try:
            # Execute XPath and extract text content
            elements = html_tree.xpath(xpath)
            
            extracted_data = []
            for element in elements:
                if isinstance(element, str):
                    # Direct text result
                    extracted_data.append(element.strip())
                elif hasattr(element, 'text_content'):
                    # Element with text content
                    text = element.text_content().strip()
                    if text:
                        extracted_data.append(text)
                elif hasattr(element, 'text'):
                    # Element with text attribute
                    text = (element.text or '').strip()
                    if text:
                        extracted_data.append(text)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_data = []
            for item in extracted_data:
                if item not in seen:
                    seen.add(item)
                    unique_data.append(item)
            
            return unique_data
            
        except Exception as e:
            self.logger.error(f"XPath execution failed: {e}")
            return []
    
    def _evaluate_extraction_result(self, extracted_data: List[str], task: str) -> Dict[str, Any]:
        """
        Evaluate the quality of extraction results.
        
        This function analyzes the extracted data to determine:
        - Whether the extraction was successful
        - Confidence score based on data quality
        - Specific issues or improvements needed
        
        Args:
            extracted_data: List of extracted text values
            task: Original extraction task
            
        Returns:
            Evaluation results dictionary
        """
        if not extracted_data:
            return {
                'success': False,
                'confidence': 0.0,
                'issues': ['No data extracted'],
                'data_count': 0
            }
        
        # Basic success criteria
        data_count = len(extracted_data)
        has_meaningful_content = any(len(item.strip()) > 2 for item in extracted_data)
        
        # Check for single element weakness - prefer multiple similar elements
        is_single_element_weak = self._is_single_element_weak_candidate(extracted_data, task)
        
        # Calculate confidence based on various factors
        confidence_factors = []
        
        # Factor 1: Data quantity - be more lenient with single elements
        if is_single_element_weak:
            confidence_factors.append(0.6)  # Increased from 0.4 - more lenient
        elif 2 <= data_count <= 20:
            confidence_factors.append(0.9)  # Good range for multiple elements
        elif data_count > 20:
            confidence_factors.append(0.7)  # Many elements, might include noise
        else:
            confidence_factors.append(0.7)  # Single element, reasonable confidence
        
        # Factor 2: Content quality
        if has_meaningful_content:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.3)
        
        # Factor 3: Task-specific validation
        task_confidence = self._validate_task_specific_requirements(extracted_data, task)
        confidence_factors.append(task_confidence)
        
        # Factor 4: Data consistency
        consistency_score = self._calculate_data_consistency(extracted_data)
        confidence_factors.append(consistency_score)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Determine success - be more lenient overall
        if is_single_element_weak:
            # For single weak elements, require moderate confidence
            success = overall_confidence >= 0.6 and has_meaningful_content  # Reduced from 0.7
        else:
            success = overall_confidence >= 0.5 and has_meaningful_content
        
        issues = self._identify_extraction_issues(extracted_data, task)
        if is_single_element_weak and not success:
            issues.append("Single element returned for task implying multiple items - needs refinement")
        
        return {
            'success': success,
            'confidence': overall_confidence,
            'data_count': data_count,
            'has_meaningful_content': has_meaningful_content,
            'is_single_element_weak': is_single_element_weak,
            'issues': issues
        }
    
    def _is_single_element_weak_candidate(self, data: List[str], task: str) -> bool:
        """
        Determine if a single element result is a weak candidate for the given task.
        
        Tasks that imply multiple items should prefer XPath expressions that return
        multiple similar elements rather than single page-level elements.
        
        Args:
            data: Extracted data items
            task: Extraction task description
            
        Returns:
            True if single element is weak for this task type
        """
        if len(data) != 1:
            return False  # Not a single element
        
        task_lower = task.lower()
        
        # Keywords that typically imply multiple items/repetition
        multiple_item_indicators = [
            'quote', 'quotes', 'items', 'products', 'articles', 'posts', 'comments',
            'reviews', 'listings', 'entries', 'records', 'results',
            'books', 'movies', 'songs', 'videos', 'images', 'photos',
            'users', 'profiles', 'contacts', 'messages', 'emails',
            'prices', 'titles', 'names', 'descriptions', 'summaries',
            'links', 'urls', 'addresses', 'locations', 'dates',
            'tags', 'categories', 'labels', 'keywords', 'topics',
            'text', 'content', 'author', 'writers'
        ]
        
        # Plural indicators
        plural_indicators = [
            'all ', 'list of', 'extract all', 'get all', 'find all',
            'collect', 'gather', 'scrape all', 'multiple', 'several',
            'extract quote', 'quote text', 'author name'
        ]
        
        # Check for multiple item indicators
        for indicator in multiple_item_indicators:
            if indicator in task_lower:
                return True
        
        # Check for plural language patterns
        for indicator in plural_indicators:
            if indicator in task_lower:
                return True
        
        # Check if task uses plural forms (simple heuristic)
        words = task_lower.split()
        for word in words:
            if len(word) > 3 and word.endswith('s') and word not in ['this', 'class', 'css']:
                # Likely plural word
                return True
        
        return False
    
    def _validate_task_specific_requirements(self, data: List[str], task: str) -> float:
        """
        Validate extracted data against task-specific requirements.
        
        Args:
            data: Extracted data
            task: Extraction task description
            
        Returns:
            Task-specific confidence score (0.0 to 1.0)
        """
        task_lower = task.lower()
        
        # Price/monetary value validation
        if any(word in task_lower for word in ['price', 'cost', 'amount', 'money']):
            price_pattern = re.compile(r'[\$€£¥]?\d+[.,]?\d*')
            price_matches = sum(1 for item in data if price_pattern.search(item))
            return min(price_matches / len(data), 1.0) if data else 0.0
        
        # Email validation
        if 'email' in task_lower:
            email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
            email_matches = sum(1 for item in data if email_pattern.search(item))
            return min(email_matches / len(data), 1.0) if data else 0.0
        
        # URL/link validation
        if any(word in task_lower for word in ['url', 'link', 'href']):
            url_pattern = re.compile(r'https?://[^\s]+')
            url_matches = sum(1 for item in data if url_pattern.search(item))
            return min(url_matches / len(data), 1.0) if data else 0.0
        
        # Default: check for non-empty, meaningful content
        meaningful_items = sum(1 for item in data if len(item.strip()) > 3)
        return min(meaningful_items / len(data), 1.0) if data else 0.0
    
    def _calculate_data_consistency(self, data: List[str]) -> float:
        """
        Calculate consistency score for extracted data.
        
        Args:
            data: List of extracted data items
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        if len(data) <= 1:
            return 1.0
        
        # Check length consistency
        lengths = [len(item) for item in data]
        length_variance = max(lengths) - min(lengths)
        length_score = max(0.0, 1.0 - (length_variance / 100))  # Penalize high variance
        
        # Check format consistency (basic pattern matching)
        patterns = set()
        for item in data:
            # Create a simple pattern based on character types
            pattern = re.sub(r'\d', 'D', item)  # Replace digits with D
            pattern = re.sub(r'[A-Za-z]', 'L', pattern)  # Replace letters with L
            pattern = re.sub(r'[^\w\s]', 'S', pattern)  # Replace symbols with S
            patterns.add(pattern[:20])  # Limit pattern length
        
        pattern_score = 1.0 / len(patterns) if patterns else 0.0
        
        return (length_score + pattern_score) / 2
    
    def _identify_extraction_issues(self, data: List[str], task: str) -> List[str]:
        """
        Identify specific issues with the extraction results.
        
        Args:
            data: Extracted data
            task: Extraction task
            
        Returns:
            List of identified issues
        """
        issues = []
        
        if not data:
            issues.append("No data extracted")
            return issues
        
        # Check for empty or whitespace-only results
        empty_count = sum(1 for item in data if not item.strip())
        if empty_count > 0:
            issues.append(f"{empty_count} empty results")
        
        # Check for very short results (might be incomplete)
        short_count = sum(1 for item in data if len(item.strip()) < 3)
        if short_count > len(data) * 0.5:
            issues.append("Many results are very short")
        
        # Check for very long results (might include unwanted content)
        long_count = sum(1 for item in data if len(item) > 200)
        if long_count > 0:
            issues.append(f"{long_count} results are very long")
        
        # Check for duplicate results
        unique_data = set(data)
        if len(unique_data) < len(data):
            issues.append(f"{len(data) - len(unique_data)} duplicate results")
        
        return issues
    
    def _apply_step_back_logic(self, html_tree: etree._Element, current_xpath: str, 
                              task: str, evaluation: Dict[str, Any]) -> str:
        """
        Apply step-back logic to refine XPath expression.
        
        Step-back logic in AUTOSCRAPER:
        1. If XPath returns no results, try parent elements
        2. If XPath returns too many results, add more specific conditions
        3. If results are low quality, try sibling or child elements
        
        Args:
            html_tree: Parsed HTML tree
            current_xpath: Current XPath expression
            task: Extraction task
            evaluation: Current evaluation results
            
        Returns:
            Refined XPath expression
        """
        self.logger.info(f"Applying step-back logic for XPath: {current_xpath}")
        
        # Create step-back action record
        step_back_action = ExtractionAction(
            action_type=ActionType.STEP_BACK,
            xpath=current_xpath,
            description=f"Step-back analysis: {evaluation.get('issues', [])}",
            success=False,
            confidence_score=evaluation.get('confidence', 0.0)
        )
        self.action_sequence.append(step_back_action)
        
        # Strategy 1: No results - try parent elements
        if evaluation['data_count'] == 0:
            return self._step_back_to_parent(current_xpath)
        
        # Strategy 2: Too many results - add specificity
        if evaluation['data_count'] > 20:
            return self._add_xpath_specificity(html_tree, current_xpath, task)
        
        # Strategy 3: Low quality results - try alternative paths
        if evaluation['confidence'] < 0.4:
            return self._try_alternative_paths(html_tree, current_xpath, task)
        
        # Strategy 4: Moderate results - fine-tune
        return self._fine_tune_xpath(html_tree, current_xpath, task, evaluation)
    
    def _step_back_to_parent(self, xpath: str) -> str:
        """
        Modify XPath to target parent elements.
        
        Args:
            xpath: Current XPath expression
            
        Returns:
            Modified XPath targeting parent elements
        """
        # Simple step-back: add /.. to go to parent
        if xpath.endswith('/text()'):
            # Remove /text() and go to parent
            base_xpath = xpath[:-7]
            return f"({base_xpath})/.."
        elif xpath.endswith('/@*'):
            # Remove attribute selection and go to parent
            base_xpath = xpath[:-3]
            return f"({base_xpath})/.."
        else:
            # General case: wrap in parentheses and add parent
            return f"({xpath})/.."
    
    def _add_xpath_specificity(self, html_tree: etree._Element, xpath: str, task: str) -> str:
        """
        Add specificity to XPath to reduce result count.
        
        Args:
            html_tree: HTML tree for analysis
            xpath: Current XPath
            task: Extraction task
            
        Returns:
            More specific XPath expression
        """
        # Strategy: Add position filters or attribute conditions
        
        # Try adding position filter for first few elements
        if '[' not in xpath:
            return f"({xpath})[position() <= 10]"
        
        # Try adding text length condition
        if 'text()' in xpath:
            return xpath.replace('text()', 'text()[string-length(.) > 2 and string-length(.) < 100]')
        
        # Try adding class or id conditions if available
        try:
            elements = html_tree.xpath(xpath)
            if elements and hasattr(elements[0], 'get'):
                elem = elements[0]
                if elem.get('class'):
                    class_name = elem.get('class').split()[0]
                    return f"{xpath}[@class='{class_name}']"
                elif elem.get('id'):
                    elem_id = elem.get('id')
                    return f"{xpath}[@id='{elem_id}']"
        except:
            pass
        
        return xpath  # Return unchanged if no improvements found
    
    def _try_alternative_paths(self, html_tree: etree._Element, xpath: str, task: str) -> str:
        """
        Try alternative XPath strategies for better results.
        
        Args:
            html_tree: HTML tree
            xpath: Current XPath
            task: Extraction task
            
        Returns:
            Alternative XPath expression
        """
        # Strategy 1: Try child elements instead of current
        if not xpath.endswith('/text()'):
            return f"{xpath}/text()"
        
        # Strategy 2: Try sibling elements
        if '//' in xpath:
            # Try following-sibling
            base = xpath.replace('/text()', '')
            return f"{base}/following-sibling::*[1]/text()"
        
        # Strategy 3: Try different text extraction method
        if '/text()' in xpath:
            base = xpath.replace('/text()', '')
            return f"{base}//text()[normalize-space()]"
        
        return xpath
    
    def _fine_tune_xpath(self, html_tree: etree._Element, xpath: str, task: str, 
                        evaluation: Dict[str, Any]) -> str:
        """
        Fine-tune XPath based on evaluation feedback.
        
        Args:
            html_tree: HTML tree
            xpath: Current XPath
            task: Extraction task
            evaluation: Current evaluation results
            
        Returns:
            Fine-tuned XPath expression
        """
        issues = evaluation.get('issues', [])
        
        # Address specific issues
        if 'empty results' in str(issues):
            # Try to get non-empty text
            if '/text()' in xpath:
                return xpath.replace('/text()', '/text()[normalize-space()]')
        
        if 'very short' in str(issues):
            # Add minimum length requirement
            if '/text()' in xpath:
                return xpath.replace('/text()', '/text()[string-length(.) > 5]')
        
        if 'very long' in str(issues):
            # Add maximum length requirement
            if '/text()' in xpath:
                return xpath.replace('/text()', '/text()[string-length(.) < 100]')
        
        return xpath
    
    def _refine_xpath_with_llm(self, xpath: str, task: str, evaluation: Dict[str, Any], 
                              analysis: Dict[str, Any]) -> str:
        """
        Use LLM to refine XPath based on evaluation feedback.
        
        Args:
            xpath: Current XPath
            task: Extraction task
            evaluation: Evaluation results
            analysis: HTML structure analysis
            
        Returns:
            LLM-refined XPath expression
        """
        # Create refinement prompt
        prompt = f"""
        TASK: Refine XPath expression based on evaluation feedback.
        
        ORIGINAL TASK: {task}
        CURRENT XPATH: {xpath}
        
        EVALUATION RESULTS:
        - Success: {evaluation['success']}
        - Confidence: {evaluation['confidence']}
        - Data Count: {evaluation['data_count']}
        - Issues: {evaluation.get('issues', [])}
        
        INSTRUCTIONS:
        1. Analyze the current XPath and its performance issues
        2. Suggest improvements to address the identified issues
        3. Consider the HTML structure patterns: {analysis.get('common_patterns', [])}
        4. Return a refined XPath that should perform better
        
        RESPONSE FORMAT:
        {{"xpath": "//refined/xpath/expression", "confidence": 0.8, "changes": "description of changes"}}
        """
        
        # Call LLM for refinement
        llm_response = self._call_llm_for_xpath_refinement(prompt)
        
        if llm_response and 'xpath' in llm_response:
            refined_xpath = llm_response['xpath']
            
            # Log the refinement action
            refine_action = ExtractionAction(
                action_type=ActionType.REFINE_XPATH,
                xpath=refined_xpath,
                description=f"LLM refinement: {llm_response.get('changes', 'No description')}",
                success=False,  # Will be updated after testing
                confidence_score=llm_response.get('confidence', 0.5)
            )
            self.action_sequence.append(refine_action)
            
            return refined_xpath
        
        return xpath  # Return unchanged if LLM refinement fails
    
    def _create_success_result(self, xpath: str, data: List[str], confidence: float) -> Dict[str, Any]:
        """Create a successful result dictionary."""
        return {
            'success': True,
            'action_sequence': self.action_sequence,
            'final_xpath': xpath,
            'extracted_data': data,
            'confidence_score': confidence,
            'total_iterations': len([a for a in self.action_sequence if a.action_type == ActionType.XPATH_EXTRACT])
        }
    
    def _create_failure_result(self, reason: str) -> Dict[str, Any]:
        """Create a failure result dictionary."""
        return {
            'success': False,
            'action_sequence': self.action_sequence,
            'final_xpath': None,
            'extracted_data': [],
            'confidence_score': 0.0,
            'failure_reason': reason,
            'total_iterations': len([a for a in self.action_sequence if a.action_type == ActionType.XPATH_EXTRACT])
        }
    
    # Placeholder LLM functions
    def _call_llm_for_xpath(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Placeholder function for LLM XPath generation.
        
        In a real implementation, this would call an actual LLM API
        (OpenAI, Anthropic, local model, etc.)
        
        Args:
            prompt: Structured prompt for XPath generation
            
        Returns:
            Dictionary with 'xpath', 'confidence', and 'reasoning' fields
        """
        # Placeholder implementation - returns a simple XPath
        # In real implementation, replace with actual LLM API call
        
        self.logger.info("Calling LLM for XPath generation (placeholder)")
        
        # Simulate LLM response based on prompt analysis
        prompt_lower = prompt.lower()
        
        # More comprehensive XPath generation based on common patterns
        if any(word in prompt_lower for word in ['name', 'title', 'product']) and any(word in prompt_lower for word in ['price', 'cost']):
            # When both names and prices are requested, target product containers more broadly
            xpaths = [
                '//div[contains(.//text(), "Rs.") or contains(.//text(), "Balance")]//text()[normalize-space() and string-length(.) > 4 and string-length(.) < 200 and not(contains(., "Add to cart")) and not(contains(., "Quick view"))]',
                '//*[contains(@class, "product") or contains(@class, "item") or contains(@class, "card") or contains(@class, "shoe")]//text()[normalize-space() and string-length(.) > 4 and string-length(.) < 200]',
                '//a[contains(@href, "product") or contains(@href, "shoe")]//text()[normalize-space() and string-length(.) > 5] | //div[contains(.//text(), "Rs.")]//text()[normalize-space() and (contains(., "Rs.") or string-length(.) > 10)]',
                '//*[contains(text(), "New Balance") or contains(text(), "Rs.")]//text()[normalize-space() and string-length(.) > 4]',
                '//text()[normalize-space() and (contains(., "New Balance") or contains(., "Rs.")) and string-length(.) > 4 and string-length(.) < 200]'
            ]
        elif any(word in prompt_lower for word in ['price', 'cost', 'amount', 'money']):
            xpaths = [
                '//*[contains(@class, "price")]/text()',
                '//*[contains(text(), "$") or contains(text(), "₹") or contains(text(), "Rs.") or contains(text(), "price")]/text()',
                '//span[contains(@class, "amount") or contains(@class, "cost")]/text()',
                '//*[@class="price-value" or @class="product-price"]/text()'
            ]
        elif any(word in prompt_lower for word in ['quote', 'quotes', 'author', 'text']):
            xpaths = [
                '//div[contains(@class, "quote")]//text()[normalize-space() and string-length(.) > 10]',
                '//*[contains(@class, "text") or contains(@class, "author")]//text()[normalize-space()]',
                '//span[contains(@class, "text")]//text() | //small[contains(@class, "author")]//text()',
                '//blockquote//text()[normalize-space()] | //cite//text()[normalize-space()]',
                '//*[contains(text(), """) or contains(@class, "quote")]//text()[normalize-space() and string-length(.) > 5]'
            ]
        elif any(word in prompt_lower for word in ['title', 'heading', 'name', 'product']):
            xpaths = [
                '//h1/text() | //h2/text() | //h3/text()',
                '//*[contains(@class, "title") or contains(@class, "name") or contains(@class, "product-name")]/text()',
                '//a[contains(@class, "product") or contains(@class, "item")]/text()',
                '//*[@class="product-title" or @class="item-name"]/text()'
            ]
        elif any(word in prompt_lower for word in ['link', 'url', 'href']):
            xpaths = [
                '//a/@href',
                '//a[contains(@class, "product") or contains(@class, "item")]/@href',
                '//*[@class="link" or @class="url"]/@href'
            ]
        elif any(word in prompt_lower for word in ['shoe', 'shoes', 'product', 'item', 'new balance']):
            xpaths = [
                '//*[contains(@class, "product") or contains(@class, "item") or contains(@class, "card") or contains(@class, "shoe")]//text()[normalize-space() and string-length(.) > 3 and string-length(.) < 200]',
                '//a[contains(@href, "product") or contains(@href, "shoe") or contains(text(), "Balance")]//text()[normalize-space() and string-length(.) > 5]',
                '//*[contains(@class, "title") or contains(@class, "name") or contains(@class, "heading")]//text()[normalize-space() and string-length(.) > 4]',
                '//h1//text()[normalize-space()] | //h2//text()[normalize-space()] | //h3//text()[normalize-space()]',
                '//*[contains(text(), "New Balance") or contains(text(), "shoe") or contains(text(), "Rs.")]//text()[normalize-space() and string-length(.) > 3]'
            ]
        else:
            # Generic content extraction
            xpaths = [
                '//*[text()]/text()[normalize-space() and string-length(.) > 5]',
                '//p/text() | //div/text() | //span/text()',
                '//*[contains(@class, "content") or contains(@class, "text")]//text()[normalize-space()]',
                '//article//text()[normalize-space()] | //section//text()[normalize-space()]'
            ]
        
        # Return the first XPath with reasonable confidence
        selected_xpath = xpaths[0] if xpaths else '//*[text()]/text()[normalize-space()]'
        
        return {
            'xpath': selected_xpath,
            'confidence': 0.7,
            'reasoning': f'Generated XPath for content extraction based on task keywords'
        }
    
    def _call_llm_for_xpath_refinement(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Placeholder function for LLM XPath refinement.
        
        Args:
            prompt: Refinement prompt with evaluation feedback
            
        Returns:
            Dictionary with refined XPath and explanation
        """
        self.logger.info("Calling LLM for XPath refinement (placeholder)")
        
        # Placeholder refinement logic
        # In real implementation, replace with actual LLM API call
        
        prompt_lower = prompt.lower()
        
        if 'no data extracted' in prompt_lower:
            return {
                'xpath': '//div//text()[normalize-space() and string-length(.) > 3]',
                'confidence': 0.6,
                'changes': 'Broadened search to all div elements with meaningful text'
            }
        elif 'too many results' in prompt_lower:
            return {
                'xpath': '(//*[normalize-space(text()) and string-length(.) > 10]/text())[position() <= 10]',
                'confidence': 0.7,
                'changes': 'Limited results to first 10 matches with longer text'
            }
        elif 'single element' in prompt_lower:
            return {
                'xpath': '//*[contains(@class, "item") or contains(@class, "product") or contains(@class, "card")]//text()[normalize-space()]',
                'confidence': 0.8,
                'changes': 'Targeting multiple product/item containers'
            }
        elif 'very short' in prompt_lower:
            return {
                'xpath': '//*[string-length(normalize-space(text())) > 10]/text()',
                'confidence': 0.7,
                'changes': 'Added minimum text length requirement of 10 characters'
            }
        else:
            return {
                'xpath': '//*[contains(@class, "content") or contains(@class, "text") or contains(@class, "description")]//text()[normalize-space()]',
                'confidence': 0.6,
                'changes': 'Targeting content-related class names'
            }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample HTML for testing
    sample_html = """
    <html>
    <body>
        <div class="product-list">
            <div class="product-item">
                <h3 class="product-title">Laptop Computer</h3>
                <span class="price">$999.99</span>
                <p class="description">High-performance laptop</p>
            </div>
            <div class="product-item">
                <h3 class="product-title">Desktop Monitor</h3>
                <span class="price">$299.99</span>
                <p class="description">24-inch LED monitor</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Test Phase 1 processing
    phase1 = Phase1Progressive(max_iterations=3, min_confidence=0.6)
    
    # Test different extraction tasks
    test_tasks = [
        "Extract product titles",
        "Extract prices",
        "Extract product descriptions"
    ]
    
    for task in test_tasks:
        print(f"\n--- Loading: {task} ---")
        result = phase1.process_extraction_task(sample_html, task)
        
        print(f"Success: {result['success']}")
        print(f"Final XPath: {result['final_xpath']}")
        print(f"Extracted Data: {result['extracted_data']}")
        print(f"Confidence: {result['confidence_score']:.2f}")
        print(f"Total Iterations: {result['total_iterations']}")
        
        if result['action_sequence']:
            print("Action Sequence:")
            for i, action in enumerate(result['action_sequence']):
                print(f"  {i+1}. {action.action_type.value}: {action.xpath}")
                print(f"     Success: {action.success}, Confidence: {action.confidence_score:.2f}")