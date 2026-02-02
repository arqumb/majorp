"""
Phase 2 Synthesis Module - AUTOSCRAPER Framework Implementation

This module implements Phase 2 (Synthesis) of the AUTOSCRAPER framework, which
improves robustness by testing Action Sequences across multiple pages and selecting
the most reusable extraction patterns.

Why Synthesis Improves Robustness:
1. Cross-validation: Tests sequences on pages they weren't trained on
2. Generalization: Identifies patterns that work across different page structures
3. Noise reduction: Filters out page-specific XPath expressions
4. Reliability: Selects sequences with consistent performance across datasets
5. Adaptability: Handles variations in HTML structure and content organization
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from lxml import html, etree
import statistics
from collections import defaultdict


class SynthesisMetric(Enum):
    """Metrics used for evaluating Action Sequence performance."""
    SUCCESS_RATE = "success_rate"
    CONSISTENCY_SCORE = "consistency_score"
    DATA_QUALITY = "data_quality"
    COVERAGE_SCORE = "coverage_score"


@dataclass
class ActionSequence:
    """
    Represents an Action Sequence from Phase 1 with metadata.
    
    This structure encapsulates the extraction logic and tracks
    its performance across different pages during synthesis.
    """
    sequence_id: str
    source_page_id: str
    extraction_task: str
    actions: List[Dict[str, Any]]
    original_confidence: float
    synthesis_scores: Dict[str, float] = field(default_factory=dict)
    cross_page_results: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    """
    Contains the results of Phase 2 synthesis process.
    
    Provides comprehensive information about sequence performance
    and the rationale for the final selection.
    """
    selected_sequence: ActionSequence
    all_sequences_performance: List[Dict[str, Any]]
    synthesis_metrics: Dict[str, float]
    cross_validation_summary: Dict[str, Any]
    robustness_score: float
    selection_rationale: str


class Phase2Synthesis:
    """
    Phase 2 of AUTOSCRAPER framework - Action Sequence Synthesis and Selection.
    
    This class implements the synthesis logic that tests multiple Action Sequences
    across different pages to identify the most robust and reusable extraction
    patterns. The synthesis process improves reliability by:
    
    1. Testing generalization across different page structures
    2. Identifying sequences that work consistently across datasets
    3. Filtering out overfitted, page-specific extraction rules
    4. Selecting patterns with the highest cross-validation performance
    """
    
    def __init__(self, min_success_threshold: float = 0.6, 
                 consistency_weight: float = 0.4,
                 coverage_weight: float = 0.3,
                 quality_weight: float = 0.3):
        """
        Initialize Phase 2 synthesis processor.
        
        Args:
            min_success_threshold: Minimum success rate for viable sequences
            consistency_weight: Weight for consistency in scoring
            coverage_weight: Weight for data coverage in scoring  
            quality_weight: Weight for data quality in scoring
        """
        self.min_success_threshold = min_success_threshold
        self.consistency_weight = consistency_weight
        self.coverage_weight = coverage_weight
        self.quality_weight = quality_weight
        self.logger = logging.getLogger(__name__)
    
    def synthesize_sequences(self, action_sequences: List[ActionSequence], 
                           page_htmls: Dict[str, str]) -> SynthesisResult:
        """
        Main synthesis process that evaluates and selects the best Action Sequence.
        
        The synthesis process works by:
        1. Cross-validating each sequence against all available pages
        2. Computing robustness metrics for each sequence
        3. Ranking sequences by their generalization performance
        4. Selecting the most robust sequence for production use
        
        This approach ensures that the selected extraction logic works reliably
        across different page variations, not just the original training page.
        
        Args:
            action_sequences: List of Action Sequences from Phase 1
            page_htmls: Dictionary mapping page_id -> HTML content
            
        Returns:
            SynthesisResult containing the selected sequence and analysis
        """
        self.logger.info(f"Starting synthesis of {len(action_sequences)} sequences across {len(page_htmls)} pages")
        
        if not action_sequences:
            raise ValueError("No Action Sequences provided for synthesis")
        
        if not page_htmls:
            raise ValueError("No HTML pages provided for cross-validation")
        
        # Step 1: Execute cross-validation for all sequences
        cross_validation_results = self._execute_cross_validation(action_sequences, page_htmls)
        
        # Step 2: Calculate synthesis metrics for each sequence
        sequence_performances = self._calculate_synthesis_metrics(cross_validation_results, page_htmls)
        
        # Step 3: Rank sequences by robustness score
        ranked_sequences = self._rank_sequences_by_robustness(sequence_performances)
        
        # Step 4: Select the best performing sequence
        selected_sequence = self._select_best_sequence(ranked_sequences)
        
        # Step 5: Generate comprehensive synthesis report
        synthesis_result = self._generate_synthesis_result(
            selected_sequence, ranked_sequences, cross_validation_results, page_htmls
        )
        
        self.logger.info(f"Synthesis complete. Selected sequence: {selected_sequence.sequence_id}")
        return synthesis_result
    
    def _execute_cross_validation(self, sequences: List[ActionSequence], 
                                 page_htmls: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Execute each Action Sequence against all available pages.
        
        Cross-validation is the core of synthesis robustness testing. By running
        each sequence on pages it wasn't trained on, we can identify which
        extraction patterns generalize well vs. those that are overfitted to
        specific page structures.
        
        Args:
            sequences: Action Sequences to test
            page_htmls: HTML content for all pages
            
        Returns:
            Dictionary mapping sequence_id -> page_id -> execution results
        """
        self.logger.info("Executing cross-validation across all sequence-page combinations")
        
        cross_validation_results = {}
        
        for sequence in sequences:
            self.logger.debug(f"Cross-validating sequence: {sequence.sequence_id}")
            sequence_results = {}
            
            for page_id, html_content in page_htmls.items():
                # Execute sequence on this page
                execution_result = self._execute_sequence_on_page(sequence, html_content, page_id)
                sequence_results[page_id] = execution_result
                
                # Store results in the sequence object for later analysis
                sequence.cross_page_results[page_id] = execution_result.get('extracted_data', [])
            
            cross_validation_results[sequence.sequence_id] = sequence_results
        
        return cross_validation_results
    
    def _execute_sequence_on_page(self, sequence: ActionSequence, html_content: str, 
                                 page_id: str) -> Dict[str, Any]:
        """
        Execute a single Action Sequence on a specific page.
        
        This function replays the extraction logic from Phase 1 on new pages
        to test generalization. It handles potential failures gracefully and
        provides detailed execution metrics.
        
        Args:
            sequence: Action Sequence to execute
            html_content: HTML content of the target page
            page_id: Identifier for the page being processed
            
        Returns:
            Dictionary containing execution results and metrics
        """
        try:
            # Parse HTML content
            html_tree = html.fromstring(html_content)
            
            # Initialize execution state
            execution_state = {
                'success': False,
                'extracted_data': [],
                'execution_steps': [],
                'error_count': 0,
                'final_xpath': None
            }
            
            # Execute each action in the sequence
            for i, action in enumerate(sequence.actions):
                step_result = self._execute_single_action(html_tree, action, i)
                execution_state['execution_steps'].append(step_result)
                
                if step_result['success']:
                    execution_state['extracted_data'] = step_result['data']
                    execution_state['final_xpath'] = step_result['xpath']
                    execution_state['success'] = True
                else:
                    execution_state['error_count'] += 1
            
            # Calculate execution quality metrics
            execution_state.update(self._calculate_execution_quality(execution_state, sequence.extraction_task))
            
            return execution_state
            
        except Exception as e:
            self.logger.error(f"Failed to execute sequence {sequence.sequence_id} on page {page_id}: {e}")
            return {
                'success': False,
                'extracted_data': [],
                'execution_steps': [],
                'error_count': 1,
                'final_xpath': None,
                'quality_score': 0.0,
                'data_consistency': 0.0,
                'error_message': str(e)
            }
    
    def _execute_single_action(self, html_tree: etree._Element, action: Dict[str, Any], 
                              step_index: int) -> Dict[str, Any]:
        """
        Execute a single action from an Action Sequence.
        
        Actions represent individual XPath operations or refinements from Phase 1.
        This function handles different action types and provides detailed
        execution feedback for synthesis analysis.
        
        Args:
            html_tree: Parsed HTML tree
            action: Single action from the sequence
            step_index: Index of this action in the sequence
            
        Returns:
            Dictionary with action execution results
        """
        action_type = action.get('action_type', 'xpath_extract')
        xpath = action.get('xpath', '')
        
        if not xpath:
            return {
                'success': False,
                'data': [],
                'xpath': xpath,
                'step_index': step_index,
                'error': 'No XPath provided'
            }
        
        try:
            # Execute XPath expression
            elements = html_tree.xpath(xpath)
            
            # Extract text content from results
            extracted_data = []
            for element in elements:
                if isinstance(element, str):
                    # Direct text result
                    text = element.strip()
                    if text:
                        extracted_data.append(text)
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
            
            return {
                'success': len(unique_data) > 0,
                'data': unique_data,
                'xpath': xpath,
                'step_index': step_index,
                'element_count': len(elements),
                'data_count': len(unique_data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'data': [],
                'xpath': xpath,
                'step_index': step_index,
                'error': str(e)
            }
    
    def _calculate_execution_quality(self, execution_state: Dict[str, Any], 
                                   extraction_task: str) -> Dict[str, float]:
        """
        Calculate quality metrics for sequence execution results.
        
        Quality assessment helps distinguish between sequences that extract
        meaningful data vs. those that return noise or irrelevant content.
        This is crucial for synthesis because a sequence might technically
        "succeed" but extract poor quality data.
        
        Args:
            execution_state: Current execution state with results
            extraction_task: Original extraction task description
            
        Returns:
            Dictionary with quality metrics
        """
        extracted_data = execution_state.get('extracted_data', [])
        
        if not extracted_data:
            return {
                'quality_score': 0.0,
                'data_consistency': 0.0,
                'content_relevance': 0.0
            }
        
        # Calculate content quality score
        quality_factors = []
        
        # Factor 1: Data meaningfulness (non-empty, reasonable length)
        meaningful_items = [item for item in extracted_data if 3 <= len(item.strip()) <= 200]
        meaningfulness_score = len(meaningful_items) / len(extracted_data) if extracted_data else 0
        quality_factors.append(meaningfulness_score)
        
        # Factor 2: Data consistency (similar formats/patterns)
        consistency_score = self._calculate_data_consistency_score(extracted_data)
        quality_factors.append(consistency_score)
        
        # Factor 3: Task relevance (basic keyword matching)
        relevance_score = self._calculate_task_relevance_score(extracted_data, extraction_task)
        quality_factors.append(relevance_score)
        
        # Factor 4: Data uniqueness (avoid excessive duplication)
        unique_ratio = len(set(extracted_data)) / len(extracted_data) if extracted_data else 0
        quality_factors.append(min(unique_ratio * 1.2, 1.0))  # Slight bonus for uniqueness
        
        overall_quality = sum(quality_factors) / len(quality_factors)
        
        return {
            'quality_score': overall_quality,
            'data_consistency': consistency_score,
            'content_relevance': relevance_score,
            'meaningfulness': meaningfulness_score,
            'uniqueness_ratio': unique_ratio
        }
    
    def _calculate_data_consistency_score(self, data: List[str]) -> float:
        """
        Calculate consistency score for extracted data items.
        
        Consistent data patterns indicate that the XPath is targeting
        the right type of content across different instances.
        
        Args:
            data: List of extracted data items
            
        Returns:
            Consistency score between 0.0 and 1.0
        """
        if len(data) <= 1:
            return 1.0
        
        # Analyze length consistency
        lengths = [len(item) for item in data]
        length_variance = statistics.variance(lengths) if len(lengths) > 1 else 0
        length_consistency = max(0.0, 1.0 - (length_variance / 10000))  # Normalize variance
        
        # Analyze character pattern consistency
        patterns = []
        for item in data:
            # Create simple pattern based on character types
            pattern = ""
            for char in item[:20]:  # Limit pattern analysis to first 20 chars
                if char.isdigit():
                    pattern += "D"
                elif char.isalpha():
                    pattern += "L"
                elif char.isspace():
                    pattern += "S"
                else:
                    pattern += "P"  # Punctuation/special
            patterns.append(pattern)
        
        # Calculate pattern similarity
        unique_patterns = set(patterns)
        pattern_consistency = 1.0 / len(unique_patterns) if unique_patterns else 0.0
        
        return (length_consistency + pattern_consistency) / 2
    
    def _calculate_task_relevance_score(self, data: List[str], task: str) -> float:
        """
        Calculate how relevant the extracted data is to the original task.
        
        This helps identify sequences that extract the right type of content
        vs. those that extract irrelevant but structurally similar data.
        
        Args:
            data: Extracted data items
            task: Original extraction task description
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not data or not task:
            return 0.0
        
        task_lower = task.lower()
        relevance_indicators = []
        
        # Check for task-specific patterns in the data
        for item in data:
            item_lower = item.lower()
            
            # Price/monetary relevance
            if any(word in task_lower for word in ['price', 'cost', 'amount', 'money']):
                if any(char in item for char in ['$', '€', '£', '¥']) or 'price' in item_lower:
                    relevance_indicators.append(1.0)
                else:
                    relevance_indicators.append(0.3)
            
            # Title/heading relevance
            elif any(word in task_lower for word in ['title', 'heading', 'name']):
                # Titles are usually not too long and don't contain special chars
                if 5 <= len(item) <= 100 and not any(char in item for char in ['@', 'http']):
                    relevance_indicators.append(0.8)
                else:
                    relevance_indicators.append(0.4)
            
            # Link/URL relevance
            elif any(word in task_lower for word in ['link', 'url', 'href']):
                if 'http' in item_lower or item.startswith('/'):
                    relevance_indicators.append(1.0)
                else:
                    relevance_indicators.append(0.2)
            
            # Email relevance
            elif 'email' in task_lower:
                if '@' in item and '.' in item:
                    relevance_indicators.append(1.0)
                else:
                    relevance_indicators.append(0.1)
            
            # Default: basic content relevance
            else:
                # Assume meaningful content should be reasonably sized
                if 3 <= len(item) <= 500:
                    relevance_indicators.append(0.6)
                else:
                    relevance_indicators.append(0.3)
        
        return sum(relevance_indicators) / len(relevance_indicators) if relevance_indicators else 0.0
    
    def _calculate_synthesis_metrics(self, cross_validation_results: Dict[str, Dict[str, Any]], 
                                   page_htmls: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Calculate comprehensive synthesis metrics for each Action Sequence.
        
        Synthesis metrics evaluate how well each sequence generalizes across
        different pages. This is the core of robustness assessment - sequences
        that work well on their training page but fail on others are overfitted
        and should be penalized.
        
        Args:
            cross_validation_results: Results from cross-validation execution
            page_htmls: Original HTML content for reference
            
        Returns:
            List of performance dictionaries for each sequence
        """
        self.logger.info("Calculating synthesis metrics for sequence ranking")
        
        sequence_performances = []
        
        for sequence_id, page_results in cross_validation_results.items():
            # Calculate success rate across all pages
            success_count = sum(1 for result in page_results.values() if result['success'])
            success_rate = success_count / len(page_results) if page_results else 0.0
            
            # Calculate consistency metrics
            consistency_metrics = self._calculate_cross_page_consistency(page_results)
            
            # Calculate coverage metrics (how much data is extracted)
            coverage_metrics = self._calculate_coverage_metrics(page_results)
            
            # Calculate overall quality across pages
            quality_scores = [result.get('quality_score', 0.0) for result in page_results.values()]
            average_quality = statistics.mean(quality_scores) if quality_scores else 0.0
            
            # Calculate robustness score (weighted combination of metrics)
            robustness_score = self._calculate_robustness_score(
                success_rate, consistency_metrics, coverage_metrics, average_quality
            )
            
            performance = {
                'sequence_id': sequence_id,
                'success_rate': success_rate,
                'consistency_score': consistency_metrics['overall_consistency'],
                'coverage_score': coverage_metrics['average_coverage'],
                'quality_score': average_quality,
                'robustness_score': robustness_score,
                'total_pages_tested': len(page_results),
                'successful_pages': success_count,
                'detailed_metrics': {
                    'consistency': consistency_metrics,
                    'coverage': coverage_metrics,
                    'quality_distribution': quality_scores
                }
            }
            
            sequence_performances.append(performance)
        
        return sequence_performances
    
    def _calculate_cross_page_consistency(self, page_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate consistency of extraction results across different pages.
        
        Consistency measures how similar the extraction patterns are across
        pages. High consistency indicates that the sequence extracts the same
        type of content regardless of minor page variations.
        
        Args:
            page_results: Results for a single sequence across all pages
            
        Returns:
            Dictionary with various consistency metrics
        """
        # Extract data from all successful pages
        successful_extractions = []
        for result in page_results.values():
            if result['success'] and result.get('extracted_data'):
                successful_extractions.append(result['extracted_data'])
        
        if len(successful_extractions) < 2:
            return {
                'overall_consistency': 1.0 if successful_extractions else 0.0,
                'data_count_consistency': 1.0,
                'content_pattern_consistency': 1.0,
                'length_consistency': 1.0
            }
        
        # Calculate data count consistency
        data_counts = [len(extraction) for extraction in successful_extractions]
        count_variance = statistics.variance(data_counts) if len(data_counts) > 1 else 0
        count_consistency = max(0.0, 1.0 - (count_variance / 100))  # Normalize variance
        
        # Calculate content pattern consistency
        all_items = [item for extraction in successful_extractions for item in extraction]
        pattern_consistency = self._calculate_data_consistency_score(all_items)
        
        # Calculate length consistency across extractions
        avg_lengths = [statistics.mean([len(item) for item in extraction]) 
                      for extraction in successful_extractions if extraction]
        length_variance = statistics.variance(avg_lengths) if len(avg_lengths) > 1 else 0
        length_consistency = max(0.0, 1.0 - (length_variance / 1000))
        
        overall_consistency = (count_consistency + pattern_consistency + length_consistency) / 3
        
        return {
            'overall_consistency': overall_consistency,
            'data_count_consistency': count_consistency,
            'content_pattern_consistency': pattern_consistency,
            'length_consistency': length_consistency
        }
    
    def _calculate_coverage_metrics(self, page_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate coverage metrics for extraction results.
        
        Coverage measures how much relevant data the sequence extracts.
        Good coverage indicates that the sequence finds most of the target
        content without missing important items.
        
        Args:
            page_results: Results for a single sequence across all pages
            
        Returns:
            Dictionary with coverage metrics
        """
        data_counts = []
        quality_weighted_counts = []
        
        for result in page_results.values():
            if result['success']:
                data_count = len(result.get('extracted_data', []))
                quality_score = result.get('quality_score', 0.0)
                
                data_counts.append(data_count)
                quality_weighted_counts.append(data_count * quality_score)
        
        if not data_counts:
            return {
                'average_coverage': 0.0,
                'coverage_consistency': 0.0,
                'quality_weighted_coverage': 0.0
            }
        
        average_coverage = statistics.mean(data_counts)
        coverage_variance = statistics.variance(data_counts) if len(data_counts) > 1 else 0
        coverage_consistency = max(0.0, 1.0 - (coverage_variance / 100))
        quality_weighted_coverage = statistics.mean(quality_weighted_counts)
        
        return {
            'average_coverage': min(average_coverage / 10, 1.0),  # Normalize to 0-1 scale
            'coverage_consistency': coverage_consistency,
            'quality_weighted_coverage': min(quality_weighted_coverage / 10, 1.0)
        }
    
    def _calculate_robustness_score(self, success_rate: float, consistency_metrics: Dict[str, float],
                                  coverage_metrics: Dict[str, float], quality_score: float) -> float:
        """
        Calculate overall robustness score for a sequence.
        
        The robustness score combines multiple metrics to provide a single
        measure of how well a sequence generalizes across different pages.
        This is the primary metric used for sequence selection.
        
        Args:
            success_rate: Proportion of pages where sequence succeeded
            consistency_metrics: Cross-page consistency measurements
            coverage_metrics: Data coverage measurements
            quality_score: Average quality of extracted data
            
        Returns:
            Overall robustness score between 0.0 and 1.0
        """
        # Weight the different components
        weighted_score = (
            success_rate * 0.4 +  # Success rate is most important
            consistency_metrics['overall_consistency'] * self.consistency_weight +
            coverage_metrics['average_coverage'] * self.coverage_weight +
            quality_score * self.quality_weight
        )
        
        # Apply penalty for low success rates (sequences must work on most pages)
        if success_rate < self.min_success_threshold:
            weighted_score *= (success_rate / self.min_success_threshold)
        
        return min(weighted_score, 1.0)
    
    def _rank_sequences_by_robustness(self, sequence_performances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank Action Sequences by their robustness scores.
        
        Ranking helps identify the most reliable sequences for production use.
        The top-ranked sequence will be selected as the final extraction rule.
        
        Args:
            sequence_performances: Performance metrics for all sequences
            
        Returns:
            List of sequences sorted by robustness score (descending)
        """
        # Sort by robustness score in descending order
        ranked_sequences = sorted(
            sequence_performances,
            key=lambda x: x['robustness_score'],
            reverse=True
        )
        
        # Add ranking information
        for i, sequence in enumerate(ranked_sequences):
            sequence['rank'] = i + 1
            sequence['is_selected'] = (i == 0)  # Top sequence is selected
        
        self.logger.info(f"Sequence ranking complete. Top sequence: {ranked_sequences[0]['sequence_id']}")
        return ranked_sequences
    
    def _select_best_sequence(self, ranked_sequences: List[Dict[str, Any]]) -> ActionSequence:
        """
        Select the best performing Action Sequence.
        
        Selection is based on the highest robustness score, but also considers
        minimum thresholds for success rate and quality to ensure the selected
        sequence meets production requirements.
        
        Args:
            ranked_sequences: Sequences ranked by performance
            
        Returns:
            The selected ActionSequence object
        """
        if not ranked_sequences:
            raise ValueError("No sequences available for selection")
        
        # Find the best sequence that meets minimum requirements
        for sequence_perf in ranked_sequences:
            if (sequence_perf['success_rate'] >= self.min_success_threshold and
                sequence_perf['quality_score'] >= 0.3):  # Minimum quality threshold
                
                # Create ActionSequence object (this would normally come from the original list)
                selected_sequence = ActionSequence(
                    sequence_id=sequence_perf['sequence_id'],
                    source_page_id="synthesis_selected",
                    extraction_task="synthesized_task",
                    actions=[],  # Would be populated from original sequence
                    original_confidence=sequence_perf['quality_score'],
                    synthesis_scores=sequence_perf
                )
                
                self.logger.info(f"Selected sequence {selected_sequence.sequence_id} with robustness score: {sequence_perf['robustness_score']:.3f}")
                return selected_sequence
        
        # If no sequence meets requirements, select the best available
        best_sequence_perf = ranked_sequences[0]
        self.logger.warning(f"No sequence meets minimum requirements. Selecting best available: {best_sequence_perf['sequence_id']}")
        
        return ActionSequence(
            sequence_id=best_sequence_perf['sequence_id'],
            source_page_id="synthesis_selected",
            extraction_task="synthesized_task",
            actions=[],
            original_confidence=best_sequence_perf['quality_score'],
            synthesis_scores=best_sequence_perf
        )
    
    def _generate_synthesis_result(self, selected_sequence: ActionSequence,
                                 ranked_sequences: List[Dict[str, Any]],
                                 cross_validation_results: Dict[str, Dict[str, Any]],
                                 page_htmls: Dict[str, str]) -> SynthesisResult:
        """
        Generate comprehensive synthesis result with analysis and rationale.
        
        The synthesis result provides detailed information about the selection
        process and performance metrics for transparency and debugging.
        
        Args:
            selected_sequence: The chosen ActionSequence
            ranked_sequences: All sequences with performance metrics
            cross_validation_results: Detailed cross-validation data
            page_htmls: Original HTML content
            
        Returns:
            Complete SynthesisResult object
        """
        # Calculate overall synthesis metrics
        all_robustness_scores = [seq['robustness_score'] for seq in ranked_sequences]
        synthesis_metrics = {
            'total_sequences_evaluated': len(ranked_sequences),
            'total_pages_tested': len(page_htmls),
            'average_robustness_score': statistics.mean(all_robustness_scores),
            'best_robustness_score': max(all_robustness_scores),
            'robustness_score_variance': statistics.variance(all_robustness_scores) if len(all_robustness_scores) > 1 else 0
        }
        
        # Generate cross-validation summary
        selected_results = cross_validation_results.get(selected_sequence.sequence_id, {})
        cross_validation_summary = {
            'pages_successful': sum(1 for result in selected_results.values() if result['success']),
            'pages_failed': sum(1 for result in selected_results.values() if not result['success']),
            'average_data_count': statistics.mean([len(result.get('extracted_data', [])) 
                                                 for result in selected_results.values()]),
            'success_by_page': {page_id: result['success'] 
                              for page_id, result in selected_results.items()}
        }
        
        # Generate selection rationale
        selected_perf = ranked_sequences[0]  # Top ranked sequence
        selection_rationale = (
            f"Selected sequence '{selected_sequence.sequence_id}' based on highest robustness score "
            f"({selected_perf['robustness_score']:.3f}). "
            f"Success rate: {selected_perf['success_rate']:.1%}, "
            f"Quality score: {selected_perf['quality_score']:.3f}, "
            f"Consistency: {selected_perf['consistency_score']:.3f}. "
            f"Sequence succeeded on {cross_validation_summary['pages_successful']}/{len(page_htmls)} pages."
        )
        
        return SynthesisResult(
            selected_sequence=selected_sequence,
            all_sequences_performance=ranked_sequences,
            synthesis_metrics=synthesis_metrics,
            cross_validation_summary=cross_validation_summary,
            robustness_score=selected_perf['robustness_score'],
            selection_rationale=selection_rationale
        )


# Utility functions for testing and integration
def create_mock_action_sequence(sequence_id: str, xpath: str, task: str) -> ActionSequence:
    """
    Create a mock ActionSequence for testing purposes.
    
    Args:
        sequence_id: Unique identifier for the sequence
        xpath: XPath expression to use
        task: Extraction task description
        
    Returns:
        Mock ActionSequence object
    """
    return ActionSequence(
        sequence_id=sequence_id,
        source_page_id=f"page_{sequence_id}",
        extraction_task=task,
        actions=[{
            'action_type': 'xpath_extract',
            'xpath': xpath,
            'description': f'Extract {task}',
            'success': True,
            'confidence_score': 0.8
        }],
        original_confidence=0.8
    )


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample HTML pages for testing
    sample_pages = {
        'page1': '''
        <html><body>
            <div class="product">
                <h2 class="title">Laptop Computer</h2>
                <span class="price">$999.99</span>
            </div>
            <div class="product">
                <h2 class="title">Desktop Monitor</h2>
                <span class="price">$299.99</span>
            </div>
        </body></html>
        ''',
        'page2': '''
        <html><body>
            <article class="item">
                <h3 class="name">Wireless Mouse</h3>
                <div class="cost">$49.99</div>
            </article>
            <article class="item">
                <h3 class="name">Keyboard</h3>
                <div class="cost">$79.99</div>
            </article>
        </body></html>
        ''',
        'page3': '''
        <html><body>
            <section class="product-card">
                <h1>Tablet Device</h1>
                <p class="price-tag">$399.99</p>
            </section>
        </body></html>
        '''
    }
    
    # Create mock Action Sequences
    sequences = [
        create_mock_action_sequence('seq1', '//h2[@class="title"]/text()', 'product titles'),
        create_mock_action_sequence('seq2', '//h3[@class="name"]/text()', 'product titles'),
        create_mock_action_sequence('seq3', '//h1/text() | //h2/text() | //h3/text()', 'product titles'),
    ]
    
    # Run synthesis
    synthesizer = Phase2Synthesis(min_success_threshold=0.5)
    result = synthesizer.synthesize_sequences(sequences, sample_pages)
    
    # Display results
    print("=== Phase 2 Synthesis Results ===")
    print(f"Selected Sequence: {result.selected_sequence.sequence_id}")
    print(f"Robustness Score: {result.robustness_score:.3f}")
    print(f"Selection Rationale: {result.selection_rationale}")
    
    print("\n=== All Sequence Performance ===")
    for seq_perf in result.all_sequences_performance:
        print(f"Sequence {seq_perf['sequence_id']}: "
              f"Rank {seq_perf['rank']}, "
              f"Robustness {seq_perf['robustness_score']:.3f}, "
              f"Success Rate {seq_perf['success_rate']:.1%}")
    
    print(f"\n=== Cross-Validation Summary ===")
    cv_summary = result.cross_validation_summary
    print(f"Pages Successful: {cv_summary['pages_successful']}")
    print(f"Pages Failed: {cv_summary['pages_failed']}")
    print(f"Average Data Count: {cv_summary['average_data_count']:.1f}")