"""
Main Entry Point for Intelligent Web Scraping System
"""

import json
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import our modules
from autoscraper_ai.loader.page_loader import PageLoader
from autoscraper_ai.preprocessing.html_preprocessor import HTMLPreprocessor
from autoscraper_ai.autoscraper.phase1_progressive import Phase1Progressive
from autoscraper_ai.autoscraper.phase2_synthesis import Phase2Synthesis
from autoscraper_ai.storage.scraper_repository import ScraperRepository, ActionSequence, create_sequence_id, extract_domain_from_url
from autoscraper_ai.executor.xpath_executor import XPathExecutor


class AutoScraperSystem:
    """Main orchestrator for the intelligent web scraping system."""
    
    def __init__(self, db_path: str = "scraper_rules.db"):
        """Initialize the AutoScraper system with all components."""
        self.page_loader = PageLoader()
        self.html_preprocessor = HTMLPreprocessor()
        self.phase1_processor = Phase1Progressive()
        self.phase2_processor = Phase2Synthesis()
        self.repository = ScraperRepository(db_path)
        self.xpath_executor = XPathExecutor(normalize_whitespace=True)
    
    def scrape(self, url: str, task: str, force_new_rule: bool = False) -> Dict[str, Any]:
        """
        Main scraping method that orchestrates the entire process.
        
        Args:
            url: Target URL to scrape
            task: Natural language description of extraction task
            force_new_rule: If True, generate new rule even if existing one exists
            
        Returns:
            Dictionary with scraping results and metadata
        """
        start_time = datetime.now()
        domain = extract_domain_from_url(url)
        
        try:
            # Step 1: Check for existing rules
            existing_sequences = []
            if not force_new_rule:
                existing_sequences = self.repository.fetch_by_domain_and_task(domain, task, limit=3)
                
                if existing_sequences:
                    return self._execute_existing_rules(url, existing_sequences, start_time)
            
            # Step 2: Load and preprocess page
            html_content = self.page_loader.load_page(url)
            cleaned_html = self.html_preprocessor.clean_html(html_content)
            
            # Step 3: Generate new rules using AUTOSCRAPER
            new_rule = self._generate_new_rule(cleaned_html, task, domain, url)
            
            if not new_rule:
                return self._create_error_result("Failed to generate extraction rule", start_time)
            
            # Step 4: Execute the new rule
            execution_result = self.xpath_executor.execute_action_sequence(
                cleaned_html, new_rule.xpath_actions
            )
            
            # Step 5: Store the rule if successful
            if execution_result.success and execution_result.final_data:
                self.repository.save_sequence(new_rule)
            
            return self._create_success_result(
                execution_result.final_data,
                new_rule,
                execution_result,
                start_time,
                is_new_rule=True
            )
            
        except Exception as e:
            return self._create_error_result(f"Scraping error: {str(e)}", start_time)
    
    def _execute_existing_rules(self, url: str, sequences: List[ActionSequence], 
                               start_time: datetime) -> Dict[str, Any]:
        """Execute existing rules against the target URL."""
        
        # Load and preprocess page
        html_content = self.page_loader.load_page(url)
        cleaned_html = self.html_preprocessor.clean_html(html_content)
        
        best_result = None
        best_score = 0.0
        
        # Try each existing sequence
        for sequence in sequences:
            execution_result = self.xpath_executor.execute_action_sequence(
                cleaned_html, sequence.xpath_actions
            )
            
            if execution_result.success and execution_result.final_data:
                # Score based on success rate and data count
                score = (execution_result.successful_steps / execution_result.total_steps) * len(execution_result.final_data)
                
                if score > best_score:
                    best_score = score
                    best_result = (sequence, execution_result)
        
        if best_result:
            sequence, execution_result = best_result
            return self._create_success_result(
                execution_result.final_data,
                sequence,
                execution_result,
                start_time,
                is_new_rule=False
            )
        else:
            return self._create_error_result("No existing rules produced results", start_time)
    
    def _generate_new_rule(self, cleaned_html: str, task: str, domain: str, url: str) -> Optional[ActionSequence]:
        """Generate a new extraction rule using AUTOSCRAPER phases."""
        
        # Phase 1: Generate initial action sequence
        phase1_result = self.phase1_processor.process_extraction_task(cleaned_html, task)
        
        if not phase1_result['success']:
            return None
        
        # Convert Phase 1 action sequence to storage format
        xpath_actions = []
        for action in phase1_result['action_sequence']:
            xpath_actions.append({
                'action_type': action.action_type.value,
                'xpath': action.xpath,
                'description': action.description,
                'success': action.success,
                'confidence_score': action.confidence_score
            })
        
        # Create ActionSequence for storage
        sequence_id = create_sequence_id(domain, task)
        confidence = phase1_result['confidence_score']
        
        return ActionSequence(
            sequence_id=sequence_id,
            domain=domain,
            extraction_task=task,
            xpath_actions=xpath_actions,
            success_rate=confidence,
            robustness_score=confidence,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata={
                'source_url': url,
                'phase1_iterations': phase1_result.get('total_iterations', 0),
                'initial_data_count': len(phase1_result['extracted_data']),
                'generation_method': 'phase1_direct'
            }
        )
    
    def _create_success_result(self, data: List[str], sequence: ActionSequence,
                              execution_result, start_time: datetime, is_new_rule: bool) -> Dict[str, Any]:
        """Create a successful scraping result."""
        end_time = datetime.now()
        
        return {
            'success': True,
            'data': data,
            'metadata': {
                'extraction_task': sequence.extraction_task,
                'domain': sequence.domain,
                'sequence_id': sequence.sequence_id,
                'is_new_rule': is_new_rule,
                'data_count': len(data),
                'execution_time_seconds': (end_time - start_time).total_seconds(),
                'success_rate': execution_result.successful_steps / execution_result.total_steps,
                'robustness_score': sequence.robustness_score,
                'timestamp': end_time.isoformat()
            }
        }
    
    def _create_error_result(self, error_message: str, start_time: datetime) -> Dict[str, Any]:
        """Create an error result."""
        end_time = datetime.now()
        
        return {
            'success': False,
            'data': [],
            'error': error_message,
            'metadata': {
                'execution_time_seconds': (end_time - start_time).total_seconds(),
                'timestamp': end_time.isoformat()
            }
        }


def main():
    """Simple CLI interface for the AutoScraper system."""
    
    if len(sys.argv) < 3:
        print("Usage: python main.py <url> <task> [--force-new] [--output file.json]")
        print("Example: python main.py 'https://example.com' 'extract product titles'")
        sys.exit(1)
    
    url = sys.argv[1]
    task = sys.argv[2]
    force_new = '--force-new' in sys.argv
    
    # Check for output file
    output_file = None
    if '--output' in sys.argv:
        try:
            output_index = sys.argv.index('--output')
            if output_index + 1 < len(sys.argv):
                output_file = sys.argv[output_index + 1]
        except (ValueError, IndexError):
            pass
    
    # Initialize system and perform scraping
    scraper = AutoScraperSystem()
    
    try:
        result = scraper.scrape(url, task, force_new_rule=force_new)
        
        # Output results as JSON
        json_output = json.dumps(result, indent=2, ensure_ascii=False)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"Results saved to {output_file}")
        else:
            print(json_output)
        
        # Exit with appropriate code
        sys.exit(0 if result.get('success', False) else 1)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        error_result = {
            'success': False,
            'error': f"System error: {str(e)}",
            'timestamp': datetime.now().isoformat()
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()