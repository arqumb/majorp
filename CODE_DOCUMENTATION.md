# 📚 AutoScraper AI - Code Documentation

Complete overview of every Python file in the project, explaining what each file does, its key functions, and how it fits into the overall system.

---

## 📁 Project Structure Overview

```
autoscraper_ai/
├── autoscraper/          # Core AUTOSCRAPER algorithm
├── config/               # Configuration and logging
├── executor/             # XPath execution engine
├── input/                # URL handling and validation
├── loader/               # Page loading (static + dynamic)
├── preprocessing/        # HTML cleaning and analysis
├── storage/              # Rule persistence
└── output/               # Output handling
```

---

## 🎯 Entry Point

### `main.py`
**Location:** Root directory  
**Purpose:** Main entry point for the AutoScraper system

**What it does:**
- Provides command-line interface (CLI) for the scraper
- Orchestrates the entire scraping workflow
- Handles user input (URL, task, options)
- Manages the scraping pipeline from start to finish
- Outputs results in JSON format

**Key Classes:**
- `AutoScraperSystem` - Main orchestrator class

**Key Methods:**
- `scrape()` - Main scraping method that coordinates all components
- `_execute_existing_rules()` - Uses cached rules for faster scraping
- `_generate_new_rule()` - Creates new extraction rules using AUTOSCRAPER
- `_create_success_result()` - Formats successful scraping results
- `_create_error_result()` - Handles and formats errors

**Usage Flow:**
1. Parse command-line arguments (URL, task, options)
2. Check for existing cached rules
3. If no rules exist, generate new ones using Phase 1 & 2
4. Execute extraction rules
5. Structure and format output
6. Save results to file or print to console

**Example:**
```python
# Command line usage
python main.py "https://example.com" "extract titles" --output results.json

# Internal flow
scraper = AutoScraperSystem()
result = scraper.scrape(url, task)
```

---

## 🧠 Core AUTOSCRAPER Algorithm

### `autoscraper/phase1_progressive.py`
**Purpose:** Implements Phase 1 of AUTOSCRAPER - Progressive XPath generation with iterative refinement

**What it does:**
- Analyzes HTML structure and extraction requirements
- Generates initial XPath suggestions using pattern recognition
- Tests XPath expressions against the HTML
- Applies step-back logic when extraction fails
- Iteratively refines XPath until successful extraction
- Builds ordered Action Sequence of extraction steps

**Key Classes:**
- `Phase1Progressive` - Main Phase 1 processor
- `ExtractionAction` - Represents a single extraction action
- `ActionType` - Enum for action types (XPATH_EXTRACT, STEP_BACK, REFINE_XPATH)

**Key Methods:**
- `process_extraction_task()` - Main entry point for Phase 1
- `_analyze_html_structure()` - Analyzes DOM structure for context
- `_generate_initial_xpath()` - Creates first XPath suggestion
- `_execute_progressive_refinement()` - Iterative refinement loop
- `_execute_xpath()` - Tests XPath against HTML
- `_evaluate_extraction_result()` - Scores extraction quality
- `_is_single_element_weak_candidate()` - Detects weak single-element results
- `_apply_step_back_logic()` - Moves to parent elements on failure
- `_call_llm_for_xpath()` - Placeholder for AI-powered XPath generation

**Algorithm Flow:**
1. Parse and analyze HTML structure
2. Generate initial XPath based on task keywords
3. Execute XPath and extract data
4. Evaluate results (confidence scoring)
5. If confidence is low, apply step-back or refinement
6. Repeat until success or max iterations reached
7. Return best Action Sequence found

**Configuration:**
- `max_iterations` - Maximum refinement attempts (default: 5)
- `min_confidence` - Minimum confidence threshold (default: 0.7)

**Example:**
```python
phase1 = Phase1Progressive(max_iterations=5, min_confidence=0.7)
result = phase1.process_extraction_task(html, "extract product prices")
# Returns: {'success': True, 'action_sequence': [...], 'confidence_score': 0.85}
```

---

### `autoscraper/phase2_synthesis.py`
**Purpose:** Implements Phase 2 of AUTOSCRAPER - Cross-page validation and rule synthesis

**What it does:**
- Tests Action Sequences across multiple similar pages
- Validates rule robustness and reliability
- Calculates success rates for each sequence
- Selects the most reusable extraction pattern
- Ensures rules work across different page variations

**Key Classes:**
- `Phase2Synthesis` - Main Phase 2 processor
- `SynthesisResult` - Stores validation results

**Key Methods:**
- `synthesize_rules()` - Main synthesis method
- `_test_sequence_on_page()` - Tests sequence on single page
- `_calculate_robustness_score()` - Calculates reliability metric
- `_select_best_sequence()` - Chooses most robust sequence

**Algorithm Flow:**
1. Receive multiple Action Sequences from Phase 1
2. Load multiple similar pages for testing
3. Execute each sequence on each page
4. Calculate success rate for each sequence
5. Rank sequences by robustness score
6. Return best performing sequence

**Robustness Scoring:**
- Success rate across pages (0.0 - 1.0)
- Data consistency check
- Element count stability
- Pattern reliability

**Example:**
```python
phase2 = Phase2Synthesis()
best_sequence = phase2.synthesize_rules(
    action_sequences=[seq1, seq2, seq3],
    test_urls=["url1", "url2", "url3"]
)
# Returns: Most robust Action Sequence
```

---

### `autoscraper/ai_assistant.py`
**Purpose:** AI-powered helper for rule generation and optimization (placeholder for future LLM integration)

**What it does:**
- Provides intelligent XPath suggestions
- Optimizes existing XPath expressions
- Diagnoses extraction issues
- Detects content patterns
- Currently uses rule-based fallbacks (ready for LLM integration)

**Key Classes:**
- `AIAssistant` - Main AI helper class

**Key Methods:**
- `generate_xpath_suggestions()` - Suggests XPath expressions
- `optimize_xpath()` - Improves existing XPath
- `diagnose_extraction_issues()` - Identifies problems
- `suggest_content_patterns()` - Finds data patterns
- `_fallback_xpath_suggestions()` - Rule-based suggestions (current)

**Future Integration Points:**
- `configure_openai()` - OpenAI GPT integration
- `configure_anthropic()` - Anthropic Claude integration
- `configure_local_model()` - Local LLM integration

**Current Implementation:**
- Uses keyword-based pattern matching
- Provides reasonable XPath suggestions
- Ready for AI model integration

**Example:**
```python
assistant = AIAssistant()
suggestions = assistant.generate_xpath_suggestions(html, "extract prices")
# Returns: [{'xpath': '//span[@class="price"]', 'confidence': 0.8}]
```

---

## 🌐 Page Loading

### `loader/page_loader.py`
**Purpose:** Main page loader that handles both static and dynamic websites

**What it does:**
- Attempts static loading first (faster)
- Detects JavaScript-rendered content
- Automatically falls back to dynamic loading if needed
- Handles errors gracefully
- Returns final HTML content

**Key Classes:**
- `PageLoader` - Main loader class

**Key Methods:**
- `load_page()` - Main loading method with auto-detection
- `_load_static()` - Loads page using requests library
- `_is_dynamic_content()` - Detects if page needs JavaScript
- `_load_dynamic()` - Falls back to Playwright for dynamic content

**Detection Logic:**
- Checks for minimal HTML content
- Looks for JavaScript framework indicators
- Detects empty body tags
- Identifies lazy-loaded content

**Example:**
```python
loader = PageLoader()
html = loader.load_page("https://example.com")
# Automatically chooses static or dynamic loading
```

---

### `loader/dynamic_loader.py`
**Purpose:** Handles JavaScript-rendered websites using Playwright

**What it does:**
- Launches headless browser (Chromium)
- Renders JavaScript content
- Handles lazy-loaded elements
- Performs controlled scrolling for infinite scroll
- Waits for dynamic content to load
- Returns fully-rendered HTML

**Key Classes:**
- `DynamicLoader` - Playwright-based loader

**Key Methods:**
- `load_page()` - Loads page with Playwright
- `_scroll_page()` - Performs controlled scrolling
- `_wait_for_content()` - Waits for dynamic elements

**Features:**
- Headless browser execution
- Automatic wait for network idle
- Controlled scrolling for lazy-load
- Error handling for missing Playwright

**Configuration:**
- `timeout` - Page load timeout (default: 30s)
- `wait_for_selector` - Optional element to wait for
- `scroll_count` - Number of scroll iterations

**Example:**
```python
loader = DynamicLoader(timeout=30000)
html = loader.load_page("https://dynamic-site.com")
# Returns fully-rendered HTML with JavaScript content
```

---

## 🧹 HTML Preprocessing

### `preprocessing/html_preprocessor.py`
**Purpose:** Cleans and normalizes HTML while preserving DOM structure

**What it does:**
- Removes script and style tags
- Cleans up whitespace
- Preserves class attributes and DOM hierarchy
- Normalizes HTML structure
- Prepares HTML for XPath extraction

**Key Classes:**
- `HTMLPreprocessor` - Main preprocessing class

**Key Methods:**
- `clean_html()` - Main cleaning method
- `_remove_unwanted_tags()` - Removes scripts, styles, etc.
- `_normalize_whitespace()` - Cleans up spacing
- `_preserve_structure()` - Maintains DOM hierarchy

**Cleaning Process:**
1. Parse HTML with BeautifulSoup
2. Remove script, style, and comment tags
3. Normalize whitespace in text nodes
4. Preserve class and id attributes
5. Maintain parent-child relationships
6. Return cleaned HTML string

**Example:**
```python
preprocessor = HTMLPreprocessor()
cleaned = preprocessor.clean_html(raw_html)
# Returns: Clean HTML ready for XPath extraction
```

---

### `preprocessing/dom_analyzer.py`
**Purpose:** Advanced DOM structure analysis for better scraping insights

**What it does:**
- Analyzes HTML structure and patterns
- Identifies repeated elements (potential data containers)
- Detects semantic class names
- Finds content patterns (prices, dates, emails)
- Provides insights for XPath generation

**Key Classes:**
- `DOMAnalyzer` - Main analysis class

**Key Methods:**
- `analyze_structure()` - Comprehensive DOM analysis
- `_get_basic_stats()` - Element counts and depth
- `_analyze_element_distribution()` - Tag frequency
- `_analyze_class_patterns()` - CSS class analysis
- `_find_semantic_classes()` - Identifies meaningful classes
- `_analyze_content_patterns()` - Detects data patterns
- `_find_repetitive_structures()` - Finds repeated containers
- `_find_price_patterns()` - Detects price-like text
- `_find_date_patterns()` - Detects date formats

**Analysis Output:**
```python
{
  'basic_stats': {'total_elements': 1234, 'max_depth': 12},
  'element_distribution': {'div': 456, 'span': 234},
  'class_patterns': {'most_common_classes': {'product': 50}},
  'content_patterns': {'prices': ['$99.99', '$149.99']},
  'repetitive_structures': {'product-card': {'count': 20}}
}
```

**Use Cases:**
- Understanding page structure
- Identifying data containers
- Improving XPath generation
- Debugging extraction issues

**Example:**
```python
analyzer = DOMAnalyzer()
analysis = analyzer.analyze_structure(html)
# Returns: Comprehensive DOM analysis report
```

---

## ⚙️ XPath Execution

### `executor/xpath_executor.py`
**Purpose:** Robust XPath execution engine with DOM-based structuring

**What it does:**
- Executes XPath expressions against HTML
- Handles various XPath result types
- Groups extracted data by parent containers
- Infers semantic field names (title, price, etc.)
- Creates structured objects from flat data
- Provides detailed execution metrics

**Key Classes:**
- `XPathExecutor` - Main execution engine
- `ExecutionStep` - Single XPath execution result
- `ExecutionResult` - Complete execution summary
- `ExecutionStatus` - Status enum (SUCCESS, FAILED, PARTIAL)

**Key Methods:**
- `execute_action_sequence()` - Executes full Action Sequence
- `_execute_single_xpath()` - Executes one XPath expression
- `_extract_text_from_elements()` - Extracts text content
- `_apply_dom_structuring()` - Groups data into objects
- `_find_source_elements()` - Locates elements that produced data
- `_identify_repeated_containers()` - Finds parent containers
- `_group_by_containers()` - Groups elements by parent
- `_create_structured_record()` - Creates structured object
- `_infer_field_name()` - Determines field type (title, price, etc.)
- `_is_navigation_element()` - Filters out UI elements
- `_find_product_containers()` - Finds e-commerce containers
- `_alternative_structuring_approach()` - Fallback structuring
- `_are_elements_related()` - Checks element relationships

**Execution Flow:**
1. Parse HTML into tree structure
2. Execute each XPath in sequence
3. Extract and clean text content
4. Apply DOM-based structuring
5. Group related elements
6. Infer field names
7. Create structured objects
8. Return execution results

**Structuring Logic:**
- Identifies repeated parent containers
- Groups child elements by container
- Infers field types from content and attributes
- Pairs related data (e.g., title with price)
- Filters out navigation/UI elements

**Field Inference:**
- Detects prices (currency symbols, numbers)
- Identifies titles (headings, product names)
- Recognizes brands (known brand names)
- Finds descriptions (longer text content)
- Detects dates, links, images

**Example:**
```python
executor = XPathExecutor(enable_structuring=True)
result = executor.execute_action_sequence(html, action_sequence)

# Returns structured data:
{
  'success': True,
  'final_data': [
    {'title': 'Product 1', 'price': '$99.99'},
    {'title': 'Product 2', 'price': '$149.99'}
  ],
  'execution_steps': [...],
  'metadata': {'success_rate': 1.0}
}
```

---

## 💾 Storage

### `storage/scraper_repository.py`
**Purpose:** Persistent storage for reusable extraction rules

**What it does:**
- Stores Action Sequences in SQLite database
- Indexes rules by domain and task
- Enables rule reuse across scraping sessions
- Provides CRUD operations for rules
- Manages rule metadata and performance metrics

**Key Classes:**
- `ScraperRepository` - Main storage class
- `ActionSequence` - Rule data structure

**Key Methods:**
- `save_sequence()` - Stores new extraction rule
- `fetch_by_domain_and_task()` - Retrieves matching rules
- `fetch_by_id()` - Gets specific rule by ID
- `list_all_sequences()` - Lists all stored rules
- `delete_sequence()` - Removes rule from storage
- `update_sequence()` - Updates existing rule

**Database Schema:**
```sql
CREATE TABLE scraper_sequences (
  sequence_id TEXT PRIMARY KEY,
  domain TEXT,
  extraction_task TEXT,
  xpath_actions TEXT,  -- JSON
  success_rate REAL,
  robustness_score REAL,
  created_at TEXT,
  updated_at TEXT,
  metadata TEXT  -- JSON
)
```

**Storage Benefits:**
- Faster subsequent scrapes (no rule generation)
- Rule reuse across similar pages
- Performance tracking
- Rule versioning
- Domain-specific optimization

**Example:**
```python
repo = ScraperRepository("scraper_rules.db")

# Save rule
sequence = ActionSequence(
  sequence_id="seq_123",
  domain="example.com",
  extraction_task="extract products",
  xpath_actions=[...],
  success_rate=0.95
)
repo.save_sequence(sequence)

# Retrieve rule
rules = repo.fetch_by_domain_and_task("example.com", "extract products")
```

---

## 🔧 Configuration

### `config/settings.py`
**Purpose:** Centralized configuration management with environment variable support

**What it does:**
- Defines default configuration values
- Loads settings from environment variables
- Provides configuration for all system components
- Supports runtime configuration updates

**Key Classes:**
- `ScrapingConfig` - Scraping behavior settings
- `LoaderConfig` - Page loading settings
- `StorageConfig` - Database settings
- `Settings` - Main settings container

**Configuration Categories:**

**Scraping Settings:**
- `max_iterations` - Maximum XPath refinement attempts
- `min_confidence` - Minimum confidence threshold
- `timeout_ms` - XPath execution timeout
- `max_results_per_xpath` - Result limit per XPath
- `enable_structuring` - Enable DOM structuring
- `normalize_whitespace` - Clean whitespace

**Loader Settings:**
- `static_timeout` - Static page load timeout
- `dynamic_timeout` - Dynamic page load timeout
- `user_agent` - HTTP user agent string
- `enable_javascript` - Enable JS rendering
- `wait_for_selector` - Optional element to wait for

**Storage Settings:**
- `db_path` - Database file location
- `cache_enabled` - Enable caching
- `max_cache_size` - Cache size limit

**Environment Variables:**
```bash
AUTOSCRAPER_MAX_ITERATIONS=5
AUTOSCRAPER_MIN_CONFIDENCE=0.7
AUTOSCRAPER_TIMEOUT_MS=5000
AUTOSCRAPER_STATIC_TIMEOUT=10
AUTOSCRAPER_DYNAMIC_TIMEOUT=30
AUTOSCRAPER_DB_PATH=scraper_rules.db
```

**Example:**
```python
from autoscraper_ai.config.settings import get_settings

settings = get_settings()
print(settings.scraping.max_iterations)  # 5
print(settings.loader.static_timeout)    # 10
```

---

### `config/logger.py`
**Purpose:** Centralized logging configuration with colored output

**What it does:**
- Sets up logging for entire application
- Provides colored console output
- Supports file logging
- Offers convenience logging functions
- Includes execution time decorator

**Key Classes:**
- `ColoredFormatter` - Custom formatter with colors

**Key Functions:**
- `setup_logging()` - Configures logging system
- `get_logger()` - Gets module-specific logger
- `log_execution_time()` - Decorator for timing functions
- `log_scraping_session()` - Logs scraping summary

**Log Levels:**
- DEBUG (Cyan) - Detailed debugging info
- INFO (Green) - General information
- WARNING (Yellow) - Warning messages
- ERROR (Red) - Error messages
- CRITICAL (Magenta) - Critical failures

**Features:**
- Colored console output (if terminal supports it)
- File logging with rotation
- Module-specific loggers
- Execution time tracking
- Session summaries

**Example:**
```python
from autoscraper_ai.config.logger import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO", log_file="scraper.log")

# Get logger
logger = get_logger(__name__)
logger.info("Starting scraping process")
logger.error("Failed to load page")

# Use decorator
@log_execution_time
def scrape_page(url):
    # Function automatically logs execution time
    pass
```

---

## 🔗 Input Handling

### `input/url_handler.py`
**Purpose:** URL validation, normalization, and security checks

**What it does:**
- Validates URL format and structure
- Normalizes URLs for consistency
- Blocks private/local IP addresses
- Extracts domain information
- Builds absolute URLs from relative paths
- Provides robots.txt URL generation

**Key Classes:**
- `URLHandler` - Main URL processing class

**Key Methods:**
- `validate_url()` - Comprehensive URL validation
- `_normalize_url()` - Standardizes URL format
- `_extract_url_metadata()` - Extracts URL components
- `build_absolute_url()` - Converts relative to absolute URLs
- `extract_domain()` - Gets domain from URL
- `is_same_domain()` - Checks if URLs share domain
- `get_robots_txt_url()` - Generates robots.txt URL

**Validation Checks:**
- URL format (regex pattern)
- Scheme (http/https only)
- Domain presence
- Blocked patterns (localhost, private IPs)
- Security restrictions

**Blocked Patterns:**
- localhost
- 127.0.0.1
- 0.0.0.0
- Private IP ranges (192.168.x.x, 10.x.x.x, 172.16-31.x.x)

**Normalization:**
- Lowercase scheme and domain
- Remove default ports (80, 443)
- Normalize path (remove trailing slashes)
- URL encode special characters
- Remove fragments (#)

**Metadata Extraction:**
```python
{
  'scheme': 'https',
  'domain': 'example.com',
  'path': '/products',
  'has_query': True,
  'is_secure': True,
  'port': None,
  'site_type': 'ecommerce'
}
```

**Example:**
```python
handler = URLHandler()

# Validate URL
result = handler.validate_url("https://example.com/products")
# Returns: {'valid': True, 'normalized': 'https://example.com/products', ...}

# Build absolute URL
absolute = handler.build_absolute_url(
  "https://example.com/products/",
  "../about.html"
)
# Returns: "https://example.com/about.html"

# Extract domain
domain = handler.extract_domain("https://shop.example.com/items")
# Returns: "shop.example.com"
```

---

## 📦 Module Initialization Files

### `__init__.py` files
**Purpose:** Python package initialization and exports

**Locations:**
- `autoscraper_ai/__init__.py`
- `autoscraper_ai/autoscraper/__init__.py`
- `autoscraper_ai/config/__init__.py`
- `autoscraper_ai/executor/__init__.py`
- `autoscraper_ai/input/__init__.py`
- `autoscraper_ai/loader/__init__.py`
- `autoscraper_ai/output/__init__.py`
- `autoscraper_ai/preprocessing/__init__.py`
- `autoscraper_ai/storage/__init__.py`

**What they do:**
- Mark directories as Python packages
- Define package exports
- Enable clean imports
- Provide package-level documentation

**Example:**
```python
# autoscraper_ai/autoscraper/__init__.py
from .phase1_progressive import Phase1Progressive
from .phase2_synthesis import Phase2Synthesis

__all__ = ['Phase1Progressive', 'Phase2Synthesis']
```

---

## 🔄 Data Flow Through the System

```
1. User Input (main.py)
   ↓
2. URL Validation (input/url_handler.py)
   ↓
3. Page Loading (loader/page_loader.py → loader/dynamic_loader.py)
   ↓
4. HTML Preprocessing (preprocessing/html_preprocessor.py)
   ↓
5. DOM Analysis (preprocessing/dom_analyzer.py)
   ↓
6. Phase 1: XPath Generation (autoscraper/phase1_progressive.py)
   ↓
7. Phase 2: Cross-Validation (autoscraper/phase2_synthesis.py)
   ↓
8. Rule Storage (storage/scraper_repository.py)
   ↓
9. XPath Execution (executor/xpath_executor.py)
   ↓
10. Structured Output (main.py)
```

---

## 🎯 Key Design Patterns

### 1. **Factory Pattern**
- `loader/page_loader.py` - Chooses static or dynamic loader

### 2. **Strategy Pattern**
- `executor/xpath_executor.py` - Different extraction strategies

### 3. **Repository Pattern**
- `storage/scraper_repository.py` - Data access abstraction

### 4. **Builder Pattern**
- `autoscraper/phase1_progressive.py` - Builds Action Sequences

### 5. **Decorator Pattern**
- `config/logger.py` - Execution time logging

---

## 📊 Performance Considerations

### Caching
- **Rule Storage**: Reuses successful extraction patterns
- **First Run**: Slower (generates rules)
- **Subsequent Runs**: Faster (uses cached rules)

### Optimization
- **Static First**: Tries fast static loading before dynamic
- **Lazy Loading**: Only loads Playwright when needed
- **Minimal Iterations**: Stops when confidence threshold met
- **Efficient XPath**: Optimizes XPath expressions

### Resource Usage
- **Memory**: ~100-200MB typical usage
- **Disk**: ~300MB for Playwright browsers
- **Network**: Depends on target website
- **CPU**: Moderate during XPath generation

---

## 🔍 Debugging Tips

### Enable Debug Logging
```python
from autoscraper_ai.config.logger import setup_logging
setup_logging(level="DEBUG", log_file="debug.log")
```

### Test Individual Components
```python
# Test page loader
from autoscraper_ai.loader.page_loader import PageLoader
loader = PageLoader()
html = loader.load_page("https://example.com")

# Test XPath executor
from autoscraper_ai.executor.xpath_executor import XPathExecutor
executor = XPathExecutor()
result = executor.execute_single_xpath(html, "//h1/text()")

# Test DOM analyzer
from autoscraper_ai.preprocessing.dom_analyzer import analyze_dom
analysis = analyze_dom(html)
```

### Check Stored Rules
```python
from autoscraper_ai.storage.scraper_repository import ScraperRepository
repo = ScraperRepository()
rules = repo.list_all_sequences()
print(rules)
```

---

## 🚀 Extension Points

### Adding New Features

**1. Custom XPath Strategies:**
- Extend `phase1_progressive.py`
- Add new pattern recognition logic

**2. Additional Loaders:**
- Create new loader in `loader/` directory
- Implement `load_page()` method

**3. Export Formats:**
- Add exporters in `output/` directory
- Support CSV, Excel, XML, etc.

**4. AI Integration:**
- Implement LLM calls in `ai_assistant.py`
- Add OpenAI/Anthropic/local model support

**5. Advanced Structuring:**
- Enhance `xpath_executor.py`
- Add machine learning for field inference

---

## 📚 Further Reading

- **AUTOSCRAPER Research**: Original algorithm papers
- **XPath Tutorial**: W3Schools XPath guide
- **Playwright Docs**: https://playwright.dev/python/
- **BeautifulSoup Docs**: https://www.crummy.com/software/BeautifulSoup/
- **lxml Documentation**: https://lxml.de/

---

**This documentation covers all Python files in the AutoScraper AI project. Each file has a specific purpose and works together to create a robust, intelligent web scraping system.** 🚀