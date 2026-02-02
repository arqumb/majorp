"""
Application Configuration and Settings

This module provides centralized configuration management for the AutoScraper system.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ScrapingConfig:
    """Configuration for scraping behavior."""
    max_iterations: int = 5
    min_confidence: float = 0.7
    timeout_ms: int = 5000
    max_results_per_xpath: int = 1000
    enable_structuring: bool = True
    normalize_whitespace: bool = True


@dataclass
class LoaderConfig:
    """Configuration for page loading."""
    static_timeout: int = 10
    dynamic_timeout: int = 30
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    enable_javascript: bool = True
    wait_for_selector: Optional[str] = None


@dataclass
class StorageConfig:
    """Configuration for data storage."""
    db_path: str = "scraper_rules.db"
    cache_enabled: bool = False
    max_cache_size: int = 100


class Settings:
    """Main settings class for AutoScraper application."""
    
    def __init__(self):
        """Initialize settings with defaults and environment overrides."""
        self.scraping = ScrapingConfig()
        self.loader = LoaderConfig()
        self.storage = StorageConfig()
        
        # Load from environment variables if available
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Scraping settings
        if os.getenv('AUTOSCRAPER_MAX_ITERATIONS'):
            self.scraping.max_iterations = int(os.getenv('AUTOSCRAPER_MAX_ITERATIONS'))
        
        if os.getenv('AUTOSCRAPER_MIN_CONFIDENCE'):
            self.scraping.min_confidence = float(os.getenv('AUTOSCRAPER_MIN_CONFIDENCE'))
        
        if os.getenv('AUTOSCRAPER_TIMEOUT_MS'):
            self.scraping.timeout_ms = int(os.getenv('AUTOSCRAPER_TIMEOUT_MS'))
        
        # Loader settings
        if os.getenv('AUTOSCRAPER_STATIC_TIMEOUT'):
            self.loader.static_timeout = int(os.getenv('AUTOSCRAPER_STATIC_TIMEOUT'))
        
        if os.getenv('AUTOSCRAPER_DYNAMIC_TIMEOUT'):
            self.loader.dynamic_timeout = int(os.getenv('AUTOSCRAPER_DYNAMIC_TIMEOUT'))
        
        if os.getenv('AUTOSCRAPER_USER_AGENT'):
            self.loader.user_agent = os.getenv('AUTOSCRAPER_USER_AGENT')
        
        # Storage settings
        if os.getenv('AUTOSCRAPER_DB_PATH'):
            self.storage.db_path = os.getenv('AUTOSCRAPER_DB_PATH')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary format."""
        return {
            'scraping': {
                'max_iterations': self.scraping.max_iterations,
                'min_confidence': self.scraping.min_confidence,
                'timeout_ms': self.scraping.timeout_ms,
                'max_results_per_xpath': self.scraping.max_results_per_xpath,
                'enable_structuring': self.scraping.enable_structuring,
                'normalize_whitespace': self.scraping.normalize_whitespace
            },
            'loader': {
                'static_timeout': self.loader.static_timeout,
                'dynamic_timeout': self.loader.dynamic_timeout,
                'user_agent': self.loader.user_agent,
                'enable_javascript': self.loader.enable_javascript,
                'wait_for_selector': self.loader.wait_for_selector
            },
            'storage': {
                'db_path': self.storage.db_path,
                'cache_enabled': self.storage.cache_enabled,
                'max_cache_size': self.storage.max_cache_size
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Settings':
        """Create Settings instance from dictionary."""
        settings = cls()
        
        if 'scraping' in config_dict:
            scraping_config = config_dict['scraping']
            settings.scraping = ScrapingConfig(**scraping_config)
        
        if 'loader' in config_dict:
            loader_config = config_dict['loader']
            settings.loader = LoaderConfig(**loader_config)
        
        if 'storage' in config_dict:
            storage_config = config_dict['storage']
            settings.storage = StorageConfig(**storage_config)
        
        return settings


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def update_settings(**kwargs) -> None:
    """Update global settings with provided values."""
    global settings
    
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)


# Environment variable documentation
ENV_VARS_HELP = """
AutoScraper Environment Variables:

Scraping Configuration:
- AUTOSCRAPER_MAX_ITERATIONS: Maximum refinement iterations (default: 5)
- AUTOSCRAPER_MIN_CONFIDENCE: Minimum confidence threshold (default: 0.7)
- AUTOSCRAPER_TIMEOUT_MS: XPath execution timeout in ms (default: 5000)

Loader Configuration:
- AUTOSCRAPER_STATIC_TIMEOUT: Static page load timeout (default: 10)
- AUTOSCRAPER_DYNAMIC_TIMEOUT: Dynamic page load timeout (default: 30)
- AUTOSCRAPER_USER_AGENT: Custom user agent string

Storage Configuration:
- AUTOSCRAPER_DB_PATH: Database file path (default: scraper_rules.db)
"""