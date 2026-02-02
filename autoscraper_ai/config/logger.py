"""
Logging Configuration and Utilities

This module provides centralized logging setup for the AutoScraper system.
"""

import logging
import sys
from typing import Optional
from datetime import datetime
import os


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        if hasattr(record, 'levelname') and record.levelname in self.COLORS:
            # Add color to level name
            colored_levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            record.levelname = colored_levelname
        
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_colors: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for AutoScraper.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        enable_colors: Whether to enable colored console output
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create root logger
    logger = logging.getLogger('autoscraper')
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if enable_colors and sys.stdout.isatty():
        console_formatter = ColoredFormatter(format_string)
    else:
        console_formatter = logging.Formatter(format_string)
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        
        # File logs don't need colors
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'autoscraper.{name}')


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_execution_time
        def my_function():
            pass
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


def log_scraping_session(url: str, task: str, success: bool, 
                        execution_time: float, data_count: int = 0):
    """
    Log a scraping session summary.
    
    Args:
        url: Target URL
        task: Extraction task description
        success: Whether scraping was successful
        execution_time: Total execution time in seconds
        data_count: Number of items extracted
    """
    logger = get_logger('session')
    
    status = "SUCCESS" if success else "FAILED"
    message = f"Scraping {status} | URL: {url} | Task: {task} | Time: {execution_time:.2f}s"
    
    if success and data_count > 0:
        message += f" | Items: {data_count}"
    
    if success:
        logger.info(message)
    else:
        logger.warning(message)


# Default logger setup
_default_logger = None


def get_default_logger() -> logging.Logger:
    """Get the default AutoScraper logger."""
    global _default_logger
    
    if _default_logger is None:
        _default_logger = setup_logging()
    
    return _default_logger


# Convenience functions for common log levels
def debug(message: str, logger_name: str = 'main'):
    """Log debug message."""
    get_logger(logger_name).debug(message)


def info(message: str, logger_name: str = 'main'):
    """Log info message."""
    get_logger(logger_name).info(message)


def warning(message: str, logger_name: str = 'main'):
    """Log warning message."""
    get_logger(logger_name).warning(message)


def error(message: str, logger_name: str = 'main'):
    """Log error message."""
    get_logger(logger_name).error(message)


def critical(message: str, logger_name: str = 'main'):
    """Log critical message."""
    get_logger(logger_name).critical(message)


# Example usage and testing
if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging(level="DEBUG", enable_colors=True)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test module logger
    module_logger = get_logger(__name__)
    module_logger.info("Module-specific logger test")
    
    # Test session logging
    log_scraping_session(
        url="https://example.com",
        task="test extraction",
        success=True,
        execution_time=2.5,
        data_count=10
    )