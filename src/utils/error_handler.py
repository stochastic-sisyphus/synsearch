
import sys
import logging
import traceback
from functools import wraps
from typing import Callable, Any, Optional
from pathlib import Path

class GlobalErrorHandler:
    """Global error handler for consistent error management"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def handle_error(self, error: Exception, context: str = "") -> None:
        """Handle errors with consistent formatting and logging"""
        error_type = type(error).__name__
        error_msg = str(error)
        trace = traceback.format_exc()
        
        self.logger.error(
            f"Error in {context}\n"
            f"Type: {error_type}\n"
            f"Message: {error_msg}\n"
            f"Trace:\n{trace}"
        )

def with_error_handling(func: Callable) -> Callable:
    """Decorator for consistent error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            handler = GlobalErrorHandler()
            handler.handle_error(e, func.__name__)
            raise
    return wrapper