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
        """
        Handle errors with consistent formatting and logging.

        Args:
            error (Exception): The exception to handle.
            context (str, optional): Additional context about where the error occurred.
        """
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
    """
    Decorator for consistent error handling.

    Args:
        func (Callable): The function to wrap with error handling.

    Returns:
        Callable: The wrapped function with error handling.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            handler = GlobalErrorHandler()
            handler.handle_error(e, func.__name__)
            raise
    return wrapper

def validate_method_implementation(obj: Any, method_name: str) -> bool:
    """
    Validate if a method is implemented in the given object.

    Args:
        obj (Any): The object to check.
        method_name (str): The name of the method to validate.

    Returns:
        bool: True if the method is implemented, False otherwise.
    """
    return callable(getattr(obj, method_name, None))

def log_method_call(logger: logging.Logger, method_name: str, *args, **kwargs) -> None:
    """
    Log the method call with parameters.

    Args:
        logger (logging.Logger): The logger instance.
        method_name (str): The name of the method being called.
        *args: Positional arguments.
        **kwargs: Keyword arguments.
    """
    logger.info(f"Calling method: {method_name}")
    logger.debug(f"Parameters: args={args}, kwargs={kwargs}")

def log_intermediate_result(logger: logging.Logger, result: Any, description: str = "") -> None:
    """
    Log intermediate results.

    Args:
        logger (logging.Logger): The logger instance.
        result (Any): The result to log.
        description (str, optional): Additional description of the result.
    """
    logger.info(f"Intermediate result: {description}")
    logger.debug(f"Result: {result}")

def ensure_structural_correctness(data: Any, expected_type: type, description: str = "") -> None:
    """
    Ensure the structural correctness of the input data.

    Args:
        data (Any): The data to validate.
        expected_type (type): The expected type of the data.
        description (str, optional): Additional description of the data.
    """
    if not isinstance(data, expected_type):
        raise ValueError(f"Incorrect data structure for {description}. Expected {expected_type}, got {type(data)}")
