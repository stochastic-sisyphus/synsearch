import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict
import json
from datetime import datetime
import sys

def setup_logging(log_file: str = None):
    """
    Configure logging to both file and console.

    Args:
        log_file (str, optional): Path to the log file. Defaults to None.

    Returns:
        logging.Logger: Configured logger instance.
    """
    try:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        # File handler if log_file specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        
        return logger
    except Exception as e:
        logging.error(f"Error setting up logging: {e}")
        raise

class StructuredLogger:
    """Enhanced logger with structured output and environment awareness"""
    
    def __init__(
        self,
        name: str,
        log_dir: Optional[Path] = None,
        level: int = logging.INFO,
        max_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize the structured logger.

        Args:
            name (str): Name of the logger.
            log_dir (Optional[Path], optional): Directory to save log files. Defaults to None.
            level (int, optional): Logging level. Defaults to logging.INFO.
            max_size (int, optional): Maximum size of log files in bytes. Defaults to 10MB.
            backup_count (int, optional): Number of backup log files to keep. Defaults to 5.
        """
        try:
            self.logger = logging.getLogger(name)
            self.logger.setLevel(level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Add console handler
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(formatter)
            self.logger.addHandler(console)
            
            # Add rotating file handler if directory provided
            if log_dir:
                log_dir = Path(log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.handlers.RotatingFileHandler(
                    log_dir / f"{name}.log",
                    maxBytes=max_size,
                    backupCount=backup_count
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
        except Exception as e:
            logging.error(f"Error initializing structured logger: {e}")
            raise
            
    def log_event(
        self,
        level: int,
        message: str,
        extra: Optional[Dict] = None,
        exc_info=None
    ):
        """
        Log structured event with context.

        Args:
            level (int): Logging level.
            message (str): Log message.
            extra (Optional[Dict], optional): Additional context. Defaults to None.
            exc_info (optional): Exception information. Defaults to None.
        """
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'extra': extra or {}
            }
            
            self.logger.log(
                level,
                json.dumps(event),
                exc_info=exc_info
            )
        except Exception as e:
            # Fallback to basic logging if structured logging fails
            self.logger.error(f"Logging failed: {e}")
            self.logger.log(level, message, exc_info=exc_info)

    def get_logger(self) -> logging.Logger:
        """
        Get the underlying logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger
