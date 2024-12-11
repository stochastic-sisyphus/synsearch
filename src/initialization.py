from pathlib import Path
import logging
import sys
from typing import Optional, Dict, Any
import yaml
from .utils.environment import EnvironmentValidator
from .utils.error_handler import GlobalErrorHandler
from .utils.performance import PerformanceOptimizer
from .utils.caching import CacheManager

class SystemInitializer:
    """Handles system initialization and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or 'config/config.yaml'
        self.logger = self._setup_logging()
        self.error_handler = GlobalErrorHandler(self.logger)
        
    def initialize(self) -> Dict[str, Any]:
        """Initialize complete system"""
        try:
            # Validate environment
            env_validator = EnvironmentValidator()
            if not env_validator.validate_environment():
                raise EnvironmentError("Environment validation failed")
                
            # Load configuration
            config = self._load_config()
            
            # Setup performance optimization
            optimizer = PerformanceOptimizer()
            config['performance'] = {
                'batch_size': optimizer.get_optimal_batch_size(),
                'num_workers': optimizer.get_optimal_workers()
            }
            
            # Initialize cache
            cache_mgr = CacheManager(config.get('cache_dir', 'cache'))
            
            return {
                'config': config,
                'cache_manager': cache_mgr,
                'performance_optimizer': optimizer
            }
            
        except Exception as e:
            self.error_handler.handle_error(e, "System initialization")
            raise
            
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger('synsearch')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_dir / 'synsearch.log')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration"""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                
            # Validate config
            self._validate_config(config)
            return config
            
        except Exception as e:
            raise ValueError(f"Configuration error: {e}")
            
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration contents"""
        required_keys = {'data', 'preprocessing', 'embedding', 'clustering'}
        missing = required_keys - set(config.keys())
        
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")
