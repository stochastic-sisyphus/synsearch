
import os
import sys
import platform
from pathlib import Path
import yaml
from typing import Dict, List, Optional
import logging
import torch

class EnvironmentValidator:
    """Validates and configures the execution environment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_environment(self) -> bool:
        """Validate complete environment setup"""
        try:
            # Check Python version
            if sys.version_info < (3, 8):
                raise EnvironmentError("Python 3.8 or higher required")
            
            # Validate critical paths
            self._validate_paths()
            
            # Check CUDA availability
            self._check_cuda()
            
            # Validate dependencies
            self._validate_dependencies()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Environment validation failed: {e}")
            return False
            
    def _validate_paths(self) -> None:
        """Validate and create required paths"""
        required_dirs = ['logs', 'outputs', 'cache', 'models']
        project_root = Path(__file__).parent.parent.parent
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            
            if not os.access(dir_path, os.W_OK):
                raise PermissionError(f"No write permission for {dir_path}")
                
    def _check_cuda(self) -> None:
        """Check CUDA availability and configuration"""
        if torch.cuda.is_available():
            self.logger.info(
                f"CUDA available: {torch.cuda.get_device_name(0)}\n"
                f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB"
            )
        else:
            self.logger.warning("CUDA not available, using CPU")
            
    def _validate_dependencies(self) -> None:
        """Validate all required dependencies"""
        required = {
            'torch': 'Deep learning',
            'transformers': 'Language models',
            'pandas': 'Data processing',
            'numpy': 'Numerical operations'
        }
        
        missing = []
        for pkg, purpose in required.items():
            try:
                __import__(pkg)
            except ImportError:
                missing.append(f"{pkg} ({purpose})")
                
        if missing:
            raise ImportError(f"Missing dependencies: {', '.join(missing)}")