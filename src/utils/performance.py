import multiprocessing
import psutil
import torch
from functools import lru_cache
import logging
from typing import Optional

class PerformanceOptimizer:
    """Handles performance optimization settings"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_optimal_batch_size(self, embedding_dim: int = 768) -> int:
        """Calculate optimal batch size based on available resources"""
        if torch.cuda.is_available():
            return self._get_cuda_batch_size(embedding_dim)
        return self._get_cpu_batch_size()
        
    def get_optimal_workers(self) -> int:
        """Get optimal number of worker processes"""
        return max(1, multiprocessing.cpu_count() - 1)
            
    @lru_cache(maxsize=1)
    def _get_cuda_batch_size(self, embedding_dim: int) -> int:
        """Calculate optimal batch size for GPU"""
        try:
            gpu = torch.cuda.get_device_properties(0)
            mem_bytes = gpu.total_memory * 0.8  # Use 80% of memory
            bytes_per_sample = embedding_dim * 4  # 4 bytes per float32
            return min(int(mem_bytes / bytes_per_sample), 64)  # Cap at 64
        except Exception as e:
            self.logger.warning(f"Error calculating GPU batch size: {e}")
            return 32
            
    def _get_cpu_batch_size(self) -> int:
        """Calculate optimal batch size for CPU"""
        mem = psutil.virtual_memory()
        available_gb = mem.available / 1e9
        return min(max(int(available_gb * 8), 8), 32)  # 8-32 range

    def clear_memory_cache(self) -> None:
        """Clear unused variables and cache to optimize memory usage"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.logger.info("Memory cache cleared successfully.")
        except Exception as e:
            self.logger.error(f"Error clearing memory cache: {e}")
