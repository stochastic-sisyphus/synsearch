
from pathlib import Path
import json
import pickle
from typing import Any, Optional, Dict
import logging
from datetime import datetime
import hashlib

class CacheManager:
    """Manages caching of intermediate results"""
    
    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def cache_exists(self, key: str) -> bool:
        """Check if cache exists for key"""
        return (self.cache_dir / f"{key}.pkl").exists()
        
    def get_cache(self, key: str) -> Optional[Any]:
        """Get cached data if available"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Cache read failed for {key}: {e}")
        return None
        
    def set_cache(
        self,
        key: str,
        data: Any,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Cache data with metadata"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            meta_file = self.cache_dir / f"{key}.meta.json"
            
            # Save data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                
            # Save metadata
            meta = {
                'timestamp': datetime.now().isoformat(),
                'type': str(type(data)),
                'size': len(pickle.dumps(data)),
                **(metadata or {})
            }
            
            with open(meta_file, 'w') as f:
                json.dump(meta, f)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Cache write failed for {key}: {e}")
            return False