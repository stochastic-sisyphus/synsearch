import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class MetricsLogger:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_dir = Path(config['logging']['output_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_metrics(self, metrics: Dict[str, float], step: str) -> None:
        """Log metrics with timestamp and step information."""
        timestamp = datetime.now().isoformat()
        
        metrics_with_meta = {
            'timestamp': timestamp,
            'step': step,
            'metrics': metrics
        }
        
        # Save to JSON file
        metrics_file = self.log_dir / f'metrics_{step}.json'
        with open(metrics_file, 'a') as f:
            json.dump(metrics_with_meta, f)
            f.write('\n')
        
        # Log to console/file
        self.logger.info(f"Step: {step} - Metrics: {metrics}") 