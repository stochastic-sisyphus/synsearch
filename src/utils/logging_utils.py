import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader, Dataset

class MetricsDataset(Dataset):
    """Custom Dataset for metrics."""
    
    def __init__(self, metrics: list):
        self.metrics = metrics
        
    def __len__(self):
        return len(self.metrics)
    
    def __getitem__(self, idx):
        return self.metrics[idx]

class MetricsLogger:
    """
    A class to log metrics with timestamp and step information.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MetricsLogger with configuration settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
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
        """
        Log metrics with timestamp and step information.

        Args:
            metrics (Dict[str, float]): Dictionary of metrics to log.
            step (str): Step information for logging.
        """
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
    
    def log_metrics_batch(self, metrics_list: list, step: str, batch_size: int = 32) -> None:
        """
        Log metrics in batches with timestamp and step information.

        Args:
            metrics_list (list): List of metrics dictionaries to log.
            step (str): Step information for logging.
            batch_size (int, optional): Batch size for processing. Defaults to 32.
        """
        dataset = MetricsDataset(metrics_list)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for batch in dataloader:
            timestamp = datetime.now().isoformat()
            
            for metrics in batch:
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

    def log_method_call(self, method_name: str, *args, **kwargs) -> None:
        """
        Log the method call with parameters.

        Args:
            method_name (str): The name of the method being called.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.logger.info(f"Calling method: {method_name}")
        self.logger.debug(f"Parameters: args={args}, kwargs={kwargs}")

    def log_intermediate_result(self, result: Any, description: str = "") -> None:
        """
        Log intermediate results.

        Args:
            result (Any): The result to log.
            description (str, optional): Additional description of the result.
        """
        self.logger.info(f"Intermediate result: {description}")
        self.logger.debug(f"Result: {result}")

    def ensure_structural_correctness(self, data: Any, expected_type: type, description: str = "") -> None:
        """
        Ensure the structural correctness of the input data.

        Args:
            data (Any): The data to validate.
            expected_type (type): The expected type of the data.
            description (str, optional): Additional description of the data.
        """
        if not isinstance(data, expected_type):
            raise ValueError(f"Incorrect data structure for {description}. Expected {expected_type}, got {type(data)}")
