from typing import Dict, List, Any
import numpy as np
from torch.utils.data import DataLoader, Dataset
from ..utils.metrics_utils import calculate_cluster_metrics, calculate_summary_metrics
from ..utils.logging_utils import MetricsLogger
from ..utils.metrics_calculator import MetricsCalculator  # Add this import
import logging

class EmbeddingDataset(Dataset):
    """Custom Dataset for embeddings."""
    
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]

class EvaluationPipeline:
    """Pipeline for evaluating clustering and summarization quality."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the EvaluationPipeline with configuration settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.config = config
        self.logger = MetricsLogger(config)
        self.log = logging.getLogger(__name__)
        
    def evaluate_clustering(self, embeddings: np.ndarray, labels: np.ndarray, batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate clustering quality.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            labels (np.ndarray): Array of cluster labels.
            batch_size (int, optional): Batch size for processing. Defaults to 32.

        Returns:
            Dict[str, float]: Dictionary of clustering metrics.
        """
        try:
            self.log.info("Starting clustering evaluation")
            self.log.debug(f"Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")

            # Ensure structural correctness of inputs
            if not isinstance(embeddings, np.ndarray):
                raise ValueError("Embeddings must be a numpy array")
            if embeddings.ndim != 2:
                raise ValueError("Embeddings must be a 2D array")
            if not isinstance(labels, np.ndarray):
                raise ValueError("Labels must be a numpy array")
            if labels.ndim != 1:
                raise ValueError("Labels must be a 1D array")

            dataset = EmbeddingDataset(embeddings)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            all_embeddings = []
            for batch in dataloader:
                all_embeddings.append(batch)

            concatenated_embeddings = np.concatenate(all_embeddings, axis=0)

            metrics = calculate_cluster_metrics(concatenated_embeddings, labels)
            self.logger.log_metrics(metrics, 'clustering')
            return metrics
        except Exception as e:
            self.log.error(f"Error evaluating clustering: {e}")
            return {}

    def evaluate_summaries(self, 
                           generated_summaries: List[str], 
                           reference_summaries: List[str]) -> Dict[str, float]:
        """
        Evaluate summary quality.

        Args:
            generated_summaries (List[str]): List of generated summaries.
            reference_summaries (List[str]): List of reference summaries.

        Returns:
            Dict[str, float]: Dictionary of summary metrics.
        """
        try:
            self.log.info("Starting summary evaluation")
            self.log.debug(f"Number of generated summaries: {len(generated_summaries)}, Number of reference summaries: {len(reference_summaries)}")

            if not isinstance(generated_summaries, list) or not all(isinstance(s, str) for s in generated_summaries):
                raise ValueError("Generated summaries must be a list of strings")
            if not isinstance(reference_summaries, list) or not all(isinstance(s, str) for s in reference_summaries):
                raise ValueError("Reference summaries must be a list of strings")

            metrics = MetricsCalculator()._calculate_summarization_metrics(generated_summaries, reference_summaries)
            self.logger.log_metrics(metrics, 'summarization')
            return metrics
        except Exception as e:
            self.log.error(f"Error evaluating summaries: {e}")
            return {}
