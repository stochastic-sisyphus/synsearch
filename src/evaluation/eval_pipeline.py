from typing import Dict, List, Any
import numpy as np
from torch.utils.data import DataLoader, Dataset
from ..utils.metrics_utils import calculate_cluster_metrics, calculate_summary_metrics
from ..utils.logging_utils import MetricsLogger

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
        dataset = EmbeddingDataset(embeddings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_embeddings = []
        for batch in dataloader:
            all_embeddings.append(batch)
        
        concatenated_embeddings = np.concatenate(all_embeddings, axis=0)
        
        metrics = calculate_cluster_metrics(concatenated_embeddings, labels)
        self.logger.log_metrics(metrics, 'clustering')
        return metrics
    
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
        metrics = {
            'summary_metrics': [
                calculate_summary_metrics(gen, ref) 
                for gen, ref in zip(generated_summaries, reference_summaries)
            ]
        }
        self.logger.log_metrics(metrics, 'summarization')
        return metrics 