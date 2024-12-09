from typing import Dict, List, Any
import numpy as np
from ..utils.metrics_utils import calculate_cluster_metrics, calculate_summary_metrics
from ..utils.logging_utils import MetricsLogger

class EvaluationPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = MetricsLogger(config)
        
    def evaluate_clustering(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality."""
        metrics = calculate_cluster_metrics(embeddings, labels)
        self.logger.log_metrics('clustering', metrics)
        return metrics
    
    def evaluate_summaries(self, 
                         generated_summaries: List[str], 
                         reference_summaries: List[str]) -> Dict[str, float]:
        """Evaluate summary quality."""
        metrics = {
            'summary_metrics': [
                calculate_summary_metrics(gen, ref) 
                for gen, ref in zip(generated_summaries, reference_summaries)
            ]
        }
        self.logger.log_metrics('summarization', metrics)
        return metrics 