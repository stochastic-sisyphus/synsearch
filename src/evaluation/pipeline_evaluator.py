from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np

class PipelineEvaluator:
    def __init__(self, config: Dict):
        self.metrics = EvaluationMetrics()
        self.output_dir = Path(config['evaluation']['output_dir'])
        
    def evaluate_pipeline(self, 
                         datasets: List[str],
                         embeddings: Dict[str, np.ndarray],
                         clusters: Dict[str, List],
                         summaries: Dict[str, str]) -> Dict:
        """Comprehensive pipeline evaluation"""
        results = {
            'datasets': self._evaluate_datasets(datasets),
            'embeddings': self._evaluate_embeddings(embeddings),
            'clustering': self._evaluate_clustering(clusters),
            'summarization': self._evaluate_summaries(summaries),
            'runtime': self._calculate_runtime()
        }
        self._save_results(results)
        return results 