from typing import List, Dict, Union, Optional
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from rouge_score import rouge_scorer
import logging
from pathlib import Path
import json
from datetime import datetime

class EvaluationMetrics:
    def __init__(self):
        """Initialize the evaluation metrics calculator"""
        self.logger = logging.getLogger(__name__)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def calculate_clustering_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clustering quality metrics"""
        try:
            # Filter out noise points (label -1) if any
            mask = labels != -1
            if not np.any(mask):
                return {
                    'silhouette_score': 0.0,
                    'davies_bouldin_score': float('inf')
                }
                
            valid_embeddings = embeddings[mask]
            valid_labels = labels[mask]
            
            # Calculate metrics
            silhouette = silhouette_score(valid_embeddings, valid_labels)
            davies_bouldin = davies_bouldin_score(valid_embeddings, valid_labels)
            
            return {
                'silhouette_score': float(silhouette),
                'davies_bouldin_score': float(davies_bouldin)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating clustering metrics: {e}")
            raise
            
    def calculate_rouge_scores(
        self,
        summaries: List[str],
        references: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate ROUGE scores for summaries"""
        try:
            scores = {
                'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
                'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
                'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
            }
            
            for summary, reference in zip(summaries, references):
                score = self.rouge_scorer.score(reference, summary)
                
                for metric, values in score.items():
                    scores[metric]['precision'].append(values.precision)
                    scores[metric]['recall'].append(values.recall)
                    scores[metric]['fmeasure'].append(values.fmeasure)
            
            # Calculate averages
            averaged_scores = {}
            for metric, values in scores.items():
                averaged_scores[metric] = {
                    k: float(np.mean(v)) for k, v in values.items()
                }
                
            return averaged_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating ROUGE scores: {e}")
            raise
            
    def save_metrics(
        self,
        metrics: Dict,
        output_dir: Union[str, Path],
        prefix: str = ''
    ) -> None:
        """Save metrics to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_metrics_{timestamp}.json" if prefix else f"metrics_{timestamp}.json"
        
        with open(output_dir / filename, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        self.logger.info(f"Saved metrics to {output_dir / filename}") 
        
    def calculate_baseline_metrics(self, dataset_name: str, metrics: Dict) -> Dict[str, float]:
        """Calculate and store baseline metrics for a dataset"""
        baseline_metrics = {
            'dataset': dataset_name,
            'runtime': metrics.get('runtime', 0),
            'rouge_scores': metrics.get('rouge_scores', {}),
            'clustering_scores': {
                'silhouette': metrics.get('silhouette_score', 0),
                'davies_bouldin': metrics.get('davies_bouldin_score', 0)
            },
            'preprocessing_time': metrics.get('preprocessing_time', 0)
        }
        return baseline_metrics 