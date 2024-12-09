from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging
from pathlib import Path
from datetime import datetime
import json

class MetricsCalculator:
    """Calculate and manage various metrics for the pipeline."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_clustering_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        metrics = {}
        
        if len(np.unique(labels)) > 1:
            metrics['silhouette_score'] = float(silhouette_score(embeddings, labels))
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(embeddings, labels))
        else:
            metrics['silhouette_score'] = 0.0
            metrics['davies_bouldin_score'] = float('inf')
            
        return metrics
    
    def calculate_summary_metrics(
        self,
        summaries: Dict[str, str],
        references: Dict[str, str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Calculate summary quality metrics."""
        metrics = {}
        
        for cluster_id, summary in summaries.items():
            cluster_metrics = {
                'length': len(summary.split()),
                'avg_sentence_length': np.mean([len(s.split()) for s in summary.split('.')])
            }
            
            if references and cluster_id in references:
                # Add reference-based metrics if available
                ref = references[cluster_id]
                cluster_metrics.update(self._calculate_reference_metrics(summary, ref))
                
            metrics[cluster_id] = cluster_metrics
            
        return metrics
    
    def _calculate_reference_metrics(
        self,
        summary: str,
        reference: str
    ) -> Dict[str, float]:
        """Calculate metrics that compare summary to reference."""
        return {
            'length_ratio': len(summary.split()) / len(reference.split()),
            'vocabulary_overlap': len(
                set(summary.lower().split()) & set(reference.lower().split())
            ) / len(set(reference.lower().split()))
        }
    
    def save_metrics(
        self,
        metrics: Dict[str, Any],
        output_dir: str,
        prefix: str = ''
    ) -> None:
        """Save metrics to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(output_dir) / f'{prefix}_metrics_{timestamp}.json'
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Saved metrics to {output_path}") 