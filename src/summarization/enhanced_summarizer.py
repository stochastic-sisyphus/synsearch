from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from typing import Dict, List, Tuple
import numpy as np
from .adaptive_style import AdaptiveStyleSelector

class EnhancedHybridSummarizer:
    def __init__(self, config: Dict):
        self.config = config
        self.style_selector = AdaptiveStyleSelector(config)
        
    def summarize_all_clusters(
        self, 
        cluster_texts: Dict[str, List[Dict]], 
        embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """Summarize all clusters with adaptive style selection."""
        summaries = {}
        
        for cluster_id, texts in cluster_texts.items():
            cluster_embeddings = embeddings[cluster_id]
            cluster_text_content = [doc['text'] for doc in texts]
            
            # Determine style adaptively
            style = self.style_selector.determine_style(
                cluster_embeddings,
                cluster_text_content
            )
            
            # Generate summary with selected style
            summary = self._batch_summarize(
                texts=cluster_text_content,
                style=style
            )
            
            summaries[cluster_id] = {
                'summary': summary,
                'style': style
            }
            
        return summaries