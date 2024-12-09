from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
from typing import List, Dict, Any
from src.utils.style_selector import AdaptiveStyleSelector
from src.utils.metrics_utils import calculate_cluster_metrics
import numpy as np
import torch

class AdaptiveSummarizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = AutoModelForSeq2SeqGeneration.from_pretrained(
            config['summarization']['model_name']
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['summarization']['model_name']
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.style_selector = AdaptiveStyleSelector(config)
    
    def summarize_cluster(
        self, 
        texts: List[str], 
        embeddings: np.ndarray,
        cluster_id: int = None
    ) -> Dict[str, Any]:
        """Generate summaries with adaptive style based on cluster characteristics.
        
        Args:
            texts: List of texts in the cluster
            embeddings: numpy array of text embeddings
            cluster_id: Optional cluster identifier
        """
        # Calculate cluster metrics
        metrics = calculate_cluster_metrics(embeddings, texts)
        
        # Determine appropriate style using the style selector
        style_info = self.style_selector.determine_cluster_style(
            embeddings=embeddings,
            texts=texts
        )
        
        # Get base parameters and adjust based on metrics
        gen_params = self._get_adaptive_params(style_info['style'], metrics)
        
        # Generate summary with attention to key sentences
        summary = self._generate_summary(texts, gen_params)
        
        return {
            "summary": summary,
            "style": style_info['style'],
            "metrics": {**metrics, **style_info['metrics']},
            "cluster_id": cluster_id
        }
    
    def _get_adaptive_params(self, style: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Dynamically adjust generation parameters based on metrics."""
        base_params = self.config['summarization']['style_params'][style].copy()
        
        # Adjust length based on cluster complexity
        if metrics['lexical_diversity'] > 0.8:  # High diversity
            base_params['min_length'] = int(base_params['min_length'] * 1.2)
            base_params['max_length'] = int(base_params['max_length'] * 1.2)
        
        # Adjust other parameters based on metrics
        if metrics['cluster_density'] < 0.5:  # Sparse cluster
            base_params['num_beams'] = max(4, base_params.get('num_beams', 4))
        
        return base_params
    
    def _generate_summary(self, texts: List[str], gen_params: Dict[str, Any]) -> str:
        """Generate summary with attention to key sentences."""
        # Combine texts with special attention to representative sentences
        combined_text = " ".join(texts)
        inputs = self.tokenizer(
            combined_text, 
            return_tensors="pt", 
            truncation=True,
            max_length=self.config['summarization'].get('max_input_length', 1024)
        ).to(self.device)
        
        summary_ids = self.model.generate(
            inputs["input_ids"],
            **gen_params
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary