from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Any
import numpy as np
import torch
from src.utils.metrics_utils import calculate_cluster_metrics

class AdaptiveSummarizer:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the adaptive summarizer with configuration."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.get('model_name', 'facebook/bart-large-cnn')
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get('model_name', 'facebook/bart-large-cnn')
        )
        
        # Style parameters
        self.style_params = config.get('style_params', {
            'concise': {'max_length': 100, 'min_length': 30},
            'detailed': {'max_length': 300, 'min_length': 100},
            'technical': {'max_length': 200, 'min_length': 50}
        })
        
        # Default parameters
        self.default_params = {
            'num_beams': config.get('num_beams', 4),
            'length_penalty': config.get('length_penalty', 2.0),
            'early_stopping': config.get('early_stopping', True)
        }

    def summarize_cluster(
        self, 
        texts: List[str], 
        embeddings: np.ndarray,
        cluster_id: int = None
    ) -> Dict[str, Any]:
        """Generate summaries with adaptive style based on cluster characteristics."""
        # Calculate cluster metrics
        metrics = calculate_cluster_metrics(embeddings, texts)
        
        # Determine style based on metrics
        style = self._determine_style(metrics)
        
        # Get generation parameters for the style
        gen_params = self._get_style_params(style)
        
        # Generate summary
        summary = self._generate_summary(texts, gen_params)
        
        return {
            "summary": summary,
            "style": style,
            "metrics": metrics,
            "cluster_id": cluster_id
        }

    def _determine_style(self, metrics: Dict[str, float]) -> str:
        """Determine appropriate summarization style based on metrics."""
        # Simple rule-based style selection
        if metrics['variance'] > 0.7:  # High variance -> detailed
            return 'detailed'
        elif metrics['lexical_diversity'] > 0.8:  # High diversity -> technical
            return 'technical'
        else:  # Default to concise
            return 'concise'

    def _get_style_params(self, style: str) -> Dict[str, Any]:
        """Get generation parameters for a given style."""
        style_params = self.style_params.get(style, self.style_params['concise'])
        return {**self.default_params, **style_params}

    def _generate_summary(self, texts: List[str], params: Dict[str, Any]) -> str:
        """Generate summary using the specified parameters."""
        # Combine texts with separator
        combined_text = " ".join(texts)
        
        # Tokenize
        inputs = self.tokenizer(
            combined_text,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        outputs = self.model.generate(
            inputs['input_ids'],
            **params
        )
        
        # Decode and return
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary