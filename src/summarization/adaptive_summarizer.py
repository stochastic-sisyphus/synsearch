from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Any
from src.utils.style_selector import AdaptiveStyleSelector
from src.utils.metrics_utils import calculate_cluster_metrics
import numpy as np
import torch

class AdaptiveSummarizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config['summarization']['model_name']
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['summarization']['model_name']
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.style_selector = AdaptiveStyleSelector(config)
        
        # Add temperature scaling for generation
        self.temperature_scaling = config['summarization'].get('temperature_scaling', True)
        
        # Add support for multi-style templates
        self.style_templates = config['summarization'].get('style_templates', {
            'technical': 'Summarize technically: ',
            'narrative': 'Provide a narrative summary: ',
            'concise': 'Summarize concisely: '
        })
    
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
        """Enhanced parameter adaptation based on content characteristics."""
        base_params = self.config['summarization']['style_params'][style].copy()
        
        # Dynamic temperature scaling based on cluster coherence
        if self.temperature_scaling:
            coherence = metrics.get('cluster_coherence', 0.5)
            base_params['temperature'] = max(0.7, min(1.0, 1.0 - coherence))
        
        # Adjust length based on content complexity and diversity
        complexity_factor = (
            metrics['lexical_diversity'] * 0.6 + 
            metrics.get('structural_complexity', 0.5) * 0.4
        )
        
        if complexity_factor > 0.7:  # High complexity content
            base_params['min_length'] = int(base_params['min_length'] * 1.3)
            base_params['max_length'] = int(base_params['max_length'] * 1.3)
            base_params['num_beams'] = max(5, base_params.get('num_beams', 4))
        
        # Adjust repetition penalty for diverse content
        if metrics['lexical_diversity'] > 0.75:
            base_params['repetition_penalty'] = max(
                1.5, 
                base_params.get('repetition_penalty', 1.0)
            )
        
        return base_params
    
    def _generate_summary(self, texts: List[str], gen_params: Dict[str, Any]) -> str:
        """Enhanced summary generation with style-specific prompting."""
        # Get style-specific template
        style = gen_params.get('style', 'narrative')
        template = self.style_templates.get(style, '')
        
        # Combine texts with special attention to key sentences
        combined_text = template + " ".join(texts)
        
        # Add length guidance
        target_length = gen_params.get('max_length', 150)
        combined_text = f"Length: {target_length} tokens. {combined_text}"
        
        inputs = self.tokenizer(
            combined_text, 
            return_tensors="pt", 
            truncation=True,
            max_length=self.config['summarization'].get('max_input_length', 1024),
            padding=True
        ).to(self.device)
        
        # Generate with enhanced parameters
        summary_ids = self.model.generate(
            inputs["input_ids"],
            do_sample=gen_params.get('do_sample', True),
            top_p=gen_params.get('top_p', 0.9),
            **gen_params
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary