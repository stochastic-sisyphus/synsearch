from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from pathlib import Path

class HybridSummarizer:
    """
    HybridSummarizer: A flexible summarization module that combines extractive and abstractive approaches.

    Features:
    - Style-aware summarization (technical, concise, detailed)
    - Configurable length and parameters per style
    - Batch processing support
    - GPU acceleration
    - Checkpoint support

    Example:
        summarizer = HybridSummarizer(
            model_name='facebook/bart-large-cnn',
            max_length=150,
            min_length=50
        )
        summary = summarizer.summarize(texts, style='technical')
    """
    def __init__(
        self,
        model_name: str = 'facebook/bart-large-cnn',
        max_length: int = 150,
        min_length: int = 50,
        batch_size: int = 4,
        device: Optional[str] = None
    ):
        """Initialize hybrid summarizer with extractive + abstractive capabilities"""
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.batch_size = batch_size
            self.max_length = max_length
            self.min_length = min_length
            
            # Initialize TF-IDF for extractive step
            self.tfidf = TfidfVectorizer(max_features=1000)
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
            
        # Add style-specific prompts and parameters
        self.style_config = {
            'technical': {
                'prompt': "Provide a technical summary focusing on methodology and results:",
                'top_k': 3,
                'length_multiplier': 1.2
            },
            'concise': {
                'prompt': "Summarize the key points briefly:",
                'top_k': 2,
                'length_multiplier': 0.8
            },
            'detailed': {
                'prompt': "Provide a comprehensive summary including background and implications:",
                'top_k': 4,
                'length_multiplier': 1.5
            },
            'balanced': {
                'prompt': "",
                'top_k': 3,
                'length_multiplier': 1.0
            }
        }

    def _extract_key_sentences(self, texts: List[str], top_k: int = 3) -> List[str]:
        """Extract most important sentences using TF-IDF scores"""
        try:
            # Calculate TF-IDF matrix
            tfidf_matrix = self.tfidf.fit_transform(texts)
            
            # Get average TF-IDF scores for each document
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=1)
            
            # Select top-k documents based on scores
            top_indices = np.argsort(avg_scores)[-top_k:]
            return [texts[i] for i in top_indices]
            
        except Exception as e:
            self.logger.error(f"Error in extractive summarization: {e}")
            raise
            
    def summarize_cluster(
        self,
        texts: List[str],
        style: str = 'balanced'
    ) -> Dict[str, str]:
        """Generate style-aware hybrid summary for a cluster of texts"""
        try:
            # Get style-specific parameters
            style_params = self.style_config.get(style, self.style_config['balanced'])
            
            # Extract key sentences with style-specific top_k
            key_texts = self._extract_key_sentences(
                texts,
                top_k=style_params['top_k']
            )
            
            # Combine key texts with style prompt
            combined_text = style_params['prompt'] + " " + " ".join(key_texts)
            
            # Adjust length based on style
            style_max_length = int(self.max_length * style_params['length_multiplier'])
            style_min_length = int(self.min_length * style_params['length_multiplier'])
            
            # Tokenize
            inputs = self.tokenizer(
                combined_text,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=style_max_length,
                min_length=style_min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return {
                'summary': summary,
                'key_texts': key_texts,
                'style': style
            }
            
        except Exception as e:
            self.logger.error(f"Error generating hybrid summary: {e}")
            raise
            
    def summarize_all_clusters(
        self,
        cluster_texts: Dict[str, List[str]],
        style: str = 'balanced'
    ) -> Dict[str, Dict[str, str]]:
        """Generate summaries for all clusters"""
        summaries = {}
        
        for cluster_id, texts in cluster_texts.items():
            summaries[cluster_id] = self.summarize_cluster(texts, style)
            
        return summaries 