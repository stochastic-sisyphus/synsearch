from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

class ClusterSummarizer:
    def __init__(
        self,
        model_name: str = 't5-base',
        device: Optional[str] = None,
        max_length: int = 512,
        min_length: int = 50,
        batch_size: int = 8
    ):
        """Initialize the summarizer"""
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            
            self.max_length = max_length
            self.min_length = min_length
            self.batch_size = batch_size
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
            
    def summarize_cluster(
        self,
        texts: List[str],
        cluster_id: Union[int, str]
    ) -> Dict[str, str]:
        """Generate summary for a cluster of texts"""
        try:
            # Concatenate texts with special tokens
            combined_text = " [DOC] ".join(texts)
            
            # Tokenize
            inputs = self.tokenizer(
                f"summarize: {combined_text}",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=self.max_length,
                min_length=self.min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )
            
            return {
                'cluster_id': cluster_id,
                'summary': summary,
                'num_docs': len(texts)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating summary for cluster {cluster_id}: {e}")
            raise
            
    def summarize_all_clusters(
        self,
        cluster_texts: Dict[str, List[str]]
    ) -> List[Dict[str, str]]:
        """Generate summaries for all clusters"""
        summaries = []
        
        for cluster_id, texts in tqdm(cluster_texts.items()):
            summary = self.summarize_cluster(texts, cluster_id)
            summaries.append(summary)
            
        return summaries
    
    def save_summaries(
        self,
        summaries: List[Dict[str, str]],
        output_dir: Union[str, Path]
    ) -> None:
        """Save summaries to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'summaries.json', 'w') as f:
            json.dump(summaries, f, indent=2)
            
        self.logger.info(f"Saved summaries to {output_dir}") 