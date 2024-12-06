import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Union, Optional
import pandas as pd
from tqdm import tqdm
import json

class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = 'all-mpnet-base-v2',
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """Initialize the embedding generator"""
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.batch_size = batch_size
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
            
    def generate_embeddings(
        self,
        texts: List[str],
        show_progress_bar: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
            
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: Dict,
        output_dir: Union[str, Path]
    ) -> None:
        """Save embeddings and metadata to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(output_dir / 'embeddings.npy', embeddings)
        
        # Save metadata
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
            
        self.logger.info(f"Saved embeddings to {output_dir}")
        
    def load_embeddings(
        self,
        input_dir: Union[str, Path]
    ) -> tuple[np.ndarray, Dict]:
        """Load embeddings and metadata from disk"""
        input_dir = Path(input_dir)
        
        try:
            # Load embeddings
            embeddings = np.load(input_dir / 'embeddings.npy')
            
            # Load metadata
            with open(input_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
                
            return embeddings, metadata
        except Exception as e:
            self.logger.error(f"Error loading embeddings from {input_dir}: {e}")
            raise