import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Union, Optional
import json
from tqdm import tqdm
import os

class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = 'all-mpnet-base-v2',
        device: Optional[str] = None,
        batch_size: int = 32,
        max_seq_length: int = 128
    ):
        """Initialize the embedding generator"""
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.logger.info(f"Using device: {self.device}")
        
        # Load model with optimizations
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model.max_seq_length = max_seq_length  # Limit sequence length
            self.batch_size = batch_size
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
            
    def generate_embeddings(
        self,
        texts: List[str],
        show_progress_bar: bool = True,
        checkpoint_dir: Optional[str] = None,
        checkpoint_frequency: int = 1000
    ) -> np.ndarray:
        """Generate embeddings with checkpoints and progress tracking"""
        try:
            total_texts = len(texts)
            embeddings_list = []
            
            # Create checkpoint directory if needed
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Load latest checkpoint if exists
                checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')])
                if checkpoints:
                    latest = checkpoints[-1]
                    start_idx = int(latest.split('_')[1].split('.')[0])
                    embeddings_list = list(np.load(os.path.join(checkpoint_dir, latest)))
                    texts = texts[start_idx:]
                    self.logger.info(f"Resuming from checkpoint {latest}")
                else:
                    start_idx = 0
            else:
                start_idx = 0
            
            # Process in batches with progress bar
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings_list.extend(batch_embeddings)
                
                # Save checkpoint if enabled
                if checkpoint_dir and (i + start_idx) % checkpoint_frequency == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{i + start_idx}.npy')
                    np.save(checkpoint_path, np.array(embeddings_list))
                    self.logger.info(f"Saved checkpoint at {i + start_idx}/{total_texts} documents")
            
            return np.array(embeddings_list)
            
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