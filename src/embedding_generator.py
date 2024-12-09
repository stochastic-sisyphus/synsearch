import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union
import logging
import gc  # Add garbage collector

class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.attention = nn.Linear(embedding_dim, 1)
    
    def forward(self, embeddings):
        attention_weights = torch.softmax(self.attention(embeddings), dim=0)
        return embeddings * attention_weights

class EnhancedEmbeddingGenerator:
    def __init__(
        self, 
        model_name: str = 'all-mpnet-base-v2',
        embedding_dim: int = 768,
        max_seq_length: int = 512,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """Initialize the embedding generator with memory management."""
        self.logger = logging.getLogger(__name__)
        
        # Determine device with fallback options
        if device is None:
            if torch.cuda.is_available():
                try:
                    # Try to get GPU memory info
                    free_memory, total_memory = torch.cuda.mem_get_info()
                    if free_memory > 2 * 1024 * 1024 * 1024:  # Check if > 2GB free
                        device = 'cuda'
                    else:
                        self.logger.warning("Insufficient GPU memory, falling back to CPU")
                        device = 'cpu'
                except:
                    device = 'cpu'
            else:
                device = 'cpu'
        
        self.device = device
        self.logger.info(f"Using device: {self.device}")
        
        # Clear any existing cached memory
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            self.model = SentenceTransformer(model_name)
            self.model.to(self.device)
            
            # Initialize attention layer
            self.attention_layer = AttentionLayer(embedding_dim).to(self.device)
            
            self.batch_size = batch_size
            self.max_seq_length = max_seq_length
            
        except RuntimeError as e:
            self.logger.error(f"Error initializing model: {e}")
            self.logger.info("Falling back to CPU")
            self.device = 'cpu'
            try:
                self.model = SentenceTransformer(model_name)
                self.model.to(self.device)
                self.attention_layer = AttentionLayer(embedding_dim).to(self.device)
            except Exception as e:
                self.logger.error(f"Critical error initializing model: {e}")
                raise
    
    def generate_embeddings(
        self,
        texts: List[str],
        apply_attention: bool = True,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Generate embeddings with memory-efficient batching."""
        if batch_size is None:
            batch_size = self.batch_size
            
        try:
            # Process in batches to manage memory
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Clear cache between batches if using CUDA
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    
                with torch.no_grad():  # Reduce memory usage during inference
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_tensor=True
                    )
                    
                    if apply_attention:
                        batch_embeddings = self.attention_layer(batch_embeddings)
                    
                    # Move to CPU and convert to numpy to free GPU memory
                    batch_embeddings = batch_embeddings.cpu().numpy()
                    all_embeddings.append(batch_embeddings)
                    
            # Concatenate all batches
            embeddings = np.concatenate(all_embeddings, axis=0)
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise

    def calculate_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Calculate cosine similarity between two embeddings."""
        return torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    def get_intracluster_similarity(self, cluster_embeddings: np.ndarray) -> float:
        """Calculate average pairwise similarity within a cluster."""
        n = len(cluster_embeddings)
        if n < 2:
            return 1.0
            
        total_similarity = 0.0
        pairs = 0
        
        # Convert to torch tensors
        embeddings = torch.tensor(cluster_embeddings).to(self.device)
        
        for i in range(n):
            for j in range(i + 1, n):
                total_similarity += self.calculate_similarity(
                    embeddings[i], 
                    embeddings[j]
                )
                pairs += 1
                
        return total_similarity / pairs if pairs > 0 else 0.0