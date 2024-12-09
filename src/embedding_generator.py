import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union
import logging

class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attention = nn.Linear(embedding_dim, 1)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism to embeddings."""
        attention_weights = torch.softmax(self.attention(embeddings), dim=0)
        return embeddings * attention_weights

class EnhancedEmbeddingGenerator:
    def __init__(
        self, 
        model_name: str = 'all-mpnet-base-v2',
        embedding_dim: int = 768,
        max_seq_length: int = 512,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        """Initialize the embedding generator with memory-efficient settings."""
        self.logger = logging.getLogger(__name__)
        
        # Set device with memory constraints in mind
        if device is None:
            # Default to CPU if CUDA memory is limited
            if torch.cuda.is_available():
                try:
                    # Test CUDA memory
                    torch.cuda.empty_cache()
                    test_tensor = torch.zeros((1, embedding_dim)).cuda()
                    test_tensor = None
                    self.device = 'cuda'
                except RuntimeError:
                    self.logger.warning("CUDA memory limited, falling back to CPU")
                    self.device = 'cpu'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        
        # Initialize the model and attention layer with memory efficiency
        try:
            self.model = SentenceTransformer(model_name)
            # Move model to CPU first
            self.model.to('cpu')
            torch.cuda.empty_cache()  # Clear CUDA memory
            # Now try moving to target device
            self.model.to(self.device)
            self.attention_layer = AttentionLayer(embedding_dim).to(self.device)
        except RuntimeError as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            self.logger.info("Falling back to CPU")
            self.device = 'cpu'
            self.model = SentenceTransformer(model_name)
            self.model.to(self.device)
            self.attention_layer = AttentionLayer(embedding_dim).to(self.device)

    def generate_embeddings(
        self,
        texts: List[str],
        apply_attention: bool = True
    ) -> np.ndarray:
        """Generate embeddings with memory-efficient batching."""
        try:
            all_embeddings = []
            
            # Process in smaller batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Generate embeddings for batch
                with torch.no_grad():  # Reduce memory usage
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_tensor=True,
                        device=self.device
                    )
                    
                    if apply_attention:
                        batch_embeddings = self.attention_layer(batch_embeddings)
                    
                    # Move to CPU and convert to numpy to free GPU memory
                    batch_embeddings = batch_embeddings.cpu().numpy()
                    all_embeddings.append(batch_embeddings)
                
                # Clear cache after each batch
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            # Combine all batches
            embeddings = np.vstack(all_embeddings)
            return embeddings
            
        except RuntimeError as e:
            self.logger.error(f"CUDA error: {str(e)}")
            # Try falling back to CPU if CUDA error occurs
            if self.device == 'cuda':
                self.logger.info("Falling back to CPU")
                self.device = 'cpu'
                self.model.to(self.device)
                self.attention_layer.to(self.device)
                return self.generate_embeddings(texts, apply_attention)
            raise
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
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