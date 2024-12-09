import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union

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
        model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        embedding_dim: int = 768
    ):
        self.batch_size = batch_size
        
        # Try GPU first, fallback to CPU if OOM
        if device is None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.device = torch.device('cuda')
                else:
                    self.device = torch.device('cpu')
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: GPU out of memory, falling back to CPU")
                    self.device = torch.device('cpu')
                else:
                    raise e
        else:
            self.device = device
            
        # Initialize model and attention layer
        self.model = SentenceTransformer(model_name).to(self.device)
        self.attention_layer = AttentionLayer(embedding_dim).to(self.device)
        
    def generate_embeddings(
        self, 
        texts: List[str], 
        apply_attention: bool = True
    ) -> torch.Tensor:
        """Generate embeddings with optional attention mechanism.
        
        Args:
            texts: List of input texts to embed
            apply_attention: Whether to apply attention mechanism
            
        Returns:
            torch.Tensor: Generated embeddings with shape (n_texts, embedding_dim)
        """
        embeddings_list = []
        
        # Process in batches to manage memory
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Generate base embeddings
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False
                ).to(self.device)
                
                # Apply attention if requested
                if apply_attention:
                    batch_embeddings = self.attention_layer(batch_embeddings)
                    
                embeddings_list.append(batch_embeddings)
        
        # Concatenate all batches
        all_embeddings = torch.cat(embeddings_list, dim=0)
        
        # Calculate intra-cluster similarity for monitoring
        if len(all_embeddings) > 1:
            similarity = self.calculate_similarity(all_embeddings[0], all_embeddings[1])
            print(f"Sample embedding similarity: {similarity:.4f}")
            
        return all_embeddings
    
    def calculate_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Calculate cosine similarity between two embeddings."""
        return torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    def get_intracluster_similarity(self, cluster_embeddings: np.ndarray) -> float:
        """Calculate average pairwise similarity within a cluster."""
        n = len(cluster_embeddings)
        if n < 2:
            return 1.0
            
        total_similarity = 0
        pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_similarity += self.calculate_similarity(
                    cluster_embeddings[i], 
                    cluster_embeddings[j]
                )
                pairs += 1
                
        return total_similarity / pairs if pairs > 0 else 0.0