import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np

class AttentionRefiner(nn.Module):
    """Refines embeddings using self-attention before clustering."""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to refine embeddings."""
        # Add batch dimension if needed
        if len(embeddings.shape) == 2:
            embeddings = embeddings.unsqueeze(0)
            
        # Apply self-attention
        attn_output, _ = self.attention(
            embeddings, 
            embeddings, 
            embeddings
        )
        
        return attn_output.squeeze(0)

class HybridClusteringModule:
    """Combines attention-refined embeddings with dynamic clustering."""
    
    def __init__(self, embedding_dim: int, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention_refiner = AttentionRefiner(embedding_dim).to(self.device) 
