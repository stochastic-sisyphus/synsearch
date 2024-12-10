import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np
from torch.utils.data import DataLoader, Dataset

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

class EmbeddingDataset(Dataset):
    """Custom Dataset for embeddings."""
    
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]

class HybridClusteringModule:
    """Combines attention-refined embeddings with dynamic clustering."""
    
    def __init__(self, embedding_dim: int, device: Optional[str] = None):
        """
        Initialize the HybridClusteringModule with embedding dimension and device.

        Args:
            embedding_dim (int): Dimension of the embeddings.
            device (Optional[str], optional): Device to use for computation. Defaults to None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention_refiner = AttentionRefiner(embedding_dim).to(self.device)
        
    def refine_embeddings(self, embeddings: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Refine embeddings using self-attention in batches.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            batch_size (int, optional): Batch size for processing. Defaults to 32.

        Returns:
            np.ndarray: Refined embeddings.
        """
        dataset = EmbeddingDataset(embeddings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        refined_embeddings = []
        for batch in dataloader:
            batch = batch.to(self.device)
            refined_batch = self.attention_refiner(batch)
            refined_embeddings.append(refined_batch.cpu().numpy())
        
        return np.concatenate(refined_embeddings, axis=0)
