import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

class EnhancedEmbeddingGenerator:
    def __init__(self, model_name: str = 'all-mpnet-base-v2', embedding_dim: int = 768):
        self.model = SentenceTransformer(model_name)
        self.attention = AttentionRefinement(embedding_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.attention.to(self.device)
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        # Generate base embeddings
        base_embeddings = self.model.encode(texts, convert_to_tensor=True)
        
        # Apply attention refinement
        refined_embeddings = self.attention(base_embeddings)
        
        return refined_embeddings.cpu().numpy()

class AttentionRefinement(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Calculate attention weights
        weights = torch.softmax(self.attention(embeddings), dim=0)
        # Apply attention weights
        refined = embeddings * weights
        return refined