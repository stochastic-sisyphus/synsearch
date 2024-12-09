import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, embeddings):
        # Convert to tensor if needed
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
            
        # Compute attention scores
        q = self.query(embeddings).unsqueeze(1)
        k = self.key(embeddings).unsqueeze(0)
        v = self.value(embeddings)
        
        # Calculate attention weights
        scores = torch.matmul(q, k.transpose(-2, -1))
        attention_weights = self.softmax(scores)
        
        # Apply attention
        refined_embeddings = torch.matmul(attention_weights, v)
        return refined_embeddings

class EnhancedEmbeddingGenerator:
    def __init__(self, model_name='all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = SentenceTransformer(model_name)
        self.attention = AttentionLayer(self.model.get_sentence_embedding_dimension())
        self.device = device
        self.model.to(device)
        self.attention.to(device)
        
    def generate_embeddings(self, texts, batch_size=32):
        embeddings = self.model.encode(texts, batch_size=batch_size)
        refined_embeddings = self.attention(embeddings)
        return refined_embeddings.cpu().numpy()