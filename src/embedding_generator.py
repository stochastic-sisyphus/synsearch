import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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
    def __init__(self, model_name, device=None, n_workers=None):
        self.device = device or get_device()
        self.n_workers = n_workers or get_optimal_workers()
        self.model = SentenceTransformer(model_name).to(self.device)
        
    def generate_embeddings(self, texts, batch_size=32):
        """Generate embeddings using parallel processing and GPU acceleration."""
        # Split texts into batches for parallel processing
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            embeddings_list = list(executor.map(
                lambda batch: self.model.encode(
                    batch,
                    device=self.device,
                    show_progress_bar=True,
                    convert_to_tensor=True
                ),
                batches
            ))
        
        # Concatenate results
        embeddings = torch.cat(embeddings_list, dim=0)
        return embeddings.cpu().numpy()