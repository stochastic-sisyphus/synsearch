import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int):
        """Initialize attention layer for embedding refinement.
        
        Args:
            embedding_dim: Dimension of input embeddings
        """
        super().__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism to input embeddings.
        
        Args:
            embeddings: Input tensor of shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            Refined embeddings with same shape as input
        """
        # Generate Q, K, V matrices
        q = self.query(embeddings)
        k = self.key(embeddings)
        v = self.value(embeddings)
        
        # Calculate attention scores
        attention_scores = torch.bmm(q, k.transpose(1, 2))
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention weights
        refined_embeddings = torch.bmm(attention_weights, v)
        return refined_embeddings

class EnhancedEmbeddingGenerator:
    def __init__(self, 
                 model_name: str = 'all-mpnet-base-v2',
                 embedding_dim: int = 768,
                 device: str = None,
                 **kwargs):
        """Initialize the enhanced embedding generator with attention mechanism.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            embedding_dim: Dimension of embeddings
            device: Device to use for computation ('cuda' or 'cpu')
            **kwargs: Additional arguments for model configuration
        """
        self.logger = logging.getLogger(__name__)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name).to(self.device)
        self.attention_layer = AttentionLayer(embedding_dim).to(self.device)
        self.embedding_dim = embedding_dim
        
    def generate_embeddings(self, 
                          texts: List[str], 
                          batch_size: int = 32,
                          show_progress: bool = True,
                          **kwargs) -> np.ndarray:
        """Generate refined embeddings for input texts using attention mechanism.
        
        Args:
            texts: List of input texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            Numpy array of refined embeddings
        """
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Process in batches
        all_embeddings = []
        iterator = tqdm(range(0, len(texts), batch_size)) if show_progress else range(0, len(texts), batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # Generate base embeddings
            with torch.no_grad():
                base_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                
                # Reshape for attention (add sequence length dimension)
                base_embeddings = base_embeddings.unsqueeze(1)
                
                # Apply attention refinement
                refined_embeddings = self.attention_layer(base_embeddings)
                
                # Remove sequence length dimension and convert to numpy
                refined_embeddings = refined_embeddings.squeeze(1).cpu().numpy()
                all_embeddings.append(refined_embeddings)
        
        # Concatenate all batches
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        self.logger.info(f"Generated embeddings with shape: {final_embeddings.shape}")
        
        return final_embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, path: str):
        """Save embeddings to disk.
        
        Args:
            embeddings: Numpy array of embeddings to save
            path: Path to save the embeddings
        """
        np.save(path, embeddings)
        self.logger.info(f"Saved embeddings to {path}")
    
    def load_embeddings(self, path: str) -> np.ndarray:
        """Load embeddings from disk.
        
        Args:
            path: Path to load the embeddings from
            
        Returns:
            Loaded embeddings as numpy array
        """
        embeddings = np.load(path)
        self.logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
        return embeddings

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    texts = [
        "This is a sample text.",
        "Another example document.",
        "Testing the embedding generator."
    ]
    
    generator = EnhancedEmbeddingGenerator()
    embeddings = generator.generate_embeddings(texts)
    print(f"Generated embeddings shape: {embeddings.shape}")

EmbeddingGenerator = EnhancedEmbeddingGenerator  # Alias for backward compatibility