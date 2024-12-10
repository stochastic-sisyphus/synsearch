import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union, Dict, Any
import logging
import gc  # Add garbage collector
from pathlib import Path
from datetime import datetime
from tqdm import tqdm  # Add tqdm import
from torch.utils.data import DataLoader, Dataset

class AttentionLayer(nn.Module):
    """Attention layer for refining embeddings."""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attention = nn.Linear(embedding_dim, 1)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism to embeddings."""
        attention_weights = torch.softmax(self.attention(embeddings), dim=0)
        return embeddings * attention_weights

class EmbeddingDataset(Dataset):
    """Custom Dataset for embeddings."""
    
    def __init__(self, texts: List[str]):
        self.texts = texts
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

class EnhancedEmbeddingGenerator:
    """Generates embeddings with memory management and attention mechanism."""
    
    def __init__(
        self, 
        model_name: str = 'all-mpnet-base-v2',
        embedding_dim: int = 768,
        max_seq_length: int = 512,
        batch_size: int = 32,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None  # Add config parameter
    ):
        """Initialize the embedding generator with memory management."""
        self.config = config  # Initialize config attribute
        self.logger = logging.getLogger(__name__)
        
        # Determine device with fallback options
        if device is None:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    # Try allocating a small tensor to verify memory
                    torch.zeros((1, embedding_dim), device='cuda')
                except Exception:
                    self.logger.warning("GPU memory issue detected, falling back to CPU")
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

    def __init__(self, **kwargs):
        self.config = kwargs.get('config')
        self.logger = logging.getLogger(__name__)
        
        try:
            self.device = kwargs.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise RuntimeError("Failed to initialize embedding model")
    
    def generate_embeddings(
        self,
        texts: List[str],
        apply_attention: bool = True,
        batch_size: Optional[int] = None,
        cache_dir: Optional[Path] = None
    ) -> np.ndarray:
        """Generate embeddings with optimized batch processing and caching."""
        if not texts:
            raise ValueError("Empty text list provided")
            
        try:
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = cache_dir / 'embeddings.npy'
                if cache_file.exists():
                    return np.load(cache_file)
            
            # Use smaller batch size if memory is limited
            if batch_size is None:
                batch_size = min(self.batch_size, len(texts))
            
            dataset = EmbeddingDataset(texts)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            all_embeddings = []
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                # Clear GPU cache between batches if using CUDA
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    
                with torch.no_grad():
                    embeddings = self.model.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_tensor=True,
                        device=self.device
                    )
                    embeddings = embeddings.cpu().numpy()
                    all_embeddings.append(embeddings)
            
            final_embeddings = np.concatenate(all_embeddings, axis=0)
            
            if cache_dir:
                np.save(cache_file, final_embeddings)
                
            return final_embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise

    def _get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on available memory."""
        if self.device == 'cuda':
            try:
                free_memory = torch.cuda.get_device_properties(0).total_memory
                # Use 80% of available memory and ensure integer result
                return max(1, int((free_memory * 0.8) // (768 * 4)))  # Add int() conversion
            except:
                return 32  # Default GPU batch size
        return 64  # Default CPU batch size

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

    def save_embeddings(self, embeddings: torch.Tensor, path: Path):
        """Save embeddings to disk with metadata."""
        torch.save({
            'embeddings': embeddings,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }, path)
        
    def _validate_checkpoint(self, checkpoint: Dict) -> bool:
        """Validate checkpoint contents."""
        required_keys = ['embeddings', 'config', 'timestamp']
        return all(key in checkpoint for key in required_keys)

    def load_embeddings(self, path: Path) -> torch.Tensor:
        """Load embeddings with validation."""
        checkpoint = torch.load(path)
        self._validate_checkpoint(checkpoint)
        return checkpoint['embeddings']
