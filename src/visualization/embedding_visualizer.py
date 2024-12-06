import numpy as np
import umap
import plotly.express as px
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional, Union

class EmbeddingVisualizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.umap_reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """Reduce embedding dimensions using UMAP"""
        try:
            if fit:
                return self.umap_reducer.fit_transform(embeddings)
            return self.umap_reducer.transform(embeddings)
        except Exception as e:
            self.logger.error(f"Failed to reduce dimensions: {e}")
            raise
    
    def visualize_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: Optional[pd.DataFrame] = None,
        color_column: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> Dict:
        """Generate interactive visualization of embeddings"""
        try:
            # Reduce dimensions
            reduced_embeddings = self.reduce_dimensions(embeddings)
            
            # Prepare visualization data
            viz_data = pd.DataFrame(
                reduced_embeddings,
                columns=['UMAP1', 'UMAP2']
            )
            
            if metadata is not None and color_column and color_column in metadata.columns:
                viz_data[color_column] = metadata[color_column]
            
            # Create plot
            fig = px.scatter(
                viz_data,
                x='UMAP1',
                y='UMAP2',
                color=color_column if color_column else None,
                title='Document Embeddings Visualization',
                template='plotly_white'
            )
            
            # Save if path provided
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(save_path)
                self.logger.info(f"Saved visualization to {save_path}")
            
            return {
                'figure': fig,
                'reduced_embeddings': reduced_embeddings
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create visualization: {e}")
            raise 