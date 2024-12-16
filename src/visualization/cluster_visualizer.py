import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

class ClusterVisualizer:
    def __init__(self):
        self.reducer = umap.UMAP(random_state=42)
        
    def plot_clusters(self, embeddings: np.ndarray, labels: np.ndarray = None):
        reduced = self.reducer.fit_transform(embeddings)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                            c=labels if labels is not None else None)
        if labels is not None:
            plt.colorbar(scatter)
        plt.show()