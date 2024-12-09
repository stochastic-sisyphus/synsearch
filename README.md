# Dynamic Summarization and Adaptive Clustering

A framework for real-time research synthesis using dynamic clustering and abstractive summarization. This project combines attention-based embedding refinement, adaptive clustering algorithms, and cluster-aware summarization to provide comprehensive research synthesis capabilities.

## Requirements

### System Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- CUDA-compatible GPU recommended for faster processing
- 2GB free disk space

### Dependencies
```bash
torch>=1.9.0
transformers>=4.11.0
sentence-transformers>=2.0.0
scikit-learn>=0.24.0
numpy>=1.19.0
pandas>=1.3.0
plotly>=5.0.0
streamlit>=1.0.0
```

## Quick Start

1. **Install the package**
```bash
pip install -r requirements.txt
```

2. **Run a simple example**
```python
from src.main import process_documents

# Process a sample document
result = process_documents(
    input_files="examples/sample.txt",
    output_dir="outputs"
)

# Expected output:
# {
#     'clusters': 3,
#     'documents_processed': 10,
#     'processing_time': '2.3s',
#     'summaries': {
#         'cluster_0': 'Summary of first cluster...',
#         'cluster_1': 'Summary of second cluster...',
#         'cluster_2': 'Summary of third cluster...'
#     }
# }
```

3. **View results**
```bash
cat outputs/results/summaries.json
```

## Key Features

### Embedding Generation
- Attention-based embedding refinement for improved semantic representation
- Device-agnostic computation (CPU/GPU support)
- Efficient batch processing
- Built on state-of-the-art Sentence Transformers

### Dynamic Clustering
- Adaptive algorithm selection based on data characteristics
- Multiple clustering methods (K-Means, DBSCAN, HDBSCAN)
- Real-time cluster quality assessment
- Comprehensive clustering metrics tracking

### Advanced Summarization
- Cluster-aware abstractive summarization
- Multiple summarization styles
- Representative text selection
- Batch processing optimization
- Fine-tuned on research paper datasets

### Visualization & Analysis
- Interactive cluster exploration
- Embedding visualization using UMAP/t-SNE
- Real-time metric tracking
- Performance analytics dashboard

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dynamic-summarization.git
cd dynamic-summarization
``` 

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Basic Usage
```python
from src.main import process_documents

# Process a single collection of documents
results = process_documents(
    input_files="path/to/documents",
    output_dir="path/to/output",
    batch_size=32
)
```

### 2. Command Line Interface

Run the pipeline from command line:

```bash
# Process documents with default settings
python -m src.main --input "path/to/documents" --output "path/to/output"

# Process with custom configuration
python -m src.main --config "path/to/custom_config.yaml"
```

### 3. Step-by-Step Guide

1. **Prepare Your Data**
   ```python
   from src.data_loader import DataLoader
   from src.data_preparation import DataPreparator
   
   # Load data
   loader = DataLoader()
   data = loader.load_xlsum()  # or load_scisummnet() for scientific papers
   
   # Prepare data
   preparator = DataPreparator()
   processed_data = preparator.process(data)
   ```

2. **Generate Embeddings**
   ```python
   from src.embedding_generator import EnhancedEmbeddingGenerator
   
   generator = EnhancedEmbeddingGenerator(model_name='all-mpnet-base-v2')
   embeddings = generator.generate_embeddings(processed_data)
   ```

3. **Perform Clustering**
   ```python
   from src.clustering.dynamic_cluster_manager import DynamicClusterManager
   
   cluster_manager = DynamicClusterManager(config={
       'clustering': {
           'min_cluster_size': 5,
           'min_samples': 3
       }
   })
   clusters = cluster_manager.fit_predict(embeddings)
   ```

4. **Generate Summaries**
   ```python
   from src.summarization.hybrid_summarizer import HybridSummarizer
   
   summarizer = HybridSummarizer(
       model_name='facebook/bart-large-cnn',
       max_length=150,
       min_length=50
   )
   summaries = summarizer.summarize_all_clusters(clusters, style='technical')
   ```

### 4. Configuration

Create a custom configuration file (`config.yaml`):

```yaml
embedding:
  model_name: 'all-mpnet-base-v2'
  batch_size: 32
  max_length: 512

clustering:
  min_cluster_size: 5
  min_samples: 3
  algorithm: 'auto'  # 'auto', 'kmeans', 'hdbscan', or 'dbscan'

summarization:
  model_name: 'facebook/bart-large-cnn'
  max_length: 150
  min_length: 50
  style: 'technical'  # 'concise', 'detailed', or 'technical'

visualization:
  enabled: true
  method: 'umap'  # 'umap' or 'tsne'
  output_dir: 'outputs/figures'
```

### 5. Advanced Features

#### Custom Preprocessing
```python
from src.preprocessor import DomainAgnosticPreprocessor

preprocessor = DomainAgnosticPreprocessor(config={
    'remove_citations': True,
    'max_length': 1000
})
processed_texts = preprocessor.process_dataset(data)
```

#### Checkpoint Management
```python
from src.utils.checkpoint_manager import CheckpointManager

checkpoint_manager = CheckpointManager(
    checkpoint_dir='outputs/checkpoints',
    enable_metrics=True
)

# Save checkpoint
checkpoint_manager.save_stage('embedding', {
    'embeddings': embeddings,
    'config': config
})

# Load checkpoint
embedding_state = checkpoint_manager.get_stage_data('embedding')
```

#### Visualization
```python
from src.visualization.embedding_visualizer import EmbeddingVisualizer

visualizer = EmbeddingVisualizer(config={
    'method': 'umap',
    'n_neighbors': 15
})
visualizer.plot_embeddings(embeddings, labels, output_path='outputs/figures')
```

### 6. Output Structure

```
outputs/
├── checkpoints/
│   ├── embedding/
│   ├── clustering/
│   └── summarization/
├── figures/
│   ├── embeddings_umap.png
│   └── cluster_distribution.png
└── results/
    ├── clusters.json
    ├── summaries.json
    └── metrics.json
```

### 7. Evaluation

```python
from src.evaluation.metrics import EvaluationMetrics

metrics = EvaluationMetrics()
scores = metrics.calculate_rouge_scores(generated_summaries, reference_summaries)
print(f"ROUGE-L F1: {scores['rougeL']['fmeasure']:.3f}")
```

## Project Structure
dynamic-summarization/
├── config/
│ └── config.yaml
├── src/
│ ├── embedding_generator.py # Attention-based embedding
│ ├── clustering/ # Dynamic clustering algorithms
│ ├── summarization/ # Enhanced summarization
│ ├── visualization/ # Dashboard components
│ └── utils/ # Helper functions
├── tests/ # Unit tests
└── examples/ # Usage examples


## Supported Datasets

- XL-Sum Dataset (via Hugging Face)
- ScisummNet
- Custom document collections (PDF, TXT, CSV)

## Evaluation Metrics

The framework tracks multiple metrics for quality assessment:

- **Embedding Quality**
  - Intra-cluster cosine similarity
  - Dimension reduction quality

- **Clustering Performance**
  - Silhouette score
  - Davies-Bouldin index
  - Cluster stability metrics

- **Summarization Quality**
  - ROUGE scores
  - BLEU scores
  - Summary coherence metrics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:
bibtex
@software{dynamic_summarization,
title = {Dynamic Summarization and Adaptive Clustering},
author = {Your Name},
year = {2024},
url = {https://github.com/yourusername/dynamic-summarization}
}
```
