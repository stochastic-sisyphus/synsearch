# Dynamic Summarization and Adaptive Clustering

A framework for real-time research synthesis using dynamic clustering and abstractive summarization. This project combines attention-based embedding refinement, adaptive clustering algorithms, and style-aware summarization to provide comprehensive research synthesis capabilities.

## Requirements

### System Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- CUDA-compatible GPU recommended for faster processing
- 2GB free disk space

### Dependencies
See `requirements.txt` for complete list of dependencies.

## Quick Start

1. **Install the package**
```bash
pip install -r requirements.txt
```

2. **Run the pipeline**
```bash
# Basic usage with default settings
python -m src.main

# With custom config file
python -m src.main --config path/to/custom_config.yaml

# With specific input and output paths
python -m src.main --input "data/documents/" --output "results/"
```

3. **Expected Output Structure**
```
outputs/
├── checkpoints/          # Processing checkpoints
├── embeddings/          # Generated embeddings
├── clusters/           # Clustering results
├── summaries/         # Generated summaries
└── figures/          # Visualizations
```

## Key Features

### Adaptive Style Selection
- Automatic style adaptation based on cluster characteristics
- Support for technical, detailed, and concise summary styles
- Real-time style optimization based on content complexity

### Domain-Agnostic Processing
- Universal preprocessing pipeline
- Adaptive feature extraction
- Cross-domain compatibility

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
│   └─��� cluster_distribution.png
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

## Detailed Instructions for Running the Pipeline

### Step-by-Step Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/dynamic-summarization.git
cd dynamic-summarization
```

2. **Create and activate a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare your data**
   - For XL-Sum dataset:
     ```python
     from datasets import load_dataset
     dataset = load_dataset('GEM/xlsum', 'english')
     ```
   - For ScisummNet dataset:
     ```python
     from src.data_loader import DataLoader
     loader = DataLoader()
     data = loader.load_scisummnet('path/to/scisummnet')
     ```

5. **Run the pipeline**
```bash
python -m src.main --input "data/documents/" --output "results/"
```

### Usage Examples for Each Module

1. **Data Preparation**
   ```python
   from src.data_preparation import DataPreparator
   preparator = DataPreparator()
   processed_data = preparator.process(raw_data)
   ```

2. **Embedding Generation**
   ```python
   from src.embedding_generator import EnhancedEmbeddingGenerator
   generator = EnhancedEmbeddingGenerator(model_name='all-mpnet-base-v2')
   embeddings = generator.generate_embeddings(processed_data)
   ```

3. **Clustering**
   ```python
   from src.clustering.dynamic_cluster_manager import DynamicClusterManager
   cluster_manager = DynamicClusterManager(config={'min_cluster_size': 5, 'min_samples': 3})
   clusters = cluster_manager.fit_predict(embeddings)
   ```

4. **Summarization**
   ```python
   from src.summarization.hybrid_summarizer import HybridSummarizer
   summarizer = HybridSummarizer(model_name='facebook/bart-large-cnn', max_length=150, min_length=50)
   summaries = summarizer.summarize_all_clusters(clusters, style='technical')
   ```

5. **Visualization**
   ```python
   from src.visualization.embedding_visualizer import EmbeddingVisualizer
   visualizer = EmbeddingVisualizer(config={'method': 'umap', 'n_neighbors': 15})
   visualizer.plot_embeddings(embeddings, labels, output_path='outputs/figures')
   ```

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

### Example Code for Evaluation

```python
from src.evaluation.metrics import EvaluationMetrics

metrics = EvaluationMetrics()
scores = metrics.calculate_rouge_scores(generated_summaries, reference_summaries)
print(f"ROUGE-L F1: {scores['rougeL']['fmeasure']:.3f}")

bleu_scores = metrics.calculate_bleu_scores(generated_summaries, reference_summaries)
print(f"BLEU Score: {bleu_scores['bleu']:.3f}")
```

# config/config.yaml
embedding:
  model_name: 'all-mpnet-base-v2'
  batch_size: 32
  max_seq_length: 512

clustering:
  algorithm: 'hdbscan'
  min_cluster_size: 5
  min_samples: 3
  metric: 'euclidean'

summarization:
  model_name: 'facebook/bart-large-cnn'
  styles:
    technical:
      max_length: 150
      min_length: 50
    concise:
      max_length: 100
      min_length: 30
