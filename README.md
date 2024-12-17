# SynSearch

SynSearch is an advanced document processing and semantic search system that combines embedding generation, clustering, and summarization capabilities to effectively process and analyze large collections of text documents.

## 🌟 Features

- **Document Processing Pipeline**
  - Domain-agnostic text preprocessing
  - Supports multiple dataset formats
  - Efficient batch processing capabilities

- **Advanced Embedding Generation**
  - Transformer-based embeddings
  - Configurable model selection
  - GPU acceleration support
  - Optimized batch processing

- **Dynamic Clustering**
  - Adaptive clustering algorithms
  - Theme-based document grouping
  - Support for multiple clustering strategies

- **Intelligent Summarization**
  - Hybrid summarization approach
  - Support for scientific and legal domains
  - Cluster-based summary generation
  - Configurable summary length

- **ArXiv Integration**
  - Direct ArXiv paper search
  - Batch paper fetching
  - Rate-limited API handling

## 📋 Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)
- Required Python packages:
  - torch
  - transformers
  - pandas
  - numpy
  - spacy
  - pyyaml

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/stochastic-sisyphus/synsearch.git
cd synsearch
```

2. Set up a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required datasets:
```bash
make download-data
```

## ⚙️ Configuration

The system is configured through YAML files located in the `config` directory. Key configuration areas include:

- Data sources and paths
- Embedding model settings
- Preprocessing parameters
- Clustering configuration
- Summarization options

Example configuration:
```yaml
data:
  datasets:
    - name: scisummnet
      enabled: true
  scisummnet_path: "path/to/dataset"

embedding:
  model_name: "bert-base-uncased"
  dimension: 768
  max_seq_length: 512
  batch_size: 32

preprocessing:
  # Preprocessing settings

clustering:
  # Clustering settings

summarization:
  # Summarization settings
```

## 🔨 Usage

1. Basic usage:
```python
from src.main import main

# Run the complete pipeline
main()
```

2. Using specific components:
```python
from src.preprocessing.domain_agnostic_preprocessor import DomainAgnosticPreprocessor
from src.embedding_generator import EnhancedEmbeddingGenerator

# Initialize components
preprocessor = DomainAgnosticPreprocessor()
embedding_generator = EnhancedEmbeddingGenerator(model_name="bert-base-uncased")

# Process texts
processed_texts = preprocessor.preprocess_texts(your_texts)
embeddings = embedding_generator.generate_embeddings(processed_texts)
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Key test areas include:
- Preprocessing functionality
- ArXiv API integration
- Embedding generation
- Clustering algorithms
- Summarization quality

## 📁 Project Structure

```
synsearch/
├── src/
│   ├── api/              # API integrations
│   ├── preprocessing/    # Text preprocessing
│   ├── clustering/       # Clustering algorithms
│   ├── summarization/    # Summary generation
│   └── utils/            # Utility functions
├── tests/               # Test suite
├── config/             # Configuration files
├── data/              # Dataset storage
└── outputs/           # Generated outputs
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License
