# Detailed Usage Guide

## 1. Basic Paper Analysis

### Using Interactive Tool
```bash
python my_analysis.py
```

Example session:
```
ArXiv Paper Analysis
==================================================

Enter your search query: quantum computing
How many papers to analyze? [50]: 25

Fetching papers...
Generating embeddings...
Clustering papers...

Found 25 papers in 4 clusters

Cluster 0 (8 papers):
- Title: Quantum Computing with Neutral Atoms
  Authors: John Smith, Jane Doe
  Published: 2024-01-15
...
```

### Search Query Examples
1. Basic topic search:
   ```
   machine learning
   ```

2. Multiple topics:
   ```
   quantum computing AND artificial intelligence
   ```

3. Title-specific search:
   ```
   ti:"deep learning"
   ```

4. Author search:
   ```
   au:Smith
   ```

## 2. Detailed Analysis

For more comprehensive analysis:
```bash
python scripts/analyze_papers.py
```

This provides:
- Detailed clustering analysis
- Visualizations
- Saved results in outputs directory

## 3. Quick Start

For a simple predefined analysis:
```bash
python quick_start.py
```

## 4. Output Files

Results are saved in:
```
outputs/
├── papers_[timestamp]/
│   ├── papers.json        # Raw paper data
│   └── metadata.json      # Query information
├── clusters_[timestamp]/
│   └── cluster_data.json  # Clustering results
└── visualizations/
    └── clusters.html      # Interactive visualization
```

## 5. Advanced Features

### Custom Analysis
```python
from src.api.arxiv_api import ArxivAPI
from src.embedding_generator import EnhancedEmbeddingGenerator

# Initialize
api = ArxivAPI()
generator = EnhancedEmbeddingGenerator()

# Fetch papers
papers = api.fetch_papers_batch("your query", max_papers=50)

# Process
embeddings = generator.generate_embeddings([p['summary'] for p in papers])
```

## 6. Best Practices

1. **Start Small**
   - Begin with 25-50 papers
   - Increase if needed

2. **Refine Searches**
   - Use specific terms
   - Add category filters
   - Combine search terms

3. **Save Results**
   - Results automatically saved
   - Check outputs directory

## 7. Troubleshooting

1. **No Results**
   - Broaden search terms
   - Check internet connection
   - Verify query syntax

2. **Slow Performance**
   - Reduce number of papers
   - Close other applications
   - Check memory usage

3. **Installation Issues**
   - Use Python 3.11
   - Create fresh virtual environment
   - Update pip and dependencies
