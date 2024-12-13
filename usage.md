# Usage Guide

## Quick Commands

### 1. Basic Operation
```bash
# Process default dataset
python run_optimized.py

# With custom dataset
python run_optimized.py --input data/my_documents/ --output results/

# With specific configuration
python run_optimized.py --config custom_config.yaml
```

### 2. Dataset Processing

```bash
# Process XL-Sum dataset
python run_optimized.py --dataset xlsum --language english

# Process ScisummNet dataset
python run_optimized.py --dataset scisummnet

# Process custom documents
python run_optimized.py --input "path/to/documents/*/*.txt" --format txt
```

### 3. Performance Optimization

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python run_optimized.py

# Optimize batch size
python run_optimized.py --batch-size auto

# Multi-worker processing
python run_optimized.py --workers 4
```

### 4. Checkpointing & Recovery

```bash
# Enable checkpointing
python run_optimized.py --enable-checkpoints

# Resume from checkpoint
python run_optimized.py --resume --checkpoint-dir checkpoints/

# Force new run
python run_optimized.py --no-resume
```

### 5. Visualization & Dashboard

```bash
# Start dashboard
python src/dashboard/app.py

# Generate static visualizations
python run_optimized.py --visualize-only

# Export results as HTML
python run_optimized.py --export html
```

## Common Use Cases

### Processing Your Own Documents

1. **Directory Structure**:
```
data/
├── input/
│   ├── documents.json
│   ├── papers.csv
│   └── texts/
│       ├── doc1.txt
│       └── doc2.txt
```

2. **JSON Format**:
```json
{
  "documents": [
    {
      "text": "Document content here",
      "metadata": {
        "id": "doc1",
        "category": "research"
      }
    }
  ]
}
```

3. **CSV Format**:
```csv
id,text,category
doc1,"Document text here",research
```

4. **Command**:
```bash
python run_optimized.py \
  --input data/input/documents.json \
  --format json \
  --output results/ \
  --batch-size 32
```

### Custom Dataset Integration

1. **Create Dataset Config**:
```yaml
# config/custom_dataset.yaml
data:
  name: "my_dataset"
  format: "json"
  text_field: "content"
  metadata_fields: ["id", "category"]
```

2. **Run Processing**:
```bash
python run_optimized.py \
  --config config/custom_dataset.yaml \
  --input data/my_dataset/ \
  --output results/my_dataset/
```

### Batch Processing Large Collections

1. **Enable Memory Optimization**:
```bash
python run_optimized.py \
  --input large_dataset/ \
  --batch-size auto \
  --optimize-memory \
  --checkpoints
```

2. **Monitor Progress**:
```bash
tail -f logs/pipeline.log
```

### Advanced Features

1. **Custom Preprocessing**:
```bash
python run_optimized.py \
  --preprocess-config config/preprocess.yaml \
  --custom-filters "remove_citations,clean_latex"
```

2. **Style-based Summarization**:
```bash
python run_optimized.py \
  --summary-style technical \
  --max-length 200 \
  --min-length 50
```

3. **Export Options**:
```bash
python run_optimized.py \
  --export-format json \
  --export-metrics \
  --include-embeddings
```

## Tips & Troubleshooting

### Memory Issues
```bash
# Reduce batch size
python run_optimized.py --batch-size 16

# Enable memory optimization
python run_optimized.py --optimize-memory
```

### CUDA Issues
```bash
# Force CPU usage
python run_optimized.py --device cpu

# Select specific GPU
CUDA_VISIBLE_DEVICES=0 python run_optimized.py
```

### Data Validation
```bash
# Validate input data
python run_optimized.py --validate-only

# Show detailed validation results
python run_optimized.py --validate-only --verbose
```

For more detailed information, refer to [README.md](README.md) and configuration examples in `config/`.
