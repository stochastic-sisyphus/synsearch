# Research Paper Analysis Tool

A tool for analyzing research papers from arXiv using clustering and summarization.

## Features
- Search arXiv papers by topic
- Cluster similar papers together
- Generate summaries for paper clusters
- Interactive paper analysis

## Setup

### Prerequisites
- Python 3.11 (recommended)
- pip or conda package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd synsearch
```

2. Create virtual environment (choose one method):

**Using venv (recommended):**
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Using conda:**
```bash
conda create -n synsearch python=3.11
conda activate synsearch
conda install numpy=1.24.3
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```

## Usage

### 1. Quick Paper Analysis
The simplest way to analyze papers:
```bash
python my_analysis.py
```
This will:
- Prompt for your search query
- Ask for number of papers to analyze
- Show clustered results
- Allow multiple searches

### 2. Detailed Analysis
For more detailed analysis with visualizations:
```bash
python scripts/analyze_papers.py
```

### 3. Quick Start Example
For a simple predefined analysis:
```bash
python quick_start.py
```

## Search Tips

### Basic Queries
- Single topic: `quantum computing`
- Multiple topics: `quantum computing AND machine learning`
- Title search: `ti:quantum`
- Author search: `au:Smith`

### Advanced Queries
- Date range: `submittedDate:[20230101 TO 20240101]`
- Categories: 
  - Computer Science: `cat:cs.AI`
  - Physics: `cat:physics`
  - Mathematics: `cat:math`

## Output Structure
```
outputs/
├── papers/              # Retrieved papers
├── clusters/           # Clustering results
├── summaries/         # Generated summaries
└── visualizations/    # Interactive plots
```

## Troubleshooting

### Common Issues
1. **Installation Problems**
   ```bash
   # If using Python 3.12, switch to 3.11:
   brew install python@3.11
   python3.11 -m venv .venv
   ```

2. **Memory Issues**
   - Reduce number of papers in search
   - Close other applications

3. **No Results**
   - Check internet connection
   - Try broader search terms
   - Verify query syntax

## Contributing
1. Fork the repository
2. Create your feature branch
3. Submit a pull request

## License
MIT License