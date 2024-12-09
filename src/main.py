import os
import sys
import logging

# Set up logging at the start of the script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler('pipeline.log')  # Save to file
    ]
)

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import yaml
import pandas as pd
from data_loader import DataLoader
from data_preparation import DataPreparator
from data_validator import DataValidator, ConfigValidator
from utils.logging_config import setup_logging
from embedding_generator import EnhancedEmbeddingGenerator
from visualization.embedding_visualizer import EmbeddingVisualizer
import numpy as np
from preprocessor import TextPreprocessor, DomainAgnosticPreprocessor
from clustering.dynamic_cluster_manager import DynamicClusterManager
from typing import List, Dict, Any
from datetime import datetime
from summarization.hybrid_summarizer import HybridSummarizer
from evaluation.metrics import EvaluationMetrics
import json
from utils.checkpoint_manager import CheckpointManager
try:
    from dashboard.app import DashboardApp
except ImportError as e:
    print(f"Warning: Dashboard functionality not available - {str(e)}")
    print("To enable dashboard, install required packages: pip install dash plotly umap-learn")
    DashboardApp = None

import torch
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from utils.style_selector import determine_cluster_style, get_style_parameters
from summarization.adaptive_summarizer import AdaptiveSummarizer
from utils.metrics_utils import calculate_cluster_variance, calculate_lexical_diversity, calculate_cluster_metrics
from datasets import load_dataset

def get_device():
    """Get the best available device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_optimal_workers():
    """Get optimal number of worker processes."""
    return multiprocessing.cpu_count()

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_texts(texts: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Process texts with adaptive summarization and enhanced metrics."""
    # Initialize components with config settings
    embedding_gen = EnhancedEmbeddingGenerator(
        model_name=config['embedding']['model_name'],
        batch_size=config['embedding']['batch_size'],
        max_seq_length=config['embedding']['max_seq_length']
    )
    
    # Generate embeddings with performance settings
    embeddings = embedding_gen(texts)
    
    # Process clusters with metrics
    results = process_clusters(texts, embeddings, config)
    
    return results

def group_texts_by_similarity(texts: List[str], embeddings: np.ndarray) -> Dict[int, List[str]]:
    """Group texts into clusters based on embedding similarity."""
    # ... existing clustering logic ...
    return {0: texts}  # Placeholder - implement your clustering logic

def validate_config(config):
    """Validate configuration and create directories."""
    required_dirs = [
        ('input_path', config['data']['input_path']),
        ('output_path', config['data']['output_path']),
        ('processed_path', config['data']['processed_path']),
        ('visualization_output', config['visualization']['output_dir']),
        ('checkpoints_dir', config['checkpoints']['dir'])
    ]
    
    for dir_key, dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        if not os.access(dir_path, os.W_OK):
            raise ValueError(f"No write permission for path: {dir_path} ({dir_key})")
    
    # Validate dataset configurations
    for dataset_config in config['data']['datasets']:
        if dataset_config['name'] == 'xlsum' and dataset_config.get('enabled', False):
            if 'language' not in dataset_config:
                raise ValueError("XL-Sum dataset requires 'language' specification in config")
            if 'dataset_name' not in dataset_config:
                raise ValueError("XL-Sum dataset requires 'dataset_name' specification in config")
        elif dataset_config['name'] == 'scisummnet' and dataset_config.get('enabled', False):
            if 'path' not in dataset_config:
                raise ValueError("ScisummNet dataset requires 'path' specification in config")

def get_dataset_path(dataset_config):
    """Helper function to resolve dataset paths"""
    if dataset_config['name'] == 'scisummnet':
        # Get the absolute path of where the script is being run from
        current_dir = os.getcwd()
        logging.info(f"Current working directory: {current_dir}")
        
        # Construct path directly to scisummnet directory
        dataset_path = os.path.join(current_dir, 'data', 'scisummnet_release1.1__20190413')
        
        # Log paths and directory contents
        logging.info(f"Looking for dataset at: {dataset_path}")
        
        # Check if data directory exists
        data_dir = os.path.join(current_dir, 'data')
        if not os.path.exists(data_dir):
            logging.error(f"Data directory not found at: {data_dir}")
            logging.error("Please create the data directory and copy the dataset:")
            logging.error("1. mkdir -p data")
            logging.error("2. Copy scisummnet_release1.1__20190413 to the data directory")
            return dataset_path
            
        # Check if dataset exists
        if os.path.exists(dataset_path):
            logging.info(f"Found dataset directory. Contents: {os.listdir(dataset_path)}")
            return dataset_path
        else:
            logging.error(f"Dataset directory not found at: {dataset_path}")
            logging.error("Please copy the dataset from your local machine:")
            logging.error("scp -r /Users/vanessa/Dropbox/synsearch/data/scisummnet_release1.1__20190413 charhub@charhub-inference-0:~/vb/synsearch/data/")
            
        return dataset_path
    return dataset_config['path']

def load_datasets(config):
    """Load and prepare datasets based on configuration"""
    datasets = []
    
    for dataset_config in config['data']['datasets']:
        if not dataset_config.get('enabled', True):
            continue
            
        if dataset_config['name'] == 'xlsum':
            try:
                dataset = load_dataset(
                    dataset_config['dataset_name'],
                    dataset_config['language']
                )
                datasets.append({
                    'name': 'xlsum',
                    'data': dataset
                })
                logging.info(f"Successfully loaded XL-Sum dataset for {dataset_config['language']}")
            except Exception as e:
                logging.error(f"Failed to load XL-Sum dataset: {str(e)}")
                continue
                
        elif dataset_config['name'] == 'scisummnet':
            dataset_path = get_dataset_path(dataset_config)
            
            # Log the actual path and check if it exists
            abs_path = os.path.abspath(dataset_path)
            logging.info(f"Checking ScisummNet path: {abs_path}")
            
            if not os.path.exists(dataset_path):
                logging.warning(f"ScisummNet dataset path not found: {dataset_path}")
                logging.warning("Expected path structure: ./data/scisummnet_release1.1__20190413")
                continue
            
            try:
                scisummnet_data = {
                    'name': 'scisummnet',
                    'path': dataset_path,
                    'data': load_scisummnet(dataset_path)
                }
                datasets.append(scisummnet_data)
                logging.info(f"Successfully loaded ScisummNet dataset from {dataset_path}")
            except Exception as e:
                logging.error(f"Failed to load ScisummNet dataset: {str(e)}")
                continue
    
    return datasets

def main():
    """Main pipeline execution."""
    try:
        logging.info("Starting pipeline execution...")
        
        # Load configuration with default path
        config = load_config()
        logging.info("Loaded configuration")
        
        # Validate configuration and create directories
        validate_config(config)
        logging.info("Validated configuration")
        
        # Load datasets
        logging.info("Loading datasets...")
        datasets = load_datasets(config)
        logging.info(f"Loaded {len(datasets)} datasets")
        
        # Check if any enabled datasets were loaded
        enabled_datasets = [d for d in config['data']['datasets'] if d.get('enabled', True)]
        if not datasets:
            if enabled_datasets:
                logging.warning("No datasets were successfully loaded. Attempting to proceed with XL-Sum only.")
                # Try loading XL-Sum as fallback
                try:
                    from datasets import load_dataset
                    dataset = load_dataset('GEM/xlsum', 'english')
                    datasets = [{'name': 'xlsum', 'data': dataset}]
                    logging.info("Successfully loaded XL-Sum dataset as fallback")
                except Exception as e:
                    raise ValueError(f"Failed to load any datasets, including fallback: {str(e)}")
            else:
                raise ValueError("No datasets were enabled in configuration")
        
        # Continue with the rest of your pipeline...
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise
    except ValueError as e:
        logging.error(f"Configuration error: {str(e)}")
        sys.exit(1)

def generate_summaries(cluster_texts: Dict[str, List[str]], config: Dict) -> Dict[str, Dict]:
    """Generate and evaluate summaries for clustered texts with adaptive style selection."""
    # Initialize adaptive summarizer
    summarizer = AdaptiveSummarizer(config)
    metrics_calculator = EvaluationMetrics()
    
    summaries = {}
    for cluster_id, texts in cluster_texts.items():
        # Get cluster characteristics
        cluster_texts = [doc['processed_text'] for doc in texts]
        cluster_embeddings = np.array([doc['embedding'] for doc in texts])
        
        # Determine appropriate style
        style = determine_cluster_style(
            embeddings=cluster_embeddings,
            texts=cluster_texts,
            config=config
        )
        style_params = get_style_parameters(style)
        
        # Generate summary with adaptive style
        summary_data = summarizer.summarize(texts=texts, **style_params)
        
        summaries[cluster_id] = {
            'summary': summary_data['summary'],
            'style': style,
            'metrics': summary_data['metrics']
        }
        
        # Calculate ROUGE if reference summaries available
        if any('reference_summary' in doc for doc in texts):
            references = [doc.get('reference_summary', '') for doc in texts]
            rouge_scores = metrics_calculator.calculate_rouge_scores(
                [summary_data['summary']], 
                references
            )
            summaries[cluster_id]['metrics'].update(rouge_scores)
    
    return summaries

def process_clusters(texts: List[str], embeddings: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process clusters with adaptive summarization and enhanced metrics."""
    # Initialize components with configuration
    cluster_manager = DynamicClusterManager(
        config={
            'clustering': {
                'hybrid_mode': True,
                'params': config['clustering']
            }
        }
    )
    
    # Perform dynamic clustering with data characteristics analysis
    labels, metrics = cluster_manager.fit_predict(embeddings)
    
    # Group documents by cluster
    clusters = cluster_manager.get_cluster_documents(
        [{'text': text} for text in texts],
        labels
    )
    
    # Initialize summarizer with adaptive configuration
    summarizer = AdaptiveSummarizer(config=config)
    
    # Process each cluster for summarization
    summaries = {}
    for cluster_id, docs in clusters.items():
        if cluster_id == -1:  # Skip noise cluster
            continue
            
        cluster_texts = [doc['text'] for doc in docs]
        cluster_embeddings = embeddings[labels == cluster_id]
        
        # Calculate cluster-specific metrics
        cluster_metrics = {
            'size': len(cluster_texts),
            'cohesion': float(np.mean([
                np.linalg.norm(emb - np.mean(cluster_embeddings, axis=0))
                for emb in cluster_embeddings
            ])),
            'variance': float(np.var(cluster_embeddings))
        }
        
        # Generate summary with metrics-based adaptation
        summary_data = summarizer.summarize(
            texts=cluster_texts,
            embeddings=cluster_embeddings,
            metrics=cluster_metrics
        )
        
        summaries[cluster_id] = {
            'summary': summary_data['summary'],
            'style': summary_data['style'],
            'metrics': {
                **summary_data['metrics'],
                **cluster_metrics
            }
        }
    
    return {
        'clusters': clusters,
        'summaries': summaries,
        'clustering_metrics': metrics,
        'data_characteristics': metrics['data_characteristics']
    }

def load_scisummnet(path: str) -> Dict:
    """Load and parse ScisummNet dataset from the given path."""
    logging.info(f"Loading ScisummNet from: {path}")
    try:
        data = []
        papers_path = os.path.join(path, 'top1000_complete')
        
        # Add logging to see what's happening
        logging.info(f"Scanning papers directory: {papers_path}")
        paper_dirs = os.listdir(papers_path)
        logging.info(f"Found {len(paper_dirs)} potential papers")
        
        for paper_dir in paper_dirs:
            paper_path = os.path.join(papers_path, paper_dir)
            if not os.path.isdir(paper_path):
                continue
                
            try:
                summary_path = os.path.join(paper_path, 'summary', f'{paper_dir}.gold.txt')
                xml_path = os.path.join(paper_path, 'Documents_xml', f'{paper_dir}.xml')
                
                # Add logging for file checks
                logging.info(f"Checking paper {paper_dir}:")
                logging.info(f"  Summary path: {summary_path} (exists: {os.path.exists(summary_path)})")
                logging.info(f"  XML path: {xml_path} (exists: {os.path.exists(xml_path)})")
                
                if not os.path.exists(summary_path) or not os.path.exists(xml_path):
                    logging.warning(f"Missing files for paper {paper_dir}")
                    continue
                    
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = f.read().strip()
                    
                with open(xml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    abstract_start = content.find('<abstract>') + len('<abstract>')
                    abstract_end = content.find('</abstract>')
                    abstract = content[abstract_start:abstract_end].strip()
                
                data.append({
                    'paper_id': paper_dir,
                    'abstract': abstract,
                    'summary': summary
                })
                logging.info(f"Successfully loaded paper {paper_dir}")
                
            except Exception as e:
                logging.warning(f"Failed to load paper {paper_dir}: {str(e)}")
                continue
        
        if not data:
            raise ValueError("No valid papers found in the dataset")
            
        logging.info(f"Successfully loaded {len(data)} papers from ScisummNet")
        return {
            'papers': data,
            'metadata': {
                'size': len(data),
                'path': path
            }
        }
        
    except Exception as e:
        logging.error(f"Failed to load ScisummNet dataset: {str(e)}")
        raise

def load_scisummnet_dataset(config):
    dataset_path = config['datasets']['scisummnet']['path']
    papers = config['datasets']['scisummnet'].get('papers', [])
    
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset directory not found at: {dataset_path}")
        logger.warning(f"Expected path: ./data/scisummnet_release1.1__20190413")
        return None

    papers_path = os.path.join(dataset_path, 'top1000_complete')
    
    # If papers list is empty, load all valid papers
    if not papers:
        papers = [d for d in os.listdir(papers_path) 
                 if os.path.isdir(os.path.join(papers_path, d))]
        logging.info(f"Loading all {len(papers)} papers from dataset")
    
    valid_papers = []
    for paper_id in papers:
        paper_path = os.path.join(papers_path, paper_id)
        summary_path = os.path.join(paper_path, 'summary', f'{paper_id}.gold.txt')
        xml_path = os.path.join(paper_path, 'Documents_xml', f'{paper_id}.xml')
        
        if os.path.exists(summary_path) and os.path.exists(xml_path):
            valid_papers.append(paper_id)
        else:
            logger.debug(f"Skipping {paper_id}: missing required files")

    if not valid_papers:
        raise ValueError("No valid papers found in the dataset")
        
    return valid_papers

if __name__ == "__main__":
    main() 