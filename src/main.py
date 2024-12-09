import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import logging
from pathlib import Path
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

def load_datasets(config):
    """Load and prepare datasets."""
    datasets = []
    loaded_count = 0
    
    for dataset_config in config['data']['datasets']:
        if not dataset_config.get('enabled', False):
            continue
            
        if dataset_config['name'] == 'scisummnet':
            # Convert to absolute path from project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            scisummnet_path = os.path.join(base_dir, dataset_config['path'], 'top1000_complete')
            if os.path.exists(scisummnet_path):
                logging.info(f"Loading ScisummNet dataset from {scisummnet_path}")
                try:
                    scisummnet_data = {
                        'name': 'scisummnet',
                        'path': scisummnet_path,
                        'data': [],  # Will be populated with documents
                        'batch_size': config['data']['batch_size']
                    }
                    datasets.append(scisummnet_data)
                    loaded_count += 1
                    logging.info("Successfully loaded ScisummNet dataset")
                except Exception as e:
                    logging.error(f"Failed to load ScisummNet dataset: {str(e)}")
            else:
                logging.warning(f"ScisummNet dataset path not found: {scisummnet_path}")
                logging.info("Please download the ScisummNet dataset and place it in:")
                logging.info(f"  {os.path.dirname(scisummnet_path)}")
                logging.info("You can download it from: https://cs.stanford.edu/~myasu/projects/scisumm_net/")
                # Continue with other datasets even if this one fails
                continue
        
        elif dataset_config['name'] == 'xlsum':
            logging.info(f"Loading XL-Sum dataset for language: {dataset_config['language']}")
            try:
                xlsum_data = load_dataset(
                    dataset_config['dataset_name'],
                    dataset_config['language'],
                    split='train'
                )
                datasets.append({
                    'name': 'xlsum',
                    'data': xlsum_data,
                    'batch_size': config['data']['batch_size']
                })
                loaded_count += 1
                logging.info(f"Successfully loaded XL-Sum dataset ({dataset_config['language']})")
            except Exception as e:
                logging.error(f"Failed to load XL-Sum dataset: {str(e)}")
    
    if loaded_count == 0:
        logging.warning("No datasets were successfully loaded")
        if any(d['enabled'] for d in config['data']['datasets']):
            logging.info("Please check your dataset configurations and paths")
    
    return datasets

def main():
    """Main pipeline execution."""
    try:
        # Load configuration with default path
        config = load_config()
        
        # Validate configuration and create directories
        validate_config(config)
        
        # Load datasets
        datasets = load_datasets(config)
        if not datasets and all(d['enabled'] for d in config['data']['datasets']):
            raise ValueError("No datasets were successfully loaded")
        
        # Continue with the rest of your pipeline...
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
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

if __name__ == "__main__":
    main() 