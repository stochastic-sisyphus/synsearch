import os
import sys
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
from typing import List, Dict, Any
from datetime import datetime
from summarization.hybrid_summarizer import HybridSummarizer
from evaluation.metrics import EvaluationMetrics
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.adaptive_summarizer import AdaptiveSummarizer
from src.utils.metrics_utils import calculate_cluster_metrics

# Set up logging with absolute paths
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Print to stdout explicitly
        logging.FileHandler(str(log_file))  # Convert Path to string for logging
    ]
)

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from utils.style_selector import determine_cluster_style, get_style_parameters
from summarization.adaptive_summarizer import AdaptiveSummarizer
from utils.metrics_utils import calculate_cluster_variance, calculate_lexical_diversity, calculate_cluster_metrics
from datasets import load_dataset
from utils.metrics_calculator import MetricsCalculator
from models.adaptive_summarizer import AdaptiveSummarizer
from models.dynamic_cluster_manager import DynamicClusterManager

def get_device():
    """Get the best available device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_optimal_workers():
    """Get optimal number of worker processes."""
    return multiprocessing.cpu_count()

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
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

def process_dataset(
    dataset: Dict[str, Any],
    cluster_manager: DynamicClusterManager,
    summarizer: AdaptiveSummarizer,
    evaluator: EvaluationMetrics,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a single dataset through the pipeline."""
    logger = logging.getLogger(__name__)
    
    # Perform clustering
    labels, clustering_metrics = cluster_manager.fit_predict(dataset['embeddings'])
    logger.info(f"Clustering completed with metrics: {clustering_metrics}")
    
    # Group documents by cluster
    clusters = cluster_manager.get_cluster_documents(dataset['documents'], labels)
    
    # Generate summaries for each cluster
    summaries = {}
    for cluster_id, docs in clusters.items():
        cluster_texts = [doc['text'] for doc in docs]
        cluster_embeddings = np.array([doc['embedding'] for doc in docs])
        
        summary_data = summarizer.summarize_cluster(
            texts=cluster_texts,
            embeddings=cluster_embeddings,
            cluster_id=cluster_id
        )
        summaries[str(cluster_id)] = summary_data
    
    # Calculate comprehensive metrics
    metrics = evaluator.calculate_comprehensive_metrics(
        summaries=summaries,
        references=dataset.get('references', {}),
        embeddings=dataset['embeddings']
    )
    
    # Save results
    results = {
        'clustering_metrics': clustering_metrics,
        'summaries': summaries,
        'evaluation_metrics': metrics
    }
    
    evaluator.save_metrics(
        metrics=results,
        output_dir=config['data']['output_path'],
        prefix=dataset.get('name', 'unnamed_dataset')
    )
    
    return results

def main():
    """Main pipeline execution."""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Load configuration
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Initialize components
        data_loader = DataLoader(config)
        preprocessor = DomainAgnosticPreprocessor(config)
        embedding_generator = EnhancedEmbeddingGenerator(
            model_name=config['embedding']['model_name'],
            embedding_dim=config['embedding']['dimension'],
            max_seq_length=config['embedding']['max_length']
        )
        cluster_manager = DynamicClusterManager(config)
        summarizer = AdaptiveSummarizer(config)
        evaluator = EvaluationMetrics()
        
        # Process pipeline
        dataset = data_loader.load_data(config['data']['source'])
        processed_data = preprocessor.process_dataset(dataset)
        embeddings = embedding_generator.generate_embeddings(processed_data['processed_text'])
        clusters = cluster_manager.cluster_documents(embeddings)
        summaries = summarizer.summarize_clusters(clusters, processed_data)
        metrics = evaluator.evaluate(summaries, processed_data)
        
        logger.info("Pipeline completed successfully")
        return metrics
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    main()