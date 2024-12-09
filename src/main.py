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
from clustering.dynamic_cluster_manager import DynamicClusterManager
from typing import List, Dict, Any
from datetime import datetime
from summarization.hybrid_summarizer import HybridSummarizer
from evaluation.metrics import EvaluationMetrics
from utils.metrics_utils import calculate_cluster_metrics

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
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_datasets(config):
    """Load and prepare datasets based on configuration."""
    datasets = {'texts': [], 'summaries': []}
    
    # Load XLSum dataset if enabled
    xlsum_config = next((d for d in config['data']['datasets'] if d['name'] == 'xlsum'), None)
    if xlsum_config and xlsum_config['enabled']:
        logger = logging.getLogger(__name__)
        logger.info("Loading XLSum dataset...")
        xlsum = load_dataset('GEM/xlsum')
        datasets['texts'].extend(xlsum['train']['text'][:config['data'].get('batch_size', 100)])
        datasets['summaries'].extend(xlsum['train']['summary'][:config['data'].get('batch_size', 100)])
    
    # Load ScisummNet dataset if enabled
    scisummnet_config = next((d for d in config['data']['datasets'] if d['name'] == 'scisummnet'), None)
    if scisummnet_config and scisummnet_config['enabled']:
        logger = logging.getLogger(__name__)
        logger.info("Loading ScisummNet dataset...")
        scisummnet_path = Path(scisummnet_config['path'])
        if scisummnet_path.exists():
            top1000_dir = scisummnet_path / scisummnet_config['top1000_dir']
            if top1000_dir.exists():
                # Load papers up to the configured limit
                paper_limit = scisummnet_config.get('papers', 100)
                paper_count = 0
                
                for paper_dir in top1000_dir.iterdir():
                    if paper_count >= paper_limit:
                        break
                    if paper_dir.is_dir():
                        try:
                            # Load abstract and summary
                            abstract_path = paper_dir / 'Abstract.txt'
                            summary_path = paper_dir / 'Summary.txt'
                            
                            if abstract_path.exists() and summary_path.exists():
                                with open(abstract_path, 'r', encoding='utf-8') as f:
                                    abstract = f.read().strip()
                                with open(summary_path, 'r', encoding='utf-8') as f:
                                    summary = f.read().strip()
                                    
                                datasets['texts'].append(abstract)
                                datasets['summaries'].append(summary)
                                paper_count += 1
                        except Exception as e:
                            logger.warning(f"Error loading paper from {paper_dir}: {str(e)}")
                            continue
    
    if not datasets['texts']:
        raise ValueError("No datasets were successfully loaded")
    
    return datasets

def save_results(results, summaries, output_path):
    """Save results and summaries to output directory."""
    os.makedirs(output_path, exist_ok=True)
    
    # Save results
    results_path = os.path.join(output_path, 'results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    
    # Save summaries
    summaries_path = os.path.join(output_path, 'summaries.txt')
    with open(summaries_path, 'w') as f:
        for cluster_id, summary in summaries.items():
            f.write(f"Cluster {cluster_id}:\n{summary}\n\n")

def main():
    # Load configuration
    config = load_config()
    
    # Configure logging
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format']
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load datasets
        logger.info("Loading datasets...")
        dataset = load_datasets(config)
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        cluster_manager = DynamicClusterManager(config)
        summarizer = AdaptiveSummarizer(config)
        metrics_calc = EvaluationMetrics()
        
        # Generate embeddings and perform clustering
        logger.info("Generating embeddings and clustering...")
        embeddings = cluster_manager.generate_embeddings(dataset['texts'])
        clusters = cluster_manager.fit_predict(embeddings)
        
        # Generate adaptive summaries
        logger.info("Generating summaries...")
        summaries = summarizer.summarize_clusters(
            clusters=clusters,
            texts=dataset['texts']
        )
        
        # Calculate metrics
        logger.info("Computing metrics...")
        results = metrics_calc.compute_all_metrics(
            embeddings=embeddings,
            clusters=clusters,
            summaries=summaries,
            references=dataset.get('summaries', None)  # Pass reference summaries if available
        )
        
        # Save results
        logger.info("Saving results...")
        save_results(results, summaries, config['data']['output_path'])
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 