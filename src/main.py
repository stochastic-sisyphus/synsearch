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
from src.summarization.hybrid_summarizer import HybridSummarizer
from evaluation.metrics import EvaluationMetrics
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.adaptive_summarizer import AdaptiveSummarizer
from src.utils.metrics_utils import calculate_cluster_metrics
from src.clustering.clustering_utils import process_clusters  # Update import path
import json  # Add json import
import multiprocessing
from tqdm import tqdm  # Add tqdm import

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
from utils.metrics_utils import calculate_cluster_variance, calculate_lexical_diversity, calculate_cluster_metrics
from datasets import load_dataset
from utils.metrics_calculator import MetricsCalculator
from src.summarization.adaptive_summarizer import AdaptiveSummarizer
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
import random
import numpy as np

def get_device():
    """Get the best available device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_optimal_workers():
    """Get optimal number of worker processes."""
    return multiprocessing.cpu_count()

def setup_logging(config):
    """Configure logging based on config settings."""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format']
    )
    return logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
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
        if (dataset_config['name'] == 'xlsum' and dataset_config.get('enabled', False)):
            if 'language' not in dataset_config:
                raise ValueError("XL-Sum dataset requires 'language' specification in config")
            if 'dataset_name' not in dataset_config:
                raise ValueError("XL-Sum dataset requires 'dataset_name' specification in config")
        elif (dataset_config['name'] == 'scisummnet' and dataset_config.get('enabled', False)):
            if not config['data'].get('scisummnet_path'):
                raise ValueError("ScisummNet dataset requires 'scisummnet_path' in config['data']")
    
    # Validate summarization configuration
    if 'summarization' not in config:
        config['summarization'] = {
            'model_name': 'facebook/bart-large-cnn',
            'style_params': {
                'concise': {'max_length': 100, 'min_length': 30},
                'detailed': {'max_length': 300, 'min_length': 100},
                'technical': {'max_length': 200, 'min_length': 50}
            },
            'num_beams': 4,
            'length_penalty': 2.0,
            'early_stopping': True
        }
    elif 'model_name' not in config['summarization']:
        config['summarization']['model_name'] = 'facebook/bart-large-cnn'
    
    return config

def process_dataset(
    dataset: Dict[str, Any],
    cluster_manager: DynamicClusterManager,
    summarizer: AdaptiveSummarizer,
    evaluator: EvaluationMetrics,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Process dataset with enhanced error handling and metrics tracking."""
    logger = logging.getLogger(__name__)
    start_time = datetime.now()
    
    try:
        # Initialize embedding generator
        embedding_generator = EnhancedEmbeddingGenerator(
            model_name=config['embedding']['model_name'],
            batch_size=config['embedding'].get('batch_size', 32),
            max_seq_length=config['embedding'].get('max_seq_length', 512)
        )

        # Split texts into manageable chunks (e.g., 1000 documents per chunk)
        chunk_size = config.get('processing', {}).get('chunk_size', 1000)
        text_chunks = [dataset['texts'][i:i + chunk_size] 
                      for i in range(0, len(dataset['texts']), chunk_size)]
        
        all_embeddings = []
        for chunk in tqdm(text_chunks, desc="Processing chunks"):
            chunk_embeddings = embedding_generator.generate_embeddings(
                chunk,
                cache_dir=Path(config['checkpoints']['dir']) / dataset['name'] / 'embeddings' 
                if config['checkpoints']['enabled'] else None
            )
            all_embeddings.append(chunk_embeddings)
        
        # Combine all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Validate dataset
        validation_results = DataValidator().validate_dataset(dataset)
        if not validation_results['is_valid']:
            raise ValueError(f"Dataset validation failed: {validation_results['checks']}")
        
        # Perform clustering with metrics
        labels, clustering_metrics = cluster_manager.fit_predict(embeddings)
        
        # Generate summaries with style adaptation
        summaries = {}
        clusters = cluster_manager.get_clusters(dataset['texts'], labels)
        
        for cluster_id, docs in clusters.items():
            optimal_style = determine_cluster_style(docs)
            summary = summarizer.summarize_cluster(docs, style=optimal_style)
            summaries[cluster_id] = summary
        
        # Calculate comprehensive metrics
        metrics = evaluator.calculate_comprehensive_metrics(
            summaries=summaries,
            references=dataset.get('references', {}),
            embeddings=embeddings
        )
        
        # Add runtime metrics
        runtime = (datetime.now() - start_time).total_seconds()
        metrics['runtime'] = {
            'total_seconds': runtime,
            'processed_documents': len(dataset['texts']),
            'documents_per_second': len(dataset['texts']) / runtime
        }
        
        results = {
            'validation': validation_results,
            'clustering': clustering_metrics,
            'summaries': summaries,
            'metrics': metrics
        }
        
        # Save results and metrics
        save_path = Path(config['data']['output_path'])
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / f"{dataset['name']}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
        
    except Exception as e:
        logger.error(f"Error processing dataset {dataset.get('name', 'unknown')}: {e}")
        raise

def get_optimal_batch_size() -> int:
    """Determine optimal batch size based on available CUDA memory."""
    if not torch.cuda.is_available():
        return 32  # Default CPU batch size
        
    total_memory = torch.cuda.get_device_properties(0).total_memory
    reserved_memory = 2 * 1024 * 1024 * 1024  # 2GB reserved
    available_memory = total_memory - reserved_memory
    return max(1, available_memory // (768 * 4))  # Based on embedding dimension

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    try:
        # Set random seeds
        set_random_seeds(42)
        
        # Initialize logger
        logger = logging.getLogger(__name__)
        
        # Load and validate configuration
        config = load_config()
        config = validate_config(config)
        
        # Setup additional logging configuration
        setup_logging(config)
        
        # Initialize pipeline components with their specific config sections
        data_loader = DataLoader(config)
        preprocessor = DomainAgnosticPreprocessor(config['preprocessing'])
        
        # Load datasets
        datasets = data_loader.load_all_datasets()
        
        # Process each dataset
        processed_datasets = {}
        for dataset_name, df in datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Handle different dataset structures
            if dataset_name == 'scisummnet':
                texts = df['summary'].tolist()  # Use summaries for ScisummNet
                ids = df['paper_id'].tolist()  # Use paper_id as ID
            else:
                texts = df['text'].tolist()
                ids = df.get('id', range(len(df))).tolist()  # Fallback to index if no ID
            
            # Preprocess texts
            processed_texts = preprocessor.preprocess_texts(texts)
            
            # Generate embeddings
            embedding_generator = EnhancedEmbeddingGenerator(
                model_name=config['embedding']['model_name'],
                embedding_dim=config['embedding'].get('dimension', 768),
                max_seq_length=config['embedding'].get('max_seq_length', 384),
                batch_size=config['embedding'].get('batch_size', 32),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            embeddings = embedding_generator.generate_embeddings(processed_texts)
            
            processed_datasets[dataset_name] = {
                'texts': processed_texts,
                'embeddings': embeddings,
                'summaries': df['summary'].tolist() if 'summary' in df else None,
                'ids': ids,
                'source': dataset_name
            }
        
        # Initialize clustering manager
        cluster_manager = DynamicClusterManager(config['clustering'])
        
        # Initialize adaptive summarizer
        summarizer = AdaptiveSummarizer(config['summarization'])
        
        # Initialize evaluation metrics
        evaluator = EvaluationMetrics()
        
        # Process each dataset through the pipeline
        for dataset_name, dataset in processed_datasets.items():
            logger.info(f"Running pipeline for dataset: {dataset_name}")
            results = process_dataset(
                dataset=dataset,
                cluster_manager=cluster_manager,
                summarizer=summarizer,
                evaluator=evaluator,
                config=config
            )
            logger.info(f"Pipeline results for {dataset_name}: {results}")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Critical error in pipeline: {str(e)}", exc_info=True)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
