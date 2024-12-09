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
from data_validator import DataValidator
from utils.logging_config import setup_logging
from embedding_generator import EmbeddingGenerator
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

def get_device():
    """Get the best available device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_optimal_workers():
    """Get optimal number of worker processes."""
    return multiprocessing.cpu_count()

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_texts(texts: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Process texts with adaptive summarization and enhanced metrics."""
    # Initialize components with config settings
    embedding_gen = EmbeddingGenerator(
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

def main():
    # Setup logging
    setup_logging('logs/processing.log')
    logger = logging.getLogger(__name__)
    
    try:
        # Load config
        config = load_config()
        
        # Initialize checkpoint manager with metrics tracking
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.get('checkpoints', {}).get('dir', 'outputs/checkpoints'),
            enable_metrics=True
        )
        
        # Process datasets with checkpointing
        if not checkpoint_manager.is_stage_complete('data_loading'):
            start_time = datetime.now()
            # Initialize components
            loader = DataLoader(config['data']['scisummnet_path'])
            preprocessor = DomainAgnosticPreprocessor(config['preprocessing'])
            
            # Load and process XL-Sum
            xlsum = loader.load_xlsum()
            if xlsum:
                # Convert Dataset to DataFrame
                xlsum_df = pd.DataFrame(xlsum['train'])
                processed_xlsum = preprocessor.process_dataset(
                    xlsum_df,
                    text_column='text',
                    summary_column='summary'
                )
                logger.info(f"Processed {len(processed_xlsum)} XL-Sum documents")
            
            # Load and process ScisummNet
            scisummnet = loader.load_scisummnet(config['data']['scisummnet_path'])
            if scisummnet is not None:
                processed_scisummnet = preprocessor.process_dataset(
                    scisummnet,
                    text_column='text',
                    summary_column='summary'
                )
                logger.info(f"Processed {len(processed_scisummnet)} ScisummNet documents")
            
            # Combine datasets if both are available
            all_texts = []
            all_metadata = []
            
            if 'processed_xlsum' in locals():
                all_texts.extend(processed_xlsum['processed_text'].tolist())
                all_metadata.extend(processed_xlsum.to_dict('records'))
                
            if 'processed_scisummnet' in locals():
                all_texts.extend(processed_scisummnet['processed_text'].tolist())
                all_metadata.extend(processed_scisummnet.to_dict('records'))
            
            checkpoint_manager.save_stage('data_loading', {
                'xlsum_size': len(processed_xlsum) if 'processed_xlsum' in locals() else 0,
                'scisummnet_size': len(processed_scisummnet) if 'processed_scisummnet' in locals() else 0,
                'runtime': (datetime.now() - start_time).total_seconds(),
                'data_path': str(config['data']['processed_path'])
            })
        else:
            # Load from checkpoint
            data_state = checkpoint_manager.get_stage_data('data_loading')
            logger.info(f"Resuming from checkpoint with {data_state['xlsum_size']} XL-Sum and {data_state['scisummnet_size']} ScisummNet documents")
            
            # Initialize components
            loader = DataLoader(config['data']['scisummnet_path'])
            preprocessor = DomainAgnosticPreprocessor(config['preprocessing'])
            
            # Load and process XL-Sum
            xlsum = loader.load_xlsum()
            if xlsum:
                # Convert Dataset to DataFrame
                xlsum_df = pd.DataFrame(xlsum['train'])
                processed_xlsum = preprocessor.process_dataset(
                    xlsum_df,
                    text_column='text',
                    summary_column='summary'
                )
                logger.info(f"Processed {len(processed_xlsum)} XL-Sum documents")
            
            # Load and process ScisummNet
            scisummnet = loader.load_scisummnet(config['data']['scisummnet_path'])
            if scisummnet is not None:
                processed_scisummnet = preprocessor.process_dataset(
                    scisummnet,
                    text_column='text',
                    summary_column='summary'
                )
                logger.info(f"Processed {len(processed_scisummnet)} ScisummNet documents")
            
            # Combine datasets if both are available
            all_texts = []
            all_metadata = []
            
            if 'processed_xlsum' in locals():
                all_texts.extend(processed_xlsum['processed_text'].tolist())
                all_metadata.extend(processed_xlsum.to_dict('records'))
                
            if 'processed_scisummnet' in locals():
                all_texts.extend(processed_scisummnet['processed_text'].tolist())
                all_metadata.extend(processed_scisummnet.to_dict('records'))
        
        # Initialize components with enhanced features
        embedding_generator = EmbeddingGenerator(
            model_name=config['embedding']['model_name'],
            embedding_dim=config['embedding']['dimension']
        )
        
        cluster_manager = DynamicClusterManager(
            config={
                'clustering': {
                    'hybrid_mode': True,
                    'params': {
                        'n_clusters': 5,
                        'min_cluster_size': 10
                    }
                }
            }
        )
        
        # Generate attention-refined embeddings
        embeddings = embedding_generator.generate_embeddings(all_texts)
        
        # Perform dynamic clustering
        labels, metrics = cluster_manager.fit_predict(embeddings)
        
        # Log enhanced metrics
        logger.info(f"Clustering metrics: {metrics}")
        logger.info(f"Data characteristics: {metrics['data_characteristics']}")
        
        # Group documents by cluster
        clusters = cluster_manager.get_cluster_documents(all_metadata, labels)
        
        # Generate summaries for clusters
        if config.get('summarization', {}).get('enabled', True):
            logger.info("Generating summaries for clusters...")
            cluster_texts = {
                label: [
                    {
                        'processed_text': doc['processed_text'],
                        'reference_summary': doc.get('summary', '')  # Add reference summary if available
                    }
                    for doc in docs
                ]
                for label, docs in clusters.items()
                if label != -1  # Skip noise cluster
            }
            summaries = generate_summaries(cluster_texts, config)
            
            # Log summary statistics
            logger.info(f"Generated {len(summaries)} cluster summaries")
            if any('metrics' in data for data in summaries.values()):
                avg_rouge_l = np.mean([
                    data['metrics']['rougeL']['fmeasure']
                    for data in summaries.values()
                    if 'metrics' in data
                ])
                logger.info(f"Average ROUGE-L F1: {avg_rouge_l:.3f}")
        
        # Save results
        logger.info("Saving results...")
        cluster_manager.save_results(
            clusters,
            metrics,
            Path(config['clustering']['output_dir'])
        )
        
        # Visualize embeddings if configured
        if config.get('visualization', {}).get('enabled', True):
            logger.info("Generating visualizations...")
            visualizer = EmbeddingVisualizer(config['visualization'])
            visualizer.plot_embeddings(
                embeddings,
                labels,
                Path(config['visualization']['output_dir'])
            )
        
        if DashboardApp is not None:
            app = DashboardApp(embedding_generator, cluster_manager)
            app.run_server(debug=True)
        else:
            print("Dashboard disabled - running in CLI mode only")
        
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

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
    summarizer = AdaptiveSummarizer(config=config)
    results = {}
    
    # Calculate cluster-specific metrics
    cluster_metrics = {
        'variance': calculate_cluster_variance(embeddings),
        'lexical_diversity': calculate_lexical_diversity(texts)
    }
    
    # Generate summary with metrics-based adaptation
    summary_data = summarizer.summarize(
        texts=texts,
        embeddings=embeddings,
        metrics=cluster_metrics
    )
    
    results['summary'] = summary_data['summary']
    results['style'] = summary_data['style']
    results['metrics'] = {
        **summary_data['metrics'],
        **cluster_metrics
    }
    
    return results

if __name__ == "__main__":
    main() 