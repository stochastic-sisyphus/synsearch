import logging
from pathlib import Path
import yaml
import pandas as pd
from src.data_loader import DataLoader
from src.data_preparation import DataPreparator
from src.data_validator import DataValidator
from src.utils.logging_config import setup_logging
from src.embedding_generator import EmbeddingGenerator
from src.visualization.embedding_visualizer import EmbeddingVisualizer
import numpy as np
from src.preprocessor import TextPreprocessor
from src.clustering.cluster_manager import ClusterManager
from typing import List, Dict
from datetime import datetime
from src.summarization.hybrid_summarizer import HybridSummarizer
from src.evaluation.metrics import EvaluationMetrics
import json
from src.utils.checkpoint_manager import CheckpointManager

def main():
    # Setup logging
    setup_logging('logs/processing.log')
    logger = logging.getLogger(__name__)
    
    try:
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
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
            preprocessor = TextPreprocessor()
            
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
            preprocessor = TextPreprocessor()
            
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
        
        # Generate embeddings
        if not checkpoint_manager.is_stage_complete('embeddings'):
            logger.info("Generating embeddings...")
            embeddings = generate_embeddings(all_texts, config)
            checkpoint_manager.save_stage('embeddings', {
                'shape': embeddings.shape,
                'path': str(Path(config['embedding']['output_dir']) / 'embeddings.npy')
            })
        else:
            # Load embeddings from checkpoint
            embedding_state = checkpoint_manager.get_stage_data('embeddings')
            embeddings = np.load(embedding_state['path'])
        
        # Initialize and run clustering
        logger.info("Performing clustering...")
        cluster_manager = ClusterManager(config)
        labels, clustering_metrics = cluster_manager.fit_predict(embeddings)
        
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
            clustering_metrics,
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
        
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

def generate_embeddings(texts: List[str], config: Dict) -> np.ndarray:
    """Generate embeddings for the input texts"""
    # Set up multi-threading
    import torch
    n_threads = config.get('embedding', {}).get('num_threads', 8)  # Default to 8 threads
    torch.set_num_threads(n_threads)
    
    embedding_generator = EmbeddingGenerator(
        model_name=config['embedding']['model_name'],
        batch_size=config['embedding']['batch_size'],
        max_seq_length=128  # Add shorter sequence length
    )
    
    # Generate embeddings with progress tracking and checkpoints
    embeddings = embedding_generator.generate_embeddings(
        texts,
        checkpoint_dir=config['embedding'].get('checkpoint_dir', 'outputs/embeddings/checkpoints'),
        checkpoint_frequency=1000  # Save every 1000 documents
    )
    
    # Save final embeddings
    if 'output_dir' in config['embedding']:
        metadata = {
            'model_name': config['embedding']['model_name'],
            'timestamp': datetime.now().isoformat(),
            'num_documents': len(texts)
        }
        embedding_generator.save_embeddings(
            embeddings,
            metadata,
            config['embedding']['output_dir']
        )
    
    return embeddings

def generate_summaries(cluster_texts: Dict[str, List[str]], config: Dict) -> Dict[str, Dict]:
    """Generate and evaluate summaries for clustered texts"""
    # Initialize summarizer and metrics
    summarizer = HybridSummarizer(
        model_name=config['summarization']['model_name'],
        max_length=config['summarization']['max_length'],
        min_length=config['summarization']['min_length'],
        batch_size=config['summarization']['batch_size']
    )
    
    metrics_calculator = EvaluationMetrics()
    
    # Generate summaries with style from config
    style = config['summarization'].get('style', 'balanced')
    summaries = summarizer.summarize_all_clusters(cluster_texts, style=style)
    
    # Calculate ROUGE scores if reference summaries are available
    if any('reference_summary' in next(iter(texts)) for texts in cluster_texts.values()):
        for cluster_id, cluster_data in summaries.items():
            generated = cluster_data['summary']
            references = [doc.get('reference_summary', '') for doc in cluster_texts[cluster_id]]
            rouge_scores = metrics_calculator.calculate_rouge_scores([generated], references)
            cluster_data['metrics'] = rouge_scores
    
    # Save summaries and metrics
    if 'output_dir' in config['summarization']:
        output_dir = Path(config['summarization']['output_dir'])
        
        # Save summaries
        with open(output_dir / 'summaries.json', 'w') as f:
            json.dump(summaries, f, indent=2)
            
        # Save metrics separately if they exist
        if any('metrics' in data for data in summaries.values()):
            metrics = {
                cluster_id: data['metrics'] 
                for cluster_id, data in summaries.items() 
                if 'metrics' in data
            }
            metrics_calculator.save_metrics(
                metrics,
                output_dir,
                prefix='summarization'
            )
    
    return summaries

if __name__ == "__main__":
    main() 