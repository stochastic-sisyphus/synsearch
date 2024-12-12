import logging
from pathlib import Path
import yaml
from data_loader import DataLoader
from data_preparation import DataPreparator
from data_validator import DataValidator
from utils.logging_config import setup_logging
from embedding_generator import EmbeddingGenerator
from visualization.embedding_visualizer import EmbeddingVisualizer
import numpy as np
from preprocessor import TextPreprocessor
from clustering.cluster_manager import ClusterManager
from typing import List, Dict
from datetime import datetime
from summarization.summarizer import ClusterSummarizer
from summarization.model_trainer import SummarizationModelTrainer
import pandas as pd
from torch.utils.data import DataLoader as TorchDataLoader, Dataset

class TextDataset(Dataset):
    """Custom Dataset for text data."""
    
    def __init__(self, texts: list):
        self.texts = texts
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

def main():
    """
    Main function to run the data processing pipeline.
    """
    # Setup logging
    setup_logging('logs/processing.log')
    logger = logging.getLogger(__name__)
    
    try:
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize components
        loader = DataLoader(config['data']['scisummnet_path'])
        preprocessor = TextPreprocessor()
        validator = DataValidator()
        
        # Process datasets
        logger.info("Loading and processing datasets...")
        
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
        
        # Validate processed datasets
        if not validator.validate_dataset(processed_xlsum):
            raise ValueError("Processed XL-Sum dataset validation failed")
        if not validator.validate_dataset(processed_scisummnet):
            raise ValueError("Processed ScisummNet dataset validation failed")
        
        # Train summarization model if enabled
        if config.get('training', {}).get('enabled', False):
            logger.info("Starting summarization model training...")
            trainer = SummarizationModelTrainer(config)
            
            # Prepare training data from processed datasets
            training_data = {
                'xlsum': processed_xlsum if 'processed_xlsum' in locals() else None,
                'scisummnet': processed_scisummnet if 'processed_scisummnet' in locals() else None
            }
            
            model, tokenizer = trainer.train_model(training_data)
            
            # Save trained model
            output_dir = Path(config['training']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved trained model to {output_dir}")
        
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
        logger.info("Generating embeddings...")
        embeddings = generate_embeddings(all_texts, config)
        
        # Validate embeddings
        if not validator.validate_embeddings(embeddings):
            raise ValueError("Generated embeddings validation failed")
        
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
                label: [doc['processed_text'] for doc in docs]
                for label, docs in clusters.items()
                if label != -1  # Skip noise cluster
            }
            summaries = generate_summaries(cluster_texts, config)
        
        # Validate summaries
        for summary in summaries:
            if not validator.validate_summary(summary['summary']):
                raise ValueError(f"Summary validation failed for cluster {summary['cluster_id']}")
        
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
    """
    Generate embeddings for the input texts.
    
    Args:
        texts (List[str]): List of input texts.
        config (Dict): Configuration dictionary.
    
    Returns:
        np.ndarray: Generated embeddings.
    """
    embedding_generator = EmbeddingGenerator(
        model_name=config['embedding']['model_name'],
        batch_size=config['embedding']['batch_size']
    )
    
    # Generate embeddings
    dataset = TextDataset(texts)
    dataloader = TorchDataLoader(dataset, batch_size=config['embedding']['batch_size'], shuffle=False)
    
    all_embeddings = []
    for batch in dataloader:
        embeddings = embedding_generator.generate_embeddings(batch)
        all_embeddings.append(embeddings)
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Save embeddings if output directory is specified
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

def generate_summaries(cluster_texts: Dict[str, List[str]], config: Dict) -> List[Dict[str, str]]:
    """
    Generate summaries for clustered texts.
    
    Args:
        cluster_texts (Dict[str, List[str]]): Dictionary of clustered texts.
        config (Dict): Configuration dictionary.
    
    Returns:
        List[Dict[str, str]]: Generated summaries.
    """
    summarizer = ClusterSummarizer(
        model_name=config['summarization']['model_name'],
        max_length=config['summarization']['max_length'],
        min_length=config['summarization']['min_length'],
        batch_size=config['summarization']['batch_size']
    )
    
    # Generate summaries
    summaries = summarizer.summarize_all_clusters(cluster_texts)
    
    # Save summaries if output directory is specified
    if 'output_dir' in config['summarization']:
        summarizer.save_summaries(
            summaries,
            config['summarization']['output_dir']
        )
    
    return summaries

if __name__ == "__main__":
    main()
