import os
import logging
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset

from src.data_validator import DataValidator
from src.embedding_generator import EnhancedEmbeddingGenerator
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.hybrid_summarizer import EnhancedHybridSummarizer
from src.visualization.embedding_visualizer import EmbeddingVisualizer
from src.evaluation.metrics import EvaluationMetrics
from src.utils.error_handler import with_error_handling

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def validate_config(config):
    """Validate the structure of the YAML config file."""
    required_keys = ['data', 'logging', 'clustering', 'summarization', 'metrics']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    logger.info("Config file validated successfully.")

@with_error_handling
def main(config):
    """Main function to run the optimized pipeline."""
    # Validate configuration
    validate_config(config)

    # Generate a unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Starting run with ID: %s", run_id)

    # Set up dataset loading
    logger.info("Loading dataset...")
    dataset = load_dataset('GEM/xlsum', 'english', cache_dir='data/cache')
    
    # Convert dataset to DataFrame
    logger.info("Converting dataset to DataFrame...")
    dataset_df = pd.DataFrame(dataset['train'])
    logger.info(f"Dataset loaded with {len(dataset_df)} records")

    # Validate the dataset
    logger.info("Validating dataset...")
    validator = DataValidator()
    validation_results = validator.validate_dataset(dataset_df)
    if not validation_results['is_valid']:
        logger.error("Dataset validation failed.")
        logger.error(validation_results)
        return
    
    logger.info("Dataset validation passed.")

    # Split dataset into manageable batches
    batch_size = config['data'].get('batch_size', 64)
    texts = dataset_df['text'].tolist()
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    # Process data batches
    logger.info(f"Processing {len(batches)} batches of data...")
    processed_texts = []
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(process_batch, batches), total=len(batches)):
            processed_texts.extend(result)

    logger.info(f"Processed {len(processed_texts)} texts.")

    # Generate embeddings
    logger.info("Generating embeddings...")
    embedding_generator = EnhancedEmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(processed_texts)

    # Save embeddings
    output_dir = Path(config['data']['output_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_file = output_dir / f"embeddings_{run_id}.npy"
    np.save(embeddings_file, embeddings)
    logger.info(f"Embeddings saved to {embeddings_file}")

    # Perform clustering
    logger.info("Performing clustering...")
    cluster_manager = DynamicClusterManager(config=config['clustering'])
    labels, metrics = cluster_manager.fit_predict(embeddings)
    logger.info(f"Clustering completed with metrics: {metrics}")

    # Summarize clusters
    logger.info("Summarizing clusters...")
    cluster_texts = {label: [] for label in set(labels)}
    for text, label in zip(processed_texts, labels):
        cluster_texts[label].append(text)

    summarizer = EnhancedHybridSummarizer()
    summaries = summarizer.summarize_all_clusters(cluster_texts)
    logger.info("Summarization completed.")

    # Visualize results
    logger.info("Generating visualization...")
    visualizer = EmbeddingVisualizer()
    visualization_file = output_dir / f"visualization_{run_id}.html"
    visualizer.visualize_embeddings(embeddings, labels, save_path=visualization_file)
    logger.info(f"Visualization saved to {visualization_file}")

    # Evaluate results
    logger.info("Evaluating results...")
    evaluator = EvaluationMetrics()
    evaluation_metrics = evaluator.calculate_all_metrics(
        embeddings=embeddings,
        labels=labels,
        summaries=summaries
    )
    logger.info(f"Evaluation metrics: {evaluation_metrics}")

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    # Load configuration
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("Config file not found. Please ensure 'config/config.yaml' exists.")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Run the main pipeline
    main(config)
