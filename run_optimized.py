import os
import sys
from pathlib import Path
import logging
import yaml
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from src.data_validator import DataValidator
from src.embedding_generator import EnhancedEmbeddingGenerator
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.hybrid_summarizer import EnhancedHybridSummarizer
from src.visualization.embedding_visualizer import EmbeddingVisualizer
from src.evaluation.metrics import EvaluationMetrics
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.error_handler import with_error_handling
from src.utils.performance import PerformanceOptimizer

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@with_error_handling
def process_batch(batch_data):
    """Process a batch of texts in parallel."""
    try:
        from src.preprocessing.domain_agnostic_preprocessor import DomainAgnosticPreprocessor
        preprocessor = DomainAgnosticPreprocessor()
        return [preprocessor.preprocess_text(text) for text in batch_data]
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return []

def validate_config(config):
    """Validate the structure of the YAML config file."""
    required_keys = ['data', 'logging', 'clustering', 'summarization', 'metrics']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    logger.info("Config file validated successfully.")

@with_error_handling
def main(config):
    """Main function to run the optimized script."""
    # Validate config file structure
    validate_config(config)

    # Generate a unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up logging
    log_dir = Path(config['logging']['file']).parent
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"run_optimized_{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=config['logging']['format'],
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file))
        ]
    )

    logger.info("Starting run with ID: %s", run_id)
    
    # Get optimal batch size and workers
    perf_optimizer = PerformanceOptimizer()
    batch_size = perf_optimizer.get_optimal_batch_size()
    n_workers = perf_optimizer.get_optimal_workers()
    
    logger.info(f"Using {n_workers} workers with batch size {batch_size}")

    # Load dataset
    dataset_dir = Path(config['data']['input_path'])
    if dataset_dir.is_dir():
        dataset_path = next(dataset_dir.glob("*.csv"), None)
        if not dataset_path:
            raise FileNotFoundError(f"No CSV file found in the directory: {dataset_dir}")
    else:
        dataset_path = dataset_dir

    logger.info(f"Loading dataset from: {dataset_path}")
    dataset_df = pd.read_csv(dataset_path)

    # Validate dataset structure
    validator = DataValidator()
    missing_before = dataset_df.isnull().sum().sum()
    logger.info(f"Missing values before handling: {missing_before}")

    if missing_before > 0:
        logger.info(f"Handling {missing_before} missing values in the dataset...")
        dataset_df.dropna(inplace=True)  # Drop rows with missing values

    missing_after = dataset_df.isnull().sum().sum()
    logger.info(f"Missing values after handling: {missing_after}")

    logger.info(f"Dataset structure after cleaning: {dataset_df.info()}")

    validation_results = validator.validate_dataset(dataset_df)
    logger.info(f"Validation results: {validation_results}")

    if not validation_results['is_valid']:
        logger.error("Dataset validation failed. Exiting.")
        sys.exit(1)

    # Initialize components
    embedding_generator = EnhancedEmbeddingGenerator()
    cluster_manager = DynamicClusterManager(config=config['clustering'])
    summarizer = EnhancedHybridSummarizer()
    visualizer = EmbeddingVisualizer()
    evaluator = EvaluationMetrics()
    checkpoint_manager = CheckpointManager()

    # Process data into batches
    texts = dataset_df['text'].tolist()
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    # Process batches in parallel
    processed_texts = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for batch_result in executor.map(process_batch, batches):
                processed_texts.extend(batch_result)
                pbar.update(1)

    logger.info(f"Processed {len(processed_texts)} texts")

    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(processed_texts)
    logger.info("Generated embeddings")

    # Save embeddings
    embeddings_path = Path(config['data']['output_path']) / f"embeddings_{run_id}.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings to {embeddings_path}")

    # Perform clustering
    labels, clustering_metrics = cluster_manager.fit_predict(embeddings)
    logger.info("Clustering completed")

    # Summarize clusters
    cluster_texts = {label: [] for label in set(labels)}
    for text, label in zip(processed_texts, labels):
        cluster_texts[label].append(text)
    summaries = summarizer.summarize_all_clusters(cluster_texts)
    logger.info("Generated summaries")

    # Evaluate results
    evaluation_metrics = evaluator.calculate_all_metrics(
        embeddings=embeddings,
        labels=labels,
        summaries=summaries
    )
    logger.info(f"Evaluation metrics: {evaluation_metrics}")

    # Visualize embeddings
    visualization_path = Path(config['data']['output_path']) / f"visualizations_{run_id}.html"
    visualizer.visualize_embeddings(embeddings, save_path=visualization_path)
    logger.info(f"Saved visualizations to {visualization_path}")

    logger.info("Pipeline execution completed successfully")

if __name__ == '__main__':
    # Load configuration
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Run the pipeline
    main(config)
