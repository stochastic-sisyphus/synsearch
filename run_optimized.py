import os
import sys
from pathlib import Path
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import yaml

from src.embedding_generator import EnhancedEmbeddingGenerator
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.hybrid_summarizer import EnhancedHybridSummarizer
from src.visualization.embedding_visualizer import EmbeddingVisualizer
from src.evaluation.metrics import EvaluationMetrics
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.error_handler import with_error_handling
from src.utils.performance import PerformanceOptimizer
from src.data_validator import DataValidator, ConfigValidator


def init_worker():
    """Initialize worker process with optimized settings."""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


@with_error_handling
def process_batch(batch_data):
    """Process a batch of texts in parallel."""
    try:
        from src.preprocessing.domain_agnostic_preprocessor import DomainAgnosticPreprocessor
        preprocessor = DomainAgnosticPreprocessor()
        processed_batch = [preprocessor.preprocess_text(text) for text in batch_data]

        # Use DataValidator to validate processed texts
        validator = DataValidator()
        if not validator.validate_texts(processed_batch)['is_valid']:
            raise ValueError("Processed batch validation failed.")
        return processed_batch
    except Exception as e:
        logging.error(f"Error processing batch: {e}")
        return []


@with_error_handling
def main(config):
    """Main function to run the optimized script."""
    # Validate config file structure
    config_validator = ConfigValidator()
    if not config_validator.validate_config(config):
        raise ValueError("Configuration validation failed.")

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

    logging.info("Starting run with ID: %s", run_id)

    # Get optimal batch size and workers
    perf_optimizer = PerformanceOptimizer()
    batch_size = perf_optimizer.get_optimal_batch_size()
    n_workers = perf_optimizer.get_optimal_workers()

    logging.info(f"Using {n_workers} workers with batch size {batch_size}")

    dataset = load_dataset(
        config['data']['datasets'][1]['dataset_name'],
        config['data']['datasets'][1]['language'],
        cache_dir=config['data']['output_path'],
        num_proc=n_workers
    )

    # Convert dataset to Pandas DataFrame and handle missing values
    dataset_df = pd.DataFrame(dataset['train'])  # Assuming train set is structured

    # Insert the block to handle missing values and validate the dataset
    missing_before = dataset_df.isnull().sum().sum()
    logging.info(f"Missing values before handling: {missing_before}")

    if missing_before > 0:
        logging.info(f"Handling {missing_before} missing values in the dataset...")
        dataset_df.dropna(inplace=True)  # Drop rows with missing values in any column

    missing_after = dataset_df.isnull().sum().sum()
    logging.info(f"Missing values after handling: {missing_after}")

    # Log the dataset's structure post-cleaning
    logging.info(f"Dataset structure after cleaning: {dataset_df.info()}")

    # Validate dataset
    validation_results = validator.validate_dataset(dataset_df)

    # Log detailed validation results
    logging.info(f"Validation results: {validation_results}")

    if not validation_results['is_valid']:
        for key, value in validation_results.items():
            logging.error(f"Validation Check - {key}: {value}")
        sys.exit(1)

    # Handle missing values in the dataset
    if 'text' in dataset_df.columns:
        missing_before = dataset_df.isnull().sum().sum()  # Total missing values across all columns
        if missing_before > 0:
            logging.info(f"Handling {missing_before} missing values in the dataset...")
            dataset_df.dropna(inplace=True)  # Drop rows with missing values in any column
        missing_after = dataset_df.isnull().sum().sum()
        logging.info(f"Missing values after handling: {missing_after}")

    # Validate dataset after handling missing values
    validation_results = validator.validate_dataset(dataset_df)
    if not validation_results['is_valid']:
        for key, value in validation_results.items():
            logging.error(f"Validation Check - {key}: {value}")
        sys.exit(1)

    texts = dataset_df['text'].tolist()

    # Split data into batches
    batches = [
        texts[i:i + batch_size]
        for i in range(0, len(texts), batch_size)
    ]

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()

    # Check for existing processed texts
    try:
        processed_texts = checkpoint_manager.get_stage_data('processed_texts')
    except json.JSONDecodeError:
        logging.error("JSONDecodeError: The state file is corrupted. Creating a new state.")
        processed_texts = None

    if processed_texts is None:
        processed_texts = []

        # Process batches in parallel
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker
        ) as executor:
            with tqdm(total=len(batches), desc="Processing batches") as pbar:
                for batch_result in executor.map(process_batch, batches):
                    processed_texts.extend(batch_result)
                    pbar.update(1)

        # Save processed texts checkpoint
        checkpoint_manager.save_stage('processed_texts', processed_texts)

    logging.info(f"Processed {len(processed_texts)} texts")

    # Validate processed texts using DataValidator
    if not validator.validate_texts(processed_texts)['is_valid']:
        raise ValueError("Processed texts validation failed.")

    # Save processed texts to output file
    output_dir = Path(config['data']['output_path'])
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"processed_texts_{run_id}.txt"
    with open(output_file, 'w') as f:
        for text in processed_texts:
            f.write(f"{text}\n")

    logging.info(f"Saved processed texts to {output_file}")

    # Initialize components
    embedding_generator = EnhancedEmbeddingGenerator()
    cluster_manager = DynamicClusterManager(config=config['clustering'])
    summarizer = EnhancedHybridSummarizer()
    visualizer = EmbeddingVisualizer()
    evaluator = EvaluationMetrics()

    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(processed_texts)
    if not validator.validate_embeddings(embeddings)['is_valid']:
        raise ValueError("Embeddings validation failed.")

    embeddings_file = output_dir / f"embeddings_{run_id}.npy"
    np.save(embeddings_file, embeddings)
    logging.info(f"Saved embeddings to {embeddings_file}")

    # Perform clustering
    labels, clustering_metrics = cluster_manager.fit_predict(embeddings)
    if not validator.validate_summaries(labels)['is_valid']:
        raise ValueError("Labels validation failed.")

    clusters_file = output_dir / f"clusters_{run_id}.json"
    with open(clusters_file, 'w') as f:
        json.dump({'labels': labels, 'metrics': clustering_metrics}, f)
    logging.info(f"Saved clusters to {clusters_file}")

    # Summarization
    cluster_texts = {label: [] for label in set(labels)}
    for text, label in zip(processed_texts, labels):
        cluster_texts[label].append(text)

    summaries = summarizer.summarize_all_clusters(cluster_texts)
    if not validator.validate_summaries(list(summaries.values()))['is_valid']:
        raise ValueError("Summaries validation failed.")

    summaries_file = output_dir / f"summaries_{run_id}.json"
    with open(summaries_file, 'w') as f:
        json.dump(summaries, f)
    logging.info(f"Saved summaries to {summaries_file}")

    # Visualization
    visualization_file = output_dir / f"visualizations_{run_id}.html"
    visualizer.visualize_embeddings(embeddings, save_path=visualization_file)
    logging.info(f"Saved visualizations to {visualization_file}")

    # Evaluation
    try:
        rouge_scores = evaluator.calculate_rouge_scores(list(summaries.values()), references=[])
        logging.info(f"ROUGE Scores: {rouge_scores}")
    except Exception as e:
        logging.error(f"Error calculating ROUGE scores: {e}")

    try:
        clustering_metrics = evaluator.calculate_clustering_metrics(embeddings, labels)
        logging.info(f"Clustering Metrics: {clustering_metrics}")
    except Exception as e:
        logging.error(f"Error calculating clustering metrics: {e}")


if __name__ == '__main__':
    # Load configuration from a YAML file
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    mp.set_start_method('spawn', force=True)
    main(config)
