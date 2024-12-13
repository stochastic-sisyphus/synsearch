import os
import sys
from pathlib import Path
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import logging
from datetime import datetime
import json
import yaml

# Imports adjusted to your repository structure
from src.embedding_generator import EnhancedEmbeddingGenerator
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.hybrid_summarizer import EnhancedHybridSummarizer
from src.visualization.embedding_visualizer import EmbeddingVisualizer
from src.evaluation.metrics import EvaluationMetrics
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.error_handler import with_error_handling
from src.utils.performance import PerformanceOptimizer
from src.data_validator import validate_text_list, validate_embeddings, validate_labels, validate_cluster_metrics


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

        # Validate processed texts
        validate_text_list(processed_batch, name="processed_batch")
        return processed_batch
    except Exception as e:
        logging.error(f"Error processing batch: {e}")
        return []


def validate_config(config):
    """Validate the structure of the YAML config file."""
    required_keys = ['data', 'logging', 'clustering', 'summarization', 'metrics']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    logging.info("Config file validated successfully.")


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

    logging.info("Starting run with ID: %s", run_id)

    # Get optimal batch size and workers
    perf_optimizer = PerformanceOptimizer()
    batch_size = perf_optimizer.get_optimal_batch_size()
    n_workers = perf_optimizer.get_optimal_workers()

    logging.info(f"Using {n_workers} workers with batch size {batch_size}")

    # Load dataset with optimized settings
    dataset = load_dataset(
        config['data']['datasets'][1]['dataset_name'],
        config['data']['datasets'][1]['language'],
        cache_dir=config['data']['output_path'],
        num_proc=n_workers
    )

    # Validate dataset
    if 'train' not in dataset or 'text' not in dataset['train']:
        raise KeyError("Dataset does not contain the required 'train' or 'text' keys.")

    texts = dataset['train']['text']
    validate_text_list(texts, name="dataset texts")

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

    # Validate processed texts
    validate_text_list(processed_texts, name="processed_texts")

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

    # Check for existing embeddings
    try:
        embeddings = checkpoint_manager.get_stage_data('embeddings')
    except json.JSONDecodeError:
        logging.error("JSONDecodeError: The state file is corrupted. Creating a new state.")
        embeddings = None

    if embeddings is None:
        # Generate embeddings
        embeddings = embedding_generator.generate_embeddings(processed_texts)
        validate_embeddings(embeddings)

        # Save embeddings
        checkpoint_manager.save_stage('embeddings', embeddings.tolist())

    embeddings_file = output_dir / f"embeddings_{run_id}.npy"
    np.save(embeddings_file, embeddings)
    logging.info(f"Saved embeddings to {embeddings_file}")

    # Clear memory cache
    perf_optimizer.clear_memory_cache()

    # Check for existing clusters
    try:
        clusters = checkpoint_manager.get_stage_data('clusters')
    except json.JSONDecodeError:
        logging.error("JSONDecodeError: The state file is corrupted. Creating a new state.")
        clusters = None

    if clusters is None:
        # Perform clustering
        labels, clustering_metrics = cluster_manager.fit_predict(embeddings)
        validate_labels(labels, len(processed_texts))
        validate_cluster_metrics(clustering_metrics)

        # Save clusters
        clusters = {'labels': labels.tolist(), 'metrics': clustering_metrics}
        checkpoint_manager.save_stage('clusters', clusters)

    clusters_file = output_dir / f"clusters_{run_id}.json"
    with open(clusters_file, 'w') as f:
        json.dump(clusters, f)
    logging.info(f"Saved clusters to {clusters_file}")

    # Summarization
    cluster_texts = {label: [] for label in set(clusters['labels'])}
    for text, label in zip(processed_texts, clusters['labels']):
        cluster_texts[label].append(text)

    summaries = summarizer.summarize_all_clusters(cluster_texts)
    validate_text_list(list(summaries.values()), name="summaries")

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
        clustering_metrics = evaluator.calculate_clustering_metrics(embeddings, clusters['labels'])
        logging.info(f"Clustering Metrics: {clustering_metrics}")
    except Exception as e:
        logging.error(f"Error calculating clustering metrics: {e}")


if __name__ == '__main__':
    # Load configuration from a YAML file
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    mp.set_start_method('spawn', force=True)
    main(config)
