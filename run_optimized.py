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

from src.embedding_generator import EnhancedEmbeddingGenerator
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.hybrid_summarizer import EnhancedHybridSummarizer
from src.visualization.embedding_visualizer import EmbeddingVisualizer
from src.evaluation.metrics import EvaluationMetrics
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.error_handler import with_error_handling
from src.utils.performance import PerformanceOptimizer
from src.utils.logging_utils import MetricsLogger


def init_worker():
    """Initialize worker process with optimized settings."""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


@with_error_handling
def process_batch(batch_data):
    """Process a batch of texts in parallel."""
    from src.preprocessor import DomainAgnosticPreprocessor
    preprocessor = DomainAgnosticPreprocessor()
    return [preprocessor.preprocess_text(text) for text in batch_data]


@with_error_handling
def main():
    """Main function to run the optimized script."""
    # Load configuration
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize MetricsLogger
    logger = MetricsLogger(config)

    # Generate a unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.logger.info("Starting run with ID: %s", run_id)

    # Optimize batch size and workers
    perf_optimizer = PerformanceOptimizer()
    batch_size = config['data'].get('batch_size', perf_optimizer.get_optimal_batch_size())
    n_workers = perf_optimizer.get_optimal_workers()

    logger.logger.info("Using %d workers with batch size %d", n_workers, batch_size)

    # Load dataset
    dataset = load_dataset('GEM/xlsum', 'english', cache_dir=config['embedding']['cache_dir'])
    texts = dataset['train']['text']

    # Split data into batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    logger.logger.info("Processing %d batches", len(batches))

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()

    # Process batches
    processed_texts = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as executor:
        for result in tqdm(executor.map(process_batch, batches), total=len(batches)):
            processed_texts.extend(result)

    logger.logger.info("Processed %d texts", len(processed_texts))

    # Save processed texts
    output_dir = Path(config['data']['output_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_file = output_dir / f"processed_texts_{run_id}.txt"
    with open(processed_file, 'w') as f:
        for text in processed_texts:
            f.write(f"{text}\n")

    logger.logger.info("Saved processed texts to %s", processed_file)

    # Generate embeddings
    embedding_generator = EnhancedEmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(processed_texts)

    embeddings_file = output_dir / f"embeddings_{run_id}.npy"
    np.save(embeddings_file, embeddings)
    logger.logger.info("Saved embeddings to %s", embeddings_file)

    # Perform clustering
    cluster_manager = DynamicClusterManager(config=config['clustering'])
    labels, clustering_metrics = cluster_manager.fit_predict(embeddings)
    logger.logger.info("Clustering completed with metrics: %s", clustering_metrics)

    # Summarize clusters
    summarizer = EnhancedHybridSummarizer()
    cluster_texts = {label: [] for label in set(labels)}
    for text, label in zip(processed_texts, labels):
        cluster_texts[label].append(text)

    summaries = summarizer.summarize_all_clusters(cluster_texts)

    summaries_file = output_dir / f"summaries_{run_id}.json"
    with open(summaries_file, 'w') as f:
        json.dump(summaries, f)
    logger.logger.info("Saved summaries to %s", summaries_file)

    # Visualize embeddings
    visualizer = EmbeddingVisualizer()
    visualization_file = output_dir / f"visualizations_{run_id}.html"
    visualizer.visualize_embeddings(embeddings, labels, save_path=visualization_file)
    logger.logger.info("Saved visualizations to %s", visualization_file)

    # Evaluate results
    evaluator = EvaluationMetrics()
    evaluation_metrics = evaluator.calculate_comprehensive_metrics(
        summaries=summaries,
        references={},  # Add reference summaries if available
        embeddings=embeddings
    )

    evaluation_file = output_dir / f"evaluation_{run_id}.json"
    with open(evaluation_file, 'w') as f:
        json.dump(evaluation_metrics, f)
    logger.logger.info("Saved evaluation metrics to %s", evaluation_file)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
