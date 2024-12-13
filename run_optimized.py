import os
import sys
from pathlib import Path
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
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
import logging


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
        from src.preprocessor import DomainAgnosticPreprocessor
        preprocessor = DomainAgnosticPreprocessor()
        return [preprocessor.preprocess_text(text) for text in batch_data]
    except Exception as e:
        logging.error(f"Error processing batch: {e}")
        return []

@with_error_handling
def main():
    """Main function to run the optimized script."""
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize MetricsLogger and get its logger
    logger = MetricsLogger(config)
    log = logger.logger

    # Generate a unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info("Starting run with ID: %s", run_id)

    # Initialize performance optimizer
    perf_optimizer = PerformanceOptimizer()
    batch_size = perf_optimizer.get_optimal_batch_size()
    n_workers = perf_optimizer.get_optimal_workers()
    
    log.info("Using %d workers with batch size %d", n_workers, batch_size)

    # Load dataset
    dataset_name = config['data']['datasets'][1]['dataset_name']
    language = config['data']['datasets'][1].get('language', 'english')
    dataset = load_dataset(
        dataset_name,
        language,
        cache_dir='data/cache',
        num_proc=n_workers
    )

    texts = dataset['train']['text']
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    log.info("Total batches: %d", len(batches))

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()

    # Load or process text batches
    try:
        processed_texts = checkpoint_manager.get_stage_data('processed_texts')
    except json.JSONDecodeError:
        log.error("JSONDecodeError: The state file is corrupted. Creating a new state.")
        processed_texts = None

    if processed_texts is None:
        processed_texts = []
        with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as executor:
            for batch_result in tqdm(executor.map(process_batch, batches), total=len(batches), desc="Processing batches"):
                processed_texts.extend(batch_result)

        checkpoint_manager.save_stage('processed_texts', processed_texts)

    log.info("Processed %d texts", len(processed_texts))

    # Save processed texts to output file
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"processed_texts_{run_id}.txt"
    with open(output_file, 'w') as f:
        for text in processed_texts:
            f.write(f"{text}\n")

    log.info("Saved processed texts to %s", output_file)

    # Initialize pipeline components
    embedding_generator = EnhancedEmbeddingGenerator()

    # Load or generate embeddings
    try:
        embeddings = checkpoint_manager.get_stage_data('embeddings')
    except json.JSONDecodeError:
        log.error("JSONDecodeError: The state file is corrupted. Creating a new state.")
        embeddings = None

    if embeddings is None:
        embeddings = embedding_generator.generate_embeddings(processed_texts)
        embeddings_list = embeddings.tolist()
        checkpoint_manager.save_stage('embeddings', embeddings_list)
    else:
        # If embeddings were loaded from checkpoint, ensure it's a numpy array
        embeddings = np.array(embeddings)
        embeddings_list = embeddings

    embeddings_file = output_dir / f"embeddings_{run_id}.npy"
    np.save(embeddings_file, embeddings)
    log.info("Saved embeddings to %s", embeddings_file)

    perf_optimizer.clear_memory_cache()
    checkpoint_manager.save_periodic_checkpoint('embeddings', embeddings_list)
    log.info("Completed embedding generation")

    cluster_config = config.get('clustering', {})
    cluster_manager = DynamicClusterManager(config=cluster_config)

    # Load or perform clustering
    try:
        clusters = checkpoint_manager.get_stage_data('clusters')
    except json.JSONDecodeError:
        log.error("JSONDecodeError: The state file is corrupted. Creating a new state.")
        clusters = None

    if clusters is None:
        labels, clustering_metrics = cluster_manager.fit_predict(embeddings)
        clusters = {'labels': labels.tolist(), 'metrics': clustering_metrics}
        checkpoint_manager.save_stage('clusters', clusters)

    clusters_file = output_dir / f"clusters_{run_id}.json"
    with open(clusters_file, 'w') as f:
        json.dump(clusters, f)
    log.info("Saved clusters to %s", clusters_file)

    perf_optimizer.clear_memory_cache()
    checkpoint_manager.save_periodic_checkpoint('clusters', clusters)
    log.info("Completed clustering")

    summarizer = EnhancedHybridSummarizer()

    # Load or generate summaries
    try:
        summaries = checkpoint_manager.get_stage_data('summaries')
    except json.JSONDecodeError:
        log.error("JSONDecodeError: The state file is corrupted. Creating a new state.")
        summaries = None

    if summaries is None:
        cluster_texts = {label: [] for label in set(clusters['labels'])}
        for text, label, embedding in zip(processed_texts, clusters['labels'], embeddings):
            cluster_texts[label].append({'processed_text': text, 'embedding': embedding})
        summaries = summarizer.summarize_all_clusters(cluster_texts)
        checkpoint_manager.save_stage('summaries', summaries)

    summaries_file = output_dir / f"summaries_{run_id}.json"
    with open(summaries_file, 'w') as f:
        json.dump(summaries, f)
    log.info("Saved summaries to %s", summaries_file)

    perf_optimizer.clear_memory_cache()
    checkpoint_manager.save_periodic_checkpoint('summaries', summaries)
    log.info("Completed summarization")

    visualizer = EmbeddingVisualizer()
    visualization_file = output_dir / f"visualizations_{run_id}.html"
    visualizer.visualize_embeddings(embeddings, save_path=visualization_file)
    log.info("Saved visualizations to %s", visualization_file)

    evaluator = EvaluationMetrics()
    evaluation_metrics = evaluator.calculate_comprehensive_metrics(
        summaries=summaries,
        references={},  # Add reference summaries if available
        embeddings=embeddings
    )
    evaluation_file = output_dir / f"evaluation_{run_id}.json"
    with open(evaluation_file, 'w') as f:
        json.dump(evaluation_metrics, f)
    log.info("Saved evaluation metrics to %s", evaluation_file)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
