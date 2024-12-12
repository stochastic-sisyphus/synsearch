import os
import sys
from pathlib import Path
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from src.utils.performance import PerformanceOptimizer
import logging
from datetime import datetime
import json

from src.embedding_generator import EnhancedEmbeddingGenerator
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.hybrid_summarizer import EnhancedHybridSummarizer  # Use EnhancedHybridSummarizer
from src.visualization.embedding_visualizer import EmbeddingVisualizer
from src.evaluation.metrics import EvaluationMetrics
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.error_handler import with_error_handling

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
    # Generate a unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"run_optimized_{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
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
        'GEM/xlsum',
        'english',
        cache_dir='data/cache',
        num_proc=n_workers
    )

    # Split data into batches
    texts = dataset['train']['text']
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

    # Save processed texts to output file
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"processed_texts_{run_id}.txt"
    with open(output_file, 'w') as f:
        for text in processed_texts:
            f.write(f"{text}\n")

    logging.info(f"Saved processed texts to {output_file}")

    # Initialize components
    embedding_generator = EnhancedEmbeddingGenerator()
    config = {  # Add your configuration here
        'clustering': {
            'min_cluster_size': 5,
            'min_samples': 3
        }
    }
    cluster_manager = DynamicClusterManager(config=config)
    summarizer = EnhancedHybridSummarizer()  # Use EnhancedHybridSummarizer here
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
        # Convert embeddings to list before saving
        embeddings_list = embeddings.tolist()
        checkpoint_manager.save_stage('embeddings', embeddings_list)
    else:
        embeddings_list = embeddings  # Ensure embeddings_list is assigned if it already exists

    embeddings_file = output_dir / f"embeddings_{run_id}.npy"
    np.save(embeddings_file, embeddings)
    logging.info(f"Saved embeddings to {embeddings_file}")

    # Clear unused variables and cache
    perf_optimizer.clear_memory_cache()
    checkpoint_manager.save_periodic_checkpoint('embeddings', embeddings_list)
    logging.info("Completed embedding generation")

    # Check for existing clusters
    try:
        clusters = checkpoint_manager.get_stage_data('clusters')
    except json.JSONDecodeError:
        logging.error("JSONDecodeError: The state file is corrupted. Creating a new state.")
        clusters = None

    if clusters is None:
        # Perform clustering
        labels, clustering_metrics = cluster_manager.fit_predict(embeddings)
        clusters = {'labels': labels.tolist(), 'metrics': clustering_metrics}
        checkpoint_manager.save_stage('clusters', clusters)

    clusters_file = output_dir / f"clusters_{run_id}.json"
    with open(clusters_file, 'w') as f:
        json.dump(clusters, f)
    logging.info(f"Saved clusters to {clusters_file}")

    # Clear unused variables and cache
    perf_optimizer.clear_memory_cache()
    checkpoint_manager.save_periodic_checkpoint('clusters', clusters)
    logging.info("Completed clustering")

    # Check for existing summaries
    try:
        summaries = checkpoint_manager.get_stage_data('summaries')
    except json.JSONDecodeError:
        logging.error("JSONDecodeError: The state file is corrupted. Creating a new state.")
        summaries = None

    if summaries is None:
        # Generate summaries
        cluster_texts = {label: [] for label in set(clusters['labels'])}
        for text, label in zip(processed_texts, clusters['labels']):
            cluster_texts[label].append(text)
        summaries = summarizer.summarize_all_clusters(cluster_texts)
        checkpoint_manager.save_stage('summaries', summaries)

    summaries_file = output_dir / f"summaries_{run_id}.json"
    with open(summaries_file, 'w') as f:
        json.dump(summaries, f)
    logging.info(f"Saved summaries to {summaries_file}")

    # Clear unused variables and cache
    perf_optimizer.clear_memory_cache()
    checkpoint_manager.save_periodic_checkpoint('summaries', summaries)
    logging.info("Completed summarization")

    # Visualize embeddings
    visualization_file = output_dir / f"visualizations_{run_id}.html"
    visualizer.visualize_embeddings(embeddings, save_path=visualization_file)
    logging.info(f"Saved visualizations to {visualization_file}")

    # Evaluate results
    evaluation_metrics = evaluator.calculate_comprehensive_metrics(
        summaries=summaries,
        references={},  # Add reference summaries if available
        embeddings=embeddings
    )
    evaluation_file = output_dir / f"evaluation_{run_id}.json"
    with open(evaluation_file, 'w') as f:
        json.dump(evaluation_metrics, f)
    logging.info(f"Saved evaluation metrics to {evaluation_file}")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
