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
import yaml

from src.embedding_generator import EnhancedEmbeddingGenerator
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.hybrid_summarizer import EnhancedHybridSummarizer
from src.visualization.embedding_visualizer import EmbeddingVisualizer
from src.evaluation.metrics import EvaluationMetrics
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.error_handler import with_error_handling, GlobalErrorHandler
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
    try:
        from src.preprocessor import DomainAgnosticPreprocessor
        preprocessor = DomainAgnosticPreprocessor()
        return [preprocessor.preprocess_text(text) for text in batch_data]
    except Exception as e:
        logging.error(f"Error processing batch: {e}")
        return []

@with_error_handling
def main(config):
    """Main function to run the optimized script."""
    # Generate a unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up logging
    log_dir = Path(config['data']['output_path'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_optimized_{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file))
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting run with ID: {run_id}")
    
    # Get optimal batch size and workers
    perf_optimizer = PerformanceOptimizer()
    batch_size = perf_optimizer.get_optimal_batch_size()
    n_workers = perf_optimizer.get_optimal_workers()
    
    logger.info(f"Using {n_workers} workers with batch size {batch_size}")

    # Load dataset with optimized settings
    dataset = load_dataset(
        config['data']['datasets'][1]['dataset_name'],
        config['data']['datasets'][1]['language'],
        cache_dir=config['data']['output_path'],
        num_proc=n_workers
    )

    # Split data into batches
    texts = dataset['train']['text']
    batches = [
        texts[i:i + batch_size] 
        for i in range(0, len(texts), batch_size)
    ]
    logger.info(f"Total batches: {len(batches)}")

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()

    # Process texts
    try:
        processed_texts = checkpoint_manager.get_stage_data('processed_texts')
    except (json.JSONDecodeError, FileNotFoundError):
        processed_texts = None

    if processed_texts is None:
        processed_texts = []
        with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as executor:
            with tqdm(total=len(batches), desc="Processing batches") as pbar:
                for batch_result in executor.map(process_batch, batches):
                    processed_texts.extend(batch_result)
                    pbar.update(1)
        checkpoint_manager.save_stage('processed_texts', processed_texts)

    logger.info(f"Processed {len(processed_texts)} texts")

    # Save processed texts
    output_dir = Path(config['data']['output_path'])
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"processed_texts_{run_id}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in processed_texts:
            f.write(f"{text}\n")
    logger.info(f"Saved processed texts to {output_file}")

    # Initialize components
    embedding_generator = EnhancedEmbeddingGenerator()
    cluster_manager = DynamicClusterManager(config=config['clustering'])
    summarizer = EnhancedHybridSummarizer()
    visualizer = EmbeddingVisualizer()
    evaluator = EvaluationMetrics()

    # Generate embeddings
    try:
        embeddings = checkpoint_manager.get_stage_data('embeddings')
        if embeddings is not None:
            embeddings = np.array(embeddings)
    except (json.JSONDecodeError, FileNotFoundError):
        embeddings = None

    if embeddings is None:
        embeddings = embedding_generator.generate_embeddings(processed_texts)
        checkpoint_manager.save_stage('embeddings', embeddings.tolist())

    embeddings_file = output_dir / f"embeddings_{run_id}.npy"
    np.save(embeddings_file, embeddings)
    logger.info(f"Saved embeddings to {embeddings_file}")

    # Clear memory
    perf_optimizer.clear_memory_cache()
    logger.info("Completed embedding generation")

    # Perform clustering
    try:
        clusters = checkpoint_manager.get_stage_data('clusters')
    except (json.JSONDecodeError, FileNotFoundError):
        clusters = None

    if clusters is None:
        labels, clustering_metrics = cluster_manager.fit_predict(embeddings)
        clusters = {'labels': labels.tolist(), 'metrics': clustering_metrics}
        checkpoint_manager.save_stage('clusters', clusters)

    clusters_file = output_dir / f"clusters_{run_id}.json"
    with open(clusters_file, 'w') as f:
        json.dump(clusters, f)
    logger.info(f"Saved clusters to {clusters_file}")

    # Clear memory
    perf_optimizer.clear_memory_cache()
    logger.info("Completed clustering")

    # Generate summaries
    try:
        summaries = checkpoint_manager.get_stage_data('summaries')
    except (json.JSONDecodeError, FileNotFoundError):
        summaries = None

    if summaries is None:
        cluster_texts = {str(label): [] for label in set(clusters['labels'])}
        for text, label, embedding in zip(processed_texts, clusters['labels'], embeddings):
            cluster_texts[str(label)].append({'processed_text': text, 'embedding': embedding.tolist()})
        summaries = summarizer.summarize_all_clusters(cluster_texts)
        checkpoint_manager.save_stage('summaries', summaries)

    # Ensure summaries is a dictionary of strings
    if isinstance(summaries, dict):
        summaries = {str(k): v['summary'] if isinstance(v, dict) else v 
                    for k, v in summaries.items()}
    else:
        logger.error("Summaries is not a dictionary")
        summaries = {}

    summaries_file = output_dir / f"summaries_{run_id}.json"
    with open(summaries_file, 'w', encoding='utf-8') as f:
        json.dump(summaries, f)
    logger.info(f"Saved summaries to {summaries_file}")

    # Clear memory
    perf_optimizer.clear_memory_cache()
    logger.info("Completed summarization")

    # Visualize embeddings
    visualization_file = output_dir / f"visualizations_{run_id}.html"
    visualizer.visualize_embeddings(embeddings, save_path=visualization_file)
    logger.info(f"Saved visualizations to {visualization_file}")

    # Create references dictionary
    references = {}
    for label in set(clusters['labels']):
        str_label = str(label)
        cluster_indices = [i for i, l in enumerate(clusters['labels']) if str(l) == str_label]
        if cluster_indices:
            references[str_label] = processed_texts[cluster_indices[0]]

    # Convert summaries and references to lists for metrics calculation
    summary_list = []
    reference_list = []
    
    # Ensure we only include pairs where both summary and reference exist
    for label in summaries.keys():
        if label in references:
            summary_list.append(summaries[label])
            reference_list.append(references[label])

    # Evaluate results with lists instead of dictionaries
    evaluation_metrics = evaluator.calculate_comprehensive_metrics(
        summaries=summary_list,
        references=reference_list,
        embeddings=embeddings,
        labels=np.array(clusters['labels']),
        batch_size=batch_size
    )
    
    evaluation_file = output_dir / f"evaluation_{run_id}.json"
    with open(evaluation_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_metrics, f)
    logger.info(f"Saved evaluation metrics to {evaluation_file}")

if __name__ == '__main__':
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    mp.set_start_method('spawn', force=True)
    main(config)
