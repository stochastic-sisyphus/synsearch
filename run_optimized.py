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
from src.summarization.hybrid_summarizer import HybridSummarizer
from src.visualization.embedding_visualizer import EmbeddingVisualizer
from src.evaluation.metrics import EvaluationMetrics

def init_worker():
    """Initialize worker process with optimized settings."""
    # Pin memory for better performance
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

def process_batch(batch_data):
    """Process a batch of texts in parallel."""
    try:
        from src.preprocessor import DomainAgnosticPreprocessor
        preprocessor = DomainAgnosticPreprocessor()
        return [preprocessor.preprocess_text(text) for text in batch_data]
    except Exception as e:
        logging.error(f"Error processing batch: {e}")
        return []

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

    # Process batches in parallel
    processed_texts = []
    with ProcessPoolExecutor(
        max_workers=n_workers, 
        initializer=init_worker
    ) as executor:
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for batch_result in executor.map(process_batch, batches):
                processed_texts.extend(batch_result)
                pbar.update(1)

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
    cluster_manager = DynamicClusterManager()
    summarizer = HybridSummarizer()
    visualizer = EmbeddingVisualizer()
    evaluator = EvaluationMetrics()

    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(processed_texts)
    embeddings_file = output_dir / f"embeddings_{run_id}.npy"
    np.save(embeddings_file, embeddings)
    logging.info(f"Saved embeddings to {embeddings_file}")

    # Perform clustering
    labels, clustering_metrics = cluster_manager.fit_predict(embeddings)
    clusters_file = output_dir / f"clusters_{run_id}.json"
    with open(clusters_file, 'w') as f:
        json.dump({'labels': labels.tolist(), 'metrics': clustering_metrics}, f)
    logging.info(f"Saved clusters to {clusters_file}")

    # Generate summaries
    cluster_texts = {label: [] for label in set(labels)}
    for text, label in zip(processed_texts, labels):
        cluster_texts[label].append(text)
    summaries = summarizer.summarize_all_clusters(cluster_texts)
    summaries_file = output_dir / f"summaries_{run_id}.json"
    with open(summaries_file, 'w') as f:
        json.dump(summaries, f)
    logging.info(f"Saved summaries to {summaries_file}")

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
