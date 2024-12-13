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
from src.utils.metrics_utils import MetricsCalculator
import gc
import logging


def init_worker():
    """Initialize worker process with optimized settings."""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

def _get_optimal_batch_size() -> int:
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        # Use 80% of available memory, assuming 768 dimensions and float32
        return max(1, int((free_memory * 0.8) // (768 * 4)))
    return 64  # Default CPU batch size

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
    try:
        torch.cuda.empty_cache()
        gc.collect()

        # Add configuration
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize logger and get its logger
        logger = MetricsLogger(config)
        log = logger.logger
        
        # Initialize performance optimizer
        perf_optimizer = PerformanceOptimizer()
        batch_size = _get_optimal_batch_size()
        n_workers = perf_optimizer.get_optimal_workers()
        
        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.get('checkpoints', {}).get('dir', 'outputs/checkpoints'),
            enable_metrics=True
        )

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
        output_dir = Path(config['data']['output_path'])
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"processed_texts_{run_id}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
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
            cluster_texts = {str(label): [] for label in set(clusters['labels'])}
            for text, label, embedding in zip(processed_texts, clusters['labels'], embeddings):
                cluster_texts[str(label)].append({'processed_text': text, 'embedding': embedding.tolist()})
            summaries = summarizer.summarize_all_clusters(cluster_texts)
            checkpoint_manager.save_stage('summaries', summaries)

        summaries_file = output_dir / f"summaries_{run_id}.json"
        with open(summaries_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f)
        log.info("Saved summaries to %s", summaries_file)

        perf_optimizer.clear_memory_cache()
        checkpoint_manager.save_periodic_checkpoint('summaries', summaries)
        log.info("Completed summarization")

        visualizer = EmbeddingVisualizer()
        visualization_file = output_dir / f"visualizations_{run_id}.html"
        visualizer.visualize_embeddings(embeddings, save_path=visualization_file)
        log.info("Saved visualizations to %s", visualization_file)

        # Create summary and reference lists for evaluation
        summary_texts = []
        reference_texts = []

        # Convert summaries to the correct format
        for label in sorted(summaries.keys()):
            if isinstance(summaries[label], dict) and 'summary' in summaries[label]:
                summary_text = summaries[label]['summary']
            else:
                summary_text = summaries[label]
            
            summary_texts.append(summary_text)
            
            # Use the first text from each cluster as reference
            cluster_indices = [i for i, l in enumerate(clusters['labels']) if str(l) == str(label)]
            if cluster_indices:
                reference_texts.append(processed_texts[cluster_indices[0]])
            else:
                reference_texts.append("")  # Empty string as fallback

        # Ensure embeddings is a numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # Convert labels to numpy array
        labels = np.array(clusters['labels'])

        evaluator = EvaluationMetrics()
        evaluation_metrics = evaluator.calculate_comprehensive_metrics(
            summaries=summary_texts,
            references=reference_texts,
            embeddings=embeddings,
            labels=labels,
            batch_size=batch_size
        )
        
        evaluation_file = output_dir / f"evaluation_{run_id}.json"
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_metrics, f)
        log.info("Saved evaluation metrics to %s", evaluation_file)

    except Exception as e:
        log.error(f"Error in main function: {e}")
        raise


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
