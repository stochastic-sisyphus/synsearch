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
from src.utils.metrics_calculator import MetricsCalculator
import gc
import logging


def init_worker():
    """Initialize worker process with optimized settings."""
    try:
        # Configure CUDA optimizations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False  # Added for speed
            
            # Set device specific settings
            device = torch.device('cuda')
            torch.cuda.set_device(device)
            
            # Optional: Set TensorFloat-32 for faster processing on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception as e:
        logging.error(f"CUDA initialization failed: {e}")
        device = torch.device('cpu')

def _get_optimal_batch_size() -> int:
    """Calculate optimal batch size with safety margin."""
    if torch.cuda.is_available():
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            free_memory = gpu_props.total_memory - torch.cuda.memory_allocated()
            embedding_size = 768  # Model embedding dimension
            dtype_size = 4  # float32 size
            safety_margin = 0.8  # Use 80% of available memory
            
            # Account for both input and gradient memory
            memory_per_sample = embedding_size * dtype_size * 3
            optimal_batch = int((free_memory * safety_margin) // memory_per_sample)
            
            return max(1, min(optimal_batch, 512))  # Cap at 512 for stability
        except Exception as e:
            logging.warning(f"Batch size calculation failed: {e}")
            return 32  # Fallback batch size for GPU
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
        
        # Define run_id
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
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

        # Convert summaries to the correct format for evaluation
        processed_summaries = {}
        for label, summary_data in summaries.items():
            if isinstance(summary_data, dict):
                processed_summaries[label] = summary_data.get('summary', '')
            else:
                processed_summaries[label] = str(summary_data)

        # Create lists for evaluation maintaining order
        summary_texts = []
        reference_texts = []
        unique_labels = sorted(set(clusters['labels']))

        for label in unique_labels:
            str_label = str(label)
            if str_label in processed_summaries:
                summary_texts.append(processed_summaries[str_label])
                
                # Get reference text from the first document in each cluster
                cluster_docs = [processed_texts[i] for i, l in enumerate(clusters['labels']) if l == label]
                reference_texts.append(cluster_docs[0] if cluster_docs else '')

        # Ensure arrays are properly formatted
        embeddings = np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings
        labels = np.array(clusters['labels'])

        # Initialize metrics evaluator with timing
        evaluator = EvaluationMetrics()
        evaluator.start_time = datetime.now()
        evaluator.num_samples = len(processed_texts)

        # Convert summaries and prepare for evaluation
        summary_reference_map = {}
        for label in unique_labels:
            str_label = str(label)
            if str_label in processed_summaries:
                summary_text = processed_summaries[str_label]
                # Get all documents in this cluster for better reference selection
                cluster_docs = [processed_texts[i] for i, l in enumerate(clusters['labels']) if str(l) == str_label]
                # Use the most representative document as reference
                reference_text = cluster_docs[0] if cluster_docs else ''
                summary_reference_map[str_label] = {
                    'summary': summary_text,
                    'reference': reference_text
                }

        # Prepare evaluation inputs
        try:
            # Convert summaries to proper format
            summary_texts = {}
            reference_texts = {}
            
            for label, cluster_docs in cluster_texts.items():
                if cluster_docs:  # Check if cluster has documents
                    # Get the summary for this cluster
                    summary = summaries.get(label, {}).get('summary', '')
                    if isinstance(summary, dict):
                        summary = summary.get('text', '')  # Handle nested dict case
                    summary_texts[label] = str(summary) if summary else ""
                    
                    # Use first document as reference
                    reference = cluster_docs[0].get('processed_text', '')
                    reference_texts[label] = str(reference) if reference else ""
            
            # Convert embeddings and labels to numpy arrays if needed
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            if isinstance(labels, list):
                labels = np.array(labels)
            
            # Validate inputs before evaluation
            if not summary_texts or not reference_texts:
                raise ValueError("No valid summaries or references for evaluation")
            
            # Calculate evaluation metrics
            evaluation_metrics = evaluator.calculate_comprehensive_metrics(
                summaries=summary_texts,
                references=reference_texts,
                embeddings=embeddings,
                labels=labels
            )
            
            # Save evaluation results
            evaluation_file = output_dir / f"evaluation_{run_id}.json"
            with open(evaluation_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_metrics, f, indent=2)
            log.info("Saved evaluation metrics to %s", evaluation_file)
            
        except Exception as e:
            log.error(f"Error during evaluation: {e}")
            evaluation_metrics = {
                'error': str(e),
                'rouge_scores': {},
                'bert_scores': {},
                'clustering': {}
            }

    except Exception as e:
        log.error(f"Error in main function: {e}")
        raise


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
