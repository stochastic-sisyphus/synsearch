
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
        print(f"Error processing batch: {e}")
        return []

def main():
    # Get optimal batch size and workers
    perf_optimizer = PerformanceOptimizer()
    batch_size = perf_optimizer.get_optimal_batch_size()
    n_workers = perf_optimizer.get_optimal_workers()
    
    print(f"Using {n_workers} workers with batch size {batch_size}")

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

    print(f"Processed {len(processed_texts)} texts")
    return processed_texts

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()