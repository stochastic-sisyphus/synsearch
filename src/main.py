import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import logging
import torch
import numpy as np
from tqdm import tqdm
import json
import multiprocessing
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import random
from datasets import load_dataset
from dataclasses import dataclass, asdict
from typing import Optional, Union, Type
from src.utils.performance import PerformanceOptimizer  # Add this

# Add project root to PYTHONPATH when running directly
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    # Update imports to absolute paths when running as script
    from src.data_loader import DataLoader
    from src.data_validator import DataValidator, ConfigValidator
    from src.embedding_generator import EnhancedEmbeddingGenerator
    from src.visualization.embedding_visualizer import EmbeddingVisualizer
    from src.preprocessor import DomainAgnosticPreprocessor
    from src.summarization.hybrid_summarizer import HybridSummarizer
    from src.evaluation.metrics import EvaluationMetrics
    from src.clustering.dynamic_cluster_manager import DynamicClusterManager
    from src.utils.metrics_utils import (
        calculate_cluster_metrics,
        calculate_cluster_variance,
        calculate_lexical_diversity
    )
    from src.utils.style_selector import determine_cluster_style, get_style_parameters
    from src.utils.logging_config import setup_logging
    from src.utils.metrics_calculator import MetricsCalculator
    from src.summarization.adaptive_summarizer import AdaptiveSummarizer
    from src.clustering.clustering_utils import process_clusters
    from src.utils.checkpoint_manager import CheckpointManager
    from src.data_validator import DataValidator
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    import signal
    import sys
else:
    # Use relative imports when imported as module
    from .data_loader import DataLoader
    from .data_validator import DataValidator, ConfigValidator
    from .embedding_generator import EnhancedEmbeddingGenerator
    from .visualization.embedding_visualizer import EmbeddingVisualizer
    from .preprocessor import DomainAgnosticPreprocessor
    from .summarization.hybrid_summarizer import HybridSummarizer
    from .evaluation.metrics import EvaluationMetrics
    from .clustering.dynamic_cluster_manager import DynamicClusterManager
    from .utils.metrics_utils import (
        calculate_cluster_metrics,
        calculate_cluster_variance,
        calculate_lexical_diversity
    )
    from .utils.style_selector import determine_cluster_style, get_style_parameters
    from .utils.logging_config import setup_logging
    from .utils.metrics_calculator import MetricsCalculator
    from .summarization.adaptive_summarizer import AdaptiveSummarizer
    from .clustering.clustering_utils import process_clusters
    from .utils.checkpoint_manager import CheckpointManager
    from .data_validator import DataValidator
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    import signal
    import sys

# Set up logging with absolute paths
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file))
    ]
)

# Initialize logger at module level
logger = logging.getLogger(__name__)

def get_device():
    """Get the best available device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_optimal_workers():
    """Get optimal number of worker processes."""
    return multiprocessing.cpu_count()

def setup_logging(config):
    """Configure logging based on config settings."""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format']
    )
    return logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_texts(texts: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Process texts with adaptive summarization and enhanced metrics."""
    # Initialize components with config settings
    embedding_gen = EnhancedEmbeddingGenerator(
        model_name=config['embedding']['model_name'],
        batch_size=config['embedding']['batch_size'],
        max_seq_length=config['embedding']['max_seq_length']
    )
    
    # Generate embeddings with performance settings
    embeddings = embedding_gen(texts)
    
    # Process clusters with metrics
    results = process_clusters(texts, embeddings, config)
    
    return results

def validate_config(config):
    """Validate configuration and create directories."""
    required_dirs = [
        ('input_path', config['data']['input_path']),
        ('output_path', config['data']['output_path']),
        ('processed_path', config['data']['processed_path']),
        ('visualization_output', config['visualization']['output_dir']),
        ('checkpoints_dir', config['checkpoints']['dir'])
    ]
    
    for dir_key, dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        if not os.access(dir_path, os.W_OK):
            raise ValueError(f"No write permission for path: {dir_path} ({dir_key})")
    
    # Validate dataset configurations
    for dataset_config in config['data']['datasets']:
        if (dataset_config['name'] == 'xlsum' and dataset_config.get('enabled', False)):
            if 'language' not in dataset_config:
                raise ValueError("XL-Sum dataset requires 'language' specification in config")
            if 'dataset_name' not in dataset_config:
                raise ValueError("XL-Sum dataset requires 'dataset_name' specification in config")
        elif (dataset_config['name'] == 'scisummnet' and dataset_config.get('enabled', False)):
            if not config['data'].get('scisummnet_path'):
                raise ValueError("ScisummNet dataset requires 'scisummnet_path' in config['data']")
    
    # Validate summarization configuration
    if 'summarization' not in config:
        config['summarization'] = {
            'model_name': 'facebook/bart-large-cnn',
            'style_params': {
                'concise': {'max_length': 100, 'min_length': 30},
                'detailed': {'max_length': 300, 'min_length': 100},
                'technical': {'max_length': 200, 'min_length': 50}
            },
            'num_beams': 4,
            'length_penalty': 2.0,
            'early_stopping': True
        }
    elif 'model_name' not in config['summarization']:
        config['summarization']['model_name'] = 'facebook/bart-large-cnn'
    
    return config

def process_dataset(
    dataset: Dict[str, Any],
    cluster_manager: DynamicClusterManager,
    summarizer: AdaptiveSummarizer,
    evaluator: EvaluationMetrics,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Process dataset with enhanced error handling and metrics tracking."""
    logger = logging.getLogger(__name__)
    start_time = datetime.now()
    
    try:
        # Initialize embedding generator with memory-aware settings
        embedding_generator = EnhancedEmbeddingGenerator(
            model_name=config['embedding']['model_name'],
            batch_size=min(
                config['embedding'].get('batch_size', 32),
                get_optimal_batch_size()
            ),
            max_seq_length=config['embedding'].get('max_seq_length', 512),
            config=config,
            device=None  # Let the class handle device selection
        )

        # Get dataset name, fallback to 'unnamed' if not provided
        dataset_name = dataset.get('name', 'unnamed')
        
        # Update cache directory path
        cache_dir = Path(config['checkpoints']['dir']) / dataset_name / 'embeddings' \
            if config['checkpoints'].get('enabled', False) else None

        # Process in smaller chunks to handle memory better
        chunk_size = min(
            config.get('processing', {}).get('chunk_size', 1000),
            5000  # Set a reasonable maximum chunk size
        )
        text_chunks = [dataset['texts'][i:i + chunk_size] 
                      for i in range(0, len(dataset['texts']), chunk_size)]
        
        all_embeddings = []
        for chunk in tqdm(text_chunks, desc="Processing chunks"):
            chunk_embeddings = embedding_generator.generate_embeddings(
                chunk,
                cache_dir=Path(config['checkpoints']['dir']) / dataset['name'] / 'embeddings' 
                if config['checkpoints']['enabled'] else None
            )
            all_embeddings.append(chunk_embeddings)
        
        # Combine all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Validate dataset
        validation_results = DataValidator().validate_dataset(dataset)
        if not validation_results['is_valid']:
            raise ValueError(f"Dataset validation failed: {validation_results['checks']}")
        
        # Perform clustering with metrics
        labels, clustering_metrics = cluster_manager.fit_predict(embeddings)
        
        # Generate summaries with style adaptation
        summaries = {}
        clusters = cluster_manager.get_clusters(dataset['texts'], labels)
        
        for cluster_id, docs in clusters.items():
            optimal_style = determine_cluster_style(docs)
            summary = summarizer.summarize_cluster(docs, style=optimal_style)
            summaries[cluster_id] = summary
        
        # Calculate comprehensive metrics
        metrics = evaluator.calculate_comprehensive_metrics(
            summaries=summaries,
            references=dataset.get('references', {}),
            embeddings=embeddings
        )
        
        # Add runtime metrics
        runtime = (datetime.now() - start_time).total_seconds()
        metrics['runtime'] = {
            'total_seconds': runtime,
            'processed_documents': len(dataset['texts']),
            'documents_per_second': len(dataset['texts']) / runtime
        }
        
        results = {
            'validation': validation_results,
            'clustering': clustering_metrics,
            'summaries': summaries,
            'metrics': metrics
        }
        
        # Save results and metrics
        save_path = Path(config['data']['output_path'])
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / f"{dataset['name']}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
        
    except Exception as e:
        logger.error(f"Error processing dataset {dataset.get('name', 'unknown')}: {e}")
        raise

def get_optimal_batch_size() -> int:
    """Determine optimal batch size based on available CUDA memory."""
    if not torch.cuda.is_available():
        return 32  # Default CPU batch size
        
    try:
        torch.cuda.reset_peak_memory_stats()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - reserved_memory - allocated_memory
        
        # Leave some buffer memory
        usable_memory = free_memory * 0.8
        
        # Estimate memory per sample (embedding dimension * 4 bytes for float32)
        memory_per_sample = 768 * 4  # Assuming 768 dimension embeddings
        
        optimal_batch_size = int(usable_memory / memory_per_sample)
        return max(1, min(optimal_batch_size, 64))  # Cap at reasonable size
        
    except Exception:
        return 32  # Fallback to default

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_environment():
    """Initialize environment settings and global configurations."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configure logging
    logging_config = {
        'level': logging.INFO,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'handlers': [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/pipeline.log')
        ]
    }
    logging.basicConfig(**logging_config)
    
    # Verify CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    return device

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    logger.info("Received interrupt signal. Cleaning up...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class PipelineManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config['checkpoints']['dir']
        )
        self.data_validator = DataValidator()

    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """Validate dataset structure and content"""
        try:
            # Check for empty dataset
            if df.empty:
                raise ValueError("Empty dataset provided")
                
            # Check for required columns
            required_columns = ['text', 'summary'] if 'summary' in df else ['text']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Check for all-null columns
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                raise ValueError(f"Columns contain all null values: {null_columns}")
                
            # Validate data types
            text_col = 'text' if 'text' in df else df.columns[0]
            if not df[text_col].dtype == object:
                raise TypeError(f"Text column '{text_col}' must be string type")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            return False

    def process_dataset_with_checkpoints(
        self,
        dataset: Dict[str, Any],
        cluster_manager: DynamicClusterManager,
        summarizer: AdaptiveSummarizer,
        evaluator: EvaluationMetrics,
    ) -> Optional[Dict[str, Any]]:
        """Process dataset with checkpointing and error handling"""
        try:
            # Validate dataset
            if not self.validate_dataset(pd.DataFrame(dataset)):
                return None
                
            # Check for existing checkpoint
            checkpoint = self.checkpoint_manager.get_stage_data(dataset['name'])
            if checkpoint:
                self.logger.info(f"Resuming from checkpoint for {dataset['name']}")
                return self._resume_from_checkpoint(checkpoint, dataset)
                
            # Process in batches with checkpointing
            batch_size = min(
                self.config['processing'].get('batch_size', 1000),
                5000
            )
            
            results = self._process_batches(
                dataset, 
                cluster_manager,
                summarizer,
                evaluator,
                batch_size
            )
            
            # Save final results
            self.checkpoint_manager.save_stage(dataset['name'], results)
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing dataset {dataset.get('name', 'unknown')}: {e}")
            raise

    def _process_batches(
        self,
        dataset: Dict[str, Any],
        cluster_manager: DynamicClusterManager,
        summarizer: AdaptiveSummarizer,
        evaluator: EvaluationMetrics,
        batch_size: int
    ) -> Dict[str, Any]:
        """Process dataset in batches with progress tracking"""
        texts = dataset['texts']
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        results = {
            'embeddings': [],
            'clusters': [],
            'summaries': {}
        }
        
        with tqdm(total=total_batches, desc=f"Processing {dataset['name']}") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Process batch
                batch_results = self._process_single_batch(
                    batch_texts,
                    cluster_manager,
                    summarizer,
                    evaluator
                )
                
                # Update results
                results['embeddings'].extend(batch_results['embeddings'])
                results['clusters'].extend(batch_results['clusters'])
                results['summaries'].update(batch_results['summaries'])
                
                # Save checkpoint
                self.checkpoint_manager.save_stage(
                    f"{dataset['name']}_batch_{i//batch_size}",
                    batch_results
                )
                
                pbar.update(1)
                
        return results

@dataclass
class PipelineConfig:
    """Strongly typed configuration for validation"""
    batch_size: int = 32
    max_seq_length: int = 512
    cache_dir: Optional[str] = None
    device: str = 'auto'
    debug_mode: bool = False
    
class EnhancedPipelineManager(PipelineManager):
    """Enhanced pipeline manager with better error handling and environment flexibility"""
    
    def __init__(self, config: Union[Dict[str, Any], PipelineConfig]):
        if isinstance(config, dict):
            # Convert dict to typed config
            config = PipelineConfig(**config)
            
        super().__init__(asdict(config))
        self.early_stopping = False
        self._setup_environment()
        
    def _setup_environment(self):
        """Validate environment and dependencies"""
        try:
            # Check Python version
            min_python = (3, 8)
            if sys.version_info < min_python:
                raise EnvironmentError(f"Python {'.'.join(map(str, min_python))} or higher required")
                
            # Verify critical dependencies
            self._verify_dependencies()
            
            # Setup device
            if self.config.device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(self.config.device)
                
            self.logger.info(f"Environment setup complete. Using device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Environment setup failed: {e}")
            raise
            
    def _verify_dependencies(self):
        """Verify all required dependencies are available"""
        required = {
            'torch': 'Deep learning',
            'transformers': 'Language models',
            'numpy': 'Numerical operations',
            'pandas': 'Data processing',
            'tqdm': 'Progress tracking'
        }
        
        missing = []
        for pkg, purpose in required.items():
            try:
                __import__(pkg)
            except ImportError:
                missing.append(f"{pkg} ({purpose})")
                
        if missing:
            raise ImportError(f"Missing required dependencies: {', '.join(missing)}")
            
    def process_with_checkpoints(
        self, 
        dataset: pd.DataFrame,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process dataset with enhanced error handling and checkpointing"""
        try:
            if self.config.debug_mode:
                # Sample dataset for debugging
                dataset = dataset.sample(min(len(dataset), 1000))
                
            # Validate dataset structure
            self._validate_dataset_structure(dataset)
            
            # Process in batches with checkpointing
            results = []
            batch_size = batch_size or self.config.batch_size
            
            for i in range(0, len(dataset), batch_size):
                if self.early_stopping:
                    self.logger.info("Early stopping requested")
                    break
                    
                batch = dataset.iloc[i:i + batch_size]
                try:
                    batch_results = self._process_batch(batch)
                    results.append(batch_results)
                    
                    # Save checkpoint after each batch
                    self._save_checkpoint(i // batch_size, batch_results)
                    
                except Exception as e:
                    self.logger.error(f"Error processing batch {i // batch_size}: {e}")
                    continue
                    
            return self._combine_results(results)
            
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user. Saving progress...")
            self._handle_interrupt()
            raise
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise

    def _validate_dataset_structure(self, df: pd.DataFrame) -> None:
        """Validate dataset structure with detailed error messages"""
        if df.empty:
            raise ValueError("Empty dataset provided")
            
        required_cols = {'text'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        # Check for null values in critical columns
        null_counts = df[list(required_cols)].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}")

def validate_dataset_paths(config: Dict[str, Any]) -> None:
    """Validate that required datasets are present"""
    # Find scisummnet dataset config if enabled
    scisummnet_config = next(
        (dataset for dataset in config['data']['datasets'] 
         if dataset['name'] == 'scisummnet' and dataset.get('enabled', False)),
        None
    )
    
    if scisummnet_config:
        scisummnet_path = Path(config['data']['scisummnet_path'])
        if not scisummnet_path.exists():
            raise FileNotFoundError(
                f"ScisummNet dataset not found at {scisummnet_path}. "
                "Please run 'make download-data' to download required datasets."
            )

def main():
    try:
        # Load and validate configuration
        config = load_config()
        config = validate_config(config)
        
        # Validate dataset paths
        validate_dataset_paths(config)
        
        # Initialize environment and performance optimizer
        device = setup_environment()
        perf_optimizer = PerformanceOptimizer()
        
        # Load and validate configuration
        config = load_config()
        config = validate_config(config)
        
        # Initialize pipeline manager
        pipeline = PipelineManager(config)
        
        # Initialize embedding generator with proper config
        embedding_generator = EnhancedEmbeddingGenerator(
            model_name=config['embedding']['model_name'],
            embedding_dim=config['embedding'].get('dimension', 768),
            max_seq_length=config['embedding'].get('max_seq_length', 512),
            batch_size=perf_optimizer.get_optimal_batch_size(),
            device=device,
            config=config['embedding']
        )

        # Initialize components with dependency injection
        data_loader = DataLoader(config)
        preprocessor = DomainAgnosticPreprocessor(config['preprocessing'])
        embedding_generator = EnhancedEmbeddingGenerator(
            model_name=config['embedding']['model_name'],
            device=device
        )
        
        # Load datasets
        datasets = data_loader.load_all_datasets()
        
        # Process each dataset
        processed_datasets = {}
        for dataset_name, df in datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Handle different dataset structures
            if (dataset_name == 'scisummnet'):
                texts = df['summary'].tolist()  # Use summaries for ScisummNet
                ids = df['paper_id'].tolist()  # Use paper_id as ID
            else:
                texts = df['text'].tolist()
                ids = df.get('id', range(len(df))).tolist()  # Fallback to index if no ID
            
            # Preprocess texts
            processed_texts = preprocessor.preprocess_texts(texts)
            
            # Generate embeddings
            embedding_generator = EnhancedEmbeddingGenerator(
                model_name=config['embedding']['model_name'],
                embedding_dim=config['embedding'].get('dimension', 768),
                max_seq_length=config['embedding'].get('max_seq_length', 384),
                batch_size=config['embedding'].get('batch_size', 32),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            embeddings = embedding_generator.generate_embeddings(processed_texts)
            
            processed_datasets[dataset_name] = {
                'name': dataset_name,  # Add dataset name
                'texts': processed_texts,
                'embeddings': embeddings,
                'summaries': df['summary'].tolist() if 'summary' in df else None,
                'ids': ids,
                'source': dataset_name
            }
        
        # Initialize clustering manager
        cluster_manager = DynamicClusterManager(config['clustering'])
        
        # Initialize adaptive summarizer
        summarizer = AdaptiveSummarizer(config['summarization'])
        
        # Initialize evaluation metrics
        evaluator = EvaluationMetrics()
        
        # Process each dataset through the pipeline
        for dataset_name, dataset in processed_datasets.items():
            logger.info(f"Running pipeline for dataset: {dataset_name}")
            batch_size = perf_optimizer.get_optimal_batch_size()
            
            results = process_dataset(
                dataset=dataset,
                cluster_manager=cluster_manager,
                summarizer=summarizer,
                evaluator=evaluator,
                config={**config, 'batch_size': batch_size}
            )
            logger.info(f"Pipeline results for {dataset_name}: {results}")
        
        logger.info("Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical error in pipeline: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
