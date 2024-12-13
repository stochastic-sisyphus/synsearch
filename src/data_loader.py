from pathlib import Path
import pandas as pd
from datasets import load_dataset
import logging
from typing import Dict, Any, Union, List, Optional
import xml.etree.ElementTree as ET
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from multiprocessing import Pool, cpu_count
from src.utils.performance import PerformanceOptimizer
from src.data_validator import DataValidator

class TextDataset(Dataset):
    """Custom Dataset for text data."""
    
    def __init__(self, texts: list):
        self.texts = texts
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

class DataLoader:
    """
    DataLoader class for loading and validating datasets.

    The DataLoader requires the following:
    - A configuration dictionary `config` with necessary keys.
    - Proper logger initialization, i.e., `self.logger` should be set up using `logging.getLogger(__name__)`.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        logger (logging.Logger): Logger instance.
        perf_optimizer (PerformanceOptimizer): Performance optimizer instance.
        batch_size (int): Optimal batch size.
        num_workers (int): Optimal number of workers.
        validator (DataValidator): Data validator instance.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize DataLoader with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize performance optimizer
        self.perf_optimizer = PerformanceOptimizer()
        
        # Get optimal batch size
        self.batch_size = min(
            config.get('data', {}).get('batch_size', 32),
            self.perf_optimizer.get_optimal_batch_size()
        )
        
        # Get optimal number of workers
        self.num_workers = self.perf_optimizer.get_optimal_workers()
        
        # Convert relative paths to absolute using project root
        project_root = Path(__file__).parent.parent
        if 'scisummnet_path' in self.config['data']:
            self.config['data']['scisummnet_path'] = str(project_root / self.config['data']['scisummnet_path'])

        # Initialize DataValidator
        self.validator = DataValidator()

        # Validate input configuration
        if not self._validate_config():
            raise ValueError("Invalid configuration format")

    def _validate_config(self) -> bool:
        """Validate input configuration"""
        required_keys = ['data', 'preprocessing', 'embedding', 'clustering']
        for key in required_keys:
            if key not in self.config:
                self.logger.error(f"Missing required configuration key: {key}")
                return False
        return True

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all configured datasets."""
        datasets = {}
        
        # Validate input configuration
        if not isinstance(self.config, dict) or 'data' not in self.config:
            raise ValueError("Invalid configuration format")
        
        for dataset_config in self.config['data']['datasets']:
            if not dataset_config.get('enabled', False):
                continue
            
            dataset_name = dataset_config['name']
            if dataset_name == 'xlsum':
                datasets.update(self._load_xlsum_dataset(dataset_config))
            elif dataset_name == 'scisummnet':
                datasets.update(self._load_scisummnet_dataset())
        
        if not datasets:
            raise ValueError("No datasets were successfully loaded")
        
        return datasets

    def _load_xlsum_dataset(self, dataset_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load XL-Sum dataset."""
        datasets = {}
        try:
            self.logger.info("Loading XL-Sum dataset...")
            language = dataset_config.get('language', 'english')
            dataset = load_dataset('GEM/xlsum', language)
            if dataset and 'train' in dataset:
                df = pd.DataFrame({
                    'text': dataset['train']['text'],
                    'summary': dataset['train']['target'],
                    'id': range(len(dataset['train'])),
                    'source': ['xlsum'] * len(dataset['train'])
                })
                validation_results = self.validator.validate_dataset(df)
                if not validation_results['is_valid']:
                    self.logger.warning(f"XL-Sum dataset validation failed: {validation_results}")
                else:
                    datasets['xlsum'] = df
                    self.logger.info(f"Successfully loaded {len(df)} documents from XL-Sum")
            else:
                self.logger.warning("XL-Sum dataset structure is not as expected")
        except Exception as e:
            self.logger.warning(f"Failed to load XL-Sum dataset: {e}")
        return datasets

    def _load_scisummnet_dataset(self) -> Dict[str, pd.DataFrame]:
        """Load ScisummNet dataset."""
        datasets = {}
        try:
            self.logger.info("Loading ScisummNet dataset...")
            scisummnet_path = Path(self.config['data']['scisummnet_path'])
            if scisummnet_path.exists():
                df = self.load_scisummnet(str(scisummnet_path))
                if df is not None and not df.empty:
                    validation_results = self.validator.validate_dataset(df)
                    if not validation_results['is_valid']:
                        self.logger.warning(f"ScisummNet dataset validation failed: {validation_results}")
                    else:
                        datasets['scisummnet'] = df
                        self.logger.info(f"Successfully loaded {len(df)} documents from ScisummNet")
                else:
                    self.logger.warning("No valid documents found in ScisummNet dataset")
            else:
                self.logger.warning(f"ScisummNet path not found: {scisummnet_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load ScisummNet dataset: {e}")
        return datasets

    def load_scisummnet(self, path: str) -> Optional[pd.DataFrame]:
        """Load ScisummNet dataset with parallel processing."""
        try:
            self.logger.info(f"Loading ScisummNet dataset from {path}...")
            top1000_dir = Path(path) / 'top1000_complete'
            
            if not top1000_dir.exists():
                raise FileNotFoundError(f"Directory not found: {top1000_dir}")
            
            doc_dirs = [d for d in top1000_dir.iterdir() if d.is_dir()]
            
            with Pool(processes=cpu_count()) as pool:
                results = list(tqdm(
                    pool.imap(self._process_document, doc_dirs),
                    total=len(doc_dirs),
                    desc="Processing documents"
                ))
                
            valid_results = [r for r in results if r is not None]
            
            if not valid_results:
                raise ValueError("No valid documents found")
                
            df = pd.DataFrame(valid_results)
            self.logger.info(f"Successfully loaded {len(df)} documents")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading ScisummNet: {e}")
            return None

    def _process_document(self, doc_dir: Path) -> Optional[Dict]:
        """Process a single document with error handling."""
        try:
            paper_id = doc_dir.name
            xml_path = doc_dir / 'Documents_xml' / f'{paper_id}.xml'
            summary_path = doc_dir / 'summary' / f'{paper_id}.gold.txt'
            
            if not xml_path.exists() or not summary_path.exists():
                return None
                
            with open(xml_path, 'rb') as f:
                tree = ET.parse(f)
                
            text = ' '.join(
                elem.text.strip()
                for elem in tree.findall('.//S')
                if elem.text and elem.text.strip()
            )
            
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = f.read().strip()
                
            if not text or not summary:
                return None
                
            return {
                'text': text,
                'summary': summary,
                'paper_id': paper_id,
                'source': 'scisummnet'
            }
            
        except Exception as e:
            self.logger.warning(f"Error processing {doc_dir.name}: {e}")
            return None

class EnhancedDataLoader:
    def __init__(self, config: Dict[str, Any]):
        """Initialize EnhancedDataLoader with configuration"""
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 4)
        self.logger = logging.getLogger(__name__)
        
    def get_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create a PyTorch DataLoader with optimal settings."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=False
        )
        
    def load_scisummnet(self, path: str) -> Optional[pd.DataFrame]:
        """Load ScisummNet dataset with progress tracking."""
        try:
            self.logger.info(f"Loading ScisummNet dataset from {path}...")
            data = []
            
            top1000_dir = Path(path) / 'top1000_complete'
            if not top1000_dir.exists():
                raise FileNotFoundError(f"Directory not found: {top1000_dir}")
            
            doc_dirs = [d for d in top1000_dir.iterdir() if d.is_dir()]
            
            # Add progress bar
            for doc_dir in tqdm(doc_dirs, desc="Loading documents"):
                # ...existing document processing code...
                pass  # Placeholder for existing document processing code
                
            return pd.DataFrame(data) if data else None
                
        except Exception as e:
            self.logger.error(f"Error loading ScisummNet dataset: {e}")
            return None

class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validator = DataValidator()

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        datasets = {}
        
        for dataset_config in self.config['data']['datasets']:
            if not dataset_config.get('enabled', False):
                continue
                
            try:
                dataset_name = dataset_config['name']
                if dataset_name == 'xlsum':
                    dataset = load_dataset(
                        dataset_config['dataset_name'], 
                        dataset_config.get('language', 'english'),
                        cache_dir=self.config['data'].get('cache_dir', 'data/cache')
                    )
                    if dataset and 'train' in dataset:
                        df = pd.DataFrame({
                            'text': dataset['train']['text'],
                            'summary': dataset['train']['target'],
                            'id': range(len(dataset['train'])),
                            'source': ['xlsum'] * len(dataset['train'])
                        })
                        datasets['xlsum'] = df
                        
                elif dataset_name == 'scisummnet':
                    df = self._load_scisummnet_dataset()
                    if df is not None:
                        datasets['scisummnet'] = df
                        
            except Exception as e:
                self.logger.error(f"Error loading dataset {dataset_name}: {e}")
                continue
                
        if not datasets:
            raise ValueError("No datasets were successfully loaded")
            
        return datasets
