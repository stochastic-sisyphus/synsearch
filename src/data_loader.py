from pathlib import Path
import pandas as pd
from datasets import load_dataset
import logging
from typing import Dict, Any, Union, List, Optional
import json
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer

class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        """Initialize DataLoader with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.batch_size = config.get('data', {}).get('batch_size', 32)
        self.supported_formats = {
            '.md': self._load_markdown,
            '.txt': self._load_text,
            '.pdf': self._load_pdf,
            '.json': self._load_json
        }

    def load_xlsum(self) -> Optional[Dict]:
        """Load XL-Sum dataset from Hugging Face"""
        try:
            self.logger.info("Loading XL-Sum dataset...")
            # Load only English subset of XL-Sum
            dataset = load_dataset('GEM/xlsum', 'english')
            if dataset and 'train' in dataset:
                self.logger.info(f"Successfully loaded XL-Sum dataset with {len(dataset['train'])} documents")
                return dataset
            else:
                self.logger.error("XL-Sum dataset structure is not as expected")
                return None
        except Exception as e:
            self.logger.error(f"Error loading XL-Sum dataset: {e}")
            return None

    def load_scisummnet(self, path: str) -> Optional[pd.DataFrame]:
        """Load ScisummNet dataset from local directory"""
        try:
            self.logger.info(f"Loading ScisummNet dataset from {path}...")
            data = []
            top1000_dir = Path(path) / 'top1000_complete'
            
            if not top1000_dir.exists():
                self.logger.error(f"Directory not found: {top1000_dir}")
                return None
            
            # List all document directories
            doc_dirs = [d for d in top1000_dir.iterdir() if d.is_dir()]
            self.logger.info(f"Found {len(doc_dirs)} potential documents")
            
            for doc_dir in doc_dirs:
                try:
                    # Load document text from Documents_xml directory
                    doc_xml_path = doc_dir / 'Documents_xml' / f'{doc_dir.name}.xml'
                    if not doc_xml_path.exists():
                        continue
                    
                    # Parse XML and extract text
                    tree = ET.parse(doc_xml_path)
                    root = tree.getroot()
                    text_elements = root.findall('.//S')
                    if not text_elements:
                        continue
                    
                    text = ' '.join([elem.text for elem in text_elements if elem.text])
                    
                    # Load summary from summary directory
                    summary_path = doc_dir / 'summary' / 'summary.txt'
                    if not summary_path.exists():
                        continue
                        
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = f.read().strip()
                    
                    if text and summary:  # Only add if both text and summary exist
                        data.append({
                            'id': doc_dir.name,
                            'text': text,
                            'summary': summary,
                            'source': 'scisummnet'
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Error processing document {doc_dir.name}: {e}")
                    continue
            
            if not data:
                self.logger.error("No valid documents found in ScisummNet dataset")
                return None
                
            df = pd.DataFrame(data)
            self.logger.info(f"Successfully loaded {len(df)} documents from ScisummNet")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading ScisummNet dataset: {e}")
            return None

    def load_data(self, source: Union[str, Path, List[str]]) -> Dict[str, Any]:
        """Universal data loading method with proper error handling."""
        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                if path.suffix.lower() in self.supported_formats:
                    return self.supported_formats[path.suffix.lower()](source)
            elif isinstance(source, list):
                return self._batch_process(source)
            raise ValueError(f"Unsupported data source: {source}")
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _load_markdown(self, source: str) -> List[Dict]:
        """Load markdown data from the given source"""
        # Implementation for markdown loading
        pass
    
    def _load_text(self, source: str) -> List[Dict]:
        """Load text data from the given source"""
        # Implementation for text loading
        pass
    
    def _load_pdf(self, source: str) -> List[Dict]:
        """Load PDF data from the given source"""
        # Implementation for PDF loading
        pass
    
    def _load_json(self, source: str) -> List[Dict]:
        """Load JSON data from the given source"""
        # Implementation for JSON loading
        pass
    
    def _load_url(self, source: str) -> List[Dict]:
        """Load data from the given URL"""
        # Implementation for URL loading
        pass
    
    def _batch_process(self, sources: List[str]) -> List[Dict]:
        """Process a batch of data sources"""
        # Implementation for batch processing
        pass

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all configured datasets."""
        datasets = {}
        
        # Load XL-Sum dataset
        try:
            self.logger.info("Loading XL-Sum dataset...")
            dataset = load_dataset('GEM/xlsum')
            # Convert to DataFrame and process
            df = pd.DataFrame({
                'text': dataset['train']['text'],
                'summary': dataset['train']['summary'],
                'id': range(len(dataset['train'])),
                'source': 'xlsum'
            })
            datasets['xlsum'] = df
        except Exception as e:
            self.logger.warning(f"Failed to load XL-Sum dataset: {e}")
        
        # Load ScisummNet dataset
        try:
            self.logger.info("Loading ScisummNet dataset...")
            scisummnet_path = self.config.get('data', {}).get('scisummnet_path')
            if not scisummnet_path:
                self.logger.warning("ScisummNet path not found in config")
            else:
                scisumm = self.load_scisummnet(scisummnet_path)
                if scisumm is not None:
                    datasets['scisummnet'] = scisumm
        except Exception as e:
            self.logger.warning(f"Failed to load ScisummNet dataset: {e}")
            
        if not datasets:
            raise ValueError("No datasets were successfully loaded")
            
        return datasets
        
    def load_xlsum_dataset(self) -> pd.DataFrame:
        """Load XL-Sum dataset from Hugging Face."""
        dataset = load_dataset('GEM/xlsum')
        # Convert to DataFrame and process as needed
        df = pd.DataFrame(dataset['train'])
        return df
        
    def load_scisummnet_dataset(self) -> pd.DataFrame:
        """Load ScisummNet dataset from local path."""
        path = self.config['data']['scisummnet_path']
        # Add your existing ScisummNet loading logic here
        # Return the loaded data as a DataFrame
        pass

def load_xlsum_dataset():
    """Load and preprocess XL-Sum dataset"""
    from datasets import load_dataset
    dataset = load_dataset('GEM/xlsum')
    return dataset

def load_scisummnet():
    """Load and preprocess ScisummNet dataset"""
    base_path = Path("/Users/vanessa/Dropbox/synsearch/data/scisummnet_release1.1__20190413")
    data_path = base_path / "top1000_complete"
    # Process ScisummNet data
    return processed_data

def load_dataset(dataset_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load and prepare dataset based on configuration."""
    logger = logging.getLogger(__name__)
    
    if dataset_config.get('source') == 'huggingface':
        # Load XL-Sum from Hugging Face
        logger.info(f"Loading {dataset_config['dataset_name']} from Hugging Face")
        dataset = load_dataset(dataset_config['dataset_name'])
        
        # Filter for English if specified
        if dataset_config.get('language') == 'english':
            dataset = dataset.filter(lambda x: x['language'] == 'english')
        
        # Prepare documents
        documents = [
            {
                'text': item['text'],
                'summary': item['summary'],
                'id': idx
            }
            for idx, item in enumerate(dataset['train'])
        ]
        
    elif 'scisummnet' in dataset_config['path']:
        # Load ScisummNet from local path
        logger.info("Loading ScisummNet dataset")
        path = Path(dataset_config['path'])
        documents = load_scisummnet(
            path / dataset_config['top1000_dir'],
            max_papers=dataset_config.get('papers', None)
        )
    
    else:
        raise ValueError(f"Unsupported dataset source: {dataset_config}")
    
    # Generate embeddings
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode([doc['text'] for doc in documents])
    
    return {
        'name': dataset_config['name'],
        'documents': documents,
        'embeddings': embeddings
    }

def load_scisummnet(path: Path, max_papers: int = None) -> List[Dict]:
    """Load papers from ScisummNet dataset."""
    documents = []
    paper_dirs = list(path.glob('*'))
    
    if max_papers:
        paper_dirs = paper_dirs[:max_papers]
    
    for paper_dir in paper_dirs:
        try:
            with open(paper_dir / 'summary.txt') as f:
                summary = f.read().strip()
            with open(paper_dir / 'paper.txt') as f:
                text = f.read().strip()
                
            documents.append({
                'text': text,
                'summary': summary,
                'id': paper_dir.name
            })
        except Exception as e:
            logging.warning(f"Error loading paper {paper_dir}: {e}")
            continue
    
    return documents