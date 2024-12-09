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
        
        # Convert relative paths to absolute using project root
        project_root = Path(__file__).parent.parent
        if 'scisummnet_path' in self.config['data']:
            self.config['data']['scisummnet_path'] = str(project_root / self.config['data']['scisummnet_path'])

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all configured datasets."""
        datasets = {}
        
        # Load XL-Sum dataset if enabled
        for dataset_config in self.config['data']['datasets']:
            if not dataset_config.get('enabled', False):
                continue
                
            if dataset_config['name'] == 'xlsum':
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
                        datasets['xlsum'] = df
                        self.logger.info(f"Successfully loaded {len(df)} documents from XL-Sum")
                    else:
                        self.logger.warning("XL-Sum dataset structure is not as expected")
                except Exception as e:
                    self.logger.warning(f"Failed to load XL-Sum dataset: {e}")
            
            # Load ScisummNet dataset if enabled
            elif dataset_config['name'] == 'scisummnet':
                try:
                    self.logger.info("Loading ScisummNet dataset...")
                    scisummnet_path = Path(self.config['data']['scisummnet_path'])
                    if scisummnet_path.exists():
                        df = self.load_scisummnet(str(scisummnet_path))
                        if df is not None and not df.empty:
                            datasets['scisummnet'] = df
                            self.logger.info(f"Successfully loaded {len(df)} documents from ScisummNet")
                        else:
                            self.logger.warning("No valid documents found in ScisummNet dataset")
                    else:
                        self.logger.warning(f"ScisummNet path not found: {scisummnet_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to load ScisummNet dataset: {e}")

        if not datasets:
            raise ValueError("No datasets were successfully loaded")
        
        return datasets

    def load_scisummnet(self, path: str) -> Optional[pd.DataFrame]:
        """Load ScisummNet dataset from local directory"""
        try:
            self.logger.info(f"Loading ScisummNet dataset from {path}...")
            data = []
            top1000_dir = Path(path)
            
            if not top1000_dir.exists():
                self.logger.error(f"Directory not found: {top1000_dir}")
                return None
            
            # List all document directories
            doc_dirs = [d for d in top1000_dir.iterdir() if d.is_dir()]
            self.logger.info(f"Found {len(doc_dirs)} potential documents")
            
            for doc_dir in doc_dirs:
                try:
                    # Load document text
                    doc_path = doc_dir / 'Documents_xml' / f'{doc_dir.name}.xml'
                    summary_path = doc_dir / 'summary' / 'summary.txt'
                    
                    if not doc_path.exists() or not summary_path.exists():
                        continue
                    
                    # Read document text from XML
                    tree = ET.parse(doc_path)
                    root = tree.getroot()
                    text_elements = root.findall('.//S')
                    text = ' '.join(elem.text.strip() for elem in text_elements if elem.text)
                    
                    # Read summary
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

    def load_scisummnet_dataset(self) -> pd.DataFrame:
        """Load ScisummNet dataset."""
        try:
            self.logger.info(f"Loading ScisummNet dataset from {self.config['data']['scisummnet_path']}...")
            
            # Get path to top1000_complete directory
            top1000_dir = Path(self.config['data']['scisummnet_path']) / 'top1000_complete'
            if not top1000_dir.exists():
                self.logger.warning(f"ScisummNet top1000_complete directory not found at {top1000_dir}")
                return None

            data = []
            # Each subdirectory is a paper ID (e.g., W05-0904)
            for paper_dir in top1000_dir.iterdir():
                if not paper_dir.is_dir():
                    continue
                    
                paper_id = paper_dir.name
                xml_path = paper_dir / 'Documents_xml' / f'{paper_id}.xml'
                summary_path = paper_dir / 'summary' / f'{paper_id}.gold.txt'  # Changed from .txt to .gold.txt
                
                if not xml_path.exists() or not summary_path.exists():
                    self.logger.warning(f"Missing files for paper {paper_id}")
                    continue
                    
                try:
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = f.read().strip()
                        
                    # Process XML and add to data
                    data.append({
                        'paper_id': paper_id,
                        'summary': summary,
                        'xml_path': str(xml_path),
                        'summary_path': str(summary_path)
                    })
                        
                except Exception as e:
                    self.logger.warning(f"Error processing paper {paper_id}: {e}")
                    continue

            if not data:
                self.logger.warning("No valid documents found in ScisummNet dataset")
                return None
            
            df = pd.DataFrame(data)
            self.logger.info(f"Successfully loaded {len(df)} documents from ScisummNet")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading ScisummNet dataset: {e}")
            return None