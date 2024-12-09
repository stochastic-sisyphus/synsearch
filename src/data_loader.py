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
                            'summary': dataset['train']['summary'],
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
                    scisummnet_path = self.config['data']['scisummnet_path']
                    if scisummnet_path:
                        df = self.load_scisummnet(scisummnet_path)
                        if df is not None and not df.empty:
                            datasets['scisummnet'] = df
                            self.logger.info(f"Successfully loaded {len(df)} documents from ScisummNet")
                        else:
                            self.logger.warning("No valid documents found in ScisummNet dataset")
                    else:
                        self.logger.warning("ScisummNet path not configured")
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
            top1000_dir = Path(path) / 'top1000_complete'
            
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