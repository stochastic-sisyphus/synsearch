from pathlib import Path
import pandas as pd
from datasets import load_dataset
import logging
from typing import Dict, Any
import json
import xml.etree.ElementTree as ET

class DataLoader:
    def __init__(self, scisummnet_path: str):
        self.scisummnet_path = Path(scisummnet_path)
        self.logger = logging.getLogger(__name__)
        
        # Verify ScisummNet path exists
        if not self.scisummnet_path.exists():
            raise FileNotFoundError(f"ScisummNet path not found: {scisummnet_path}")
    
    def load_xlsum(self, language='english'):
        """Load XL-Sum dataset for a specific language"""
        try:
            self.logger.info(f"Loading XL-Sum dataset for language: {language}")
            dataset = load_dataset('GEM/xlsum', language)
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load XL-Sum dataset: {e}")
            raise
    
    def load_scisummnet(self, path: str) -> pd.DataFrame:
        """Load ScisummNet dataset from the given path"""
        try:
            self.logger.info(f"Loading ScisummNet from {path}")
            papers = []
            top1000_path = Path(path) / 'top1000_complete'
            
            for doc_dir in top1000_path.iterdir():
                if not doc_dir.is_dir():
                    continue
                    
                try:
                    doc_id = doc_dir.name
                    
                    # Load XML document
                    xml_dir = doc_dir / 'Documents_xml'
                    if not xml_dir.exists():
                        continue
                    
                    # Find main document XML
                    xml_files = list(xml_dir.glob('*.xml'))
                    if not xml_files:
                        continue
                    
                    # Load citation data
                    citation_path = doc_dir / 'citing_sentences_annotated.json'
                    citations = []
                    if citation_path.exists():
                        with open(citation_path, 'r') as f:
                            citations = json.load(f)
                    
                    # Load summary
                    summary_path = doc_dir / 'summary' / 'summary.txt'
                    summary = ''
                    if summary_path.exists():
                        with open(summary_path, 'r') as f:
                            summary = f.read().strip()
                    
                    # Parse XML
                    tree = ET.parse(xml_files[0])
                    root = tree.getroot()
                    
                    # Extract data with better error handling
                    title = root.find('.//title')
                    abstract = root.find('.//abstract')
                    
                    paper_data = {
                        'doc_id': doc_id,
                        'title': title.text.strip() if title is not None and title.text else '',
                        'text': abstract.text.strip() if abstract is not None and abstract.text else '',
                        'summary': summary,
                        'citation_count': len(citations),
                        'citations': citations
                    }
                    papers.append(paper_data)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {doc_dir}: {str(e)}")
                    continue
            
            df = pd.DataFrame(papers)
            self.logger.info(f"Loaded {len(df)} documents from ScisummNet")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load ScisummNet: {e}")
            return pd.DataFrame()