import pandas as pd
import spacy
import nltk
from pathlib import Path
import logging
from typing import List, Dict, Union, Optional, Any
import re
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from src.utils.performance import PerformanceOptimizer

class TextDataset(Dataset):
    """Custom Dataset for text data."""
    
    def __init__(self, texts: list):
        self.texts = texts
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

class TextPreprocessor:
    def __init__(self, language: str = 'english'):
        """Initialize the preprocessor with specified language."""
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            self.logger.warning(f"Could not download NLTK data: {e}")
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except Exception as e:
            self.logger.error(f"Could not load spaCy model: {e}")
            raise
            
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.perf_optimizer = PerformanceOptimizer()
        
    def clean_xml(self, text: str) -> str:
        """Remove XML tags and clean the text."""
        try:
            # Remove XML tags
            text = BeautifulSoup(text, "xml").get_text()
            return text.strip()
        except Exception as e:
            self.logger.warning(f"XML cleaning failed: {e}")
            return text
    
    def preprocess_text(self, text: str, remove_citations: bool = True) -> str:
        """Clean and normalize text with advanced options."""
        try:
            if not isinstance(text, str) or not text.strip():
                return ""
                
            # Basic cleaning
            text = text.lower().strip()
            
            # Remove citations if requested
            if remove_citations:
                text = re.sub(r'\[\d+\]|\[[\w\s,]+\]', '', text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://\S+', '', text)
            
            # Remove special characters but keep sentence structure
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!]', ' ', text)
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # SpaCy processing for advanced NLP
            doc = self.nlp(text)
            
            # Remove stopwords, lemmatize, and filter tokens
            tokens = []
            for token in doc:
                if (not token.is_stop and 
                    not token.is_punct and 
                    not token.is_space and 
                    len(token.text) > 1):  # Filter single characters
                    tokens.append(token.lemma_)
            
            return " ".join(tokens)
        except Exception as e:
            self.logger.error(f"Error in preprocessing text: {e}")
            return ""
    
    def process_dataset(
        self, 
        data: Union[pd.DataFrame, Dict], 
        text_column: str,
        summary_column: Optional[str] = None,
        batch_size: int = 1000,
        n_jobs: int = -1
    ) -> pd.DataFrame:
        """Process dataset using multiprocessing."""
        self.logger.info("Starting dataset processing...")
        
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        if n_jobs == -1:
            n_jobs = self.perf_optimizer.get_optimal_workers()

        with Pool(processes=n_jobs) as pool:
            # Process text column
            processed_texts = list(tqdm(
                pool.imap(self.preprocess_text, df[text_column]),
                total=len(df),
                desc="Processing texts"
            ))
            
            # Process summaries if available
            if summary_column and summary_column in df.columns:
                process_summary = partial(self.preprocess_text, remove_citations=False)
                processed_summaries = list(tqdm(
                    pool.imap(process_summary, df[summary_column]),
                    total=len(df),
                    desc="Processing summaries"
                ))
                df['processed_summary'] = processed_summaries
        
        df['processed_text'] = processed_texts
        df['token_count'] = df['processed_text'].apply(len)
        
        return df

    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Get preprocessing statistics."""
        stats = {
            'total_documents': len(df),
            'avg_token_count': df['token_count'].mean(),
            'documents_with_summary': df['has_summary'].sum() if 'has_summary' in df.columns else 0,
            'empty_documents': (df['processed_text'] == '').sum()
        }
        return stats
    
    def clean_scientific_text(self, text: str) -> str:
        """Clean scientific text with special handling for citations and formulas"""
        # Remove citations
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\([A-Za-z\s]+,\s+\d{4}\)', '', text)
        
        # Handle mathematical expressions
        text = re.sub(r'\$.*?\$', '[MATH]', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '[URL]', text)
        
        # Clean whitespace
        text = ' '.join(text.split())
        return text
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from scientific text"""
        doc = self.nlp(text)
        
        return {
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'noun_phrases': [chunk.text for chunk in doc.noun_chunks],
            'key_terms': [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop]
        }
    
    def process_document(self, doc: Dict[str, str]) -> Dict[str, Any]:
        """Process a single document"""
        processed = {}
        
        # Clean text
        if 'text' in doc:
            processed['cleaned_text'] = self.clean_scientific_text(doc['text'])
            processed['metadata'] = self.extract_metadata(doc['text'])
        
        # Clean summary if available
        if 'summary' in doc:
            processed['cleaned_summary'] = self.clean_scientific_text(doc['summary'])
        
        return processed

class DomainAgnosticPreprocessor:
    def preprocess_texts(self, texts: List[str], batch_size: int = 32) -> List[str]:
        try:
            processed_texts = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_processed = []
                
                for text in batch:
                    if not isinstance(text, str):
                        self.logger.warning(f"Skipping non-string input: {type(text)}")
                        continue
                        
                    # Basic cleaning
                    cleaned = text.strip()
                    if not cleaned:
                        continue
                        
                    # Remove URLs and special characters
                    cleaned = re.sub(r'http\S+|www.\S+', '', cleaned)
                    cleaned = re.sub(r'[^\w\s.,!?-]', '', cleaned)
                    
                    # Normalize whitespace
                    cleaned = ' '.join(cleaned.split())
                    
                    batch_processed.append(cleaned)
                    
                processed_texts.extend(batch_processed)
                
            return processed_texts
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            raise

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

# Example usage
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Initialize
    loader = DataLoader("/path/to/scisummnet")
    preprocessor = TextPreprocessor()
    
    # Load and preprocess XL-Sum
    xlsum = loader.load_xlsum()
    if xlsum:
        # Convert to DataFrame for consistent processing
        train_df = pd.DataFrame(xlsum['train'])
        processed_train = preprocessor.process_dataset(train_df, 'text')
        print("\nXL-Sum Processing Complete:")
        print(f"Total documents: {len(processed_train)}")
        print("Sample processed text:")
        print(processed_train['processed_text'].iloc[0][:200])
    
    # Load and preprocess ScisummNet
    scisummnet = loader.load_scisummnet()
    if scisummnet is not None:
        processed_sci = preprocessor.process_dataset(scisummnet, 'text')
        print("\nScisummNet Processing Complete:")
        print(f"Total documents: {len(processed_sci)}")
        print("Sample processed text:")
        print(processed_sci['processed_text'].iloc[0][:200])
