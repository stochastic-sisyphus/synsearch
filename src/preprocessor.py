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
from src.utils.performance import PerformanceOptimizer  # Add this

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
    """Enhanced preprocessor for handling various text domains."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            self.logger.warning("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')
            
    def preprocess_text(
        self, 
        text: str, 
        domain: str = 'general'
    ) -> str:
        """Preprocess single text with domain-specific settings."""
        try:
            # Basic cleaning
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)
            
            # Domain-specific processing
            if domain == 'scientific':
                text = self._process_scientific(text)
            elif domain == 'legal':
                text = self._process_legal(text)
                
            # Extract key information using spaCy
            doc = self.nlp(text)
            
            # Keep only relevant tokens
            tokens = [
                token.text for token in doc
                if not token.is_stop and not token.is_punct
                and len(token.text.strip()) > 1
            ]
            
            return ' '.join(tokens)
            
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            raise

    def preprocess_texts(
        self,
        texts: List[str],
        domain: str = 'general',
        batch_size: int = 32
    ) -> List[str]:
        """Preprocess multiple texts in parallel."""
        try:
            with ThreadPoolExecutor() as executor:
                processed_texts = list(
                    tqdm(
                        executor.map(
                            lambda x: self.preprocess_text(x, domain),
                            texts
                        ),
                        total=len(texts),
                        desc="Preprocessing texts"
                    )
                )
            return processed_texts
            
        except Exception as e:
            self.logger.error(f"Error in batch preprocessing: {e}")
            raise

    def _process_scientific(self, text: str) -> str:
        """Process scientific text with special handling for technical terms."""
        # Replace numerical expressions with placeholders
        text = re.sub(r'\d+\.\d+', '[NUM]', text)
        text = re.sub(r'\d+%', '[PERCENT]', text)
        
        # Preserve equations and formulas
        text = re.sub(r'\$.*?\$', '[EQUATION]', text)
        
        return text

    def _process_legal(self, text: str) -> str:
        """Process legal text with special handling for citations and references."""
        # Replace legal citations
        text = re.sub(r'(\d+\s+U\.S\.C\.\s+§\s+\d+)', '[LEGAL_REF]', text)
        
        # Replace section numbers
        text = re.sub(r'Section\s+\d+', '[SECTION]', text)
        
        return text

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        return entities

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
