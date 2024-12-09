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
    
    def process_dataset(self, 
                       data: Union[pd.DataFrame, Dict], 
                       text_column: str,
                       summary_column: Optional[str] = None,
                       batch_size: int = 1000) -> pd.DataFrame:
        """Process entire dataset with batching support."""
        self.logger.info("Starting dataset processing...")
        
        # Convert to DataFrame if necessary
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        # Process in batches
        total_rows = len(df)
        processed_texts = []
        processed_summaries = []
        
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Process text column
            batch_texts = batch[text_column].apply(self.preprocess_text)
            processed_texts.extend(batch_texts)
            
            # Process summary column if provided
            if summary_column and summary_column in df.columns:
                batch_summaries = batch[summary_column].apply(
                    lambda x: self.preprocess_text(x, remove_citations=False)
                )
                processed_summaries.extend(batch_summaries)
            
            self.logger.info(f"Processed {min(i+batch_size, total_rows)}/{total_rows} documents")
        
        # Update DataFrame
        df['processed_text'] = processed_texts
        if summary_column and summary_column in df.columns:
            df['processed_summary'] = processed_summaries
            
        # Add metadata
        df['token_count'] = df['processed_text'].apply(lambda x: len(x.split()))
        df['has_summary'] = df['processed_summary'].notna() if 'processed_summary' in df.columns else False
        
        self.logger.info("Dataset processing complete")
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
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')  # or any other model
        
    def preprocess(self, text, domain=None):
        """Domain-agnostic preprocessing"""
        # Basic cleaning
        text = self._clean_text(text)
        
        # Universal tokenization
        tokens = self.tokenizer.encode(text).tokens
        
        # Handle different input formats
        if domain == 'scientific':
            return self._handle_scientific(tokens)
        elif domain == 'news':
            return self._handle_news(tokens)
        else:
            return self._handle_generic(tokens)

    def process_dataset(self, dataset, text_column='text', summary_column='summary'):
        """Process a dataset with text and summary columns.
        
        Args:
            dataset: pandas DataFrame containing the dataset
            text_column: name of the column containing the text to process
            summary_column: name of the column containing the summary (if available)
        
        Returns:
            pandas DataFrame with processed text and summaries
        """
        # Create a copy to avoid modifying the original
        processed_df = dataset.copy()
        
        # Process the main text
        processed_df['processed_text'] = processed_df[text_column].apply(self.preprocess_text)
        
        # Process the summary if it exists
        if summary_column in processed_df.columns:
            processed_df['processed_summary'] = processed_df[summary_column].apply(self.preprocess_text)
        
        return processed_df

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