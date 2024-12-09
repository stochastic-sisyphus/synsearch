import re
import spacy
from typing import List, Dict, Optional
import logging

class DomainAgnosticPreprocessor:
    def __init__(self):
        """Initialize the domain-agnostic preprocessor with spaCy model."""
        self.nlp = spacy.load("en_core_web_sm")
        self.logger = logging.getLogger(__name__)

    def preprocess_text(self, text: str, domain: str = 'general') -> str:
        """Process text with domain-specific configurations."""
        try:
            # Basic cleaning
            text = self._clean_general_text(text)
            
            # Apply domain-specific processing if needed
            if domain == 'scientific':
                text = self._process_scientific_text(text)
            elif domain == 'legal':
                text = self._process_legal_text(text)
            
            # Apply spaCy processing
            doc = self.nlp(text)
            
            # Remove stopwords and lemmatize
            processed_text = " ".join([token.lemma_ for token in doc 
                                    if not token.is_stop and not token.is_punct])
            
            return processed_text
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            raise

    def _clean_general_text(self, text: str) -> str:
        """Clean text using domain-agnostic rules."""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Standardize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def _process_scientific_text(self, text: str) -> str:
        """Process scientific text (e.g., papers, technical documents)."""
        # Handle common scientific notation
        text = re.sub(r'\d+\.\d+e[+-]\d+', '[NUM]', text)
        # Standardize section headers
        text = re.sub(r'\b(?:ABSTRACT|INTRODUCTION|METHODOLOGY|RESULTS|CONCLUSION)\b', '', text)
        return text

    def _process_legal_text(self, text: str) -> str:
        """Process legal text (e.g., contracts, regulations)."""
        # Remove legal references
        text = re.sub(r'\b\d+\s*U\.?S\.?C\.?\s*§*\s*\d+\b', '[LEGAL_REF]', text)
        # Standardize section markers
        text = re.sub(r'Section\s+\d+', '[SECTION]', text)
        return text
