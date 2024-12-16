import re
from bs4 import BeautifulSoup
from typing import Optional

class XMLCleaner:
    """Clean and normalize XML-derived text from arXiv papers."""
    
    def __init__(self):
        self.math_pattern = re.compile(r'\$.*?\$|\\\[.*?\\\]')
        self.citation_pattern = re.compile(r'\[\d+\]|\[[\w\s,]+\]')
        
    def clean_text(self, text: str) -> str:
        """Clean text from XML/HTML tags and normalize content."""
        if not text:
            return ""
            
        # Remove XML/HTML tags
        soup = BeautifulSoup(text, 'lxml')
        text = soup.get_text()
        
        # Remove math expressions
        text = self.math_pattern.sub('[MATH]', text)
        
        # Remove citations
        text = self.citation_pattern.sub('', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
        
    def clean_title(self, title: str) -> str:
        """Clean paper title."""
        title = self.clean_text(title)
        # Remove common paper-specific prefixes
        title = re.sub(r'^(arxiv:|paper:)\s*', '', title, flags=re.IGNORECASE)
        return title
        
    def clean_abstract(self, abstract: str) -> str:
        """Clean paper abstract."""
        abstract = self.clean_text(abstract)
        # Remove common abstract prefixes
        abstract = re.sub(r'^(abstract:)\s*', '', abstract, flags=re.IGNORECASE)
        return abstract 