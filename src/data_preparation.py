from datasets import load_dataset
import pandas as pd
import nltk
import spacy
from pathlib import Path

class DataPreparator:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        # Load spaCy model
        self.nlp = spacy.load('en_core_web_sm')
        
    def load_xlsum(self):
        """Load XL-Sum dataset from HuggingFace"""
        return load_dataset('GEM/xlsum')
    
    def load_scisummnet(self, path):
        """Load ScisummNet dataset from local path"""
        scisummnet_path = Path(path)
        data = []
        
        # Walk through the directory structure
        for paper_dir in scisummnet_path.glob('top1000_complete/*'):
            if not paper_dir.is_dir():
                continue
            
            try:
                # Load abstract
                abstract_path = paper_dir / 'Documents_xml' / 'abstract.txt'
                if abstract_path.exists():
                    with open(abstract_path, 'r', encoding='utf-8') as f:
                        abstract = f.read().strip()
                    
                # Load summary
                summary_path = paper_dir / 'summary' / 'summary.txt'
                if summary_path.exists():
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = f.read().strip()
                    
                # Add to dataset
                if abstract and summary:
                    data.append({
                        'paper_id': paper_dir.name,
                        'text': abstract,
                        'summary': summary
                    })
                    
            except Exception as e:
                print(f"Error processing {paper_dir.name}: {e}")
                
        return pd.DataFrame(data)
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Remove special characters
        text = ' '.join(text.split())
        # Tokenize
        doc = self.nlp(text)
        # Basic cleaning
        tokens = [
            token.text.lower() 
            for token in doc 
            if not token.is_stop and not token.is_punct
        ]
        return ' '.join(tokens)
    
    def process_dataset(self, dataset, save_path):
        """Process and save dataset"""
        processed_data = []
        
        # For each document
        for doc in dataset:
            processed_doc = {
                'id': doc.get('id', ''),
                'text': self.preprocess_text(doc['text']),
                'summary': self.preprocess_text(doc['summary']),
                'metadata': {
                    'source': doc.get('source', ''),
                    'length': len(doc['text'].split())
                }
            }
            processed_data.append(processed_doc)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(processed_data)
        
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        
        return df

def main():
    data_prep = DataPreparator()
    
    # Process XL-Sum
    xlsum = data_prep.load_xlsum()
    data_prep.process_dataset(
        xlsum, 
        'data/processed/xlsum_processed.csv'
    )
    
    # Process ScisummNet
    scisummnet = data_prep.load_scisummnet(
        '/Users/vanessa/Dropbox/synsearch/data/scisummnet_release1.1__20190413'
    )
    data_prep.process_dataset(
        scisummnet,
        'data/processed/scisummnet_processed.csv'
    )

if __name__ == "__main__":
    main()
