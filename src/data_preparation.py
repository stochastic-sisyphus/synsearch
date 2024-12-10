from datasets import load_dataset
import pandas as pd
import nltk
import spacy
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    """Custom Dataset for text data."""
    
    def __init__(self, texts: list):
        self.texts = texts
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

class DataPreparator:
    def __init__(self):
        """Initialize the DataPreparator with required resources."""
        nltk.download('punkt')
        nltk.download('stopwords')
        self.nlp = spacy.load('en_core_web_sm')
        
    def load_xlsum(self) -> dict:
        """Load XL-Sum dataset from HuggingFace."""
        return load_dataset('GEM/xlsum')
    
    def load_scisummnet(self, path: str) -> pd.DataFrame:
        """Load ScisummNet dataset from local path."""
        scisummnet_path = Path(path)
        data = []
        
        for paper_dir in scisummnet_path.glob('top1000_complete/*'):
            if not paper_dir.is_dir():
                continue
            
            try:
                abstract_path = paper_dir / 'Documents_xml' / 'abstract.txt'
                if abstract_path.exists():
                    with open(abstract_path, 'r', encoding='utf-8') as f:
                        abstract = f.read().strip()
                    
                summary_path = paper_dir / 'summary' / 'summary.txt'
                if summary_path.exists():
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = f.read().strip()
                    
                if abstract and summary:
                    data.append({
                        'paper_id': paper_dir.name,
                        'text': abstract,
                        'summary': summary
                    })
                    
            except Exception as e:
                print(f"Error processing {paper_dir.name}: {e}")
                
        return pd.DataFrame(data)
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        text = ' '.join(text.split())
        doc = self.nlp(text)
        tokens = [
            token.text.lower() 
            for token in doc 
            if not token.is_stop and not token.is_punct
        ]
        return ' '.join(tokens)
    
    def process_dataset(self, dataset: list, save_path: str, batch_size: int = 32) -> pd.DataFrame:
        """Process and save dataset using batch processing."""
        processed_data = []
        text_dataset = TextDataset([doc['text'] for doc in dataset])
        dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False)
        
        for batch in dataloader:
            for text in batch:
                processed_doc = {
                    'text': self.preprocess_text(text),
                    'metadata': {
                        'length': len(text.split())
                    }
                }
                processed_data.append(processed_doc)
        
        df = pd.DataFrame(processed_data)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        
        return df

def main():
    data_prep = DataPreparator()
    
    xlsum = data_prep.load_xlsum()
    data_prep.process_dataset(
        xlsum, 
        'data/processed/xlsum_processed.csv'
    )
    
    scisummnet = data_prep.load_scisummnet(
        '/Users/vanessa/Dropbox/synsearch/data/scisummnet_release1.1__20190413'
    )
    data_prep.process_dataset(
        scisummnet,
        'data/processed/scisummnet_processed.csv'
    )

if __name__ == "__main__":
    main()
