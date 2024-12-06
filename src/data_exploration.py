import sys
sys.path.append('..')

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from data_preparation import DataPreparator
import logging
import json
import re
import os
import xml.etree.ElementTree as ET
from typing import Dict, Any

logger = logging.getLogger(__name__)

def analyze_xlsum():
    """Load and analyze XL-Sum dataset"""
    try:
        # Load English dataset
        dataset = load_dataset('GEM/xlsum', 'english')
        
        # Convert target to summary for consistency
        train_df = pd.DataFrame(dataset['train'])
        train_df = train_df.rename(columns={'target': 'summary'})
        
        logger.info("\n=== XL-Sum Dataset Analysis ===\n")
        logger.info("Dataset Statistics:")
        logger.info(f"Train set size: {len(dataset['train'])}")
        logger.info(f"Validation set size: {len(dataset['validation'])}")
        logger.info(f"Test set size: {len(dataset['test'])}")
        logger.info(f"\nFeatures: {list(dataset['train'].features.keys())}")
        
        return train_df
        
    except Exception as e:
        logger.error(f'Error loading XL-Sum dataset: "{str(e)}"')
        return pd.DataFrame()  # Return empty DataFrame instead of None

def process_document(doc_path):
    try:
        # Initialize variables
        abstract = ""
        article_text = ""
        summary = ""
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find the abstract section
        abstract_match = re.search(r'Abstract\n\n(.*?)\n\n', content, re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            
        # Find the article text (everything after Introduction until References/Acknowledgments)
        text_match = re.search(r'Introduction\n\n(.*?)(?:\n\nReferences|\n\nAcknowledgments)', content, re.DOTALL)
        if text_match:
            article_text = text_match.group(1).strip()
            
        # Find the summary (usually at the start of the file)
        summary_match = re.search(r'^(.*?)\n\nAbstract', content, re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip()
            
        return {
            'abstract': abstract,
            'text': article_text,
            'summary': summary
        }
        
    except Exception as e:
        print(f"Error processing {os.path.basename(doc_path)}: {str(e)}")
        return None

def analyze_scisummnet(data_path):
    """Analyze the ScisummNet dataset."""
    documents = []
    root_path = Path(data_path) / 'top1000_complete'
    
    logger.info("\n=== ScisummNet Dataset Analysis ===\n")
    
    total_papers = len(list(root_path.iterdir()))
    processed_papers = 0
    
    for paper_dir in root_path.iterdir():
        try:
            paper_data = {'paper_id': paper_dir.name}
            
            # Load citing sentences JSON
            json_path = paper_dir / 'citing_sentences_annotated.json'
            if json_path.exists():
                with open(json_path) as f:
                    citations = json.load(f)
                paper_data['total_citations'] = len(citations)
                paper_data['gold_citations'] = len([c for c in citations if c.get('keep_for_gold', False)])
            
            # Load summary
            summary_path = paper_dir / 'summary' / 'summary.txt'
            if summary_path.exists():
                with open(summary_path) as f:
                    paper_data['summary'] = f.read().strip()
            
            # Load XML and extract abstract
            xml_files = list((paper_dir / 'Documents_xml').glob('*.xml'))
            if xml_files:
                xml_path = xml_files[0]
                tree = ET.parse(xml_path)
                root = tree.getroot()
                abstract = root.find('.//abstract')
                if abstract is not None:
                    paper_data['abstract'] = ' '.join(abstract.itertext()).strip()
            
            if 'abstract' in paper_data or 'summary' in paper_data:
                documents.append(paper_data)
                processed_papers += 1
            
        except Exception as e:
            logger.debug(f"Error processing {paper_dir.name}: {e}")
    
    df = pd.DataFrame(documents)
    
    # Calculate statistics
    stats = {
        'total_papers': total_papers,
        'processed_papers': processed_papers,
        'success_rate': f"{(processed_papers/total_papers)*100:.1f}%"
    }
    
    # Add optional statistics if columns exist
    if 'abstract' in df.columns:
        stats['avg_abstract_length'] = df['abstract'].str.split().str.len().mean()
    if 'total_citations' in df.columns:
        stats['avg_citations'] = df['total_citations'].mean()
    if 'gold_citations' in df.columns:
        stats['avg_gold_citations'] = df['gold_citations'].mean()
    if 'summary' in df.columns:
        stats['avg_summary_length'] = df['summary'].str.split().str.len().mean()
    
    # Log statistics
    logger.info("\nDataset Statistics:")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    return df, stats

def plot_distributions(xlsum_data, scisummnet_data):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot length distributions
        if not xlsum_data.empty and 'text' in xlsum_data.columns:
            text_lengths = xlsum_data['text'].str.split().str.len()
            ax1.hist(text_lengths, bins=50, alpha=0.5, label='XL-Sum')
            logger.info(f"\nXL-Sum text lengths: mean={text_lengths.mean():.1f}, median={text_lengths.median():.1f}")
            
        if not scisummnet_data.empty and 'text' in scisummnet_data.columns:
            text_lengths = scisummnet_data['text'].str.split().str.len()
            ax1.hist(text_lengths, bins=50, alpha=0.5, label='ScisummNet')
            logger.info(f"ScisummNet text lengths: mean={text_lengths.mean():.1f}, median={text_lengths.median():.1f}")
        
        ax1.set_title('Document Length Distribution')
        ax1.set_xlabel('Number of Words')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Plot summary length distributions
        if not xlsum_data.empty and 'summary' in xlsum_data.columns:
            summary_lengths = xlsum_data['summary'].str.split().str.len()
            ax2.hist(summary_lengths, bins=50, alpha=0.5, label='XL-Sum')
            logger.info(f"\nXL-Sum summary lengths: mean={summary_lengths.mean():.1f}, median={summary_lengths.median():.1f}")
            
        if not scisummnet_data.empty and 'summary' in scisummnet_data.columns:
            summary_lengths = scisummnet_data['summary'].str.split().str.len()
            ax2.hist(summary_lengths, bins=50, alpha=0.5, label='ScisummNet')
            logger.info(f"ScisummNet summary lengths: mean={summary_lengths.mean():.1f}, median={summary_lengths.median():.1f}")
        
        ax2.set_title('Summary Length Distribution')
        ax2.set_xlabel('Number of Words')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = Path('outputs/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_dir / 'length_distributions.png')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting distributions: {str(e)}")

def explore_scisummnet(data_path: str):
    """Explore the structure of ScisummNet dataset"""
    root = Path(data_path)
    logger = logging.getLogger(__name__)
    
    # Look at the top1000_complete directory
    complete_dir = root / 'top1000_complete'
    
    # List first few directories/files
    logger.info("\nDirectory structure:")
    for item in list(complete_dir.iterdir())[:5]:
        logger.info(f"- {item.name}")
        if item.is_dir():
            # List contents of first few subdirectories
            for subitem in list(item.iterdir())[:3]:
                logger.info(f"  └── {subitem.name}")
    
    # Read documentation
    doc_path = root / 'Dataset_Documentation.txt'
    if doc_path.exists():
        logger.info("\nDataset Documentation:")
        with open(doc_path, 'r') as f:
            logger.info(f.read()[:500] + "...")  # First 500 chars

    # Try to read one example paper's data
    if list(complete_dir.iterdir()):
        example_dir = next(complete_dir.iterdir())
        logger.info(f"\nExample paper directory ({example_dir.name}):")
        for file in example_dir.iterdir():
            logger.info(f"- {file.name}")
            if file.suffix == '.json':
                with open(file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"\nJSON structure:")
                    logger.info(json.dumps(data, indent=2)[:500] + "...")

def analyze_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive statistics for the dataset"""
    stats = {
        'document_count': len(df),
        'text_length_stats': {
            'mean': df['text'].str.len().mean(),
            'std': df['text'].str.len().std(),
            'min': df['text'].str.len().min(),
            'max': df['text'].str.len().max()
        },
        'word_count_stats': {
            'mean': df['text'].str.split().str.len().mean(),
            'std': df['text'].str.split().str.len().std()
        },
        'missing_values': df.isnull().sum().to_dict(),
        'citation_stats': {
            'mean': df['citation_count'].mean() if 'citation_count' in df else None,
            'max': df['citation_count'].max() if 'citation_count' in df else None
        }
    }
    return stats

def plot_distributions(df: pd.DataFrame, save_path: Path = None):
    """Generate and save distribution plots"""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Text length distribution
    sns.histplot(data=df['text'].str.len(), ax=axes[0,0])
    axes[0,0].set_title('Text Length Distribution')
    
    # Word count distribution
    sns.histplot(data=df['text'].str.split().str.len(), ax=axes[0,1])
    axes[0,1].set_title('Word Count Distribution')
    
    # Citation count distribution (if available)
    if 'citation_count' in df:
        sns.histplot(data=df['citation_count'], ax=axes[1,0])
        axes[1,0].set_title('Citation Count Distribution')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    return fig

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Analyze datasets
    xlsum_data = analyze_xlsum()
    scisummnet_data, stats = analyze_scisummnet(config['data']['scisummnet_path'])
    
    # Generate plots
    plot_distributions(xlsum_data, scisummnet_data)
    
    logger.info("\nAnalysis complete. Plots saved to outputs/figures/")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    explore_scisummnet("/Users/vanessa/Dropbox/synsearch/data/scisummnet_release1.1__20190413")
    main() 