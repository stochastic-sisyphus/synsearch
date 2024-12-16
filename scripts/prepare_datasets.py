import sys
from pathlib import Path
import yaml
import logging
from datasets import load_dataset
from src.data_loader import DataLoader

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        # Create data directories
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        # Download XL-Sum
        logger.info("Downloading XL-Sum dataset...")
        xlsum = load_dataset('GEM/xlsum', 'english')
        
        # Download ScisummNet
        logger.info("Loading ScisummNet dataset...")
        data_loader = DataLoader(config)
        scisummnet = data_loader.load_scisummnet()
        
        logger.info("Datasets prepared successfully!")
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 