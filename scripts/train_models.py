import sys
from pathlib import Path
import yaml
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.model_trainer import SummarizationModelTrainer

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        # Initialize trainer
        trainer = SummarizationModelTrainer(config)
        
        # Train on ScisummNet
        if config['training']['datasets']['scisummnet']['enabled']:
            logger.info("Fine-tuning on ScisummNet dataset...")
            trainer.train(dataset_name='scisummnet')
            
        # Train on XL-Sum
        if config['training']['datasets']['xlsum']['enabled']:
            logger.info("Fine-tuning on XL-Sum dataset...")
            trainer.train(dataset_name='xlsum')
            
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 