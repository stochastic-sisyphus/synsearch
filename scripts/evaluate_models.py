import sys
from pathlib import Path
import yaml
import logging
from tabulate import tabulate
import json
from datetime import datetime

from src.training.model_trainer import SummarizationModelTrainer
from src.evaluation.metrics import EvaluationMetrics

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        # Initialize components
        trainer = SummarizationModelTrainer(config)
        metrics = EvaluationMetrics()
        
        # Load test data
        test_texts = [
            """Recent advances in machine learning have revolutionized the field...""",
            """The study of quantum mechanics reveals fundamental principles..."""
        ]
        
        # Evaluate each model
        results = []
        for model_type in ['pretrained', 'scisummnet', 'xlsum']:
            summaries = []
            for text in test_texts:
                summary = trainer._generate_summary(
                    getattr(trainer, f"{model_type}_model"), 
                    text
                )
                summaries.append(summary)
            
            # Calculate metrics
            scores = metrics.calculate_comprehensive_metrics(
                summaries=summaries,
                references=test_texts
            )
            
            results.append([
                model_type,
                scores['rouge1']['f1'],
                scores['rouge2']['f1'],
                scores['rougeL']['f1'],
                scores['bert_score']['f1']
            ])
        
        # Display results
        headers = ['Model', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']
        print("\nModel Evaluation Results:")
        print(tabulate(results, headers=headers, tablefmt='grid'))
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('outputs/evaluations')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / f'evaluation_{timestamp}.json', 'w') as f:
            json.dump({
                'results': results,
                'headers': headers,
                'timestamp': timestamp
            }, f, indent=2)
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 