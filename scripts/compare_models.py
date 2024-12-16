import sys
from pathlib import Path
import yaml
import logging
from tabulate import tabulate

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.model_trainer import SummarizationModelTrainer
from src.evaluation.metrics import EvaluationMetrics

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
            
        # Initialize trainer and metrics
        trainer = SummarizationModelTrainer(config)
        metrics = EvaluationMetrics()
        
        # Example text for comparison
        text = """
        Recent advances in machine learning have revolutionized natural language processing...
        """
        
        # Generate summaries using different models
        summaries = trainer.compare_models(text)
        
        # Calculate metrics
        results = []
        for model_name, summary in summaries.items():
            scores = metrics.calculate_comprehensive_metrics(
                summary=summary,
                reference=text
            )
            results.append([
                model_name,
                scores['rouge1']['f1'],
                scores['rouge2']['f1'],
                scores['rougeL']['f1']
            ])
            
        # Display results
        headers = ['Model', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        print("\nModel Comparison Results:")
        print(tabulate(results, headers=headers, tablefmt='grid'))
        
        # Save results
        output_dir = Path('outputs/comparisons')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'model_comparison.txt', 'w') as f:
            f.write("Original Text:\n\n")
            f.write(text)
            f.write("\n\nSummaries:\n\n")
            for model_name, summary in summaries.items():
                f.write(f"{model_name}:\n{summary}\n\n")
            f.write("\nMetrics:\n")
            f.write(tabulate(results, headers=headers, tablefmt='grid'))
            
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 