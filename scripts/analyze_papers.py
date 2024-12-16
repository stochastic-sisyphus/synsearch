import logging
from pathlib import Path
import yaml
from tabulate import tabulate
from datetime import datetime

from src.api.arxiv_api import ArxivAPI
from src.embedding_generator import EnhancedEmbeddingGenerator
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.hybrid_summarizer import EnhancedHybridSummarizer
from src.visualization.embedding_visualizer import EmbeddingVisualizer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    
    try:
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize components
        logger.info("Initializing components...")
        api = ArxivAPI()
        embedding_generator = EnhancedEmbeddingGenerator()
        cluster_manager = DynamicClusterManager({
            'min_cluster_size': 5,
            'min_samples': 2
        })
        summarizer = EnhancedHybridSummarizer({
            'model_name': 'facebook/bart-large-cnn',
            'max_length': 150,
            'min_length': 50
        })
        visualizer = EmbeddingVisualizer()
        
        # Fetch papers
        query = "machine learning"  # Change this to your topic
        max_papers = 50  # Adjust as needed
        
        logger.info(f"Fetching papers for query: {query}")
        papers = api.fetch_papers_batch(query, max_papers=max_papers)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [p['summary'] for p in papers]
        embeddings = embedding_generator.generate_embeddings(texts)
        
        # Perform clustering
        logger.info("Clustering papers...")
        labels, metrics = cluster_manager.fit_predict(embeddings)
        
        # Generate summaries for each cluster
        logger.info("Generating cluster summaries...")
        cluster_texts = {}
        for i, label in enumerate(labels):
            if label not in cluster_texts:
                cluster_texts[label] = []
            cluster_texts[label].append({
                'text': texts[i],
                'paper': papers[i]
            })
        
        summaries = summarizer.summarize_all_clusters(cluster_texts)
        
        # Create visualization
        logger.info("Creating visualization...")
        viz_results = visualizer.visualize_embeddings(
            embeddings,
            labels=labels,
            save_path="outputs/visualizations/clusters.html"
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f'outputs/analysis_{timestamp}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cluster information
        cluster_info = []
        for label in set(labels):
            if label == -1:  # Skip noise
                continue
            cluster_papers = cluster_texts[label]
            cluster_info.append({
                'cluster_id': label,
                'size': len(cluster_papers),
                'summary': summaries.get(label, ''),
                'papers': [p['paper']['title'] for p in cluster_papers]
            })
        
        # Display results
        print("\nAnalysis Results:")
        print(f"Total papers analyzed: {len(papers)}")
        print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
        
        print("\nCluster Summaries:")
        for cluster in cluster_info:
            print(f"\nCluster {cluster['cluster_id']} ({cluster['size']} papers):")
            print("Summary:", cluster['summary'])
            print("Papers:")
            for title in cluster['papers'][:3]:  # Show first 3 papers
                print(f"- {title}")
            if len(cluster['papers']) > 3:
                print(f"  ... and {len(cluster['papers'])-3} more")
        
        # Save detailed results
        with open(output_dir / 'analysis_results.txt', 'w') as f:
            f.write(f"Analysis Results for query: {query}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            for cluster in cluster_info:
                f.write(f"\nCluster {cluster['cluster_id']} ({cluster['size']} papers):\n")
                f.write(f"Summary: {cluster['summary']}\n")
                f.write("Papers:\n")
                for title in cluster['papers']:
                    f.write(f"- {title}\n")
        
        logger.info(f"Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 