from src.api.arxiv_api import ArxivAPI
from src.embedding_generator import EnhancedEmbeddingGenerator
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.hybrid_summarizer import EnhancedHybridSummarizer

def analyze_topic(query: str, max_papers: int = 50):
    # Initialize components
    api = ArxivAPI()
    embedding_generator = EnhancedEmbeddingGenerator()
    cluster_manager = DynamicClusterManager({'min_cluster_size': 5})
    summarizer = EnhancedHybridSummarizer()

    # Fetch papers
    print(f"\nFetching papers for: {query}")
    papers = api.fetch_papers_batch(query, max_papers=max_papers)
    
    if not papers:
        print("No papers found for the given query.")
        return
    
    # Process papers
    print("Generating embeddings...")
    # Ensure texts are not empty and have valid length
    texts = [p['summary'] for p in papers if p.get('summary') and len(p['summary'].strip()) > 10]
    
    if not texts:
        print("No valid texts found for embedding generation.")
        return
        
    try:
        embeddings = embedding_generator.generate_embeddings(texts)
        
        print("Clustering papers...")
        labels, metrics = cluster_manager.fit_predict(embeddings)
        
        # Print results
        print(f"\nFound {len(papers)} papers in {len(set(labels))} clusters")
        
        # Print papers by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(papers[i])
        
        for label, cluster_papers in clusters.items():
            print(f"\nCluster {label} ({len(cluster_papers)} papers):")
            for paper in cluster_papers[:3]:  # Show first 3 papers
                print(f"- {paper['title']}")
                print(f"  Authors: {', '.join(paper['authors'])}")
                print(f"  Published: {paper['published']}")
            if len(cluster_papers) > 3:
                print(f"  ... and {len(cluster_papers)-3} more papers")
                
    except Exception as e:
        print(f"Error processing papers: {str(e)}")

def get_user_input():
    print("\nArXiv Paper Analysis")
    print("=" * 50)
    
    # Get query
    while True:
        query = input("\nEnter your search query (e.g., 'quantum computing', 'machine learning'): ").strip()
        if query:
            break
        print("Please enter a valid query.")
    
    # Get number of papers
    while True:
        try:
            max_papers = input("\nHow many papers to analyze? (default: 50): ").strip()
            if not max_papers:
                max_papers = 50
                break
            max_papers = int(max_papers)
            if max_papers > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    return query, max_papers

if __name__ == "__main__":
    try:
        # Get user input
        query, max_papers = get_user_input()
        
        # Run analysis
        analyze_topic(query, max_papers)
        
        # Ask if user wants to search again
        while input("\nWould you like to search again? (y/n): ").lower().strip() == 'y':
            query, max_papers = get_user_input()
            analyze_topic(query, max_papers)
            
        print("\nThank you for using the paper analysis tool!")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
