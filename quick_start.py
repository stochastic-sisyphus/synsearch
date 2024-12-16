from src.api.arxiv_api import ArxivAPI
from src.embedding_generator import EnhancedEmbeddingGenerator
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.hybrid_summarizer import EnhancedHybridSummarizer

# Initialize
api = ArxivAPI()
embedding_generator = EnhancedEmbeddingGenerator()
cluster_manager = DynamicClusterManager({'min_cluster_size': 5})
summarizer = EnhancedHybridSummarizer()

# Process papers
papers = api.fetch_papers_batch("machine learning", max_papers=100)
texts = [p['summary'] for p in papers]
embeddings = embedding_generator.generate_embeddings(texts)
labels, metrics = cluster_manager.fit_predict(embeddings)
summaries = summarizer.summarize_all_clusters(texts, labels)

# View results
print(f"Found {len(papers)} papers in {len(set(labels))} clusters")
