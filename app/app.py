import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from pathlib import Path
import sys
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.api.arxiv_api import ArxivAPI
from src.embedding_generator import EnhancedEmbeddingGenerator
from src.clustering.dynamic_cluster_manager import DynamicClusterManager
from src.summarization.hybrid_summarizer import EnhancedHybridSummarizer
from src.visualization.embedding_visualizer import EmbeddingVisualizer

# Initialize components
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

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    # Header
    html.H1("Research Paper Analysis Dashboard"),
    
    # Search Section
    html.Div([
        html.H3("Search Papers"),
        dcc.Input(
            id='search-input',
            type='text',
            placeholder='Enter search query...',
            style={'width': '50%'}
        ),
        html.Button('Search', id='search-button', n_clicks=0),
        dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id="loading-output")
        )
    ]),
    
    # Results Section
    html.Div([
        # Paper List
        html.Div([
            html.H3("Retrieved Papers"),
            html.Div(id='paper-list')
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        # Visualization
        html.Div([
            html.H3("Paper Embeddings"),
            dcc.Graph(id='embedding-plot')
        ], style={'width': '70%', 'display': 'inline-block'}),
    ]),
    
    # Cluster Summaries
    html.Div([
        html.H3("Cluster Summaries"),
        html.Div(id='cluster-summaries')
    ])
])

@app.callback(
    [Output('paper-list', 'children'),
     Output('embedding-plot', 'figure'),
     Output('cluster-summaries', 'children'),
     Output('loading-output', 'children')],
    [Input('search-button', 'n_clicks')],
    [State('search-input', 'value')]
)
def update_output(n_clicks, value):
    if not n_clicks or not value:
        return [], {}, [], ""
        
    try:
        # Fetch papers
        papers = api.fetch_papers_batch(value, max_papers=100)
        
        # Generate embeddings
        texts = [p['summary'] for p in papers]
        embeddings = embedding_generator.generate_embeddings(texts)
        
        # Perform clustering
        labels, metrics = cluster_manager.fit_predict(embeddings)
        
        # Generate cluster summaries
        cluster_texts = {}
        for i, label in enumerate(labels):
            if label not in cluster_texts:
                cluster_texts[label] = []
            cluster_texts[label].append({
                'processed_text': texts[i],
                'embedding': embeddings[i]
            })
            
        summaries = summarizer.summarize_all_clusters(cluster_texts)
        
        # Create visualization
        viz_results = visualizer.visualize_embeddings(
            embeddings,
            labels=labels,
            save_path=None
        )
        
        # Create paper list
        paper_list = html.Div([
            html.Div([
                html.H4(paper['title']),
                html.P(f"Authors: {', '.join(paper['authors'])}"),
                html.P(f"Published: {paper['published'].strftime('%Y-%m-%d')}")
            ]) for paper in papers[:10]  # Show first 10 papers
        ])
        
        # Create summary cards
        summary_cards = html.Div([
            html.Div([
                html.H4(f"Cluster {cluster_id}"),
                html.P(summary['summary'])
            ]) for cluster_id, summary in summaries.items()
        ])
        
        return paper_list, viz_results['figure'], summary_cards, ""
        
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return [], {}, [], f"Error: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True) 