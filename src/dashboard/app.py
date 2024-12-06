import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import json
import plotly.graph_objects as go
from src.clustering.cluster_manager import ClusterManager
from src.embedding_generator import EmbeddingGenerator

# Initialize app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    # Header
    html.H1("Research Synthesis Dashboard", className="header"),
    
    # Controls
    html.Div([
        html.Label("Clustering Method:"),
        dcc.Dropdown(
            id='clustering-method',
            options=[
                {'label': 'HDBSCAN', 'value': 'hdbscan'},
                {'label': 'K-Means', 'value': 'kmeans'}
            ],
            value='hdbscan'
        )
    ], className="controls"),
    
    # Clustering parameters
    html.Div([
        html.Label("Min Cluster Size:"),
        dcc.Slider(
            id='min-cluster-size',
            min=2,
            max=20,
            step=1,
            value=5,
            marks={i: str(i) for i in range(2, 21, 2)}
        ),
        html.Label("Min Samples:"),
        dcc.Slider(
            id='min-samples',
            min=1,
            max=10,
            step=1,
            value=3,
            marks={i: str(i) for i in range(1, 11)}
        )
    ], className="parameter-controls"),
    
    # Metrics display
    html.Div([
        html.H3("Clustering Metrics"),
        html.Div(id='metrics-display')
    ], className="metrics-panel"),
    
    # Cluster details
    html.Div([
        html.H3("Cluster Details"),
        html.Div(id='cluster-details')
    ], className="cluster-details"),
    
    # Graph
    dcc.Graph(id='clustering-graph')
])

# Callback
@callback(
    [Output('clustering-graph', 'figure'),
     Output('metrics-display', 'children'),
     Output('cluster-details', 'children')],
    [Input('clustering-method', 'value'),
     Input('min-cluster-size', 'value'),
     Input('min-samples', 'value')]
)
def update_clustering(method, min_cluster_size, min_samples):
    # Initialize clustering
    cluster_manager = ClusterManager(
        method=method,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    
    # Get embeddings and perform clustering
    embeddings = load_embeddings()  # Implement this function
    labels = cluster_manager.fit_predict(embeddings)
    metrics = cluster_manager.evaluate_clusters(embeddings, labels)
    
    # Create visualization
    fig = create_cluster_visualization(embeddings, labels)
    
    # Create metrics display
    metrics_display = create_metrics_display(metrics)
    
    # Create cluster details
    cluster_details = create_cluster_details(labels, metrics)
    
    return fig, metrics_display, cluster_details

# Run app
if __name__ == '__main__':
    app.run_server(debug=True) 