import dash
from dash import html, dcc
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class DashboardApp:
    def __init__(self, config_path="config/config.yaml"):
        """Initialize dashboard with configuration."""
        self.app = dash.Dash(__name__)
        self.load_config(config_path)
        self.setup_layout()
        self.setup_callbacks()
        
    def load_config(self, config_path):
        """Load dashboard configuration."""
        import yaml
        with open(config_path) as f:
            self.config = yaml.safe_load(f).get('dashboard', {})
            
    def setup_layout(self):
        """Create the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.H1("Research Synthesis Dashboard", className="header"),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.H3("Dataset Selection"),
                    dcc.Dropdown(
                        id='dataset-selector',
                        options=[
                            {'label': 'XL-Sum', 'value': 'xlsum'},
                            {'label': 'ScisummNet', 'value': 'scisummnet'}
                        ],
                        value='xlsum'
                    ),
                ], className="control-panel"),
                
                html.Div([
                    html.H3("Visualization Settings"),
                    dcc.Checklist(
                        id='viz-options',
                        options=[
                            {'label': 'Show Cluster Labels', 'value': 'show_labels'},
                            {'label': 'Show Centroids', 'value': 'show_centroids'}
                        ],
                        value=['show_labels']
                    )
                ], className="control-panel")
            ], className="controls-container"),
            
            # Main Content
            html.Div([
                # Visualization Panel
                html.Div([
                    html.H2("Cluster Visualization"),
                    dcc.Graph(id='cluster-plot')
                ], className="viz-panel"),
                
                # Metrics Panel
                html.Div([
                    html.H2("Metrics"),
                    html.Div(id='metrics-display')
                ], className="metrics-panel")
            ], className="main-content"),
            
            # Summary Panel
            html.Div([
                html.H2("Cluster Summaries"),
                html.Div(id='summaries-display')
            ], className="summary-panel")
        ])
        
    def setup_callbacks(self):
        """Set up interactive callbacks."""
        @self.app.callback(
            [Output('cluster-plot', 'figure'),
             Output('metrics-display', 'children'),
             Output('summaries-display', 'children')],
            [Input('dataset-selector', 'value'),
             Input('viz-options', 'value')]
        )
        def update_dashboard(dataset, viz_options):
            try:
                # Load results for selected dataset
                results = self.load_results(dataset)
                if not results:
                    return self.create_empty_plot(), "No data available", "No summaries available"
                
                # Create visualization
                fig = self.create_cluster_plot(results, viz_options)
                
                # Format metrics display
                metrics_html = self.format_metrics(results.get('metrics', {}))
                
                # Format summaries display
                summaries_html = self.format_summaries(results.get('summaries', {}))
                
                return fig, metrics_html, summaries_html
                
            except Exception as e:
                print(f"Error updating dashboard: {e}")
                return self.create_empty_plot(), "Error loading metrics", "Error loading summaries"
    
    def load_results(self, dataset):
        """Load results from the specified dataset."""
        results_path = Path(self.config.get('data_dir', 'data/output')) / f"{dataset}_results.json"
        if not results_path.exists():
            return None
        with open(results_path) as f:
            return json.load(f)
            
    def create_cluster_plot(self, results, viz_options):
        """Create interactive cluster visualization."""
        return px.scatter(
            x=results.get('embeddings_2d', [])[:, 0],
            y=results.get('embeddings_2d', [])[:, 1],
            color=results.get('labels', []),
            title="Document Clusters"
        )
    
    def create_empty_plot(self):
        """Create empty plot for error states."""
        return go.Figure()
    
    def format_metrics(self, metrics):
        """Format metrics for display."""
        return html.Div([
            html.H3("Clustering Metrics"),
            html.P(f"Silhouette Score: {metrics.get('silhouette_score', 'N/A')}"),
            html.P(f"Davies-Bouldin Index: {metrics.get('davies_bouldin_score', 'N/A')}"),
            html.H3("ROUGE Scores"),
            html.P(f"ROUGE-L F1: {metrics.get('rouge_scores', {}).get('rougeL', {}).get('fmeasure', 'N/A')}")
        ])
    
    def format_summaries(self, summaries):
        """Format summaries for display."""
        return html.Div([
            html.Div([
                html.H4(f"Cluster {cluster_id}"),
                html.P(summary.get('summary', ''))
            ]) for cluster_id, summary in summaries.items()
        ])
        
    def run_server(self, debug=True, port=None):
        """Run the dashboard server."""
        port = port or self.config.get('port', 8050)
        self.app.run_server(debug=debug, port=port)

if __name__ == "__main__":
    dashboard = DashboardApp()
    dashboard.run_server(debug=True)
