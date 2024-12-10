try:
    import dash
    from dash import html, dcc
    import plotly.express as px
    from dash.dependencies import Input, Output, State
    import numpy as np
    from umap import UMAP
    DASHBOARD_ENABLED = True
except ImportError:
    DASHBOARD_ENABLED = False
    
class DashboardApp:
    def __init__(self, embedding_generator, cluster_manager):
        if not DASHBOARD_ENABLED:
            raise ImportError("Dashboard dependencies not installed. Run: pip install dash plotly umap-learn")
            
        self.app = dash.Dash(__name__)
        self.embedding_generator = embedding_generator
        self.cluster_manager = cluster_manager
        
        self.app.layout = self._create_dashboard_layout()
        self._setup_callbacks()
        
    def _create_dashboard_layout(self):
        return html.Div([
            html.H1("Research Synthesis Dashboard"),
            
            # Input section
            html.Div([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                    multiple=False
                ),
                
                # Clustering parameters
                html.Div([
                    html.Label('Min Cluster Size'),
                    dcc.Slider(
                        id='min-cluster-size',
                        min=2, max=20, value=5, step=1,
                        marks={i: str(i) for i in range(2, 21, 2)}
                    )
                ]),
                
                # Visualization parameters
                html.Div([
                    html.Label('UMAP Neighbors'),
                    dcc.Slider(
                        id='umap-neighbors',
                        min=2, max=100, value=15, step=1
                    )
                ])
            ]),
            
            # Results section
            html.Div([
                # Cluster visualization
                dcc.Graph(id='cluster-plot'),
                
                # Metrics display
                html.Div(id='metrics-display'),
                
                # Cluster summaries
                html.Div(id='cluster-summaries')
            ])
        ])
        
    def _setup_callbacks(self):
        @self.app.callback(
            [Output('cluster-plot', 'figure'),
             Output('metrics-display', 'children')],
            [Input('upload-data', 'contents'),
             Input('min-cluster-size', 'value'),
             Input('umap-neighbors', 'value')]
        )
        def update_output(contents, min_cluster_size, n_neighbors):
            if contents is None:
                return dash.no_update
                
            # Process uploaded data
            data = self._parse_contents(contents)
            
            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(data)
            
            # Update clustering parameters
            self.cluster_manager.config['thresholds']['min_cluster_size'] = min_cluster_size
            
            # Perform clustering
            labels, metrics = self.cluster_manager.fit_predict(embeddings)
            
            # Generate UMAP visualization
            umap = UMAP(n_neighbors=n_neighbors)
            embedding_2d = umap.fit_transform(embeddings)
            
            # Create figure
            fig = px.scatter(
                x=embedding_2d[:, 0], y=embedding_2d[:, 1],
                color=labels,
                title='Cluster Visualization'
            )
            
            # Format metrics display
            metrics_html = html.Div([
                html.H3("Clustering Metrics"),
                html.P(f"Silhouette Score: {metrics['silhouette']:.3f}"),
                html.P(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.3f}"),
                html.P(f"Algorithm Selected: {metrics['algorithm']}")
            ])
            
            return fig, metrics_html
            
    def run_server(self, debug=True, port=8050):
        self.app.run_server(debug=debug, port=8050)

import streamlit as st
import plotly.express as px
from pathlib import Path
import json
from typing import Dict, Any
import pandas as pd
from src.visualization.embedding_visualizer import EmbeddingVisualizer
from src.utils.metrics_calculator import MetricsCalculator

class Dashboard:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.visualizer = EmbeddingVisualizer(config)
        self.metrics = MetricsCalculator()
        
    def run(self):
        st.title("Dynamic Summarization Dashboard")
        
        # Sidebar controls
        st.sidebar.header("Controls")
        dataset = st.sidebar.selectbox(
            "Select Dataset",
            ["xlsum", "scisummnet"]
        )
        
        # Load results
        results = self._load_results(dataset)
        if not results:
            st.error("No results found for selected dataset")
            return
            
        # Display metrics
        self._show_metrics(results)
        
        # Show visualizations
        self._show_visualizations(results)
        
        # Display summaries
        self._show_summaries(results)
    
    def _load_results(self, dataset: str) -> Dict:
        results_path = Path(self.config['data']['output_path']) / f"{dataset}_results.json"
        if not results_path.exists():
            return None
        with open(results_path) as f:
            return json.load(f)
    
    def _show_metrics(self, results: Dict):
        st.header("Evaluation Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Clustering Quality")
            metrics = results['metrics']['clustering']
            st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
            st.metric("Davies-Bouldin Index", f"{metrics['davies_bouldin_score']:.3f}")
            
        with col2:
            st.subheader("Summarization Quality")
            rouge = results['metrics']['summarization']['rouge']['rougeL']
            st.metric("ROUGE-L F1", f"{rouge['fmeasure']:.3f}")
            st.metric("ROUGE-L Precision", f"{rouge['precision']:.3f}")
            
    def _show_visualizations(self, results: Dict):
        st.header("Visualizations")
        
        # UMAP plot of embeddings
        fig = self.visualizer.plot_embeddings(
            results['embeddings'],
            results['clustering']['labels']
        )
        st.plotly_chart(fig)
        
    def _show_summaries(self, results: Dict):
        st.header("Generated Summaries")
        
        for cluster_id, summary in results['summaries'].items():
            with st.expander(f"Cluster {cluster_id}"):
                st.markdown(summary['summary'])
                st.caption(f"Style: {summary['style']}")

if __name__ == "__main__":
    import yaml
    
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
        
    dashboard = Dashboard(config)
    dashboard.run()
