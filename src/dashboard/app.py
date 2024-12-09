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
        self.app.run_server(debug=debug, port=port)
