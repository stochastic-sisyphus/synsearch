import importlib
import sys
import logging
from pathlib import Path

def check_dependencies():
    required_packages = [
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical operations'),
        ('torch', 'Deep learning'),
        ('sentence_transformers', 'Text embeddings'),
        ('yaml', 'Configuration'),
        ('pytest', 'Testing'),
        ('umap', 'Dimensionality reduction'),
        ('plotly', 'Visualization'),
        ('networkx', 'Graph operations'),
        ('community', 'Community detection'),
        ('python-louvain', 'Alternative community detection'),
        ('dash', 'Dashboard framework'),
        ('dash_bootstrap_components', 'Dashboard components'),
        ('plotly', 'Interactive visualization'),
        ('streamlit', 'Alternative dashboard framework'),
        ('dash_core_components', 'Core dash components'),
        ('dash_html_components', 'HTML components for dash'),
        ('dash_table', 'Table components for dash'),
        ('jupyter_dash', 'Jupyter integration for dash'),
        ('cachetools', 'Caching utilities'),
        ('joblib', 'Performance optimization')
    ]
    
    missing = []
    for package, purpose in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package:<20} - {purpose}")
        except ImportError:
            missing.append(package)
            print(f"✗ {package:<20} - {purpose}")
    
    return missing

if __name__ == "__main__":
    print("\nChecking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print("\nMissing packages:")
        print("pip install " + " ".join(missing))
        sys.exit(1)
    else:
        print("\nAll dependencies satisfied!")