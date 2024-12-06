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
        ('plotly', 'Visualization')
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