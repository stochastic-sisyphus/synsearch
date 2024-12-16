import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.app import app

if __name__ == '__main__':
    # Load configuration
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get dashboard config
    dashboard_config = config.get('dashboard', {})
    
    # Run server
    app.run_server(
        host=dashboard_config.get('host', '0.0.0.0'),
        port=dashboard_config.get('port', 8050),
        debug=dashboard_config.get('debug', True)
    ) 