from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class CheckpointManager:
    def __init__(self, checkpoint_dir: str = 'outputs/checkpoints', enable_metrics: bool = True):
        self.logger = logging.getLogger(__name__)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / 'pipeline_state.json'
        self.state = self._load_state()
        self.enable_metrics = enable_metrics
        
    def _load_state(self) -> Dict:
        """Load existing pipeline state or create new one"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            'last_completed_stage': None,
            'stages': {},
            'timestamp': None
        }
        
    def save_stage(self, stage_name: str, data: Dict[str, Any]) -> None:
        """Save checkpoint for a pipeline stage"""
        self.state['last_completed_stage'] = stage_name
        self.state['stages'][stage_name] = data
        self.state['timestamp'] = datetime.now().isoformat()
        
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
            
        self.logger.info(f"Saved checkpoint for stage: {stage_name}")
        
    def get_last_stage(self) -> Optional[str]:
        """Get the last completed pipeline stage"""
        return self.state['last_completed_stage']
        
    def get_stage_data(self, stage_name: str) -> Optional[Dict]:
        """Get data for a specific pipeline stage"""
        return self.state['stages'].get(stage_name) 
        
    def is_stage_complete(self, stage_name: str) -> bool:
        """Check if a pipeline stage is complete"""
        return stage_name in self.state['stages']
        
    def get_stage_metrics(self, stage_name: str) -> Optional[Dict]:
        """Get performance metrics for a stage"""
        stage_data = self.get_stage_data(stage_name)
        return stage_data.get('metrics') if stage_data else None 