
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os
import tempfile
from src.main import EnhancedPipelineManager, PipelineConfig

@pytest.fixture
def test_data():
    """Create test dataset with edge cases"""
    return pd.DataFrame({
        'text': ['Normal text', '', 'Special chars: @#$', None, 'A' * 1000],
        'summary': ['Summary 1', None, 'Summary 3', 'Summary 4', 'Summary 5']
    })

@pytest.fixture
def pipeline_config():
    """Create test configuration"""
    return PipelineConfig(
        batch_size=2,
        debug_mode=True
    )

def test_empty_dataset():
    """Test handling of empty dataset"""
    pipeline = EnhancedPipelineManager(pipeline_config())
    empty_df = pd.DataFrame(columns=['text', 'summary'])
    
    with pytest.raises(ValueError, match="Empty dataset"):
        pipeline.process_with_checkpoints(empty_df)

def test_missing_columns():
    """Test handling of missing required columns"""
    pipeline = EnhancedPipelineManager(pipeline_config())
    invalid_df = pd.DataFrame({'wrong_column': ['text']})
    
    with pytest.raises(ValueError, match="Missing required columns"):
        pipeline.process_with_checkpoints(invalid_df)

def test_checkpoint_recovery():
    """Test checkpoint saving and recovery"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(
            batch_size=2,
            cache_dir=tmpdir
        )
        pipeline = EnhancedPipelineManager(config)
        
        # Process data
        df = pd.DataFrame({
            'text': ['Text 1', 'Text 2', 'Text 3'],
            'summary': ['Sum 1', 'Sum 2', 'Sum 3']
        })
        
        # Simulate interrupt after first batch
        try:
            pipeline.process_with_checkpoints(df)
        except KeyboardInterrupt:
            pass
            
        # Check checkpoint exists
        assert list(Path(tmpdir).glob('checkpoint_*.json'))
        
        # Test recovery
        pipeline2 = EnhancedPipelineManager(config)
        results = pipeline2.process_with_checkpoints(df)
        assert results is not None