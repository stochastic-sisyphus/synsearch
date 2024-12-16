import pytest
from pathlib import Path
import torch
from src.training.model_trainer import SummarizationModelTrainer

@pytest.fixture
def config():
    return {
        'training': {
            'base_model': 'facebook/bart-large-cnn',
            'output_dir': 'tests/test_models',
            'epochs': 1,
            'batch_size': 2,
            'max_input_length': 512,
            'max_output_length': 128,
            'min_output_length': 30,
            'learning_rate': 2e-5,
            'datasets': {
                'scisummnet': {'enabled': True, 'validation_split': 0.1},
                'xlsum': {'enabled': True, 'validation_split': 0.1}
            }
        }
    }

@pytest.fixture
def trainer(config):
    return SummarizationModelTrainer(config)

def test_model_initialization(trainer):
    assert trainer.model is not None
    assert trainer.tokenizer is not None
    assert trainer.device in ['cuda', 'cpu']

def test_data_preparation(trainer):
    # Test XL-Sum dataset preparation
    xlsum_dataset = trainer.prepare_xlsum_dataset()
    assert isinstance(xlsum_dataset, torch.utils.data.Dataset)
    
    # Test ScisummNet dataset preparation
    scisummnet_dataset = trainer.prepare_scientific_dataset()
    assert isinstance(scisummnet_dataset, torch.utils.data.Dataset)

def test_model_comparison(trainer):
    text = "This is a test document for summarization."
    summaries = trainer.compare_models(text)
    
    assert 'pretrained' in summaries
    assert 'scisummnet' in summaries
    assert 'xlsum' in summaries
    
    for summary in summaries.values():
        assert isinstance(summary, str)
        assert len(summary) > 0 