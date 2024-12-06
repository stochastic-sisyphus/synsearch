import pytest
import pandas as pd
from src.data_validator import DataValidator

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'text': ['This is a test document.', 'Another test document here.'],
        'summary': ['Test summary.', 'Another summary.'],
        'id': [1, 2]
    })

def test_missing_values_check():
    validator = DataValidator()
    df = pd.DataFrame({
        'text': ['Text 1', 'Text 2', None],
        'summary': ['Sum 1', 'Sum 2', 'Sum 3']
    })
    assert not validator._check_missing_values(df)

def test_text_length_check():
    validator = DataValidator()
    df = pd.DataFrame({
        'text': ['Short text', 'This is a longer text that should pass the minimum length requirement']
    })
    assert not validator._check_text_length(df)  # Should fail due to short text

def test_language_check():
    validator = DataValidator()
    df = pd.DataFrame({
        'text': ['This is English text', 'More English text here']
    })
    assert validator._check_language(df) 