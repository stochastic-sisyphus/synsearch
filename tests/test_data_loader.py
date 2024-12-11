import pytest
from pathlib import Path
from src.data_loader import DataLoader
import pandas as pd

@pytest.fixture
def data_loader():
    # Use the actual path from your config
    scisummnet_path = "/Users/vanessa/Dropbox/synsearch/data/scisummnet_release1.1__20190413"
    return DataLoader(scisummnet_path)

def test_init_data_loader(data_loader):
    """Test DataLoader initialization"""
    assert data_loader is not None
    assert isinstance(data_loader.scisummnet_path, Path)

def test_load_xlsum(data_loader):
    """Test loading XL-Sum dataset"""
    dataset = data_loader.load_xlsum('english')
    assert dataset is not None
    assert 'train' in dataset
    assert len(dataset['train']) > 0

def test_load_scisummnet(data_loader):
    """Test loading ScisummNet dataset"""
    df = data_loader.load_scisummnet(str(data_loader.scisummnet_path))
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    
    # Print debug information if the DataFrame is empty
    if len(df) == 0:
        print(f"\nDebug info:")
        print(f"ScisummNet path: {data_loader.scisummnet_path}")
        print(f"Path exists: {data_loader.scisummnet_path.exists()}")
        print(f"Is directory: {data_loader.scisummnet_path.is_dir()}")
        if data_loader.scisummnet_path.exists():
            print("Directory contents:")
            for item in data_loader.scisummnet_path.iterdir():
                print(f"  - {item}")
    
    assert len(df) > 0, "DataFrame is empty - no documents were loaded"

def test_scisummnet_content(data_loader):
    """Test content of loaded ScisummNet data"""
    df = data_loader.load_scisummnet(str(data_loader.scisummnet_path))
    assert df is not None
    
    # Check for non-empty content
    assert not df['text'].empty
    assert not df['title'].empty
    
    # Check data types
    assert df['doc_id'].dtype == object
    assert df['title'].dtype == object
    assert df['text'].dtype == object
    
    # Check for no completely empty rows
    assert not df['text'].isna().all()
    assert not df['title'].isna().all()

def test_load_all_datasets(data_loader):
    """Test loading all datasets"""
    datasets = data_loader.load_all_datasets()
    assert datasets is not None
    assert 'xlsum' in datasets
    assert 'scisummnet' in datasets
    assert isinstance(datasets['xlsum'], pd.DataFrame)
    assert isinstance(datasets['scisummnet'], pd.DataFrame)
    assert len(datasets['xlsum']) > 0
    assert len(datasets['scisummnet']) > 0
