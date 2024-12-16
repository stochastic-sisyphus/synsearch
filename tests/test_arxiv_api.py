import pytest
from src.api.arxiv_api import ArxivAPI
from datetime import datetime

@pytest.fixture
def api():
    return ArxivAPI()

def test_search_basic(api):
    """Test basic search functionality"""
    results = api.search("machine learning", max_results=5)
    assert len(results) <= 5
    assert all(isinstance(paper, dict) for paper in results)
    
    # Check required fields
    required_fields = ['title', 'summary', 'authors', 'published', 'link']
    for paper in results:
        assert all(field in paper for field in required_fields)
        assert isinstance(paper['published'], datetime)
        assert isinstance(paper['authors'], list)

def test_search_empty_query(api):
    """Test handling of empty query"""
    results = api.search("")
    assert results == []

def test_batch_fetch(api):
    """Test batch fetching functionality"""
    results = api.fetch_papers_batch("deep learning", batch_size=10, max_papers=25)
    assert len(results) <= 25
    assert all(isinstance(paper, dict) for paper in results)

def test_rate_limiting(api):
    """Test rate limiting behavior"""
    import time
    start_time = time.time()
    api.search("test", max_results=2)
    api.search("test", max_results=2)
    elapsed = time.time() - start_time
    assert elapsed >= api.rate_limit_delay

def test_error_handling(api):
    """Test error handling with invalid parameters"""
    results = api.search("machine learning", max_results=-1)
    assert results == [] 