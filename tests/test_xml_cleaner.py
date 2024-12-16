import pytest
from src.preprocessing.xml_cleaner import XMLCleaner

@pytest.fixture
def cleaner():
    return XMLCleaner()

def test_clean_text_basic(cleaner):
    text = "<p>This is a test</p>"
    assert cleaner.clean_text(text) == "This is a test"

def test_clean_text_math(cleaner):
    text = "Test with math $x^2$ and more text"
    assert cleaner.clean_text(text) == "Test with math [MATH] and more text"

def test_clean_text_citations(cleaner):
    text = "As shown in [1] and [Smith et al.]"
    assert cleaner.clean_text(text) == "As shown in and"

def test_clean_title(cleaner):
    title = "arXiv: A Study of Machine Learning"
    assert cleaner.clean_title(title) == "A Study of Machine Learning"

def test_clean_abstract(cleaner):
    abstract = "Abstract: We present a study..."
    assert cleaner.clean_abstract(abstract) == "We present a study..." 