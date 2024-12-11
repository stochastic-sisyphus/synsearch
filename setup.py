from setuptools import setup, find_packages

setup(
    name="synsearch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "torch>=2.0.0",
        "transformers>=4.15.0",
        "scikit-learn>=0.24.0",
        "sentence-transformers>=2.2.0",
        "hdbscan>=0.8.29",
        "plotly>=5.3.0",
        "streamlit>=1.2.0",
        "pytest>=6.0.0",
        "dash>=2.0.0",
        "dash-bootstrap-components>=1.0.0",
        "dash-core-components>=2.0.0",
        "dash-html-components>=2.0.0",
        "dash-table>=5.0.0",
        "jupyter-dash>=0.4.0",
        "spacy>=3.5.0",  # Added spacy
        "cachetools>=5.0.0",
        "joblib>=1.1.0"
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Dynamic Summarization and Adaptive Clustering Framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)