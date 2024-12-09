from setuptools import setup, find_packages

setup(
    name="synsearch",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'transformers',
        'scikit-learn',
        'hdbscan',
        'pyyaml',
        'nltk',
        'rouge_score'
    ]
) 