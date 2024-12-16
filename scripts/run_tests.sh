#!/bin/bash

# Run all tests
pytest tests/test_arxiv_api.py -v
pytest tests/test_pipeline.py -v
pytest tests/test_xml_cleaner.py -v 