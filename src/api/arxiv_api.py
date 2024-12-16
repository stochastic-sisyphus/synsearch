import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
import logging
from datetime import datetime
import time
from urllib.parse import urlencode

class ArxivAPI:
    """Handler for arXiv API requests and responses."""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.rate_limit_delay = self.config.get('rate_limit_delay', 3)  # seconds between requests
        
    def search(
        self,
        query: str,
        start: int = 0,
        max_results: int = 10,
        sort_by: str = 'relevance',
        sort_order: str = 'descending'
    ) -> List[Dict]:
        """
        Search arXiv papers using the API.

        Args:
            query (str): Search query string
            start (int, optional): Starting index. Defaults to 0.
            max_results (int, optional): Maximum results to return. Defaults to 10.
            sort_by (str, optional): Sort method. Defaults to 'relevance'.
            sort_order (str, optional): Sort order. Defaults to 'descending'.

        Returns:
            List[Dict]: List of paper metadata dictionaries
        """
        try:
            # Construct query parameters
            params = {
                'search_query': query,
                'start': start,
                'max_results': max_results,
                'sortBy': sort_by,
                'sortOrder': sort_order
            }
            
            # Make request
            response = self._make_request(params)
            
            # Parse response
            papers = self._parse_response(response)
            
            self.logger.info(f"Retrieved {len(papers)} papers for query: {query}")
            return papers
            
        except Exception as e:
            self.logger.error(f"Error searching arXiv: {e}")
            return []
            
    def _make_request(self, params: Dict) -> str:
        """Make HTTP request to arXiv API with rate limiting."""
        url = f"{self.BASE_URL}?{urlencode(params)}"
        
        # Rate limiting
        time.sleep(self.rate_limit_delay)
        
        response = requests.get(url)
        response.raise_for_status()
        
        return response.text
        
    def _parse_response(self, response_text: str) -> List[Dict]:
        """Parse XML response from arXiv API."""
        root = ET.fromstring(response_text)
        
        # Define namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            try:
                paper = {
                    'title': entry.find('atom:title', ns).text.strip(),
                    'summary': entry.find('atom:summary', ns).text.strip(),
                    'authors': [author.find('atom:name', ns).text 
                              for author in entry.findall('atom:author', ns)],
                    'published': datetime.strptime(
                        entry.find('atom:published', ns).text,
                        '%Y-%m-%dT%H:%M:%SZ'
                    ),
                    'link': entry.find('atom:id', ns).text,
                    'categories': [cat.get('term') 
                                 for cat in entry.findall('atom:category', ns)]
                }
                papers.append(paper)
                
            except Exception as e:
                self.logger.warning(f"Error parsing paper entry: {e}")
                continue
                
        return papers

    def fetch_papers_batch(
        self,
        query: str,
        batch_size: int = 100,
        max_papers: int = 1000
    ) -> List[Dict]:
        """
        Fetch multiple batches of papers with rate limiting.

        Args:
            query (str): Search query
            batch_size (int, optional): Papers per batch. Defaults to 100.
            max_papers (int, optional): Maximum total papers. Defaults to 1000.

        Returns:
            List[Dict]: Combined list of paper metadata
        """
        all_papers = []
        start = 0
        
        while len(all_papers) < max_papers:
            batch = self.search(
                query=query,
                start=start,
                max_results=min(batch_size, max_papers - len(all_papers))
            )
            
            if not batch:  # No more results
                break
                
            all_papers.extend(batch)
            start += len(batch)
            
            self.logger.info(f"Fetched {len(all_papers)} papers so far")
            
        return all_papers[:max_papers] 