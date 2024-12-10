from typing import Dict, List, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from collections import Counter
import logging
from multiprocessing import Pool, cpu_count

class ClusterExplainer:
    """Explains cluster characteristics and key features."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ClusterExplainer with configuration settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.nlp = spacy.load('en_core_web_sm')
        self.vectorizer = TfidfVectorizer(
            max_features=config.get('max_features', 1000),
            stop_words='english'
        )
        
    def explain_clusters(
        self,
        texts: List[str],
        labels: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate explanations for each cluster.

        Args:
            texts (List[str]): List of texts.
            labels (np.ndarray): Array of cluster labels.

        Returns:
            Dict[str, Dict[str, Any]]: Explanations for each cluster.
        """
        try:
            explanations = {}
            unique_labels = np.unique(labels)
            
            # Calculate TF-IDF for all texts
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            feature_names = self.vectorizer.get_feature_names_out()
            
            with Pool(processes=cpu_count()) as pool:
                results = pool.starmap(
                    self._process_cluster,
                    [(label, texts, labels, tfidf_matrix, feature_names) for label in unique_labels if label != -1]
                )
            
            for label, explanation in results:
                explanations[str(label)] = explanation
                
            return explanations
            
        except Exception as e:
            self.logger.error(f"Error generating explanations: {e}")
            raise
            
    def _process_cluster(
        self,
        label: int,
        texts: List[str],
        labels: np.ndarray,
        tfidf_matrix: np.ndarray,
        feature_names: np.ndarray
    ) -> (int, Dict[str, Any]):
        """
        Process a single cluster to generate explanations.

        Args:
            label (int): Cluster label.
            texts (List[str]): List of texts.
            labels (np.ndarray): Array of cluster labels.
            tfidf_matrix (np.ndarray): TF-IDF matrix.
            feature_names (np.ndarray): Feature names from TF-IDF vectorizer.

        Returns:
            (int, Dict[str, Any]): Cluster label and its explanation.
        """
        cluster_texts = [text for text, l in zip(texts, labels) if l == label]
        cluster_indices = np.where(labels == label)[0]
        
        explanation = {
            'size': len(cluster_texts),
            'key_terms': self._get_key_terms(
                tfidf_matrix[cluster_indices],
                feature_names
            ),
            'entities': self._extract_entities(cluster_texts),
            'summary_stats': self._calculate_summary_stats(cluster_texts)
        }
        
        return label, explanation
            
    def _get_key_terms(
        self,
        cluster_tfidf: np.ndarray,
        feature_names: np.ndarray,
        top_n: int = 5
    ) -> List[Dict[str, float]]:
        """
        Extract key terms using TF-IDF scores.

        Args:
            cluster_tfidf (np.ndarray): TF-IDF matrix for the cluster.
            feature_names (np.ndarray): Feature names from TF-IDF vectorizer.
            top_n (int, optional): Number of top terms to extract. Defaults to 5.

        Returns:
            List[Dict[str, float]]: List of key terms and their scores.
        """
        avg_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).ravel()
        top_indices = avg_tfidf.argsort()[-top_n:][::-1]
        
        return [
            {'term': feature_names[i], 'score': float(avg_tfidf[i])}
            for i in top_indices
        ]
        
    def _extract_entities(self, texts: List[str]) -> Dict[str, List[str]]:
        """
        Extract named entities from cluster texts.

        Args:
            texts (List[str]): List of texts in the cluster.

        Returns:
            Dict[str, List[str]]: Most frequent named entities in the cluster.
        """
        entities = {'ORG': [], 'PERSON': [], 'GPE': [], 'TOPIC': []}
        
        for text in texts:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
                    
        # Get most frequent entities
        return {
            label: [item for item, _ in Counter(items).most_common(3)]
            for label, items in entities.items() if items
        }
        
    def _calculate_summary_stats(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculate summary statistics for cluster texts.

        Args:
            texts (List[str]): List of texts in the cluster.

        Returns:
            Dict[str, float]: Summary statistics for the cluster texts.
        """
        lengths = [len(text.split()) for text in texts]
        return {
            'avg_length': float(np.mean(lengths)),
            'std_length': float(np.std(lengths)),
            'min_length': float(min(lengths)),
            'max_length': float(max(lengths))
        }
