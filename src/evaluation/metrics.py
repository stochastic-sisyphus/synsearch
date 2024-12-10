from typing import List, Dict, Union, Optional
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from rouge_score import rouge_scorer
import bert_score
import logging
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import re

class EvaluationMetrics:
    def __init__(self):
        """Initialize the evaluation metrics calculator"""
        self.logger = logging.getLogger(__name__)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def calculate_clustering_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clustering quality metrics"""
        try:
            # Filter out noise points (label -1) if any
            mask = labels != -1
            if not np.any(mask):
                return {
                    'silhouette_score': 0.0,
                    'davies_bouldin_score': float('inf')
                }
                
            valid_embeddings = embeddings[mask]
            valid_labels = labels[mask]
            
            # Calculate metrics
            silhouette = silhouette_score(valid_embeddings, valid_labels)
            davies_bouldin = davies_bouldin_score(valid_embeddings, valid_labels)
            
            return {
                'silhouette_score': float(silhouette),
                'davies_bouldin_score': float(davies_bouldin)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating clustering metrics: {e}")
            raise
            
    def calculate_rouge_scores(
        self,
        summaries: List[str],
        references: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate ROUGE scores for summaries"""
        try:
            scores = {
                'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
                'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
                'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
            }
            
            for summary, reference in zip(summaries, references):
                score = self.rouge_scorer.score(reference, summary)
                
                for metric, values in score.items():
                    scores[metric]['precision'].append(values.precision)
                    scores[metric]['recall'].append(values.recall)
                    scores[metric]['fmeasure'].append(values.fmeasure)
            
            # Calculate averages
            averaged_scores = {}
            for metric, values in scores.items():
                averaged_scores[metric] = {
                    k: float(np.mean(v)) for k, v in values.items()
                }
                
            return averaged_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating ROUGE scores: {e}")
            raise
            
    def save_metrics(
        self,
        metrics: Dict,
        output_dir: Union[str, Path],
        prefix: str = ''
    ) -> None:
        """Save metrics to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_metrics_{timestamp}.json" if prefix else f"metrics_{timestamp}.json"
        
        with open(output_dir / filename, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        self.logger.info(f"Saved metrics to {output_dir / filename}") 
        
    def calculate_baseline_metrics(self, dataset_name: str, metrics: Dict) -> Dict[str, float]:
        """Calculate and store baseline metrics for a dataset"""
        baseline_metrics = {
            'dataset': dataset_name,
            'runtime': metrics.get('runtime', 0),
            'rouge_scores': metrics.get('rouge_scores', {}),
            'clustering_scores': {
                'silhouette': metrics.get('silhouette_score', 0),
                'davies_bouldin': metrics.get('davies_bouldin_score', 0)
            },
            'preprocessing_time': metrics.get('preprocessing_time', 0)
        }
        return baseline_metrics 

    def calculate_comprehensive_metrics(
        self, 
        summaries: Dict[str, Dict],
        references: Dict[str, Dict[str, str]], 
        embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive evaluation metrics."""
        try:
            metrics = {
                'summarization': {
                    'rouge': self.calculate_rouge_scores(summaries, references),
                    'bert_score': self.calculate_bert_scores(summaries, references),
                    'style': self._calculate_style_metrics(summaries)
                }
            }
            
            if embeddings is not None:
                metrics['embedding'] = {
                    'quality': self._calculate_embedding_quality(embeddings),
                    'stability': self._calculate_embedding_stability(embeddings)
                }
                
            metrics['runtime'] = self._calculate_runtime_metrics()
            metrics['timestamp'] = datetime.now().isoformat()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            raise

    def _calculate_embedding_quality(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Calculate embedding quality metrics."""
        try:
            # Calculate cosine similarities
            similarities = cosine_similarity(embeddings)
            
            return {
                'mean_similarity': float(np.mean(similarities)),
                'std_similarity': float(np.std(similarities)),
                'min_similarity': float(np.min(similarities)),
                'max_similarity': float(np.max(similarities))
            }
        except Exception as e:
            self.logger.error(f"Error calculating embedding quality: {e}")
            raise

    def calculate_bert_scores(
        self,
        summaries: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate BERTScore for summaries."""
        try:
            P, R, F1 = bert_score.score(summaries, references, lang='en', verbose=False)
            return {
                'precision': float(P.mean()),
                'recall': float(R.mean()),
                'f1': float(F1.mean())
            }
        except Exception as e:
            self.logger.error(f"Error calculating BERTScore: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def _calculate_style_metrics(
        self, 
        summaries: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Calculate metrics specific to different summary styles."""
        style_metrics = {
            'technical_accuracy': 0.0,
            'conciseness_ratio': 0.0,
            'detail_coverage': 0.0
        }
        
        # Implementation of style-specific metrics
        # This would vary based on the style of each summary
        
        return style_metrics

    def calculate_dataset_metrics(summaries, references):
        """Calculate dataset-specific metrics"""
        metrics = {
            'xlsum': calculate_xlsum_metrics(summaries, references),
            'scisummnet': calculate_scientific_metrics(summaries, references)
        }
        return metrics

def calculate_xlsum_metrics(summaries: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate XL-Sum specific metrics"""
    metrics = {
        'average_compression_ratio': np.mean([
            len(summary.split()) / len(reference.split())
            for summary, reference in zip(summaries, references)
        ]),
        'coverage': _calculate_coverage_score(summaries, references),
        'factual_consistency': _calculate_factual_consistency(summaries, references)
    }
    return metrics

def calculate_scientific_metrics(summaries: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate scientific text specific metrics"""
    metrics = {
        'technical_term_preservation': _calculate_term_preservation(summaries, references),
        'citation_accuracy': _calculate_citation_accuracy(summaries, references),
        'methods_coverage': _calculate_methods_coverage(summaries, references),
        'results_accuracy': _calculate_results_accuracy(summaries, references)
    }
    return metrics

def _calculate_coverage_score(summaries: List[str], references: List[str]) -> float:
    """Calculate content coverage score using token overlap"""
    total_coverage = 0.0
    for summary, reference in zip(summaries, references):
        summary_tokens = set(summary.lower().split())
        reference_tokens = set(reference.lower().split())
        coverage = len(summary_tokens.intersection(reference_tokens)) / len(reference_tokens)
        total_coverage += coverage
    return total_coverage / len(summaries)

def _calculate_factual_consistency(summaries: List[str], references: List[str]) -> float:
    """Calculate factual consistency using named entity overlap"""
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        
        total_consistency = 0.0
        for summary, reference in zip(summaries, references):
            summary_doc = nlp(summary)
            reference_doc = nlp(reference)
            
            summary_entities = {ent.text.lower() for ent in summary_doc.ents}
            reference_entities = {ent.text.lower() for ent in reference_doc.ents}
            
            if reference_entities:
                consistency = len(summary_entities.intersection(reference_entities)) / len(reference_entities)
                total_consistency += consistency
                
        return total_consistency / len(summaries)
    except Exception:
        return 0.0

def _calculate_term_preservation(summaries: List[str], references: List[str]) -> float:
    """Calculate technical term preservation ratio"""
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        
        total_preservation = 0.0
        for summary, reference in zip(summaries, references):
            summary_doc = nlp(summary)
            reference_doc = nlp(reference)
            
            # Get technical terms (nouns and noun phrases)
            summary_terms = {chunk.text.lower() for chunk in summary_doc.noun_chunks}
            reference_terms = {chunk.text.lower() for chunk in reference_doc.noun_chunks}
            
            if reference_terms:
                preservation = len(summary_terms.intersection(reference_terms)) / len(reference_terms)
                total_preservation += preservation
                
        return total_preservation / len(summaries)
    except Exception:
        return 0.0

def _calculate_citation_accuracy(summaries: List[str], references: List[str]) -> float:
    """Calculate accuracy of citation preservation"""
    citation_pattern = r'\[\d+\]|\(\w+\s+et\s+al\.\s*,\s*\d{4}\)'
    
    total_accuracy = 0.0
    for summary, reference in zip(summaries, references):
        ref_citations = set(re.findall(citation_pattern, reference))
        sum_citations = set(re.findall(citation_pattern, summary))
        
        if ref_citations:
            accuracy = len(sum_citations.intersection(ref_citations)) / len(ref_citations)
            total_accuracy += accuracy
            
    return total_accuracy / len(summaries)

def _calculate_methods_coverage(summaries: List[str], references: List[str]) -> float:
    """Calculate coverage of methodology-related content"""
    methods_keywords = {
        'method', 'approach', 'technique', 'algorithm', 'procedure',
        'methodology', 'implementation', 'process', 'analysis', 'experiment'
    }
    
    total_coverage = 0.0
    for summary, reference in zip(summaries, references):
        summary_words = set(summary.lower().split())
        reference_words = set(reference.lower().split())
        
        summary_methods = summary_words.intersection(methods_keywords)
        reference_methods = reference_words.intersection(methods_keywords)
        
        if reference_methods:
            coverage = len(summary_methods) / len(reference_methods)
            total_coverage += coverage
            
    return total_coverage / len(summaries)

def _calculate_results_accuracy(summaries: List[str], references: List[str]) -> float:
    """Calculate accuracy of reported results and findings"""
    # Match numerical values and percentages
    number_pattern = r'\d+(?:\.\d+)?%?'
    
    total_accuracy = 0.0
    for summary, reference in zip(summaries, references):
        ref_numbers = set(re.findall(number_pattern, reference))
        sum_numbers = set(re.findall(number_pattern, summary))
        
        if ref_numbers:
            accuracy = len(sum_numbers.intersection(ref_numbers)) / len(ref_numbers)
            total_accuracy += accuracy
            
    return total_accuracy / len(summaries)