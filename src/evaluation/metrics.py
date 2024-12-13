from typing import List, Dict, Union, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import logging
import json
import re
import bert_score
from pathlib import Path
from datetime import datetime

class EmbeddingDataset(Dataset):
    """Custom Dataset for embeddings."""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]

class EvaluationMetrics:
    """Enhanced metrics calculation with better performance and error handling."""

    def __init__(self, device: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        # Initialize timing attributes
        self.start_time = None
        self.end_time = None
        self.num_samples = 0

        # Initialize BERTScore model with caching
        self._initialize_bert_score()

    def _initialize_bert_score(self):
        """Initialize and cache BERTScore model."""
        try:
            # Set up model parameters without cache_dir
            self.bert_scorer = bert_score.BERTScorer(
                model_type='roberta-large',
                num_layers=17,
                batch_size=32,
                nthreads=4,
                all_layers=False,
                lang='en',
                rescale_with_baseline=True,
                device=self.device,
                use_fast_tokenizer=True
            )
            
            # Warm up the model to ensure weights are loaded
            _ = self.bert_scorer.score(['test'], ['test'])
            
            self.logger.info("BERTScore model initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"BERTScore initialization failed: {e}. Falling back to default settings.")
            self.bert_scorer = None

    def calculate_clustering_metrics(
        self,
        embeddings: Union[np.ndarray, List],
        labels: Union[np.ndarray, List]
    ) -> Dict[str, float]:
        """Calculate clustering metrics with robust input handling."""
        try:
            # Convert inputs to numpy arrays if they're lists
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            if isinstance(labels, list):
                labels = np.array(labels)
                
            # Ensure proper shapes
            if embeddings.ndim != 2:
                raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")
            if labels.ndim != 1:
                raise ValueError(f"Labels must be 1D array, got shape {labels.shape}")
                
            # Remove noise points (label -1)
            mask = labels != -1
            if not np.any(mask):
                raise ValueError("No valid clusters found (all points are noise)")
                
            valid_embeddings = embeddings[mask]
            valid_labels = labels[mask]
            
            if len(set(valid_labels)) < 2:
                raise ValueError("Need at least 2 clusters for metrics calculation")
                
            metrics = {
                'silhouette_score': float(silhouette_score(valid_embeddings, valid_labels)),
                'davies_bouldin_score': float(davies_bouldin_score(valid_embeddings, valid_labels))
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating clustering metrics: {e}")
            return {
                'silhouette_score': 0.0,
                'davies_bouldin_score': float('inf')
            }

    def calculate_rouge_scores(
        self,
        summaries: Union[List[str], Dict[str, str]],
        references: Union[List[str], Dict[str, str]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate ROUGE scores with robust input handling."""
        try:
            self.logger.info("Starting ROUGE scores calculation")
            
            # Convert inputs to lists if they're dictionaries
            if isinstance(summaries, dict):
                summaries = list(summaries.values())
            if isinstance(references, dict):
                references = list(references.values())
                
            # Ensure inputs are strings
            summaries = [str(s) if s is not None else "" for s in summaries]
            references = [str(r) if r is not None else "" for r in references]
            
            # Validate lengths
            if len(summaries) != len(references):
                raise ValueError("Number of summaries and references must match")
                
            scores = {
                'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
                'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
                'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
            }

            for summary, reference in zip(summaries, references):
                try:
                    score = self.rouge_scorer.score(reference, summary)
                    for metric, values in score.items():
                        scores[metric]['precision'].append(values.precision)
                        scores[metric]['recall'].append(values.recall)
                        scores[metric]['fmeasure'].append(values.fmeasure)
                except Exception as e:
                    self.logger.warning(f"Error calculating ROUGE for single pair: {e}")
                    # Add zero scores for failed calculations
                    for metric in scores:
                        scores[metric]['precision'].append(0.0)
                        scores[metric]['recall'].append(0.0)
                        scores[metric]['fmeasure'].append(0.0)

            # Calculate averages
            averaged_scores = {}
            for metric, values in scores.items():
                averaged_scores[metric] = {
                    k: float(np.mean(v)) if v else 0.0 
                    for k, v in values.items()
                }

            return averaged_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating ROUGE scores: {e}")
            return {
                'rouge1': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rouge2': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rougeL': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
            }

    def calculate_bert_scores(
        self,
        summaries: Union[List[str], Dict[str, str]],
        references: Union[List[str], Dict[str, str]]
    ) -> Dict[str, float]:
        """Calculate BERTScore with robust input handling."""
        try:
            # Convert inputs to lists if they're dictionaries
            if isinstance(summaries, dict):
                summaries = list(summaries.values())
            if isinstance(references, dict):
                references = list(references.values())
                
            # Ensure inputs are strings
            summaries = [str(s) if s is not None else "" for s in summaries]
            references = [str(r) if r is not None else "" for r in references]
            
            if not summaries or not references:
                raise ValueError("Empty input for BERTScore calculation")
                
            # Initialize BERTScorer if not already done
            if not hasattr(self, 'bert_scorer'):
                self._initialize_bert_score()
                
            P, R, F1 = self.bert_scorer.score(summaries, references)
            
            return {
                'precision': float(P.mean()),
                'recall': float(R.mean()),
                'f1': float(F1.mean())
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating BERTScore: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def save_metrics(
        self,
        metrics: Dict,
        output_dir: Union[str, Path],
        prefix: str = ''
    ) -> None:
        """
        Save metrics to disk.

        Args:
            metrics (Dict): Dictionary of metrics to save.
            output_dir (Union[str, Path]): Directory to save the metrics.
            prefix (str, optional): Prefix for the filename. Defaults to ''.
        """
        try:
            self.logger.info("Starting metrics saving")
            self.logger.debug(f"Metrics: {metrics}, Output directory: {output_dir}, Prefix: {prefix}")

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{prefix}_metrics_{timestamp}.json" if prefix else f"metrics_{timestamp}.json"

            with open(output_dir / filename, 'w') as f:
                json.dump(metrics, f, indent=2)

            self.logger.info(f"Saved metrics to {output_dir / filename}")

        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")

    def calculate_baseline_metrics(self, dataset_name: str, metrics: Dict) -> Dict[str, float]:
        """
        Calculate and store baseline metrics for a dataset.

        Args:
            dataset_name (str): Name of the dataset.
            metrics (Dict): Dictionary of metrics.

        Returns:
            Dict[str, float]: Dictionary of baseline metrics.
        """
        try:
            self.logger.info("Starting baseline metrics calculation")
            self.logger.debug(f"Dataset name: {dataset_name}, Metrics: {metrics}")

            baseline_metrics = {
                'dataset': dataset_name,
                'runtime': metrics.get('runtime', {}),
                'rouge_scores': metrics.get('rouge_scores', {}),
                'clustering_scores': metrics.get('clustering', {}),
                'preprocessing_time': metrics.get('preprocessing_time', 0)
            }

            self.logger.info("Completed baseline metrics calculation")
            self.logger.debug(f"Baseline metrics: {baseline_metrics}")

            return baseline_metrics

        except Exception as e:
            self.logger.error(f"Error calculating baseline metrics: {e}")
            return {
                'dataset': dataset_name,
                'runtime': {},
                'rouge_scores': {},
                'clustering_scores': {},
                'preprocessing_time': 0
            }

    def calculate_comprehensive_metrics(
        self,
        summaries: Dict[str, str],
        references: Dict[str, str],
        embeddings: Optional[Union[np.ndarray, List]] = None,
        labels: Optional[Union[np.ndarray, List]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate all metrics with improved error handling."""
        try:
            metrics = {}
            
            # Calculate ROUGE scores
            rouge_scores = self.calculate_rouge_scores(summaries, references)
            metrics['rouge_scores'] = rouge_scores
            
            # Calculate BERT scores
            bert_scores = self.calculate_bert_scores(summaries, references)
            metrics['bert_scores'] = bert_scores
            
            # Calculate clustering metrics if provided
            if embeddings is not None and labels is not None:
                clustering_metrics = self.calculate_clustering_metrics(embeddings, labels)
                metrics['clustering'] = clustering_metrics
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {e}")
            return {
                'rouge_scores': {},
                'bert_scores': {},
                'clustering': {}
            }

    def _calculate_runtime_metrics(self) -> Dict[str, float]:
        """
        Calculate runtime metrics for the evaluation.

        Returns:
            Dict[str, float]: Dictionary of runtime metrics.
        """
        try:
            self.end_time = datetime.now()
            total_runtime = (self.end_time - self.start_time).total_seconds()
            average_runtime_per_sample = total_runtime / self.num_samples if self.num_samples else 0.0

            runtime_metrics = {
                'total_runtime_seconds': total_runtime,
                'average_runtime_per_sample_seconds': average_runtime_per_sample
            }

            self.logger.info(f"Runtime metrics: {runtime_metrics}")
            return runtime_metrics

        except Exception as e:
            self.logger.error(f"Error calculating runtime metrics: {e}")
            return {
                'total_runtime_seconds': 0.0,
                'average_runtime_per_sample_seconds': 0.0
            }

    def calculate_bert_scores(self, summaries: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore with proper error handling."""
        try:
            self.logger.info("Starting BERT scores calculation")
            
            if self.bert_scorer is None:
                # Create scorer inline if initialization failed
                self.bert_scorer = bert_score.BERTScorer(
                    model_type='roberta-large',
                    num_layers=17,
                    batch_size=32,
                    nthreads=4,
                    all_layers=False,
                    lang='en',
                    rescale_with_baseline=True,
                    device=self.device
                )
            
            P, R, F1 = self.bert_scorer.score(summaries, references)
            
            precision = float(P.mean())
            recall = float(R.mean())
            f1_score = float(F1.mean())

            self.logger.info("Completed BERT scores calculation")
            self.logger.debug(f"BERT scores: precision={precision}, recall={recall}, f1={f1_score}")

            return {
                'precision': precision,
                'recall': recall,
                'f1': f1_score
            }
        except Exception as e:
            self.logger.error(f"Error calculating BERTScore: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def _calculate_embedding_quality(self, embeddings: np.ndarray, batch_size: int = 32) -> Dict[str, float]:
        """
        Calculate embedding quality metrics.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            batch_size (int, optional): Batch size for processing. Defaults to 32.

        Returns:
            Dict[str, float]: Dictionary of embedding quality metrics.
        """
        try:
            self.logger.info("Starting embedding quality calculation")
            self.logger.debug(f"Embeddings shape: {embeddings.shape}, Batch size: {batch_size}")

            # Calculate cosine similarities
            similarities = cosine_similarity(embeddings)
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            min_similarity = np.min(similarities)
            max_similarity = np.max(similarities)

            self.logger.info("Completed embedding quality calculation")
            self.logger.debug(f"Embedding quality metrics: mean_similarity={mean_similarity}, std_similarity={std_similarity}, min_similarity={min_similarity}, max_similarity={max_similarity}")

            return {
                'mean_similarity': float(mean_similarity),
                'std_similarity': float(std_similarity),
                'min_similarity': float(min_similarity),
                'max_similarity': float(max_similarity)
            }
        except Exception as e:
            self.logger.error(f"Error calculating embedding quality: {e}")
            return {
                'mean_similarity': 0.0,
                'std_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0
            }

    def _calculate_style_metrics(
        self,
        summaries: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate metrics specific to different summary styles.

        Args:
            summaries (Dict[str, str]): Dictionary of generated summaries.

        Returns:
            Dict[str, float]: Dictionary of style metrics.
        """
        # Implement style-specific metrics here if needed
        style_metrics = {
            'technical_accuracy': 0.0,
            'conciseness_ratio': 0.0,
            'detail_coverage': 0.0
        }

        # Example implementation:
        total_length = 0
        total_sentences = 0
        for summary in summaries.values():
            total_length += len(summary.split())
            total_sentences += summary.count('.')

        if total_sentences > 0:
            conciseness_ratio = total_length / total_sentences
        else:
            conciseness_ratio = 0.0

        style_metrics['conciseness_ratio'] = conciseness_ratio

        return style_metrics

    def calculate_dataset_metrics(self, summaries: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate dataset-specific metrics.

        Args:
            summaries (List[str]): List of generated summaries.
            references (List[str]): List of reference summaries.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of dataset-specific metrics.
        """
        metrics = {
            'xlsum': calculate_xlsum_metrics(summaries, references),
            'scisummnet': calculate_scientific_metrics(summaries, references)
        }
        return metrics

# Standalone functions for dataset-specific metrics

def calculate_xlsum_metrics(summaries: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate XL-Sum specific metrics.

    Args:
        summaries (List[str]): List of generated summaries.
        references (List[str]): List of reference summaries.

    Returns:
        Dict[str, float]: Dictionary of XL-Sum specific metrics.
    """
    metrics = {
        'average_compression_ratio': np.mean([
            len(summary.split()) / len(reference.split()) if len(reference.split()) > 0 else 0
            for summary, reference in zip(summaries, references)
        ]),
        'coverage': _calculate_coverage_score(summaries, references),
        'factual_consistency': _calculate_factual_consistency(summaries, references)
    }
    return metrics

def calculate_scientific_metrics(summaries: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate scientific text specific metrics.

    Args:
        summaries (List[str]): List of generated summaries.
        references (List[str]): List of reference summaries.

    Returns:
        Dict[str, float]: Dictionary of scientific text specific metrics.
    """
    metrics = {
        'technical_term_preservation': _calculate_term_preservation(summaries, references),
        'citation_accuracy': _calculate_citation_accuracy(summaries, references),
        # Implement methods_coverage and results_accuracy as needed
        'methods_coverage': 0.0,
        'results_accuracy': 0.0
    }
    return metrics

def _calculate_coverage_score(summaries: List[str], references: List[str]) -> float:
    """
    Calculate content coverage score using token overlap.

    Args:
        summaries (List[str]): List of generated summaries.
        references (List[str]): List of reference summaries.

    Returns:
        float: Content coverage score.
    """
    total_coverage = 0.0
    for summary, reference in zip(summaries, references):
        summary_tokens = set(summary.lower().split())
        reference_tokens = set(reference.lower().split())
        if len(reference_tokens) > 0:
            coverage = len(summary_tokens.intersection(reference_tokens)) / len(reference_tokens)
        else:
            coverage = 0.0
        total_coverage += coverage
    return total_coverage / len(summaries) if summaries else 0.0

def _calculate_factual_consistency(summaries: List[str], references: List[str]) -> float:
    """
    Calculate factual consistency using named entity overlap.

    Args:
        summaries (List[str]): List of generated summaries.
        references (List[str]): List of reference summaries.

    Returns:
        float: Factual consistency score.
    """
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

        return total_consistency / len(summaries) if summaries else 0.0
    except Exception as e:
        logging.error(f"Error in _calculate_factual_consistency: {e}")
        return 0.0

def _calculate_term_preservation(summaries: List[str], references: List[str]) -> float:
    """
    Calculate technical term preservation ratio.

    Args:
        summaries (List[str]): List of generated summaries.
        references (List[str]): List of reference summaries.

    Returns:
        float: Technical term preservation ratio.
    """
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

        return total_preservation / len(summaries) if summaries else 0.0
    except Exception as e:
        logging.error(f"Error in _calculate_term_preservation: {e}")
        return 0.0

def _calculate_citation_accuracy(summaries: List[str], references: List[str]) -> float:
    """
    Calculate accuracy of citation preservation.

    Args:
        summaries (List[str]): List of generated summaries.
        references (List[str]): List of reference summaries.

    Returns:
        float: Citation accuracy score.
    """
    citation_pattern = r'\[\d+\]|\([A-Za-z]+\s+et\s+al\.,?\s*\d{4}\)'

    total_accuracy = 0.0
    for summary, reference in zip(summaries, references):
        ref_citations = set(re.findall(citation_pattern, reference))
        sum_citations = set(re.findall(citation_pattern, summary))

        if ref_citations:
            accuracy = len(sum_citations.intersection(ref_citations)) / len(ref_citations)
            total_accuracy += accuracy

    return total_accuracy / len(summaries) if summaries else 0.0
