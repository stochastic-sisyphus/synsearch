from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from src.utils.performance import PerformanceOptimizer
from src.data_validator import DataValidator

class HybridSummarizer:
    """Base class for hybrid summarization approaches."""
    
    def __init__(self, model_name='t5-base', tokenizer=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_name = model_name
        self.device = device
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.validator = DataValidator()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def summarize(self, texts: List[str], max_length: int = 150) -> List[str]:
        """Generate summaries for input texts."""
        # Validate input texts
        validation_results = self.validator.validate_texts(texts)
        if not validation_results['is_valid']:
            raise ValueError(f"Input texts validation failed: {validation_results}")

        # Log input texts and validation results
        self.logger.info(f"Input texts: {texts}")
        self.logger.info(f"Input validation results: {validation_results}")

        summaries = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summaries.append(summary)

        # Validate generated summaries
        validation_results = self.validator.validate_summaries(summaries)
        if not validation_results['is_valid']:
            raise ValueError(f"Generated summaries validation failed: {validation_results}")

        # Log generated summaries and validation results
        self.logger.info(f"Generated summaries: {summaries}")
        self.logger.info(f"Output validation results: {validation_results}")

        return summaries

class EnhancedHybridSummarizer(HybridSummarizer):
    """
    EnhancedHybridSummarizer: A flexible summarization module that combines extractive and abstractive approaches.

    Features:
    - Style-aware summarization (technical, concise, detailed)
    - Configurable length and parameters per style
    - Batch processing support
    - GPU acceleration
    - Checkpoint support

    Example:
        summarizer = EnhancedHybridSummarizer(
            model_name='facebook/bart-large-cnn',
            max_length=150,
            min_length=50
        )
        summary = summarizer.summarize(texts, style='technical')
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.perf_optimizer = PerformanceOptimizer()
        self.batch_size = self.perf_optimizer.get_optimal_batch_size()

    def __init__(
        self,
        model_name: str = 'facebook/bart-large-cnn',
        max_length: int = 150,
        min_length: int = 50,
        batch_size: int = 4,
        device: Optional[str] = None
    ):
        """Initialize hybrid summarizer with extractive + abstractive capabilities"""
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.batch_size = batch_size
            self.max_length = max_length
            self.min_length = min_length
            
            # Initialize TF-IDF for extractive step
            self.tfidf = TfidfVectorizer(max_features=1000)
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
            
        # Enhanced style configurations with domain-specific parameters
        self.style_config = {
            'technical': {
                'prompt': "Provide a technical summary focusing on methodology and results:",
                'top_k': 3,
                'length_multiplier': 1.2,
                'domain_weights': {
                    'methodology': 0.4,
                    'results': 0.4,
                    'background': 0.2
                }
            },
            'academic': {
                'prompt': "Summarize the academic research findings and implications:",
                'top_k': 4,
                'length_multiplier': 1.3,
                'domain_weights': {
                    'findings': 0.5,
                    'implications': 0.3,
                    'methods': 0.2
                }
            },
            'concise': {
                'prompt': "Summarize the key points briefly:",
                'top_k': 2,
                'length_multiplier': 0.8
            },
            'detailed': {
                'prompt': "Provide a comprehensive summary including background and implications:",
                'top_k': 4,
                'length_multiplier': 1.5
            },
            'balanced': {
                'prompt': "",
                'top_k': 3,
                'length_multiplier': 1.0
            }
        }

    def _extract_key_sentences(self, texts: List[str], top_k: int = 3, style: str = 'balanced') -> Tuple[List[str], Dict]:
        """Enhanced extraction with style-aware sentence selection"""
        try:
            # Calculate TF-IDF matrix
            tfidf_matrix = self.tfidf.fit_transform(texts)
            
            # Get domain-specific weights if available
            domain_weights = self.style_config.get(style, {}).get('domain_weights', None)
            
            if domain_weights:
                # Apply domain-specific weighting to TF-IDF scores
                weighted_scores = np.zeros(len(texts))
                for domain, weight in domain_weights.items():
                    domain_terms = self._get_domain_terms(domain)
                    domain_mask = self.tfidf.get_feature_names_out() in domain_terms
                    domain_scores = np.mean(tfidf_matrix[:, domain_mask].toarray(), axis=1)
                    weighted_scores += weight * domain_scores
            else:
                weighted_scores = np.mean(tfidf_matrix.toarray(), axis=1)
            
            # Select top-k documents based on weighted scores
            top_indices = np.argsort(weighted_scores)[-top_k:]
            
            return [texts[i] for i in top_indices], {
                'scores': weighted_scores[top_indices].tolist(),
                'style': style
            }
            
        except Exception as e:
            self.logger.error(f"Error in extractive summarization: {e}")
            raise

    def _get_domain_terms(self, domain: str) -> List[str]:
        """Get domain-specific terms for weighted extraction"""
        domain_terms = {
            'methodology': ['method', 'approach', 'technique', 'algorithm', 'procedure'],
            'results': ['results', 'findings', 'outcome', 'performance', 'accuracy'],
            'implications': ['implications', 'impact', 'significance', 'importance'],
            'findings': ['discovered', 'observed', 'found', 'demonstrated'],
            'background': ['background', 'context', 'previous', 'existing']
        }
        return domain_terms.get(domain, [])

    def _get_representative_texts(self, texts: List[Dict], n_samples=3) -> List[str]:
        """Select most representative texts from cluster using embedding similarity"""
        embeddings = [text['embedding'] for text in texts]
        similarities = cosine_similarity(embeddings)
        centrality_scores = similarities.mean(axis=0)
        top_indices = np.argsort(centrality_scores)[-n_samples:]
        return [texts[i]['processed_text'] for i in top_indices]
        
    def summarize_all_clusters(
        self,
        cluster_texts: Dict[str, List[Dict]],
        style: str = 'balanced'
    ) -> Dict[str, Dict]:
        """Generate summaries for all clusters with batched processing."""
        try:
            summaries = {}
            style_config = self.style_config.get(style, self.style_config['balanced'])
            
            for cluster_id, texts in tqdm(cluster_texts.items(), desc="Summarizing clusters"):
                rep_texts = self._get_representative_texts(texts)
                
                # Prepare prompt
                prompt = self._create_style_prompt(style, rep_texts)
                
                # Generate summary
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=1024
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_length=int(self.max_length * style_config['length_multiplier']),
                    min_length=self.min_length,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
                
                summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                summaries[cluster_id] = {
                    'summary': summary,
                    'style': style,
                    'num_docs': len(texts),
                    'metadata': self._get_summary_metadata(summary)
                }
                
                # Save intermediate outputs
                self._save_intermediate_outputs(cluster_id, summary, style, len(texts))
                
            return summaries
            
        except Exception as e:
            self.logger.error(f"Error in summarize_all_clusters: {e}")
            raise

    def _create_style_prompt(self, style: str, texts: List[str]) -> str:
        """Create style-specific prompts."""
        style_prompts = {
            'technical': "Provide a technical summary focusing on methodology and results:\n",
            'concise': "Summarize the key points briefly:\n",
            'detailed': "Provide a comprehensive summary including background and implications:\n"
        }
        
        base_prompt = style_prompts.get(style, "")
        return f"{base_prompt}{' '.join(texts)}"

    def summarize_batch(self, texts, max_length=150):
        """Summarize a batch of texts using GPU acceleration."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            
        summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return summaries
        
    def summarize_with_clusters(self, cluster_texts, cluster_features, batch_size=8):
        """Process clusters in batches for better GPU utilization."""
        summaries = {}
        for cluster_id, texts in cluster_texts.items():
            # Process texts in batches
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            cluster_summaries = []
            
            for batch in batches:
                batch_summaries = self.summarize_batch(batch)
                cluster_summaries.extend(batch_summaries)
                
            summaries[cluster_id] = cluster_summaries
            
        return summaries

    def summarize_cluster(
        self,
        texts: List[str],
        style: str = 'auto'
    ) -> Dict[str, Any]:
        if style == 'auto':
            style = self._determine_optimal_style(texts)
        
        config = self.style_configs[style]
        summary = self._generate_summary(
            texts,
            max_length=config['max_length'],
            min_length=config['min_length']
        )
        
        return {
            'summary': summary,
            'style': style,
            'metadata': self._get_summary_metadata(summary)
        }

    def _preprocess_cluster_texts(self, texts: List[str], style: str) -> List[str]:
        """Preprocess texts based on style requirements"""
        try:
            # Apply basic preprocessing
            processed_texts = [
                text.strip()
                for text in texts
                if text and len(text.strip()) > 0
            ]
            
            # Apply style-specific preprocessing
            if style == 'technical':
                # Preserve technical terms and numbers
                return processed_texts
            elif style == 'concise':
                # Limit to first few sentences for conciseness
                return [' '.join(text.split('.')[:3]) + '.' for text in processed_texts]
            else:
                return processed_texts
            
        except Exception as e:
            self.logger.error(f"Error in text preprocessing: {e}")
            raise

    def _batch_summarize(
        self,
        texts: List[str],
        max_length: int,
        min_length: int
    ) -> List[str]:
        """Generate summaries in batches for efficiency"""
        try:
            summaries = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    max_length=1024,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate summaries
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
                
                # Decode summaries
                batch_summaries = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in summary_ids
                ]
                summaries.extend(batch_summaries)
                
            return summaries
            
        except Exception as e:
            self.logger.error(f"Error in batch summarization: {e}")
            raise

    def _combine_summaries(
        self,
        summaries: List[str],
        style: str
    ) -> str:
        """Combine multiple summaries based on style"""
        try:
            if style == 'technical':
                # Preserve technical details in combination
                return " Furthermore, ".join(summaries)
            elif style == 'concise':
                # Take key points only
                return " In summary, " + ". ".join(summaries)
            else:
                # Default combination
                return " Moreover, ".join(summaries)
            
        except Exception as e:
            self.logger.error(f"Error combining summaries: {e}")
            raise

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint and configuration"""
        try:
            checkpoint = {
                'model_state': self.model.state_dict(),
                'tokenizer_state': self.tokenizer.save_pretrained(path / 'tokenizer'),
                'config': {
                    'max_length': self.max_length,
                    'min_length': self.min_length,
                    'batch_size': self.batch_size,
                    'device': self.device,
                    'style_config': self.style_config
                }
            }
            torch.save(checkpoint, path / 'summarizer_checkpoint.pt')
            self.logger.info(f"Saved checkpoint to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            raise

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint and configuration"""
        try:
            checkpoint = torch.load(path / 'summarizer_checkpoint.pt')
            self.model.load_state_dict(checkpoint['model_state'])
            self.tokenizer = AutoTokenizer.from_pretrained(path / 'tokenizer')
            
            # Update configuration
            config = checkpoint['config']
            self.max_length = config['max_length']
            self.min_length = config['min_length']
            self.batch_size = config['batch_size']
            self.device = config['device']
            self.style_config = config['style_config']
            
            self.logger.info(f"Loaded checkpoint from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            raise

    def _save_intermediate_outputs(self, cluster_id: str, summary: str, style: str, num_docs: int) -> None:
        """Save intermediate outputs after summarization."""
        output_dir = Path("outputs/summarization")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_file = output_dir / f"summary_{cluster_id}.txt"
        metadata_file = output_dir / f"metadata_{cluster_id}.json"
        
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        metadata = {
            'cluster_id': cluster_id,
            'style': style,
            'num_docs': num_docs
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        self.logger.info(f"Saved intermediate outputs for cluster {cluster_id} to {output_dir}")

    def _get_summary_metadata(self, summary: str) -> Dict[str, Any]:
        """
        Extract metadata from the generated summary.

        Args:
            summary (str): The generated summary text.

        Returns:
            Dict[str, Any]: Metadata information.
        """
        # Example metadata extraction logic
        metadata = {
            'length': len(summary),
            'num_sentences': summary.count('.'),
            'num_words': len(summary.split())
        }
        return metadata
