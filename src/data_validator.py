import pandas as pd
from typing import Dict, List, Any
import logging
import yaml
from torch.utils.data import DataLoader, Dataset

class DataValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate processed dataset against quality criteria.

        Args:
            df (pd.DataFrame): DataFrame containing the dataset to validate.

        Returns:
            Dict[str, bool]: Dictionary with validation results and statistics.
        """
        try:
            validation_results = {
                'missing_values': self._check_missing_values(df),
                'text_length': self._check_text_length(df),
                'language': self._check_language(df),
                'duplicates': self._check_duplicates(df)
            }

            # Log validation results
            self.logger.info(f"Dataset validation results: {validation_results}")

            # Overall validation status
            is_valid = all(validation_results.values())

            if not is_valid:
                self.logger.warning("Dataset failed validation checks")

            return {
                'is_valid': is_valid,
                'checks': validation_results,
                'stats': self.get_detailed_stats(df)
            }

        except Exception as e:
            self.logger.error(f"Error during dataset validation: {e}")
            raise

    def _check_missing_values(self, df: pd.DataFrame) -> bool:
        """
        Check if missing values are below threshold (5%).

        Args:
            df (pd.DataFrame): DataFrame to check for missing values.

        Returns:
            bool: True if missing values are below threshold, False otherwise.
        """
        missing_pct = df.isnull().sum() / len(df) * 100
        return all(missing_pct < 5)

    def _check_text_length(self, df: pd.DataFrame) -> bool:
        """
        Check if text lengths meet minimum requirements.

        Args:
            df (pd.DataFrame): DataFrame to check for text lengths.

        Returns:
            bool: True if text lengths meet minimum requirements, False otherwise.
        """
        min_length = 100  # Configurable
        text_lengths = df['text'].str.split().str.len()
        return all(text_lengths >= min_length)

    def _check_language(self, df: pd.DataFrame) -> bool:
        """
        Check if texts are in English using spacy's language detector.

        Args:
            df (pd.DataFrame): DataFrame to check for language.

        Returns:
            bool: True if texts are in English, False otherwise.
        """
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')

            # Sample a subset of texts for efficiency
            sample_size = min(100, len(df))
            sample_texts = df['text'].sample(n=sample_size)

            english_count = sum(
                1 for text in sample_texts
                if nlp(text[:100]).lang_ == 'en'  # Check first 100 chars
            )

            # Require 95% of sampled texts to be English
            return (english_count / sample_size) >= 0.95

        except Exception as e:
            self.logger.error(f"Language check failed: {e}")
            return False

    def _check_duplicates(self, df: pd.DataFrame) -> bool:
        """
        Check for duplicate entries.

        Args:
            df (pd.DataFrame): DataFrame to check for duplicates.

        Returns:
            bool: True if duplicates are below threshold, False otherwise.
        """
        duplicate_ratio = df.duplicated(subset=['text']).sum() / len(df)
        return duplicate_ratio < 0.05  # Allow up to 5% duplicates

    def get_detailed_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate detailed statistics about the dataset.

        Args:
            df (pd.DataFrame): DataFrame to generate statistics for.

        Returns:
            Dict[str, float]: Dictionary with detailed statistics.
        """
        try:
            text_lengths = df['processed_text'].str.len()
            return {
                'total_documents': len(df),
                'avg_text_length': float(text_lengths.mean()),
                'std_text_length': float(text_lengths.std()),
                'missing_ratio': float(df.isnull().mean().mean()),
                'duplicate_ratio': float(df.duplicated().mean())
            }
        except Exception as e:
            self.logger.error(f"Error calculating dataset stats: {e}")
            raise

    def validate_with_thresholds(self, df: pd.DataFrame, config: Dict) -> Dict[str, bool]:
        """
        Validate dataset against configurable thresholds.

        Args:
            df (pd.DataFrame): DataFrame to validate.
            config (Dict): Configuration dictionary with validation thresholds.

        Returns:
            Dict[str, bool]: Dictionary with validation results.
        """
        thresholds = config.get('preprocessing', {}).get('validation', {})

        checks = {
            'missing_values': all(
                pct < thresholds.get('missing_threshold', 5.0)
                for pct in (df.isnull().sum() / len(df) * 100)
            ),
            'dataset_size': len(df) >= thresholds.get('min_dataset_size', 10000),
            'text_length': all(
                thresholds.get('min_text_length', 100) <= length <= thresholds.get('max_text_length', 1000)
                for length in df['text'].str.split().str.len()
            )
        }

        self.logger.info(f"Validation results with thresholds: {checks}")
        return checks

    def validate_batch(self, df: pd.DataFrame, batch_size: int = 32) -> Dict[str, bool]:
        """
        Validate dataset in batches using PyTorch’s DataLoader.

        Args:
            df (pd.DataFrame): DataFrame to validate.
            batch_size (int, optional): Batch size for processing. Defaults to 32.

        Returns:
            Dict[str, bool]: Dictionary with validation results.
        """
        dataset = DataFrameDataset(df)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        validation_results = {
            'missing_values': True,
            'text_length': True,
            'language': True,
            'duplicates': True
        }

        for batch in dataloader:
            batch_df = pd.DataFrame(batch)
            validation_results['missing_values'] &= self._check_missing_values(batch_df)
            validation_results['text_length'] &= self._check_text_length(batch_df)
            validation_results['language'] &= self._check_language(batch_df)
            validation_results['duplicates'] &= self._check_duplicates(batch_df)

        is_valid = all(validation_results.values())
        return {
            'is_valid': is_valid,
            'checks': validation_results,
            'stats': self.get_detailed_stats(df)
        }

class DataFrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()

class ConfigValidator:
    """
    Validates configuration settings for the pipeline.
    """

    REQUIRED_FIELDS = {
        'data': {
            'input_path': str,
            'output_path': str,
            'scisummnet_path': str,
            'processed_path': str,
            'batch_size': int
        },
        'preprocessing': {
            'min_length': int,
            'max_length': int,
            'validation': dict  # For validation thresholds
        },
        'embedding': {
            'model_name': str,
            'dimension': int,  # Added this required field
            'batch_size': int,
            'max_seq_length': int,
            'device': str
        },
        'clustering': {
            'algorithm': str,
            'min_cluster_size': int,
            'min_samples': int,
            'metric': str,
            'params': dict,
            'output_dir': str
        },
        'visualization': {
            'enabled': bool,
            'output_dir': str
        },
        'summarization': {
            'model_name': str,
            'max_length': int,
            'min_length': int,
            'batch_size': int
        },
        'logging': {
            'level': str,
            'format': str
        },
        'checkpoints': {
            'dir': str
        }
    }

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validates the configuration dictionary against required fields.
        Returns True if valid, raises ValueError if invalid.

        Args:
            config (Dict[str, Any]): Configuration dictionary to validate.

        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        try:
            self._validate_section(config, self.REQUIRED_FIELDS)
            return True
        except ValueError as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")

    def _validate_section(self, config: Dict[str, Any], required: Dict[str, Any], path: str = "") -> None:
        """
        Recursively validates configuration sections.

        Args:
            config (Dict[str, Any]): Configuration dictionary to validate.
            required (Dict[str, Any]): Dictionary of required fields and their types.
            path (str, optional): Current path in the configuration dictionary. Defaults to "".

        Raises:
            ValueError: If a required field is missing or has an incorrect type.
        """
        for key, value_type in required.items():
            current_path = f"{path}.{key}" if path else key

            if key not in config:
                raise ValueError(f"Missing required field: {current_path}")

            if isinstance(value_type, dict):
                if not isinstance(config[key], dict):
                    raise ValueError(f"Field {current_path} must be a dictionary")
                self._validate_section(config[key], value_type, current_path)
            else:
                if not isinstance(config[key], value_type):
                    raise ValueError(
                        f"Field {current_path} must be of type {value_type.__name__}, "
                        f"got {type(config[key]).__name__}"
                    )
