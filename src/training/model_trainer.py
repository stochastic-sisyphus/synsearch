import torch
from transformers import AutoModelForSeq2SeqGeneration, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from typing import Dict, Optional, List
import logging
from pathlib import Path
from src.data_loader import DataLoader

class SummarizationModelTrainer:
    """Handles model fine-tuning on scientific paper datasets."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and tokenizer
        self.model_name = config['training']['base_model']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqGeneration.from_pretrained(self.model_name)
        
    def prepare_scientific_dataset(self):
        """Load and prepare ScisummNet dataset."""
        data_loader = DataLoader(self.config)
        scisummnet = data_loader.load_scisummnet()
        
        return self._prepare_training_data(
            texts=scisummnet['text'].tolist(),
            summaries=scisummnet['summary'].tolist()
        )
        
    def prepare_xlsum_dataset(self):
        """Load and prepare XL-Sum dataset."""
        dataset = load_dataset('GEM/xlsum', 'english')
        
        return self._prepare_training_data(
            texts=dataset['train']['text'],
            summaries=dataset['train']['summary']
        )
        
    def _prepare_training_data(self, texts: List[str], summaries: List[str]):
        """Prepare data for training."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config['training']['max_input_length'],
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            summaries,
            padding=True,
            truncation=True,
            max_length=self.config['training']['max_output_length'],
            return_tensors="pt"
        )
        
        return torch.utils.data.TensorDataset(
            inputs['input_ids'],
            inputs['attention_mask'],
            targets['input_ids'],
            targets['attention_mask']
        )
        
    def train(self, dataset_name: str = 'scisummnet'):
        """Fine-tune the model on specified dataset."""
        try:
            # Prepare dataset
            if dataset_name == 'scisummnet':
                dataset = self.prepare_scientific_dataset()
            else:
                dataset = self.prepare_xlsum_dataset()
                
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=self.config['training']['output_dir'],
                num_train_epochs=self.config['training']['epochs'],
                per_device_train_batch_size=self.config['training']['batch_size'],
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=100,
                save_strategy="epoch"
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=self.tokenizer
            )
            
            # Train model
            trainer.train()
            
            # Save fine-tuned model
            model_output_dir = Path(self.config['training']['output_dir']) / dataset_name
            self.model.save_pretrained(model_output_dir)
            self.tokenizer.save_pretrained(model_output_dir)
            
            self.logger.info(f"Model fine-tuned and saved to {model_output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
            
    def compare_models(self, text: str) -> Dict[str, str]:
        """Compare summaries from pre-trained and fine-tuned models."""
        try:
            # Load fine-tuned models
            scisummnet_model = AutoModelForSeq2SeqGeneration.from_pretrained(
                Path(self.config['training']['output_dir']) / 'scisummnet'
            )
            xlsum_model = AutoModelForSeq2SeqGeneration.from_pretrained(
                Path(self.config['training']['output_dir']) / 'xlsum'
            )
            
            # Generate summaries
            summaries = {
                'pretrained': self._generate_summary(self.model, text),
                'scisummnet': self._generate_summary(scisummnet_model, text),
                'xlsum': self._generate_summary(xlsum_model, text)
            }
            
            return summaries
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {e}")
            raise
            
    def _generate_summary(self, model: AutoModelForSeq2SeqGeneration, text: str) -> str:
        """Generate summary using specified model."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config['training']['max_input_length'],
            truncation=True
        ).to(self.device)
        
        outputs = model.generate(
            **inputs,
            max_length=self.config['training']['max_output_length'],
            min_length=self.config['training']['min_output_length'],
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True) 