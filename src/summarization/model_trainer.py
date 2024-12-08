class SummarizationModelTrainer:
    def __init__(self, config: Dict):
        self.model_name = config['summarization']['model_name']
        self.device = config['summarization']['device']
        self.batch_size = config['summarization']['batch_size']
        
    def prepare_training_data(self):
        """Load and prepare training datasets"""
        # Load XL-Sum for general summarization capabilities
        xlsum = load_dataset('GEM/xlsum')
        
        # Load ScisummNet for scientific summarization style
        scisummnet = DataLoader(self.config['data']['scisummnet_path']).load_scisummnet()
        
        return self._combine_datasets(xlsum, scisummnet)
        
    def train_model(self):
        """Fine-tune the model on scientific summarization"""
        training_data = self.prepare_training_data()
        
        # Initialize model
        model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Training loop
        # ... training code ...
        
        return model, tokenizer 