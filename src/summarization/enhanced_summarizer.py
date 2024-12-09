from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class EnhancedSummarizer:
    def __init__(self, model_name='t5-base'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
    def enhance_summary_with_cluster_insights(self, texts, cluster_info):
        """Generate enhanced summaries using cluster information."""
        cluster_context = f"Cluster size: {len(texts)}, Key themes: {cluster_info['themes']}"
        input_text = f"summarize: {cluster_context} {' '.join(texts)}"
        
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.model.generate(inputs.input_ids, max_length=150, min_length=40)
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True) 