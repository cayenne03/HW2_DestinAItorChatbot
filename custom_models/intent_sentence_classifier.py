import torch
from sentence_transformers import SentenceTransformer, util

class IntentClassifier:
    def __init__(self, categories, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize the intent classifier.
        
        Args:
            categories (list): List of category strings to classify intents into
            model_name (str): Name of the sentence transformer model to use
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        
        if not categories or not isinstance(categories, list):
            raise ValueError("Categories must be provided as a non-empty list")
            
        self.categories = categories
        
        # create category map dynamically
        self.category_map = {i: category for i, category in enumerate(self.categories)}
        self.category_embeddings = None
        
        # pre-compute category embeddings
        self._compute_category_embeddings()
        
    def _compute_category_embeddings(self):
        """Pre-compute the embeddings for all categories"""
        self.category_embeddings = self.model.encode(
            self.categories, 
            convert_to_tensor=True, 
            device=self.device
        )
    
    def classify(self, text):
        """Classify the text into one of the predefined categories"""
        # encode the input text
        text_embedding = self.model.encode(
            text, 
            convert_to_tensor=True, 
            device=self.device
        )
        
        # compute similarity scores
        similarities = util.cos_sim(text_embedding, self.category_embeddings).squeeze()
        
        # get the top category
        top_idx = similarities.argmax().item()
        
        return {
            "category_id": top_idx,
            "category": self.categories[top_idx],
            "confidence": round(similarities[top_idx].item(), 4)
        }