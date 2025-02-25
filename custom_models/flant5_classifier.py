from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
from typing import Tuple
from mylogger import get_logger

logger = get_logger(__name__)

class FlanT5Classifier:
    def __init__(self, model_name: str = "google/flan-t5-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        start_time = time.time()
        logger.info(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        load_time = time.time() - start_time
        logger.info(f"Model loading time: {load_time:.2f} seconds")


    @staticmethod
    def create_prompt(sentence: str) -> str:
        return f"""
You are an AI assistant that classifies user sentences into one of the following categories:

find_compare_flights: Sentences where the user is looking to find, book, or compare flights. This includes round-trip, one-way flights, or general flight information.
suggest_hotels: Sentences where the user wants to find, book, or get recommendations for hotels or accommodations.
explore_activities_places: Sentences where the user wants to explore activities, attractions, restaurants, or places in a city. This includes museums, cultural spots, and sightseeing.
out_of_scope: Sentences unrelated to travel, flights, hotels, exploring places, restaurants or finding nearby locations.

Examples:
User: I want to book a flight to Rome next week.
Category: find_compare_flights

User: Can you suggest a hotel in Paris?
Category: suggest_hotels

User: What museums can I visit in Athens?
Category: explore_activities_places

User: What's the weather like tomorrow?
Category: out_of_scope

User: I need to compare flights to Tokyo.
Category: find_compare_flights

User: Where can I stay in Berlin?
Category: suggest_hotels

User: Recommend a good restaurant in Rome.
Category: explore_activities_places

User: Where is the best ravioli in Napoly?
Category: explore_activities_places

User: I want to watch a movie tonight.
Category: out_of_scope

Now classify the following sentence:
User: {sentence}
Category:"""


    def classify(self, prompt: str) -> Tuple[str, float]:
        try:
            start_time = time.time()
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_length=20,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False   # deterministic generation (greedy)
            )
            
            prediction = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip().lower()
            
            # calculate confidence
            scores = torch.stack(outputs.scores, dim=0)
            sequence_scores = torch.softmax(scores, dim=-1)
            confidence = sequence_scores.max(dim=-1).values.mean().item()
            
            generation_time = time.time() - start_time
            logger.info(f"Generation time: {generation_time:.2f} seconds")
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error: {e}")
            raise