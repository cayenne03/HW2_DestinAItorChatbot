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
You are an AI assistant that classifies user sentences into one of the following categories/options numerically ordered:
Category 1. => trip_planning: Sentences where the user is organizing a trip. This includes booking flights, reserving hotels, arranging transportation, or creating travel itineraries. It focuses on how to get to the destination and where to stay.
Category 2. => city_explorer: Sentences where the user wants to explore attractions, food, and activities in a city. This includes visiting monuments, museums, restaurants, or finding local events and cultural experiences.
Category 3. => around_me: Sentences where the user wants to find nearby places based on their current location. This includes searching for nearby restaurants, gas stations, pharmacies, or other local services.
Category 4 => out_of_scope: Sentences unrelated to travel, trip planning, or exploring places. This includes personal statements, general questions, or any topics not connected to travel.

Examples:
User: I want to book a flight to Rome.
Category: trip_planning

User: Show me restaurants in Paris.
Category: city_explorer

User: What's near me right now?
Category: around_me

User: Let's watch movie tonight
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