from typing import Dict
import time
import json
from datetime import date

from openai import OpenAI


class OpenAIClient:
    """Handles OpenAI API interactions for airport information."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    

    @staticmethod
    def create_airport_prompt(city: str) -> str:
        return f'''You are a travel and aviation expert. 
Return information about all major unique commercial airports in {city} in the exact JSON format below.
Only include actively operating commercial airports.
Return JSON only, no other text.

Required format:
{{
   "city": city_name,
   "city_code": city_code,
   "country": country_name,
   "country_2": country_code,
   "country_3": country_code_3,
   "airports": [
       {{
           "name": "full_airport_name",
           "IATA_CODE": "3_letter_code"
       }}
   ]
}}'''

    @staticmethod
    def create_flight_extraction_prompt(request: str) -> str:
        today = date.today().strftime("%Y-%m-%d")
        return f'''
This is a request for booking flights: "{request}".
Please extract the following information if it exists.

Rules:
- For dates, use the format YYYY-MM-DD, while today's date is {today}.
- For cities be sure to write the official city name, eg. if "big Apple" is mentioned write New York
if "capital of France" is mentioned write Paris.
- If a field is not mentioned, set its value to `null`.
- Return JSON only, no other text.

Required format:
{{
  "departure_city": city_name,
  "arrival_city": city_name,
  "departure_date": date,
  "return_date": date,
  "num_passengers": number
}}'''


    def get_completion(self, prompt: str, model: str = "gpt-4") -> Dict:
        """Get completion from OpenAI API with retry logic and JSON parsing."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                result = json.loads(response.choices[0].message.content)
                return result
                
            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    raise ValueError("Failed to get valid JSON response")
                time.sleep(3)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(3)