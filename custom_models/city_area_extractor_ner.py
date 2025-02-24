from transformers import pipeline

class CityAreaExtractor:
    def __init__(self, model_name="dslim/bert-base-NER"):
        """Initialize the city extractor with a specified NER model."""
        self.ner = pipeline("ner", model=model_name, use_fast=False)

    def extract_city(self, sentence):
        # convert to title case for better NER detection,
        # this converts words like "london" to "London"
        title_case_sentence = ' '.join(word.capitalize() for word in sentence.split())

        results = self.ner(title_case_sentence)

        # group consecutive LOC entities to handle multi-word cities
        cities = []
        current_city = []

        for entity in results:
            if entity['entity'] in ['B-LOC', 'I-LOC']:
                # handle word pieces that start with ##
                if entity['word'].startswith('##'):
                    if current_city:
                        current_city[-1] += entity['word'][2:]
                else:
                    current_city.append(entity['word'])
            else:
                if current_city:
                    cities.append(' '.join(current_city))
                    current_city = []

        # add the last city if there's one being processed
        if current_city:
            cities.append(' '.join(current_city))

        return cities[0] if cities else None