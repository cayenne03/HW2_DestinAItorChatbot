from typing import Any, Text, Dict, List, Optional, Tuple
import os
import json

from dotenv import load_dotenv
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, FollowupAction, UserUtteranceReverted
from rasa_sdk.types import DomainDict

from mylogger import get_logger
from custom_models.flant5_classifier import FlanT5Classifier
from custom_models.spacy_nlp_md import nlp
from utils.apis.openai_client_api import OpenAIClient
from utils.date_utils import parse_date_to_iso


load_dotenv()

logger = get_logger(__name__)
logger.debug("Actions module loaded")

openai_client = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))

class ActionSessionStart(Action):
    def name(self) -> Text:
        return "action_session_start"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(response="utter_welcome_menu")

        logger.info("Session started....")

        return [SessionStarted()]


class ActionValidateIntent(Action):
    def __init__(self):
        self.classifier = FlanT5Classifier()

    def name(self) -> Text:
        return "action_validate_intent"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        logger.debug("ActionValidateIntent started")
        logger.debug(f"Active loop: {tracker.active_loop}")
        logger.debug(f"All slots: {tracker.current_slot_values()}")

        active_loop = tracker.active_loop.get('name')
        if active_loop:
            logger.debug(f"Skipping intent validation - active form: {active_loop}")
            return []
                
        logger.debug("Starting intent validation...")

        latest_message = tracker.latest_message.get('text')
        rasa_intent = tracker.latest_message.get('intent').get('name')
        rasa_confidence = tracker.latest_message.get('intent').get('confidence', 0.0)

        intents_to_validate = ['find_compare_flights', 'suggest_hotels', 'explore_activities_places', 'around_me']
        if rasa_intent not in intents_to_validate:
            return []

        # get FlanT5 prediction
        prompt = self.classifier.create_prompt(latest_message)
        predicted_intent, confidence_score = self.classifier.classify(prompt)

        logger.info(f"User message: {latest_message}")
        logger.info(f"RASA Intent: {rasa_intent} (confidence: {rasa_confidence})")
        logger.info(f"FlanT5 Few-shot Intent: {predicted_intent} (confidence: {confidence_score})")
        # logger.info(f"Confidence Score: {confidence_score:.2f}")

        # all_intents = tracker.latest_message.get('intent_ranking', [])
        # logger.info("All intent confidences:")
        # for intent in all_intents:
        #     logger.info(f"Intent: {intent['name']}, Confidence: {intent['confidence']}")

        if predicted_intent != rasa_intent and predicted_intent == "out_of_scope":
            logger.info(f"Reverting to FlanT5 Few-shot Intent: {predicted_intent}")
            dispatcher.utter_message(text="Seems like you are asking something out of my scope. Could you try something else from the options available?")
            # preventing Rasa from going forward with its intent
            return [UserUtteranceReverted()]
        
            # return [{"intent": predicted_intent}]
            #
            # return [
            #     {"event": "revert_user_message"},
            #     {"event": "user", "intent": "nlu_fallback"}
            # ]

        return []

class ActionExtractFlightEntities(Action):
    def name(self) -> Text:
        return "action_extract_flight_entities"
    
    def _extract_flight_type(self, doc) -> Optional[str]:
        """Extract flight type (oneway/round_trip)"""
        for ent in doc.ents:
            if ent.label_ == "FLIGHT_TYPE" and ent._.id:
                logger.info(f"Found flight type: {ent._.id}")

                return ent._.id
        return None


    def _extract_cities(self, doc) -> Tuple[Optional[str], Optional[str]]:
        """Extract departure and arrival cities"""
        departure_city = None
        arrival_city = None
        used_cities = set()

        logger.debug(f"Processing text for city extraction: {doc.text}")

        # 1st look for explicit from/to indicators
        found_indicators = []

        for ent in doc.ents:
            if ent.label_ == "LOCATION_INDICATOR" and ent._.id:
                found_indicators.append((ent._.id, ent.start, ent.end))

        logger.debug(f"Found indicators: {found_indicators}")

        # 2nd look for cities after each indicator
        for indicator_id, start, end in found_indicators:
            next_idx = end
            while next_idx < len(doc) and doc[next_idx].is_stop:
                next_idx += 1

            if next_idx >= len(doc):
                continue

            # try to find a city (up to 3 tokens)
            for end_idx in range(next_idx + 1, min(next_idx + 4, len(doc))):
                potential_city = doc[next_idx:end_idx].text
                validation_doc = nlp(potential_city)

                if any(val_ent.label_ == "GPE" for val_ent in validation_doc.ents):
                    if indicator_id == "departure" and not departure_city:
                        departure_city = potential_city
                        used_cities.add(potential_city)
                        logger.info(f"Found departure city with indicator: {departure_city}")
                        break
                    elif indicator_id == "arrival" and not arrival_city:
                        arrival_city = potential_city
                        used_cities.add(potential_city)
                        logger.info(f"Found arrival city with indicator: {arrival_city}")
                        break

        # if we're still missing cities, look for GPEs
        if not (departure_city and arrival_city):
            gpe_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"
                        and ent.text not in used_cities]
            logger.debug(f"Found unused GPE entities: {gpe_entities}")

            # Use context to determine city roles
            if gpe_entities:
                # If we have 'to' or arrow, second GPE is arrival
                if ("to" in doc.text.lower() or "->" in doc.text or "→" in doc.text) and len(gpe_entities) >= 2:
                    if not departure_city:
                        departure_city = gpe_entities[0]
                    if not arrival_city:
                        arrival_city = gpe_entities[1]
                # If we only found one GPE
                elif len(gpe_entities) == 1:
                    if "to" in doc.text.lower() and not arrival_city:
                        arrival_city = gpe_entities[0]
                    elif "from" in doc.text.lower() and not departure_city:
                        departure_city = gpe_entities[0]
                    # If no indicator, default to arrival (most common case)
                    elif not arrival_city:
                        arrival_city = gpe_entities[0]

        logger.info(f"Final cities - departure: {departure_city}, arrival: {arrival_city}")

        return departure_city, arrival_city


    def _extract_dates(self, doc, flight_type: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Extract departure and return dates"""
        departure_date = None
        return_date = None

        # getting all potential date texts (both from spaCy and text patterns)
        date_texts = []

        # getting dates from spaCy entities
        date_texts.extend([ent.text for ent in doc.ents if ent.label_ == "DATE"])

        # also try to parse any potential date expressions
        for token in doc:
            potential_date = parse_date_to_iso(token.text)
            if potential_date:
                date_texts.append(token.text)

        logger.debug(f"Found potential date expressions: {date_texts}")

        # try to parse dates in order found
        parsed_dates = []
        for date_text in date_texts:
            parsed_date = parse_date_to_iso(date_text)
            if parsed_date and parsed_date not in parsed_dates:
                parsed_dates.append(parsed_date)

        logger.debug(f"Successfully parsed dates: {parsed_dates}")

        # assign dates based on order and context
        if parsed_dates:
            departure_date = parsed_dates[0]
            if len(parsed_dates) > 1:
                return_date = parsed_dates[1]

        logger.info(f"Final dates - departure: {departure_date}, return: {return_date}")

        return departure_date, return_date


    def _extract_passengers(self, doc) -> Optional[str]:
        """Extract number of passengers"""
        for ent in doc.ents:
            if ent.label_ == "PASSENGERS":
                for token in ent:
                    if token.like_num:
                        try:
                            num = int(token.text)
                            if num > 0:
                                logger.info(f"Found {num} passengers")
                                return str(num)
                        except ValueError:
                            pass
        return None

    # TODO: extraction stuff logic could just be an LLM call 🤷‍♀️
    def extract_entities(self, text: str, domain: Dict[Text, Any]) -> Dict[str, Any]:
        forms = domain.get('forms', {})
        required_slots = forms.get('flight_booking_form', {}).get('required_slots', [])

        try:
            doc = nlp(text)
            extracted = {slot: None for slot in required_slots}

            # Extract all entities
            flight_type = self._extract_flight_type(doc)
            departure_city, arrival_city = self._extract_cities(doc)
            departure_date, return_date = self._extract_dates(doc, flight_type)
            num_passengers = self._extract_passengers(doc)

            # Update extracted dict with found values
            if "departure_city" in extracted:
                extracted["departure_city"] = departure_city
            if "arrival_city" in extracted:
                extracted["arrival_city"] = arrival_city
            if "departure_date" in extracted:
                extracted["departure_date"] = departure_date
            if "return_date" in extracted:
                extracted["return_date"] = return_date
            if "num_passengers" in extracted:
                extracted["num_passengers"] = num_passengers
            # if "flight_type" in extracted:
            #     extracted["flight_type"] = flight_type

            logger.info(f"Extracted entities: {extracted}")
            return extracted

        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return {slot: None for slot in required_slots}


    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            
        active_loop = tracker.active_loop.get('name')
        if active_loop:
            logger.debug(f"Skipping entity extraction - active form: {active_loop}")
            return []
            
        latest_message = tracker.latest_message.get('text')
        events = []
        forms = domain.get('forms', {})
        required_slots = forms.get('flight_booking_form', {}).get('required_slots', [])
        logger.info(f"Required slots: {required_slots}")

        # reset all slots at the start
        for slot in required_slots:
            events.append(SlotSet(slot, None))

        try:
            prompt = openai_client.create_flight_extraction_prompt(latest_message)
            # extracted_entities = self.extract_entities(latest_message, domain)
            extracted_entities = openai_client.get_completion(prompt)
            logger.info(f"Extracted entities: {extracted_entities}")

            # set slots and log each one
            for entity, value in extracted_entities.items():
                if entity in required_slots and value is not None:
                    logger.info(f"Setting slot {entity} = {value}")
                    events.append(SlotSet(entity, value))
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")

            # if extraction fails, just reset slots and return
            return events

        # Check missing slots based on extracted entities
        missing_slots = [slot for slot in required_slots
                    if slot not in extracted_entities or extracted_entities[slot] is None]
        logger.info(f"Missing slots: {missing_slots}")
        
        if missing_slots:
            logger.info("Activating form to collect missing slots")
            events.append(FollowupAction("flight_booking_form"))
        
        return events
    

class ValidateFlightBookingForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_flight_booking_form"

    def validate_departure_city(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate departure_city value."""
        logger.info(f"Validating departure_city with value: {slot_value}")
        current_value = tracker.get_slot('departure_city')
        logger.info(f"Current departure_city value: {current_value}")

        if current_value:
            logger.info(f"Keeping existing departure_city: {current_value}")
            return {"departure_city": current_value}

        latest_message = tracker.latest_message.get('text')
        logger.info(f"Latest message for departure_city: {latest_message}")
        
        if latest_message and slot_value:
            logger.info(f"Setting departure_city from text: {slot_value}")
            return {"departure_city": slot_value}
        
        logger.info("No valid departure_city found")
        return {"departure_city": None}

    def validate_arrival_city(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate arrival_city value."""
        logger.info(f"Validating arrival_city with value: {slot_value}")
        current_value = tracker.get_slot('arrival_city')
        logger.info(f"Current arrival_city value: {current_value}")

        if current_value:
            logger.info(f"Keeping existing arrival_city: {current_value}")
            return {"arrival_city": current_value}

        latest_message = tracker.latest_message.get('text')
        logger.info(f"Latest message for arrival_city: {latest_message}")
        
        if latest_message and slot_value:
            logger.info(f"Setting arrival_city from text: {slot_value}")
            return {"arrival_city": slot_value}
        
        logger.info("No valid arrival_city found")
        return {"arrival_city": None}

    def validate_departure_date(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate departure_date value."""
        logger.info(f"Validating departure_date with value: {slot_value}")
        current_value = tracker.get_slot('departure_date')
        logger.info(f"Current departure_date value: {current_value}")

        if current_value:
            logger.info(f"Keeping existing departure_date: {current_value}")
            return {"departure_date": current_value}

        latest_message = tracker.latest_message.get('text')
        logger.info(f"Latest message for departure_date: {latest_message}")
        
        if slot_value and slot_value != latest_message:
            try:
                parsed_date = parse_date_to_iso(slot_value)
                logger.info(f"Setting departure_date: {parsed_date}")
                return {"departure_date": parsed_date}
            except:
                logger.info("Failed to parse departure date")
                dispatcher.utter_message(text="Please provide a valid date (e.g., YYYY-MM-DD or 'next Friday')")
        
        logger.info("No valid departure_date found")
        return {"departure_date": None}

    def validate_return_date(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate return_date value."""
        logger.info(f"Validating return_date with value: {slot_value}")
        current_value = tracker.get_slot('return_date')
        logger.info(f"Current return_date value: {current_value}")

        if current_value:
            logger.info(f"Keeping existing return_date: {current_value}")
            return {"return_date": current_value}

        latest_message = tracker.latest_message.get('text')
        logger.info(f"Latest message for return_date: {latest_message}")
        
        if slot_value and slot_value != latest_message:
            try:
                parsed_date = parse_date_to_iso(slot_value)
                departure_date = tracker.get_slot('departure_date')
                logger.info(f"Checking return_date {parsed_date} against departure_date {departure_date}")
                
                if departure_date and parsed_date < departure_date:
                    logger.info("Return date before departure date")
                    dispatcher.utter_message(text="Return date must be after departure date")
                    return {"return_date": None}
                    
                logger.info(f"Setting return_date: {parsed_date}")
                return {"return_date": parsed_date}
            except:
                logger.info("Failed to parse return date")
                dispatcher.utter_message(text="Please provide a valid date (e.g., YYYY-MM-DD or 'next Friday')")
        
        logger.info("No valid return_date found")
        return {"return_date": None}

    def validate_num_passengers(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate num_passengers value."""
        logger.info(f"Validating num_passengers with value: {slot_value}")
        current_value = tracker.get_slot('num_passengers')
        logger.info(f"Current num_passengers value: {current_value}")

        if current_value:
            logger.info(f"Keeping existing num_passengers: {current_value}")
            return {"num_passengers": current_value}

        latest_message = tracker.latest_message.get('text')
        logger.info(f"Latest message for num_passengers: {latest_message}")
        
        if slot_value and slot_value != latest_message:
            try:
                num = int(slot_value)
                if num > 0:
                    logger.info(f"Setting num_passengers: {num}")
                    return {"num_passengers": str(num)}
                else:
                    logger.info("Invalid number of passengers (<=0)")
                    dispatcher.utter_message(text="Number of passengers must be greater than 0")
            except ValueError:
                logger.info("Failed to parse number of passengers")
                dispatcher.utter_message(text="Please provide a valid number of passengers")
        
        logger.info("No valid num_passengers found")
        return {"num_passengers": None}