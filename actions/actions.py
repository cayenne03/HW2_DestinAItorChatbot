from typing import Any, Text, Dict, List
from mylogger import get_logger

from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, FollowupAction, UserUtteranceReverted
from rasa_sdk.types import DomainDict

from mylogger import get_logger
from custom_models.flant5_classifier import FlanT5Classifier
from custom_models.spacy_nlp_md import nlp
from utils.date_utils import parse_date_to_iso


logger = get_logger(__name__)
logger.debug("Actions module loaded")

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
    def __init__(self):
        # fake values to simulate extractor missing some values
        self.extracted_fake = {
            "departure_city": "London",
            "arrival_city": "Rome",
            "departure_date": None,
            "return_date": None,
            "num_passengers": None
        }

    def name(self) -> Text:
        return "action_extract_flight_entities"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            
        active_loop = tracker.active_loop.get('name')
        if active_loop:
            logger.debug(f"Skipping entity extraction - active form: {active_loop}")
            return []
            
        latest_message = tracker.latest_message.get('text')
        extracted_entities = self.extracted_fake 
        
        events = []
        forms = domain.get('forms', {})
        required_slots = forms.get('flight_booking_form', {}).get('required_slots', [])
        logger.info(f"Required slots: {required_slots}")

        # log the extracted entities before setting slots
        logger.info(f"Extracted entities: {extracted_entities}")

        # set slots and log each one
        for entity, value in extracted_entities.items():
            if entity in required_slots and value is not None:
                logger.info(f"Setting slot {entity} = {value}")
                events.append(SlotSet(entity, value))
        
        # check which required slots are still empty
        current_slots = tracker.current_slot_values()
        logger.info(f"Current slots after setting: {current_slots}")
        missing_slots = [slot for slot in required_slots if not current_slots.get(slot)]
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