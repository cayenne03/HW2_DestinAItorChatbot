from typing import Any, Text, Dict, List
from mylogger import get_logger

from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted

from mylogger import get_logger
from custom_models.flant5_classifier import FlanT5Classifier


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

        logger.debug("Starting intent validation...")

        latest_message = tracker.latest_message.get('text')
        rasa_intent = tracker.latest_message.get('intent').get('name')
        rasa_confidence = tracker.latest_message.get('intent').get('confidence', 0.0)

        # get FlanT5 prediction
        prompt = self.classifier.create_prompt(latest_message)
        predicted_intent, confidence_score = self.classifier.classify(prompt)

        logger.info(f"User message: {latest_message}")
        logger.info(f"RASA Intent: {rasa_intent} (confidence: {rasa_confidence})")
        logger.info(f"FlanT5 Few-shot Intent: {predicted_intent} (confidence: {confidence_score})")
        # logger.info(f"Confidence Score: {confidence_score:.2f}")

        # TODO: check the out_of_scope and set some rules/fallback RASA vs FlanT5?
        #
        # if confidence_score > 0.95 and predicted_intent != rasa_intent:
        #     # return [{"intent": predicted_intent}]
        #     logger.info(f"Reverting to FlanT5 Few-shot Intent: {predicted_intent}")
        #     return [
        #         {"event": "revert_user_message"},
        #         {"event": "user", "intent": predicted_intent}
        #     ]

        return []
