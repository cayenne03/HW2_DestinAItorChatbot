from typing import Any, Text, Dict, List
import logging

from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionSessionStart(Action):
    def name(self) -> Text:
        return "action_session_start"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(response="utter_welcome_menu")

        logger.info("Session started....")

        return [SessionStarted()]