from typing import Any, Text, Dict, List, Optional, Tuple
import os
import json
from datetime import datetime

from dotenv import load_dotenv
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, FollowupAction, UserUtteranceReverted
from rasa_sdk.types import DomainDict

from mylogger import get_logger
from custom_models.flant5_classifier import FlanT5Classifier
from custom_models.spacy_nlp_md import nlp
from custom_models.city_area_extractor_ner import CityAreaExtractor
from utils.apis.openai_client_api import OpenAIClient
from utils.apis.amadeus_api import AmadeusAPI
from utils.apis.tripadvisor_api import TripAdvisorAPI
from utils.date_utils import parse_date_to_iso


load_dotenv()

logger = get_logger(__name__)
logger.debug("Actions module loaded")

openai_client = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))
amadeus = AmadeusAPI(client_id=os.getenv('AMADEUS_API_KEY'), client_secret=os.getenv('AMADEUS_API_SECRET'))
tripadvisor = TripAdvisorAPI()

city_extractor = CityAreaExtractor()

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

    # extraction stuff logic can just be an LLM call 🤷‍♀️
    def extract_entities(self, text: str, domain: Dict[Text, Any]) -> Dict[str, Any]:
        forms = domain.get('forms', {})
        required_slots = forms.get('flight_searching_form', {}).get('required_slots', [])

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
        required_slots = forms.get('flight_searching_form', {}).get('required_slots', [])
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

        # check missing slots based on extracted entities
        missing_slots = [slot for slot in required_slots
                    if slot not in extracted_entities or extracted_entities[slot] is None]
        logger.info(f"Missing slots: {missing_slots}")
        
        if missing_slots:
            logger.info("Activating form to collect missing slots")
            events.append(FollowupAction("flight_searching_form"))
        
        return events

class ActionExtractHotelEntities(Action):
    def name(self) -> Text:
        return "action_extract_hotel_entities"

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
        required_slots = forms.get('hotel_searching_form', {}).get('required_slots', [])
        logger.info(f"Required slots: {required_slots}")

        # reset all slots at the start
        for slot in required_slots:
            events.append(SlotSet(slot, None))

        try:
            # 1. extract city using the transformer-based NER (better for city names)
            transformer_city = city_extractor.extract_city(latest_message)
            logger.info(f"Transformer city: {transformer_city}")
            # 2. process with spaCy to find organizations and facilities
            doc = nlp(latest_message)

            # get the first ORG/FAC
            first_org_fac = next((ent.text for ent in doc.ents if ent.label_ in ["ORG", "FAC"]), None)
            logger.info(f"First ORG/FAC: {first_org_fac}")

            # get the first GPE/LOC (if transformer didn't find a city)
            first_gpe_loc = None
            if not transformer_city:
                first_gpe_loc = next((ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]), None)
                logger.info(f"First GPE/LOC: {first_gpe_loc}")

            # construct hotel_city_text based on available information
            hotel_city_text = None

            if transformer_city and first_org_fac:
                # if we have both city from transformer and facility from spaCy
                hotel_city_text = f"in {transformer_city} around {first_org_fac}"
                logger.info(f"1. Hotel city text: {hotel_city_text}")
            elif first_gpe_loc and first_org_fac:
                # if we have both location from spaCy and organization/facility
                hotel_city_text = f"in {first_gpe_loc} around {first_org_fac}"
                logger.info(f"2. Hotel city text: {hotel_city_text}")
            elif transformer_city:
                # if we only have city from transformer
                hotel_city_text = transformer_city
                logger.info(f"3. Hotel city text: {hotel_city_text}")
            elif first_gpe_loc:
                # if we only have location from spaCy
                hotel_city_text = first_gpe_loc
                logger.info(f"4. Hotel city text: {hotel_city_text}")
            elif first_org_fac:
                # if we only have organization/facility
                hotel_city_text = f"around {first_org_fac}"
                logger.info(f"5. Hotel city text: {hotel_city_text}")

            # only set the slot if we actually found an entity
            if "hotel_city" in required_slots and hotel_city_text is not None:
                logger.info(f"Setting slot hotel_city = {hotel_city_text}")
                events.append(SlotSet("hotel_city", hotel_city_text))

        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            # if extraction fails, just return events with reset slots
            return events

        # check which required slots are still missing
        # this determines if we need to activate the form for user input
        missing_slots = [slot for slot in required_slots
                        if tracker.get_slot(slot) is None]
        logger.info(f"Missing slots: {missing_slots}")

        # if there are missing required slots, activate the form to collect them from the user
        if missing_slots:
            logger.info("Activating form to collect missing slots")
            events.append(FollowupAction("hotel_searching_form"))

        return events


class ActionContinuePromptSearch(Action):
    def name(self) -> Text:
        return "action_continue_prompt_search"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # check which form was just completed
        previous_events = list(tracker.events)
        previous_form = None

        # find the most recent form that was active
        for event in reversed(previous_events):
            if event.get('event') == 'active_loop' and event.get('name') is None:
                # this indicates a form was deactivated
                for earlier_event in reversed(previous_events[:previous_events.index(event)]):
                    if earlier_event.get('event') == 'active_loop' and earlier_event.get('name'):
                        previous_form = earlier_event.get('name')
                        break
                break

        # get relevant slots based on which form was completed
        if previous_form == "flight_searching_form":
            city = tracker.get_slot("arrival_city")
            message = f"You can continue prompting me to explore flights, hotels and activities in {city} or any other city."
        elif previous_form == "hotel_searching_form":
            area = tracker.get_slot("hotel_city")
            message = f"You can continue prompting me like `hotels around or near {area}` or explore flights and activities in any other city/area."
        else:
            # default message if we can't determine the form
            message = "You can continue prompting me to explore flights, hotels and activities in any city."

        dispatcher.utter_message(text=message)
        return []


class ActionSearchFlights(Action):
    def name(self) -> Text:
        return "action_search_flights"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        departure_city = tracker.get_slot('departure_city')
        arrival_city = tracker.get_slot('arrival_city')
        departure_date = tracker.get_slot('departure_date')
        return_date = tracker.get_slot('return_date')
        num_passengers = tracker.get_slot('num_passengers')

        # convert to int if not already
        try:
            adults = int(num_passengers) if num_passengers else 1
        except ValueError:
            adults = 1

        if not all([departure_city, arrival_city, departure_date, return_date]):
            logger.info("Missing required information for flight search")
            dispatcher.utter_message(text="Sorry, I need all flight details to search for flights.")
            return []

        try:
            # get departure airports & arrival airports
            departure_prompt = openai_client.create_airport_prompt(departure_city)
            departure_data = openai_client.get_completion(departure_prompt)
            departure_airports = departure_data['airports'][:2]

            arrival_prompt = openai_client.create_airport_prompt(arrival_city)
            arrival_data = openai_client.get_completion(arrival_prompt)
            arrival_airports = arrival_data['airports'][:2]

            dispatcher.utter_message(text=f"🔍 Searching flights from {departure_city} to {arrival_city} for {adults} passenger{'' if adults == 1 else 's'}...")

            for dep_airport in departure_airports:
                for arr_airport in arrival_airports:
                    try:
                        two_way_response = amadeus.search_flights(
                            origin=dep_airport['IATA_CODE'],
                            destination=arr_airport['IATA_CODE'],
                            departure_date=departure_date,
                            return_date=return_date,
                            adults=adults
                        )

                        two_way_formatted = amadeus.parse_flight_offers(two_way_response, is_round_trip=True)

                        if two_way_formatted:
                            # header for this route
                            route_msg = f"\n✈️ Route: <u>{dep_airport['name']}</u> ({dep_airport['IATA_CODE']}) 🔄 <u>{arr_airport['name']}</u> ({arr_airport['IATA_CODE']})"
                            dispatcher.utter_message(text=route_msg)

                            # Since the message already comes formatted, we don't need to reformat the lines,
                            # just add the passengers info at the end
                            for offer in two_way_formatted:
                                # extract first line with price info
                                price_line = offer.split('\n')[0]

                                # add passenger count before "Price:"
                                if "Price:" in price_line:
                                    parts = price_line.split("Price:")
                                    price_line = f"{parts[0]}({adults} passenger{'' if adults == 1 else 's'}) Price:<b>{parts[1]}</b>"

                                    # rebuild offer with updated price line
                                    offer_lines = offer.split('\n')
                                    offer_lines[0] = price_line
                                    offer = '\n'.join(offer_lines)

                                # re-format the message with emojis & separator
                                flights = offer.split('\n')
                                if len(flights) >= 2:
                                    message = (
                                        f"🛫 Outbound: {flights[0]}\n"
                                        f"🛬 Return: {flights[1]}\n"
                                        f"{'_' * 40}"
                                    )
                                    dispatcher.utter_message(text=message)
                                else:
                                    dispatcher.utter_message(text=offer)
                        else:
                            dispatcher.utter_message(text=f"No flights found between <u>{dep_airport['name']}</u> ({dep_airport['IATA_CODE']}) 🔄 <u>{arr_airport['name']}</u> ({arr_airport['IATA_CODE']})")

                    except Exception as e:
                        logger.error(f"Error searching flights for {dep_airport['IATA_CODE']} 🔄 {arr_airport['IATA_CODE']}: {e}")
                        dispatcher.utter_message(text=f"❌ Couldn't find flights from {dep_airport['name']} to {arr_airport['name']}")

            return []

        except Exception as e:
            logger.error(f"Error in flight search: {e}")
            dispatcher.utter_message(text="❌ Sorry, I encountered an error while searching for flights.")
            return []


class ActionSearchHotels(Action):
    def name(self) -> Text:
        return "action_search_hotels"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        hotel_city = tracker.get_slot('hotel_city')

        if not hotel_city:
            logger.info("Missing required city information for hotel search")
            dispatcher.utter_message(text="Sorry, I need a city to search for hotels.")
            return []

        try:
            # get location ids
            dispatcher.utter_message(text=f"🔍 Searching hotels using the term: <b>{hotel_city}</b> ...")
            location_ids = tripadvisor.get_location_ids(query=hotel_city, category="hotels")

            # log safely...
            log_entries = []
            for loc in location_ids:
                log_entries.append(f"id: {loc['location_id']} | name: {loc['name']} | address: {loc['address_string']}")
            logger.info(f"Location IDs: {' | '.join(log_entries)}")

            # no hotels found
            if not location_ids:
                dispatcher.utter_message(text=f"❌ Sorry, I couldn't find any hotels using the above term. Try another one.")
                return []

            # get hotel details for each location (our limit is up to 3 anyways)
            found_hotels = False

            for loc_id in location_ids[:3]:
                try:
                    hotel_details = tripadvisor.get_location_details(location_id=loc_id['location_id'], category="hotels")

                    if hotel_details:
                        found_hotels = True

                        # create hotel message with relevant information - ONE ITEM PER LINE
                        hotel_info = f"🏨 <b>{hotel_details.get('name', 'Hotel')}</b>\n\n"

                        # Address
                        address_parts = []
                        if hotel_details.get('street'):
                            address_parts.append(hotel_details['street'])
                        if hotel_details.get('city'):
                            address_parts.append(hotel_details['city'])
                        if hotel_details.get('country'):
                            address_parts.append(hotel_details['country'])
                        if hotel_details.get('postal_code'):
                            address_parts.append(hotel_details['postal_code'])

                        if address_parts:
                            hotel_info += f"📍 {', '.join(address_parts)}\n"

                        if hotel_details.get('rating'):
                            reviews_text = f" ({hotel_details.get('num_reviews', '')} reviews)" if hotel_details.get('num_reviews') else ""
                            hotel_info += f"⭐ {hotel_details['rating']}/5{reviews_text}\n"

                        if hotel_details.get('ranking_string'):
                            hotel_info += f"🏆 {hotel_details['ranking_string']}\n"

                        if hotel_details.get('price_level'):
                            price_level = hotel_details.get('price_level', '')
                            # Count dollar signs and replace with same number of euro emojis
                            euro_count = price_level.count('$')
                            euro_symbols = '💲' * euro_count
                            hotel_info += f"💰 {euro_symbols}\n"

                        if hotel_details.get('subratings') and isinstance(hotel_details['subratings'], list):
                            subratings_text = []
                            for rating in hotel_details['subratings'][:4]:
                                subratings_text.append(rating)

                            hotel_info += f"📊 {' | '.join(subratings_text)}\n"

                        if hotel_details.get('amenities'):
                            amenities_list = []
                            if isinstance(hotel_details['amenities'], str):
                                amenities_list = hotel_details['amenities'].split(', ')
                            elif isinstance(hotel_details['amenities'], list):
                                amenities_list = hotel_details['amenities']

                            if amenities_list:
                                top_amenities = amenities_list[:5]
                                hotel_info += f"✨ {', '.join(top_amenities)}\n"

                        if hotel_details.get('google_maps_url'):
                            hotel_info += f"🗺️ <a href='{hotel_details['google_maps_url']}'>View on Google Maps</a>"

                        # dispatch the message and add separator
                        dispatcher.utter_message(text=hotel_info)
                        dispatcher.utter_message(text=f"{'_' * 40}")

                except Exception as e:
                    logger.error(f"Error getting details for hotel ID {loc_id['location_id']}: {e}")
                    import traceback
                    logger.debug(f"Detailed error: {traceback.format_exc()}")
                    continue

            if not found_hotels:
                dispatcher.utter_message(text=f"❌ Sorry, I couldn't retrieve details for hotels in {hotel_city}.")

            return []

        except Exception as e:
            logger.error(f"Error in hotel search: {e}")
            dispatcher.utter_message(text=f"❌ Sorry, I encountered an error while searching for hotels in {hotel_city}.")
            return []


class ValidateFlightSearchingForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_flight_searching_form"


    def _validate_city(
        self,
        city: Text,
        slot_name: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker
    ) -> Optional[str]:
        """Validate and format city name.
        Returns formatted city name if valid, None if invalid."""
        if not city or not city.strip():
            logger.info(f"Empty {slot_name} value")
            return None

        formatted_value = " ".join(word.capitalize() for word in city.strip().split())
        doc = nlp(formatted_value)

        if not any(ent.label_ == "GPE" for ent in doc.ents):
            logger.info(f"{formatted_value} is not recognized as a GPE valid city ({slot_name})")
            return None

        # check for duplicate cities
        other_slot = 'arrival_city' if slot_name == 'departure_city' else 'departure_city'
        other_city = tracker.get_slot(other_slot)
        if other_city and other_city.lower() == formatted_value.lower():
            logger.info(f"Duplicate cities: {formatted_value} already set as {other_slot}")
            dispatcher.utter_message(text="The departure and arrival cities cannot be the same.")
            return None

        return formatted_value


    def validate_departure_city(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate departure_city value."""
        logger.info(f"Validating departure_city with value: {slot_value}")

        # 1st check existing value
        current_value = tracker.get_slot('departure_city')
        if current_value:
            logger.info(f"Found existing departure_city: {current_value}")
            validated_city = self._validate_city(current_value, "departure_city", dispatcher, tracker)
            if validated_city:
                logger.info(f"Keeping valid existing departure_city: {validated_city}")
                return {"departure_city": validated_city}
            logger.info("Invalid existing departure_city")
            dispatcher.utter_message(text="I need a valid city name for departure.")
            return {"departure_city": None}

        # then validate new input
        if slot_value:
            validated_city = self._validate_city(slot_value, "departure_city", dispatcher, tracker)
            if validated_city:
                logger.info(f"Setting validated departure_city: {validated_city}")
                return {"departure_city": validated_city}
            logger.info("Invalid departure_city input")
            dispatcher.utter_message(text="Please provide a valid city name for departure.")
        
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

        # 1st check existing value
        current_value = tracker.get_slot('arrival_city')
        if current_value:
            logger.info(f"Found existing arrival_city: {current_value}")
            validated_city = self._validate_city(current_value, "arrival_city", dispatcher, tracker)
            if validated_city:
                logger.info(f"Keeping valid existing arrival_city: {validated_city}")
                return {"arrival_city": validated_city}
            logger.info("Invalid existing arrival_city")
            dispatcher.utter_message(text="I need a valid city name for arrival.")
            return {"arrival_city": None}

        # then validate new input
        if slot_value:
            validated_city = self._validate_city(slot_value, "arrival_city", dispatcher, tracker)
            if validated_city:
                logger.info(f"Setting validated arrival_city: {validated_city}")
                return {"arrival_city": validated_city}
            logger.info("Invalid arrival_city input")
            dispatcher.utter_message(text="Please provide a valid city name for arrival.")
        
        return {"arrival_city": None}


    def _validate_date(
        self,
        date_value: Any,
        slot_name: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker
    ) -> Optional[str]:
        """Validate and format date.
        Returns ISO formatted date string if valid, None if invalid."""

        if not date_value or not str(date_value).strip():
            logger.info(f"Empty {slot_name} value")
            return None

        try:
            # 1st try to parse to ISO
            parsed_date_str = parse_date_to_iso(str(date_value))
            if not parsed_date_str:  # parse_date_to_iso returns None on failure
                logger.info(f"Failed to parse date: {date_value}")
                dispatcher.utter_message(text="Please provide a valid date (e.g., YYYY-MM-DD or 'next Friday')")
                return None

            # convert to datetime for comparisons
            parsed_date = datetime.strptime(parsed_date_str, "%Y-%m-%d").date()

            # then check if it's a DATE using spacy
            doc = nlp(str(date_value))
            if not any(ent.label_ == "DATE" for ent in doc.ents):
                logger.info(f"{date_value} is not recognized as a valid date ({slot_name})")
                dispatcher.utter_message(text="Please provide a valid date (e.g., YYYY-MM-DD or 'next Friday')")
                return None

            # check date constraints
            departure_date = tracker.get_slot('departure_date')
            return_date = tracker.get_slot('return_date')

            if departure_date:
                departure_date = datetime.strptime(parse_date_to_iso(departure_date), "%Y-%m-%d").date()
            if return_date:
                return_date = datetime.strptime(parse_date_to_iso(return_date), "%Y-%m-%d").date()

            if slot_name == 'departure_date' and return_date and parsed_date > return_date:
                logger.info(f"Departure date {parsed_date_str} is after return date {return_date}")
                dispatcher.utter_message(text="Departure date must be before or on the return date")
                return None
            elif slot_name == 'return_date' and departure_date and parsed_date < departure_date:
                logger.info(f"Return date {parsed_date_str} is before departure date {departure_date}")
                dispatcher.utter_message(text="Return date must be after or on the departure date")
                return None

            return parsed_date_str

        except Exception as e:
            logger.info(f"Failed to parse date: {e}")
            dispatcher.utter_message(text="Please provide a valid date (e.g., YYYY-MM-DD or 'next Friday')")
            return None


    def validate_departure_date(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate departure_date value."""
        logger.info(f"Validating departure_date with value: {slot_value}")

        # 1st check existing value
        current_value = tracker.get_slot('departure_date')
        if current_value:
            logger.info(f"Found existing departure_date: {current_value}")
            validated_date = self._validate_date(current_value, "departure_date", dispatcher, tracker)
            if validated_date:
                logger.info(f"Keeping valid existing departure_date: {validated_date}")
                return {"departure_date": validated_date}
            logger.info("Invalid existing departure_date")
            return {"departure_date": None}

        # then validate new input
        if slot_value:
            validated_date = self._validate_date(slot_value, "departure_date", dispatcher, tracker)
            if validated_date:
                logger.info(f"Setting validated departure_date: {validated_date}")
                return {"departure_date": validated_date}
            logger.info("Invalid departure_date input")
        
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

        # 1st check existing value
        current_value = tracker.get_slot('return_date')
        if current_value:
            logger.info(f"Found existing return_date: {current_value}")
            validated_date = self._validate_date(current_value, "return_date", dispatcher, tracker)
            if validated_date:
                logger.info(f"Keeping valid existing return_date: {validated_date}")
                return {"return_date": validated_date}
            logger.info("Invalid existing return_date")
            return {"return_date": None}

        # then validate new input
        if slot_value:
            validated_date = self._validate_date(slot_value, "return_date", dispatcher, tracker)
            if validated_date:
                logger.info(f"Setting validated return_date: {validated_date}")
                return {"return_date": validated_date}
            logger.info("Invalid return_date input")
        
        return {"return_date": None}


    def _validate_passengers(
        self,
        value: Any,
        slot_name: Text,
        dispatcher: CollectingDispatcher
    ) -> Optional[int]:
        """Validate passenger number.
        Returns validated integer if valid, None if invalid."""
        if not value:
            logger.info("Empty passenger value")
            return None

        try:
            num = int(str(value).strip())
            if not 0 < num <= 5:
                logger.info(f"Passenger number {num} outside valid range (1-5)")
                dispatcher.utter_message(text="Number of passengers must be between 1 and 5")
                return None
            return num
        except ValueError:
            logger.info(f"Failed to parse passenger number: {value}")
            dispatcher.utter_message(text="Please provide a valid number of passengers (1-5)")
            return None


    def validate_num_passengers(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate num_passengers value."""
        logger.info(f"Validating num_passengers with value: {slot_value}")

        # 1st check existing value
        current_value = tracker.get_slot('num_passengers')
        if current_value:
            logger.info(f"Found existing num_passengers: {current_value}")
            validated_num = self._validate_passengers(current_value, "num_passengers", dispatcher)
            if validated_num:
                logger.info(f"Keeping valid existing num_passengers: {validated_num}")
                return {"num_passengers": validated_num}
            logger.info("Invalid existing num_passengers")
            return {"num_passengers": None}

        # then validate new input
        if slot_value:
            validated_num = self._validate_passengers(slot_value, "num_passengers", dispatcher)
            if validated_num:
                logger.info(f"Setting validated num_passengers: {validated_num}")
                return {"num_passengers": validated_num}
            logger.info("Invalid num_passengers input")

        return {"num_passengers": None}


###########################################################
# TODO:
#
# 1) Remove validators away from here
# 2) Turn to extract_* functions for speed instead of LLM
# 3) Use a DB for frequent queries (airports, cities, etc.)
###########################################################