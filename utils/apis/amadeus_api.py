from typing import Dict, Optional, List
import requests


class AmadeusAPI:
    """Handles Amadeus API interactions for flight searches."""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        self.search_url = "https://test.api.amadeus.com/v2/shopping/flight-offers"


    def get_access_token(self) -> str:
        """Get Amadeus API access token."""
        auth_data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        auth_response = requests.post(self.auth_url, data=auth_data)
        auth_response.raise_for_status()

        return auth_response.json()["access_token"]


    def search_flights(
        self,
        origin: str,
        destination: str, 
        departure_date: str,
        return_date: Optional[str] = None,
        adults: int = 1,
    ) -> Dict:
        """Search flight offers using Amadeus API."""
        try:
            access_token = self.get_access_token()
            headers = {"Authorization": f"Bearer {access_token}"}
            
            params = {
                "originLocationCode": origin,
                "destinationLocationCode": destination,
                "departureDate": departure_date,
                "adults": adults,
                "max": 3,
                "nonStop": "true"
            }
            
            if return_date:
                params["returnDate"] = return_date
            
            search_response = requests.get(self.search_url, headers=headers, params=params)
            search_response.raise_for_status()
            
            return search_response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            raise


    @staticmethod
    def parse_flight_offers(response: Dict, is_round_trip: bool = False) -> List[str]:
        """Parse Amadeus flight offers into formatted strings."""
        formatted_offers = []
        carriers = response.get('dictionaries', {}).get('carriers', {})
        
        for offer in response.get('data', []):
            output_lines = []
            itineraries = offer.get('itineraries', [])
            price = offer.get('price', {})
            
            # outbound flight
            outbound = itineraries[0]['segments'][0]
            carrier_name = carriers.get(outbound['carrierCode'], outbound['carrierCode'])
            
            departure_time = outbound['departure']['at'].replace('T', ' ')[:16]
            arrival_time = outbound['arrival']['at'].replace('T', ' ')[:16]
            
            outbound_line = (
                f"({departure_time}) "
                f"{outbound['departure']['iataCode']} ➡️ "
                f"{outbound['arrival']['iataCode']} "
                f"({arrival_time}), "
                f"Duration: {itineraries[0]['duration'][2:-1].replace('H', 'h ')}m, "
                f"Carrier: {carrier_name}, "
                f"{'Total' if is_round_trip else ''} Price: {price['total']} {price['currency']}"
            )
            output_lines.append(outbound_line)
            
            # return flight (if exists and is round trip)
            if is_round_trip and len(itineraries) > 1:
                return_segment = itineraries[1]['segments'][0]
                return_carrier = carriers.get(return_segment['carrierCode'], return_segment['carrierCode'])
                
                return_departure = return_segment['departure']['at'].replace('T', ' ')[:16]
                return_arrival = return_segment['arrival']['at'].replace('T', ' ')[:16]
                
                return_line = (
                    f"({return_departure}) "
                    f"{return_segment['departure']['iataCode']} ➡️ "
                    f"{return_segment['arrival']['iataCode']} "
                    f"({return_arrival}), "
                    f"Duration: {itineraries[1]['duration'][2:-1].replace('H', 'h ')}m, "
                    f"Carrier: {return_carrier}"
                )
                output_lines.append(return_line)
            
            formatted_offers.append('\n'.join(output_lines))
        
        return formatted_offers