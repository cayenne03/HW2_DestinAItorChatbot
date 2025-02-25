import os
import requests
import json
from typing import Dict, List, Optional

from mylogger import get_logger

logger = get_logger(__name__)
logger.debug("TripAdvisorAPI module loaded")

class TripAdvisorAPI:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize TripAdvisor API with API key"""
        self.api_key = api_key or os.getenv('TRIPADVISOR_API_KEY')
        if not self.api_key:
            raise ValueError("TripAdvisor API key is required")
        
        self.base_url = "https://api.content.tripadvisor.com/api/v1"
        self.headers = {"Accept": "application/json"}
        self.TA_CATEGORIES = set(["geos", "restaurants", "attractions", "hotels"])


    def get_location_ids(self, query: str, category: str = None) -> List[Dict]:
        """
        Search for a location by query string and return up to 3 most relevant location results.

        args:
            query: the location to search for.
            category: optional category to filter by (geos, restaurants, attractions, hotels).

        returns: list of dictionaries containing location_id, name, and address_string for up to 3 results.
        """
        if not query:
            raise ValueError("Search query cannot be empty")

        url = f"{self.base_url}/location/search"
        params = {
            'searchQuery': query,
            'language': 'en',
            'key': self.api_key,
            'radiusUnit': 'km',
        }

        if category and category in self.TA_CATEGORIES:
            params['category'] = category

        try:
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code != 200:
                logger.error(f"Error: API returned status code {response.status_code}")
                return []

            location_data = response.json()

            # check if results exist
            if not location_data.get('data'):
                return []

            # get the top 3 most relevant results
            top_results = location_data['data'][:3]

            # extract only the needed fields
            simplified_results = [
                {
                    "location_id": result.get("location_id"),
                    "name": result.get("name"),
                    "address_string": result.get("address_obj", {}).get("address_string")
                }
                for result in top_results
            ]

            return simplified_results

        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return []


    def get_location_details(self, location_id: str, category: str) -> Dict:
        """Get details for a specific location"""
        if not location_id:
            raise ValueError("Location ID is required")

        url = f"{self.base_url}/location/{location_id}/details"
        url = f"{url}?language=en&key={self.api_key}"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                logger.error(f"Error: API returned status code {response.status_code}")
                return None
            
            data = response.json()
            
            # parse based on category
            if category == "restaurants":
                return self.parse_restaurant_details(data)
            elif category == "attractions":
                return self.parse_attraction_details(data)
            elif category == "hotels":
                return self.parse_hotel_details(data)
            else:
                return self.parse_geos_details(data)
                
        except Exception as e:
            logger.error(f"Error fetching location details: {e}")
            return None


    def get_location_photos(self, location_id: str) -> Dict:
        """Get photos for a specific location"""
        if not location_id:
            raise ValueError("Location ID is required")

        url = f"{self.base_url}/location/{location_id}/photos"
        url = f"{url}?language=en&key={self.api_key}"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                logger.error(f"Error: API returned status code {response.status_code}")
                return None
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching photos: {e}")
            return None


    def parse_photos(self, location_id: str, limit: int = 5) -> List[str]:
        """Get small photo URLs for a location"""
        photos_data = self.get_location_photos(location_id)
        if not photos_data or "data" not in photos_data:
            return []
        
        return [photo["images"]["small"]["url"] 
                for photo in photos_data["data"][:limit] 
                if "images" in photo and "small" in photo["images"]]


    def create_google_maps_url(self, lat: str, long: str) -> str:
        """Create a Google Maps URL from latitude and longitude"""
        if not lat or not long:
            return None
        return f"https://www.google.com/maps/search/?api=1&query={lat},{long}"


    def parse_restaurant_details(self, data: Dict) -> Dict:
        """Parse restaurant details"""
        ancestors = []
        if "ancestors" in data:
            for ancestor in reversed(data["ancestors"]):
                ancestors.append(f"{ancestor['level']}::{ancestor['name']}::{ancestor['location_id']}")
        
        subratings = []
        if "subratings" in data:
            for _, rating in data["subratings"].items():
                subratings.append(f"{rating['value']} for {rating['localized_name']}")
        
        features_str = " | ".join(data.get("features", []))
        
        subcategories = [sub["localized_name"] for sub in data.get("subcategory", [])]
        subcategory_str = ", ".join(subcategories)
        
        return {
            "location_id": data.get("location_id"),
            "name": data.get("name"),
            "street": data.get("address_obj", {}).get("street1"),
            "city": data.get("address_obj", {}).get("city"),
            "state": data.get("address_obj", {}).get("state"),
            "country": data.get("address_obj", {}).get("country"),
            "postal_code": data.get("address_obj", {}).get("postalcode"),
            "ancestors": ancestors,
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "phone": data.get("phone"),
            "website": data.get("website"),
            "rating": data.get("rating"),
            "num_reviews": data.get("num_reviews"),
            "ranking_string": data.get("ranking_data", {}).get("ranking_string"),
            "subratings": subratings,
            "price_level": data.get("price_level"),
            "business_hours": "Business Hours:\n" + "\n".join(data.get("hours", {}).get("weekday_text", [])) if data.get("hours") else None,
            "features": features_str,
            "cuisine": " ".join([f"#{c['localized_name']}" for c in data.get("cuisine", [])]),
            "category": data.get("category", {}).get("localized_name"),
            "subcategory": subcategory_str,
            "google_maps_url": self.create_google_maps_url(data.get("latitude"), data.get("longitude")),
            "photos": self.parse_photos(data.get("location_id"))
        }


    def parse_attraction_details(self, data: Dict) -> Dict:
        """Parse attraction details"""
        ancestors = []
        if "ancestors" in data:
            for ancestor in reversed(data["ancestors"]):
                ancestors.append(f"{ancestor['level']}::{ancestor['name']}::{ancestor['location_id']}")
        
        attraction_types = []
        if "groups" in data:
            for group in data["groups"]:
                if "categories" in group:
                    attraction_types.extend(cat["localized_name"] for cat in group["categories"])
        attraction_types_str = ", ".join(attraction_types)
        
        return {
            "location_id": data.get("location_id"),
            "name": data.get("name"),
            "description": data.get("description"),
            "street": data.get("address_obj", {}).get("street1"),
            "city": data.get("address_obj", {}).get("city"),
            "country": data.get("address_obj", {}).get("country"),
            "postal_code": data.get("address_obj", {}).get("postalcode"),
            "ancestors": ancestors,
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "phone": data.get("phone"),
            "website": data.get("website"),
            "ranking_string": data.get("ranking_data", {}).get("ranking_string"),
            "rating": data.get("rating"),
            "num_reviews": data.get("num_reviews"),
            "business_hours": "Business Hours:\n" + "\n".join(data.get("hours", {}).get("weekday_text", [])) if data.get("hours") else None,
            "attraction_types": attraction_types_str,
            "google_maps_url": self.create_google_maps_url(data.get("latitude"), data.get("longitude")),
            "photos": self.parse_photos(data.get("location_id"))
        }


    def parse_hotel_details(self, data: Dict) -> Dict:
        """Parse hotel details"""
        ancestors = []
        if "ancestors" in data:
            for ancestor in reversed(data["ancestors"]):
                ancestors.append(f"{ancestor['level']}::{ancestor['name']}::{ancestor['location_id']}")
        
        subratings = []
        if "subratings" in data:
            for _, rating in data["subratings"].items():
                subratings.append(f"{rating['value']} for {rating['localized_name']}")
        
        amenities = ", ".join(data.get("amenities", [])[:10]) if data.get("amenities") else None
        
        return {
            "location_id": data.get("location_id"),
            "name": data.get("name"),
            "street": data.get("address_obj", {}).get("street1"),
            "city": data.get("address_obj", {}).get("city"),
            "country": data.get("address_obj", {}).get("country"),
            "postal_code": data.get("address_obj", {}).get("postalcode"),
            "ancestors": ancestors,
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "phone": data.get("phone"),
            "ranking_string": data.get("ranking_data", {}).get("ranking_string"),
            "rating": data.get("rating"),
            "num_reviews": data.get("num_reviews"),
            "subratings": subratings,
            "price_level": data.get("price_level"),
            "amenities": amenities,
            "google_maps_url": self.create_google_maps_url(data.get("latitude"), data.get("longitude")),
            "photos": self.parse_photos(data.get("location_id"))
        }


    def parse_geos_details(self, data: Dict) -> Dict:
        """Parse geographical location details"""
        ancestors = []
        if "ancestors" in data:
            for ancestor in reversed(data["ancestors"]):
                ancestors.append(f"{ancestor['level']}::{ancestor['name']}::{ancestor['location_id']}")
        
        return {
            "location_id": data.get("location_id"),
            "name": data.get("name"),
            "description": data.get("description"),
            "street": data.get("address_obj", {}).get("street1"),
            "city": data.get("address_obj", {}).get("city"),
            "country": data.get("address_obj", {}).get("country"),
            "postal_code": data.get("address_obj", {}).get("postalcode"),
            "ancestors": ancestors,
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
            "category": data.get("category", {}).get("name"),
            "subcategory": data.get("subcategory", [{}])[0].get("name") if data.get("subcategory") else None,
            "google_maps_url": self.create_google_maps_url(data.get("latitude"), data.get("longitude")),
            "photos": self.parse_photos(data.get("location_id"))
        }